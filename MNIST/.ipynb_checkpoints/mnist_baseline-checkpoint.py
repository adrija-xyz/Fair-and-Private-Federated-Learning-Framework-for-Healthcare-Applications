# baseline_random.py
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from client import CNN
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch.nn.functional as F
import pandas as pd

# ---------------------------- Run tag & outputs ---------------------------- #
SCENARIO = "Baseline-Random"  # <-- label used in saved CSVs
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CLASSWISE_CSV = os.path.join(RESULTS_DIR, f"classwise_{SCENARIO}.csv")
MACRO_CSV = os.path.join(RESULTS_DIR, f"macro_{SCENARIO}.csv")

def append_df_to_csv(path, df):
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))

# ---------------------------- Dataset prep ---------------------------- #
def create_noniid_shards(dataset, num_clients=10, shards_per_client=2, total_samples=60000):
    labels = np.array(dataset.targets)
    shuffled_indices = np.random.permutation(len(labels))
    selected_indices = shuffled_indices[:total_samples]
    sorted_indices = selected_indices[np.argsort(labels[selected_indices])]
    num_shards = num_clients * shards_per_client
    shard_size = total_samples // num_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)
    client_indices = [[] for _ in range(num_clients)]
    shard_idx = 0
    for client_id in range(num_clients):
        for _ in range(shards_per_client):
            client_indices[client_id].extend(shards[shard_idx])
            shard_idx += 1
    return [Subset(dataset, indices) for indices in client_indices]

# (Optional) Hybrid partition with variable shards; not used by default
def create_hybrid_partition_variable_shards(
    dataset, num_clients=100, iid_fraction=0.4, min_shards=1, max_shards=5, total_samples=60000
):
    num_iid_clients = int(num_clients * iid_fraction)
    num_non_iid_clients = num_clients - num_iid_clients
    samples_per_client = total_samples // num_clients
    all_indices = np.random.permutation(len(dataset))[:total_samples]

    iid_indices = all_indices[:num_iid_clients * samples_per_client]
    iid_client_indices = [
        iid_indices[i * samples_per_client:(i + 1) * samples_per_client].tolist()
        for i in range(num_iid_clients)
    ]

    non_iid_indices = all_indices[num_iid_clients * samples_per_client:]
    non_iid_labels = np.array(dataset.targets)[non_iid_indices]
    sorted_indices = non_iid_indices[np.argsort(non_iid_labels)]

    shard_counts = np.random.randint(min_shards, max_shards + 1, size=num_non_iid_clients)
    total_shards = shard_counts.sum()
    shard_size = len(sorted_indices) // total_shards

    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(total_shards)]
    np.random.shuffle(shards)

    non_iid_client_indices = []
    shard_ptr = 0
    for count in shard_counts:
        client_shards = shards[shard_ptr:shard_ptr + count]
        merged_indices = [idx for shard in client_shards for idx in shard]
        non_iid_client_indices.append(merged_indices)
        shard_ptr += count

    all_client_indices = iid_client_indices + non_iid_client_indices
    client_datasets = [Subset(dataset, indices) for indices in all_client_indices]
    return client_datasets

# ---------------------------- Data ---------------------------- #
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)

num_clients = 20
client_datasets = create_noniid_shards(
    mnist_train, num_clients=num_clients, shards_per_client=6, total_samples=60000
)

# Small aux set for fast eval on clients (10 samples/class)
aux_indices = []
labels_test = np.array(mnist_test.targets)
for cls in range(10):
    aux_indices.extend(np.where(labels_test == cls)[0][:10])
aux_loader = DataLoader(Subset(mnist_test, aux_indices), batch_size=32, shuffle=False)

# ---------------------------- Metrics helpers ---------------------------- #
def compute_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1

# ---------------------------- Strategy: Random Client Selection ---------------------------- #
class BaselineRandom(FedAvg):
    """FedAvg with purely random client selection (no MAB, no DP)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_participation = {}
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        agg_result = super().aggregate_fit(server_round, results, failures)
        if agg_result:
            self.final_parameters = agg_result[0]
        return agg_result

    def configure_fit(self, server_round, parameters, client_manager):
        # Plain random sampling each round
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        for client in clients:
            cid = str(client.cid)
            self.client_participation[cid] = self.client_participation.get(cid, 0) + 1
        return [(client, FitIns(parameters, {})) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_evaluate_clients,
        )
        return [(client, EvaluateIns(parameters, {"server_round": server_round})) for client in clients]

# ---------------------------- Flower Client ---------------------------- #
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, strategy):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.strategy = strategy

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # FedProx-like proximal term to stabilize (no DP)
        global_params = [param.clone().detach() for param in self.model.parameters()]
        mu = 0.01

        for data, target in self.train_loader:
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            prox_reg = sum(((param - gp) ** 2).sum() for param, gp in zip(self.model.parameters(), global_params))
            loss = loss + (mu / 2) * prox_reg
            loss.backward()
            optimizer.step()

        updated_params = self.get_parameters()
        return updated_params, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        y_true, y_pred = [], []
        loss_sum, total = 0.0, 0

        with torch.no_grad():
            for data, target in aux_loader:
                output = self.model(data)
                loss_sum += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                y_true.extend(target.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
                total += target.size(0)

        avg_loss = loss_sum / max(total, 1)

        # Macro metrics
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")
        # Class-wise metrics
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        round_id = int(config.get("server_round", -1)) if config else -1

        # Write macro row
        macro_df = pd.DataFrame([{
            "scenario": SCENARIO, "round": round_id, "client_id": int(self.cid),
            "loss": float(avg_loss), "accuracy": float(acc),
            "precision": float(prec), "recall": float(rec), "f1": float(f1)
        }])
        append_df_to_csv(MACRO_CSV, macro_df)

        # Write class-wise rows
        class_rows = []
        for c in range(len(prec_arr)):
            class_rows.append({
                "scenario": SCENARIO, "round": round_id, "client_id": int(self.cid),
                "class": c, "precision": float(prec_arr[c]),
                "recall": float(rec_arr[c]), "f1": float(f1_arr[c])
            })
        append_df_to_csv(CLASSWISE_CSV, pd.DataFrame(class_rows))

        print(f"[Baseline-Random] Client {self.cid} r{round_id} "
              f"loss: {avg_loss:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}")

        return avg_loss, total, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------------------- Client factory ---------------------------- #
def client_fn(cid: str):
    model = CNN()
    train_loader = DataLoader(client_datasets[int(cid)], batch_size=32, shuffle=True)
    return FlowerClient(cid, model, train_loader, strategy).to_client()

# ---------------------------- Run ---------------------------- #
if __name__ == "__main__":
    strategy = BaselineRandom(
        fraction_fit=0.2,            # maintains same eval cadence as your MAB run
        min_fit_clients=2,
        min_available_clients=num_clients,
        fraction_evaluate=0.2,
        min_evaluate_clients=2,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=50),
        client_resources={"num_cpus": 1},
        strategy=strategy
    )

    print(f"\nClient Participation Counts: {strategy.client_participation}")

    # -------- Final global model evaluation + per-class CSV --------
    print("\n--- Evaluating global model ---")
    if strategy.final_parameters is not None:
        final_parameters = parameters_to_ndarrays(strategy.final_parameters)
    else:
        raise RuntimeError("No final global parameters found in strategy.")

    global_model = CNN()
    state_dict = global_model.state_dict()
    for k, v in zip(state_dict.keys(), final_parameters):
        state_dict[k] = torch.tensor(v)
    global_model.load_state_dict(state_dict)

    global_model.eval()
    y_true, y_pred = [], []
    loss_sum, total = 0.0, 0
    with torch.no_grad():
        for data, target in DataLoader(mnist_test, batch_size=64):
            output = global_model(data)
            loss_sum += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            y_true.extend(target.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            total += target.size(0)

    avg_loss = loss_sum / total
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    print(f"\nGlobal test -> loss: {avg_loss:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}")

    global_cw = pd.DataFrame({
        "scenario": [SCENARIO]*len(prec_arr),
        "round": [50]*len(prec_arr),   # final round tag (adjust if needed)
        "class": list(range(len(prec_arr))),
        "precision": prec_arr,
        "recall": rec_arr,
        "f1": f1_arr
    })
    append_df_to_csv(os.path.join(RESULTS_DIR, f"global_classwise_{SCENARIO}.csv"), global_cw)