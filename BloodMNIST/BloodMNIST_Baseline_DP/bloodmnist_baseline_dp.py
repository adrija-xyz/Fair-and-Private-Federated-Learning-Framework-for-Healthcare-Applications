import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, parameters_to_ndarrays

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from medmnist import BloodMNIST

num_classes = 8
num_clients = 20

SCENARIO = "BloodMNIST-BaselineRandom-DP"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
MACRO_CSV = os.path.join(RESULTS_DIR, f"macro_{SCENARIO}.csv")
CLASSWISE_CSV = os.path.join(RESULTS_DIR, f"classwise_{SCENARIO}.csv")
GLOBAL_CLASSWISE_CSV = os.path.join(RESULTS_DIR, f"global_classwise_{SCENARIO}.csv")

def append_df_to_csv(path, df):
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 3-channel
])

train_dataset = BloodMNIST(split='train', transform=transform, download=True)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True)

def get_labels(ds):
    return np.array(ds.labels).squeeze()

def compute_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return acc, prec, rec, f1

def create_hybrid_partition_variable_shards(
    dataset, num_clients=20, iid_fraction=0.4, min_shards=1, max_shards=5, total_samples=None
):
    labels_all = get_labels(dataset)
    all_idx = np.arange(len(labels_all))
    if total_samples is not None:
        all_idx = np.random.permutation(all_idx)[:total_samples]

    num_iid = int(num_clients * iid_fraction)
    num_non_iid = num_clients - num_iid
    samples_per_client = len(all_idx) // num_clients

    iid_block = all_idx[:num_iid * samples_per_client]
    iid_clients = [
        iid_block[i*samples_per_client:(i+1)*samples_per_client].tolist()
        for i in range(num_iid)
    ]

    non_iid_block = all_idx[num_iid * samples_per_client:]
    non_iid_labels = labels_all[non_iid_block]
    sorted_idx = non_iid_block[np.argsort(non_iid_labels)]

    shard_counts = np.random.randint(min_shards, max_shards + 1, size=num_non_iid)
    total_shards = int(shard_counts.sum())
    shard_size = len(sorted_idx) // max(total_shards, 1)
    shards = [sorted_idx[i*shard_size:(i+1)*shard_size] for i in range(total_shards)]
    np.random.shuffle(shards)

    non_iid_clients, ptr = [], 0
    for c in shard_counts:
        merged = []
        for _ in range(c):
            if ptr < len(shards):
                merged.extend(shards[ptr]); ptr += 1
        non_iid_clients.append(merged)

    all_client_indices = iid_clients + non_iid_clients
    return [Subset(dataset, idxs) for idxs in all_client_indices]

client_datasets = create_hybrid_partition_variable_shards(
    train_dataset,
    num_clients=num_clients,
    iid_fraction=0.4,
    min_shards=1,
    max_shards=5,
    total_samples=None
)

NOISE_STD = 0.001
MAX_CLIP_NORM = 5.0

class BaselineRandomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None
        self.client_participation = {}

    def aggregate_fit(self, server_round, results, failures):
        agg_result = super().aggregate_fit(server_round, results, failures)
        if agg_result:
            self.final_parameters = agg_result[0]
        return agg_result

    def configure_fit(self, server_round, parameters, client_manager):
        available = client_manager.all()
        num_available = len(available)
        n_sample = max(self.min_fit_clients, int(np.ceil(self.fraction_fit * num_available)))
        clients = client_manager.sample(num_clients=n_sample, min_num_clients=self.min_fit_clients)

        for c in clients:
            cid = str(c.cid)
            self.client_participation[cid] = self.client_participation.get(cid, 0) + 1

        return [(c, FitIns(parameters, {})) for c in clients]

from client import CNN

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        sd = self.model.state_dict()
        for k, v in zip(sd.keys(), parameters):
            sd[k] = torch.tensor(v)
        self.model.load_state_dict(sd)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)

        mu = 0.01
        global_params = [p.clone().detach() for p in self.model.parameters()]

        for data, target in self.train_loader:
            target = target.squeeze().long()
            opt.zero_grad()
            out = self.model(data)
            loss = F.cross_entropy(out, target)
            prox = sum(((p - gp) ** 2).sum() for p, gp in zip(self.model.parameters(), global_params))
            (loss + (mu/2)*prox).backward()
            opt.step()

        updated_params = self.get_parameters()
        total_norm = np.sqrt(sum(np.sum(p**2) for p in updated_params))
        clip_coef = min(1.0, MAX_CLIP_NORM / (total_norm + 1e-6))
        clipped_params = [p * clip_coef for p in updated_params]
        noisy_params = [p + np.random.normal(0, NOISE_STD, p.shape) for p in clipped_params]

        return noisy_params, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        y_true, y_pred = [], []
        loss_sum, total = 0.0, 0

        with torch.no_grad():
            for data, target in DataLoader(test_dataset, batch_size=64, shuffle=False):
                target = target.squeeze().long()
                out = self.model(data)
                loss_sum += F.cross_entropy(out, target, reduction='sum').item()
                pred = out.argmax(dim=1)
                y_true.extend(target.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
                total += target.size(0)

        avg_loss = loss_sum / max(total, 1)
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")

        round_id = int((config or {}).get("server_round", -1))
        macro_df = pd.DataFrame([{
            "scenario": SCENARIO, "round": round_id, "client_id": int(self.cid),
            "loss": float(avg_loss), "accuracy": float(acc),
            "precision": float(prec), "recall": float(rec), "f1": float(f1)
        }])
        append_df_to_csv(MACRO_CSV, macro_df)

        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        class_rows = []
        for c in range(len(prec_arr)):
            class_rows.append({
                "scenario": SCENARIO, "round": round_id, "client_id": int(self.cid),
                "class": c, "precision": float(prec_arr[c]),
                "recall": float(rec_arr[c]), "f1": float(f1_arr[c])
            })
        append_df_to_csv(CLASSWISE_CSV, pd.DataFrame(class_rows))

        print(f"[{SCENARIO}] Client {self.cid} r{round_id} "
              f"loss: {avg_loss:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}")

        return avg_loss, total, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def client_fn(cid: str):
    model = CNN()
    train_loader = DataLoader(client_datasets[int(cid)], batch_size=32, shuffle=True)
    return FlowerClient(cid, model, train_loader).to_client()

if __name__ == "__main__":
    strategy = BaselineRandomFedAvg(
        fraction_fit=0.2,
        min_fit_clients=2,
        min_available_clients=num_clients,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=500),
        client_resources={"num_cpus": 1},
        strategy=strategy
    )

    print(f"\nClient Participation Counts: {strategy.client_participation}")

    print("\n--- Evaluating global model (BloodMNIST test) ---")
    if strategy.final_parameters is None:
        raise RuntimeError("No final global parameters found in strategy.")
    final_parameters = parameters_to_ndarrays(strategy.final_parameters)

    from client import CNN
    global_model = CNN()
    sd = global_model.state_dict()
    for k, v in zip(sd.keys(), final_parameters):
        sd[k] = torch.tensor(v)
    global_model.load_state_dict(sd)

    global_model.eval()
    y_true, y_pred = [], []
    loss_sum, total = 0.0, 0
    with torch.no_grad():
        for data, target in DataLoader(test_dataset, batch_size=64, shuffle=False):
            target = target.squeeze().long()
            out = global_model(data)
            loss_sum += F.cross_entropy(out, target, reduction='sum').item()
            pred = out.argmax(dim=1)
            y_true.extend(target.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            total += target.size(0)

    avg_loss = loss_sum / max(total, 1)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")
    print(f"Global test -> loss: {avg_loss:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}")

    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    global_cw = pd.DataFrame({
        "scenario": [SCENARIO]*len(prec_arr),
        "round": [200]*len(prec_arr),
        "class": list(range(len(prec_arr))),
        "precision": prec_arr,
        "recall": rec_arr,
        "f1": f1_arr
    })
    append_df_to_csv(GLOBAL_CLASSWISE_CSV, global_cw)