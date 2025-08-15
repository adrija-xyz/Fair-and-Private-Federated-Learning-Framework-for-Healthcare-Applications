import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from client import CNN
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch.nn.functional as F
import pandas as pd
import random
import copy

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
pinmem = device.type == "cuda"

SCENARIO = "Non-IID"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CLASSWISE_CSV = os.path.join(RESULTS_DIR, f"classwise_{SCENARIO}.csv")
MACRO_CSV = os.path.join(RESULTS_DIR, f"macro_{SCENARIO}.csv")

def append_df_to_csv(path, df):
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))

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

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)

num_clients = 20
client_datasets = create_noniid_shards(mnist_train, num_clients, shards_per_client=6, total_samples=60000)

aux_indices = []
labels_test = np.array(mnist_test.targets)
for cls in range(10):
    aux_indices.extend(np.where(labels_test == cls)[0][:10])
aux_loader = DataLoader(Subset(mnist_test, aux_indices), batch_size=32, shuffle=False, pin_memory=pinmem)

def compute_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1

global_class_counts = np.zeros(10, dtype=np.float64)

def compute_class_distribution(model, device):
    """Compute R (class distribution proxy) via per-class gradient norms on aux set."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    class_grad_norms = np.zeros(10, dtype=np.float64)
    class_counts = np.zeros(10, dtype=np.float64)

    for data, target in aux_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        for idx in range(len(target)):
            model.zero_grad(set_to_none=True)
            loss[idx].backward(retain_graph=True)
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach())
            if grads:
                grad = torch.cat([g.flatten() for g in grads])
                cls = int(target[idx].item())
                class_grad_norms[cls] += torch.norm(grad).item()
                class_counts[cls] += 1

    avg_grad_norms = class_grad_norms / np.maximum(class_counts, 1.0)
    beta = 1.0
    exp_norms = np.exp(beta * avg_grad_norms)
    R = exp_norms / np.sum(exp_norms)
    return R

def compute_kl_divergence(R):
    U = np.ones_like(R) / len(R)
    return np.sum(R * np.log(R / (U + 1e-12) + 1e-12))

class FedCIR_MAB_KL(FedAvg):
    def __init__(self, num_clusters=5, epsilon=0.1, lambda_weight=0.7, noise_std=0.1, max_clip_norm=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        self.lambda_weight = lambda_weight
        self.noise_std = noise_std
        self.max_clip_norm = max_clip_norm
        self.client_distributions = {}
        self.client_rewards = {}
        self.client_counts = {}
        self.client_participation = {}
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        agg_result = super().aggregate_fit(server_round, results, failures)
        if agg_result:
            self.final_parameters = agg_result[0]
        return agg_result

    def configure_fit(self, server_round, parameters, client_manager):
        if self.client_distributions:
            dist_matrix = np.array(list(self.client_distributions.values()))
            cids = list(self.client_distributions.keys())
            try:
                cluster_labels = KMeans(n_clusters=self.num_clusters, n_init="auto").fit_predict(dist_matrix)
            except TypeError:
                cluster_labels = KMeans(n_clusters=self.num_clusters, n_init=10).fit_predict(dist_matrix)

            selected_cids = []
            for cluster in range(self.num_clusters):
                cluster_cids = [cid for cid, label in zip(cids, cluster_labels) if label == cluster]
                if not cluster_cids:
                    continue
                n_select = max(1, int(self.fraction_fit * len(cluster_cids)))
                if np.random.rand() < self.epsilon:
                    selected = np.random.choice(cluster_cids, n_select, replace=False).tolist()
                else:
                    cluster_rewards = [(cid, self.client_rewards.get(cid, 0.0)) for cid in cluster_cids]
                    cluster_rewards.sort(key=lambda x: x[1], reverse=True)
                    selected = [cid for cid, _ in cluster_rewards[:n_select]]
                selected_cids.extend(selected)

            if not selected_cids:
                clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients)
            else:
                clients = [client for client in client_manager.all().values() if str(client.cid) in selected_cids]
        else:
            clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients)

        for client in clients:
            cid = str(client.cid)
            self.client_participation[cid] = self.client_participation.get(cid, 0) + 1

        return [(client, FitIns(parameters, {})) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=self.min_evaluate_clients, min_num_clients=self.min_evaluate_clients)
        return [(client, EvaluateIns(parameters, {"server_round": server_round})) for client in clients]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, strategy, device):
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.strategy = strategy
        self.device = device

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v, device=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        global_params = [p.detach().clone() for p in self.model.parameters()]
        mu = 0.01

        local_class_counts = np.zeros(10, dtype=np.float64)

        for data, target in self.train_loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)

            prox_reg = sum(((p - gp) ** 2).sum() for p, gp in zip(self.model.parameters(), global_params))
            loss = loss + (mu / 2) * prox_reg

            loss.backward()
            optimizer.step()

            for cls in target.detach().cpu().numpy():
                local_class_counts[int(cls)] += 1

        global global_class_counts
        global_class_counts = global_class_counts + local_class_counts

        w_c = 1.0 / (global_class_counts + 1e-6)
        w_c = w_c / w_c.sum()
        total_local = local_class_counts.sum()
        prop = local_class_counts / (total_local + 1e-6)
        rare_class_score = float((prop * w_c).sum())

        R = compute_class_distribution(self.model, self.device)
        kl_div = compute_kl_divergence(R)
        lambda_weight = self.strategy.lambda_weight
        composite_reward = lambda_weight * kl_div + (1 - lambda_weight) * rare_class_score

        self.strategy.client_distributions[self.cid] = R
        self.strategy.client_rewards[self.cid] = composite_reward
        self.strategy.client_counts[self.cid] = self.strategy.client_counts.get(self.cid, 0) + 1

        updated_params = self.get_parameters()
        total_norm = np.sqrt(sum(np.sum(p**2) for p in updated_params))
        clip_coef = min(1.0, self.strategy.max_clip_norm / (total_norm + 1e-6))
        clipped_params = [p * clip_coef for p in updated_params]
        noisy_params = [p + np.random.normal(0, self.strategy.noise_std, p.shape) for p in clipped_params]

        return noisy_params, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        y_true, y_pred = [], []
        loss_sum, total = 0.0, 0

        with torch.no_grad():
            for data, target in aux_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(data)
                loss_sum += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                y_true.extend(target.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())
                total += target.size(0)

        avg_loss = loss_sum / max(total, 1)

        acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

        round_id = int(config.get("server_round", -1)) if config else -1

        macro_df = pd.DataFrame([{
            "scenario": SCENARIO, "round": round_id, "client_id": int(self.cid),
            "loss": float(avg_loss), "accuracy": float(acc),
            "precision": float(prec), "recall": float(rec), "f1": float(f1)
        }])
        append_df_to_csv(MACRO_CSV, macro_df)

        class_rows = []
        for c in range(len(prec_arr)):
            class_rows.append({
                "scenario": SCENARIO, "round": round_id, "client_id": int(self.cid),
                "class": c, "precision": float(prec_arr[c]), "recall": float(rec_arr[c]), "f1": float(f1_arr[c])
            })
        append_df_to_csv(CLASSWISE_CSV, pd.DataFrame(class_rows))

        print(f"[FedCIR-MAB-KL] Client {self.cid} r{round_id} "
              f"loss: {avg_loss:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}")

        return avg_loss, total, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def client_fn(cid: str):
    model = CNN().to(device)
    train_loader = DataLoader(
        client_datasets[int(cid)],
        batch_size=32,
        shuffle=True,
        pin_memory=pinmem
    )
    return FlowerClient(cid, model, train_loader, strategy, device).to_client()

if __name__ == "__main__":
    strategy = FedCIR_MAB_KL(
        fraction_fit=0.2,
        min_fit_clients=2,
        min_available_clients=num_clients,
        num_clusters=5,
        epsilon=0.05,
        lambda_weight=0.7,
        noise_std=0.001,
        max_clip_norm=5.0,
        fraction_evaluate=0.2,
        min_evaluate_clients=2
    )

    client_resources = {"num_cpus": 1}
    if device.type == "cuda":
        client_resources["num_gpus"] = 1

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=500),
        client_resources=client_resources,
        strategy=strategy
    )

    print(f"\nClient Participation Counts: {strategy.client_participation}")

    print("\n--- Evaluating global model ---")
    if strategy.final_parameters is not None:
        final_parameters = parameters_to_ndarrays(strategy.final_parameters)
    else:
        raise RuntimeError("No final global parameters found in strategy.")

    global_model = CNN().to(device)
    state_dict = global_model.state_dict()
    for k, v in zip(state_dict.keys(), final_parameters):
        state_dict[k] = torch.tensor(v, device=device)
    global_model.load_state_dict(state_dict)

    global_model.eval()
    y_true, y_pred = [], []
    loss_sum, total = 0.0, 0

    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False, pin_memory=pinmem)

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = global_model(data)
            loss_sum += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            y_true.extend(target.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())
            total += target.size(0)

    avg_loss = loss_sum / total
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    print(f"\nGlobal test -> loss: {avg_loss:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}")

    global_cw = pd.DataFrame({
        "scenario": [SCENARIO]*len(prec_arr),
        "round": [50]*len(prec_arr),
        "class": list(range(len(prec_arr))),
        "precision": prec_arr,
        "recall": rec_arr,
        "f1": f1_arr
    })
    append_df_to_csv(os.path.join(RESULTS_DIR, f"global_classwise_{SCENARIO}.csv"), global_cw)