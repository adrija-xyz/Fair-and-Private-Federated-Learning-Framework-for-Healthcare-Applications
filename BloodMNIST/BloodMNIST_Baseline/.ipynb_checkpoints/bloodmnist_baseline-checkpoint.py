import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, parameters_to_ndarrays

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from medmnist import BloodMNIST

# ---------------------- Config ---------------------- #
num_classes = 8
num_clients = 20

# ---------------------- Load BloodMNIST ---------------------- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 3-channel
])

train_dataset = BloodMNIST(split='train', transform=transform, download=True)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True)

def get_labels(ds):
    # MedMNIST stores labels as Nx1 array of ints
    return np.array(ds.labels).squeeze()

# ---------------------------- Metrics Helper ---------------------------- #
def compute_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return acc, prec, rec, f1

# ---------------------------- Hybrid Partition (IID + Non-IID) ---------------------------- #
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

    # IID split
    iid_block = all_idx[:num_iid * samples_per_client]
    iid_clients = [
        iid_block[i*samples_per_client:(i+1)*samples_per_client].tolist()
        for i in range(num_iid)
    ]

    # Non‑IID via class‑sorted shards (variable shards per client)
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
                merged.extend(shards[ptr])
                ptr += 1
        non_iid_clients.append(merged)

    all_client_indices = iid_clients + non_iid_clients
    return [Subset(dataset, idxs) for idxs in all_client_indices]

# ---------------------------- Build Client Datasets ---------------------------- #
client_datasets = create_hybrid_partition_variable_shards(
    train_dataset,
    num_clients=num_clients,
    iid_fraction=0.4,
    min_shards=1,
    max_shards=5,
    total_samples=None  # use full BloodMNIST train
)

# ---------------------------- Baseline Strategy (Random Selection) ---------------------------- #
class BaselineRandomFedAvg(FedAvg):
    """
    A thin wrapper over FedAvg:
    - Randomly samples clients each round using fraction_fit/min_fit_clients.
    - Stores final global parameters for post-training evaluation.
    - Tracks per-client participation counts.
    """
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
        # Uniform random sampling handled via client_manager.sample
        # Use the same sampling logic as FedAvg but we also record participation
        available = client_manager.all()  # dict of cid -> ClientProxy
        num_available = len(available)

        # Number of clients to sample this round
        n_sample = max(
            self.min_fit_clients,
            int(np.ceil(self.fraction_fit * num_available))
        )

        clients = client_manager.sample(
            num_clients=n_sample,
            min_num_clients=self.min_fit_clients
        )

        # Track participation
        for c in clients:
            cid = str(c.cid)
            self.client_participation[cid] = self.client_participation.get(cid, 0) + 1

        # Build fit instructions
        fit_ins = FitIns(parameters, {})
        return [(c, fit_ins) for c in clients]

# ---------------------------- Flower Client ---------------------------- #
from client import CNN  # your 3-channel CNN

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

        # Optional: FedProx-style proximal term for stability (mu small)
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

        return self.get_parameters(), len(self.train_loader.dataset), {}

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

        avg_loss = loss_sum / total
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")
        print(f"[Client {self.cid}] loss: {avg_loss:.4f}, acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")
        return avg_loss, total, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------------------- Client Function ---------------------------- #
def client_fn(cid: str):
    model = CNN()
    train_loader = DataLoader(client_datasets[int(cid)], batch_size=32, shuffle=True)
    return FlowerClient(cid, model, train_loader).to_client()

# ---------------------------- Simulation ---------------------------- #
if __name__ == "__main__":
    strategy = BaselineRandomFedAvg(
        fraction_fit=0.2,            # sample 20% of available clients each round
        min_fit_clients=2,           # at least 2 per round
        min_available_clients=num_clients,  # ensure all are registered
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=500),
        client_resources={"num_cpus": 1},
        strategy=strategy
    )

    print(f"\nClient Participation Counts: {strategy.client_participation}")

    # ------------------ Evaluate Global Model on BloodMNIST ------------------ #
    print("\n--- Evaluating global model (BloodMNIST test) ---")
    if strategy.final_parameters is None:
        raise RuntimeError("No final global parameters found in strategy.")
    final_parameters = parameters_to_ndarrays(strategy.final_parameters)

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

    avg_loss = loss_sum / total
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")
    print(f"Global test -> loss: {avg_loss:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}")