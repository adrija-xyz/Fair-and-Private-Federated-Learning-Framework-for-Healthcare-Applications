import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, parameters_to_ndarrays
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from medmnist import BloodMNIST  # <-- import MedMNIST
# from medmnist import INFO  # optional

# ---------------------- Load BloodMNIST ---------------------- #
num_classes = 8
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # RGB normalize
])

train_dataset = BloodMNIST(split='train', transform=transform, download=True)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True)

def get_labels(ds):
    # MedMNIST stores labels as Nx1 array of ints
    return np.array(ds.labels).squeeze()

# ---------------------------- Dataset Preparation ---------------------------- #
def create_noniid_shards(dataset, num_clients=10, shards_per_client=2, total_samples=None):
    labels = get_labels(dataset)
    all_idx = np.arange(len(labels))
    if total_samples is not None:
        all_idx = np.random.permutation(all_idx)[:total_samples]
    # sort by label -> class-pure shards
    sorted_idx = all_idx[np.argsort(labels[all_idx])]
    num_shards = num_clients * shards_per_client
    shard_size = len(sorted_idx) // num_shards
    shards = [sorted_idx[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)
    client_indices = [[] for _ in range(num_clients)]
    ptr = 0
    for cid in range(num_clients):
        for _ in range(shards_per_client):
            client_indices[cid].extend(shards[ptr])
            ptr += 1
    return [Subset(dataset, idxs) for idxs in client_indices]

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

    # Non-IID via class-sorted shards (variable shards per client)
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

# ---------------------------- Metrics Helper ---------------------------- #
def compute_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return acc, prec, rec, f1

# ---------------------------- Build Client Datasets ---------------------------- #
num_clients = 20
client_datasets = create_hybrid_partition_variable_shards(
    train_dataset,
    num_clients=num_clients,
    iid_fraction=0.4,
    min_shards=1,
    max_shards=5,
    total_samples=None  # use full BloodMNIST train
)

# ---------------------------- Aux loader for class sensitivity ---------------------------- #
# Take ~10 samples per class from test set
labels_test = get_labels(test_dataset)
aux_indices = []
for cls in range(num_classes):
    cls_idx = np.where(labels_test == cls)[0][:10]
    aux_indices.extend(cls_idx)
aux_loader = DataLoader(Subset(test_dataset, aux_indices), batch_size=32, shuffle=False)

# ---------------------------- Class sensitivity profile ---------------------------- #
def compute_class_distribution(model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    class_grad_norms = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)

    for data, target in aux_loader:
        # target arrives as tensor shape [B,1] on MedMNIST; squeeze to [B]
        target = target.squeeze().long()
        model.zero_grad()
        output = self_model_forward(model, data)  # handle potential channels etc.
        loss = criterion(output, target)
        for i in range(len(target)):
            loss[i].backward(retain_graph=True)
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grad_flat = torch.cat([g.reshape(-1) for g in grads])
            cls = int(target[i].item())
            class_grad_norms[cls] += torch.norm(grad_flat).item()
            class_counts[cls] += 1
            model.zero_grad()

    avg = class_grad_norms / np.maximum(class_counts, 1)
    beta = 1.0
    exp_norms = np.exp(beta * avg)
    R = exp_norms / np.sum(exp_norms)
    return R

def compute_kl_divergence(R):
    U = np.ones_like(R) / len(R)
    return float(np.sum(R * np.log(R / (U + 1e-12) + 1e-12)))

def self_model_forward(model, x):
    # Helper in case you later add device placement or normalization tweaks
    return model(x)

# ---------------------------- Global class counts (for proportion-based rare score) ---------------------------- #
global_class_counts = np.zeros(num_classes, dtype=np.float64)

# ---------------------------- Strategy ---------------------------- #
class FedCIR_MAB_KL(FedAvg):
    def __init__(self, num_clusters=5, epsilon=0.1, lambda_weight=0.7,
                 noise_std=0.001, max_clip_norm=5.0, **kwargs):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        self.lambda_weight = lambda_weight
        # DP parameters (client-side param clipping + Gaussian noise)
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
            cids = list(self.client_distributions.keys())
            dist_matrix = np.array([self.client_distributions[cid] for cid in cids])
            labels = KMeans(n_clusters=self.num_clusters, n_init=10).fit_predict(dist_matrix)

            selected_cids = []
            for cl in range(self.num_clusters):
                cluster_cids = [cid for cid, lab in zip(cids, labels) if lab == cl]
                if not cluster_cids:
                    continue
                n_select = max(1, int(self.fraction_fit * len(cluster_cids)))
                if np.random.rand() < self.epsilon:
                    chosen = np.random.choice(cluster_cids, n_select, replace=False).tolist()
                else:
                    cluster_rewards = [(cid, self.client_rewards.get(cid, 0.0)) for cid in cluster_cids]
                    cluster_rewards.sort(key=lambda x: x[1], reverse=True)
                    chosen = [cid for cid, _ in cluster_rewards[:n_select]]
                selected_cids.extend(chosen)

            clients = [c for c in client_manager.all().values() if str(c.cid) in selected_cids]
            if not clients:  # fallback
                clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients)
        else:
            clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients)

        for client in clients:
            cid = str(client.cid)
            self.client_participation[cid] = self.client_participation.get(cid, 0) + 1

        return [(client, FitIns(parameters, {})) for client in clients]

# ---------------------------- Flower Client ---------------------------- #
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, strategy):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.strategy = strategy

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

        local_class_counts = np.zeros(num_classes, dtype=np.int64)
        for data, target in self.train_loader:
            # MedMNIST targets are Nx1 tensors; squeeze to 1D and long
            target = target.squeeze().long()
            opt.zero_grad()
            out = self.model(data)
            loss = F.cross_entropy(out, target)
            prox = sum(((p - gp) ** 2).sum() for p, gp in zip(self.model.parameters(), global_params))
            (loss + (mu/2)*prox).backward()
            opt.step()

            for cls in target.cpu().numpy():
                local_class_counts[int(cls)] += 1

        # ------- proportion-based rare class score (like your MNIST example) -------
        global global_class_counts
        global_class_counts = global_class_counts + local_class_counts
        w_c = 1.0 / (global_class_counts + 1e-6)          # rarer classes get higher weight
        w_c = w_c / w_c.sum()
        total_local = local_class_counts.sum()
        prop = local_class_counts / (total_local + 1e-6)   # client’s class proportions
        rare_class_score = float((prop * w_c).sum())

        # ------- KL part from gradient sensitivity profile -------
        R = compute_class_distribution(self.model)
        kl = compute_kl_divergence(R)
        lam = self.strategy.lambda_weight
        reward = lam * kl + (1 - lam) * rare_class_score

        self.strategy.client_distributions[self.cid] = R
        self.strategy.client_rewards[self.cid] = reward
        self.strategy.client_counts[self.cid] = self.strategy.client_counts.get(self.cid, 0) + 1

        # ---------------- DP step: clip model update and add Gaussian noise ----------------
        updated_params = self.get_parameters()
        # Compute global L2 norm of the vectorized params
        total_norm = np.sqrt(sum(np.sum(p**2) for p in updated_params))
        clip_coef = min(1.0, self.strategy.max_clip_norm / (total_norm + 1e-6))
        clipped_params = [p * clip_coef for p in updated_params]
        noisy_params = [p + np.random.normal(0, self.strategy.noise_std, p.shape) for p in clipped_params]

        # Return NOISY parameters (simulated DP)
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

        avg_loss = loss_sum / total
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred, average="macro")
        print(f"[Client {self.cid}] loss: {avg_loss:.4f}, acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")
        return avg_loss, total, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------------------- Client Function ---------------------------- #
from client import CNN  # your 3-channel CNN

def client_fn(cid: str):
    model = CNN()
    train_loader = DataLoader(client_datasets[int(cid)], batch_size=32, shuffle=True)
    return FlowerClient(cid, model, train_loader, strategy).to_client()

# ---------------------------- Simulation ---------------------------- #
if __name__ == "__main__":
    strategy = FedCIR_MAB_KL(
        fraction_fit=0.2,
        min_fit_clients=2,
        min_available_clients=num_clients,
        num_clusters=5,
        epsilon=0.05,
        lambda_weight=0.7,
        # ---- DP knobs (match your MNIST example style) ----
        noise_std=0.001,
        max_clip_norm=5.0,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=200),
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