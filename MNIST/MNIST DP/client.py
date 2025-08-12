# import flwr as fl
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from flwr.common import NDArrays, Scalar

# # Simple CNN model
# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(1, 10, kernel_size=5)
#         self.fc = nn.Linear(1440, 10)

#     def forward(self, x):
#         x = F.relu(self.conv(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(-1, 1440)
#         return self.fc(x)

# # Flower-compatible NumPy client
# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, model, train_loader):
#         self.model = model
#         self.train_loader = train_loader
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

#     def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
#         return [val.cpu().numpy() for val in self.model.state_dict().values()]

#     def set_parameters(self, parameters: NDArrays) -> None:
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = {k: torch.tensor(v) for k, v in params_dict}
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
#         self.set_parameters(parameters)
#         self.model.train()
#         for epoch in range(1):
#             for batch in self.train_loader:
#                 x, y = batch
#                 self.optimizer.zero_grad()
#                 loss = self.criterion(self.model(x), y)
#                 loss.backward()
#                 self.optimizer.step()
#         return self.get_parameters(config={}), len(self.train_loader.dataset), {}

#     def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
#         self.set_parameters(parameters)
#         self.model.eval()
#         loss = 0
#         correct = 0
#         for x, y in self.train_loader:
#             output = self.model(x)
#             loss += self.criterion(output, y).item()
#             correct += (output.argmax(1) == y).sum().item()
#         return float(loss), len(self.train_loader.dataset), {"accuracy": correct / len(self.train_loader.dataset)}


import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl

# Example CNN for MNIST
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Flower client wrapper
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # One epoch per round
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        for data, target in self.train_loader:
            output = self.model(data)
            loss += self.criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        return float(loss), len(self.train_loader.dataset), {"accuracy": correct / len(self.train_loader.dataset)}