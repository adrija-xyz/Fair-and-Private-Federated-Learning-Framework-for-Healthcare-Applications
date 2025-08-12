import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 28, 28)
            dummy_output = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy_input)))))
            self.flat_dim = dummy_output.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)