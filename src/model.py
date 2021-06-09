import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedELMModel(nn.Module):
    def __init__(self):
        super(StackedELMModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 1)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.dropout2d(x, p=0.3)
        x = F.relu(self.conv2(x))
        x = F.dropout2d(x, p=0.3)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = StackedELMModel()
    x = torch.rand(1, 1, 8, 8)
    print(f"Output shape: {model(x).shape}")
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
