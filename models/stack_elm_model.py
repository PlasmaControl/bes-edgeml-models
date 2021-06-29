import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedELMModel(nn.Module):
    def __init__(self):
        super(StackedELMModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(self.conv2(x))
        x = F.dropout2d(x, p=0.3)
        x = self.pool(x)
        x = F.gelu(self.conv3(x))
        x = F.dropout2d(x, p=0.3)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = StackedELMModel()
    input_size = (1, 1, 32, 32)
    x = torch.rand(*input_size)
    print(f"Output shape: {model(x).shape}")
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
