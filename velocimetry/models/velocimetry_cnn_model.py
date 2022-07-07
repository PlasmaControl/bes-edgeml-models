import argparse
import torch
from elm_prediction.models.cnn_model import CNNModel

class VelocimetryCNNModel(CNNModel):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        ll = [m for m in self.children() if type(m).__name__ == 'Linear']
        self.fc1 = torch.nn.Linear(in_features=ll[-3].in_features, out_features=1024)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=512)
        self.fc3 = torch.nn.Linear(in_features=512, out_features=128)
        return