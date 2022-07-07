import argparse
import torch
from models.bes_edgeml_models.models.multi_features_ds_v2_model import MultiFeaturesDsV2Model

class MultiFeaturesClassificationModel(MultiFeaturesDsV2Model):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        ll = [m for m in self.children() if type(m).__name__ == 'Linear'][-1]
        self.fc3 = torch.nn.Linear(in_features=ll.in_features, out_features=4)
        return

