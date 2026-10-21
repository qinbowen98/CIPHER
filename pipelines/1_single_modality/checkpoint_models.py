"""Architectures used by the frozen endpoint checkpoints."""
import importlib
import torch
from torch import nn
import timm

SomaTokenEncoder = importlib.import_module('pipelines.1_single_modality.model').SomaTokenEncoder


class ScalarTransformer(nn.Module):
    def __init__(self, input_dim, layers):
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, batch_first=True), layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
        self.mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.proj(x)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        return self.mlp(self.encoder(x)[:, 0]).squeeze(-1)


class SomaCheckpoint(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.encoder = SomaTokenEncoder(features, hidden_dim=256, nhead=4, nlayers=8)
        self.head = nn.Module()
        self.head.fc = nn.Linear(256, 1)

    def forward(self, x):
        return self.head.fc(self.encoder(x)).squeeze(-1)


class FundusCheckpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
        self.head = nn.Linear(self.vit.num_features, 1)

    def forward(self, x):
        return self.head(self.vit(x)).squeeze(-1)


def build_checkpoint_model(modality, snp_dim, features):
    if modality in {'snp', 'ecg'}:
        return ScalarTransformer(snp_dim if modality == 'snp' else 12, 4 if modality == 'snp' else 3)
    if modality == 'soma':
        return SomaCheckpoint(features)
    if modality == 'fundus':
        return FundusCheckpoint()
    raise ValueError('Unknown modality.')
