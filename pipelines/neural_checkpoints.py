"""Strict loading of endpoint-specific single-modality and fusion weights."""
import importlib
import re
import torch
from torch import nn
from pipelines.checkpoint_io import read_state, load_matching

single = importlib.import_module('pipelines.1_single_modality.checkpoint_models')
encoders = importlib.import_module('pipelines.1_single_modality.model')
fusion = importlib.import_module('pipelines.2_multi_modality.model')


def canonical_state(path):
    result = {}
    for key, value in read_state(path).items():
        while key.startswith(('module.', 'model.')):
            key = key.split('.', 1)[1]
        if key in result:
            raise ValueError('Duplicate normalized checkpoint key: ' + key)
        result[key] = value
    return result


def layers(state, prefix):
    indices = {int(m.group(1)) for key in state for m in [re.match(re.escape(prefix) + r'(\d+)\.', key)] if m}
    if not indices or indices != set(range(max(indices) + 1)):
        raise ValueError('Incomplete Transformer layers in checkpoint.')
    return len(indices)


class WrappedTransformer(nn.Module):
    def __init__(self, modality, input_dim, hidden, depth):
        super().__init__()
        encoder = encoders.TransformerModule(input_dim, hidden_dim=hidden, nhead=4, nlayers=depth) if modality == 'snp' else encoders.ECGTransformer(input_dim, hidden_dim=hidden, nhead=4, nlayers=depth)
        setattr(self, 'enc' if modality == 'snp' else 'backbone', encoder)
        self.modality = modality
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        encoder = self.enc if self.modality == 'snp' else self.backbone
        return self.head(encoder(x)).squeeze(-1)


def load_single_model(path, modality, snp_dim=None, features=None, device='cpu'):
    state = canonical_state(path)
    if modality in {'snp', 'ecg'}:
        prefix = 'enc.' if modality == 'snp' else 'backbone.'
        projection = prefix + ('proj.weight' if modality == 'snp' else 'embedding.weight')
        if projection in state and 'head.weight' in state:
            hidden, input_dim = state[projection].shape
            model = WrappedTransformer(modality, input_dim, hidden, layers(state, prefix + 'encoder.layers.'))
        else:
            hidden, input_dim = state['proj.weight'].shape
            head_hidden = state['mlp.0.weight'].shape[0]
            model = single.ScalarTransformer(input_dim, layers(state, 'encoder.layers.'), hidden, head_hidden)
        if modality == 'snp' and snp_dim is not None and input_dim != snp_dim:
            raise ValueError(f'SNP feature count mismatch: checkpoint={input_dim}, input={snp_dim}.')
        if modality == 'ecg' and input_dim != 12:
            raise ValueError('Expected 12 ECG channels.')
    elif modality == 'soma':
        feature_count, hidden = state['encoder.protein_emb.weight'].shape
        if features is not None and len(features) != feature_count:
            raise ValueError(f'Protein feature count mismatch: checkpoint={feature_count}, input={len(features)}.')
        model = single.SomaCheckpoint(list(range(feature_count)) if features is None else features,
                                      hidden, layers(state, 'encoder.encoder.layers.'))
    elif modality == 'fundus':
        model = single.FundusCheckpoint()
    else:
        raise ValueError('Unknown modality.')
    info = load_matching(model, path)
    return model.to(device).eval(), info


def load_fusion_model(path, snp_dim=None, features=None, device='cpu'):
    state = canonical_state(path)
    size = state['snp_enc.proj.weight'].shape[1]
    count = state['soma_enc.protein_emb.weight'].shape[0]
    if snp_dim is not None and snp_dim != size:
        raise ValueError(f'SNP feature count mismatch: checkpoint={size}, input={snp_dim}.')
    if features is not None and len(features) != count:
        raise ValueError(f'Protein feature count mismatch: checkpoint={count}, input={len(features)}.')
    model = fusion.FusionModel(True, True, True, True, snp_dim=size,
                              feature_cols=list(range(count)) if features is None else features,
                              use_pretrained_ckpt=False)
    info = load_matching(model, path)
    return model.to(device).eval(), info
