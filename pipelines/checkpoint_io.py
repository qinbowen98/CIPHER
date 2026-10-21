"""Load complete encoder tensors from standalone or wrapped checkpoints."""
from pathlib import Path
import torch


def read_state(path):
    value = torch.load(Path(path), map_location='cpu')
    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(value, dict) and key in value and isinstance(value[key], dict):
            value = value[key]
            break
    if not isinstance(value, dict):
        raise ValueError('Checkpoint must contain a state dictionary.')
    return value


def load_matching(module, path, encoder=False, aliases=None):
    source, target = read_state(path), module.state_dict()
    prefixes = ('module.', 'model.', 'backbone.', 'encoder.', 'enc.', 'net.',
                'fundus_enc.', 'snp_enc.', 'soma_enc.', 'ecg_enc.', 'vit.')
    matched, used = {}, set()
    for key, tensor in source.items():
        if not torch.is_tensor(tensor):
            continue
        candidates, pending = [], [key]
        while pending:
            name = pending.pop(0)
            if name in candidates:
                continue
            candidates.append(name)
            pending.extend(name[len(prefix):] for prefix in prefixes if name.startswith(prefix))
        if aliases:
            candidates += [new + name[len(old):] for name in list(candidates)
                           for old, new in aliases.items() if name.startswith(old)]
        names = [name for name in candidates if name in target and target[name].shape == tensor.shape]
        if names:
            name = names[0]
            if name in matched:
                raise ValueError('Multiple checkpoint tensors map to ' + name)
            matched[name] = tensor
            used.add(key)
    missing = sorted(set(target) - set(matched))
    unused = sorted(key for key, tensor in source.items() if torch.is_tensor(tensor) and key not in used)
    if missing or (unused and not encoder):
        raise ValueError('Checkpoint mismatch: missing=' + str(missing) + ', unused=' + str(unused))
    module.load_state_dict(matched, strict=True)
    return {'file': Path(path).name, 'loaded_tensors': len(matched), 'unused_tensors': unused}


def endpoint_checkpoints(root, endpoint):
    root = Path(root)
    return {'fundus': root / 'FUNDUS' / (endpoint + '_vit_small_finetune.pt'),
            'snp': root / 'SNP' / (endpoint + '_transformer_h256_l4.pt'),
            'soma': root / 'SOMA' / (endpoint + '_transformer_h256_l8.pt'),
            'ecg': root / 'ECG' / (endpoint + '_transformer_h256_l3.pt')}
