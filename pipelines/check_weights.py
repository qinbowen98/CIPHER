"""Check local single-modality and fusion checkpoints without cohort data."""
import argparse
import gc
import json
from pathlib import Path
import torch
from pipelines.neural_checkpoints import load_single_model, load_fusion_model


def check(single_root, fusion_dir, forward=True, device='cpu'):
    records = []
    paths = [(p, modality) for modality, folder in [('snp', 'SNP'), ('soma', 'SOMA'), ('ecg', 'ECG'), ('fundus', 'FUNDUS')]
             for p in sorted((Path(single_root) / folder).glob('*.pt'))]
    paths += [(p, 'multimodal') for p in sorted(Path(fusion_dir).glob('*_model.pt'))]
    if not paths or not any(mod == 'multimodal' for _, mod in paths):
        raise ValueError('Provide the single checkpoint root and a fusion checkpoint directory.')
    if any(not any(mod == required for _, mod in paths) for required in ['snp','soma','ecg','fundus']):
        raise ValueError('A single-modality checkpoint directory is empty.')
    for path, modality in paths:
        model, info = load_fusion_model(path, device=device) if modality == 'multimodal' else load_single_model(path, modality, device=device)
        record = {'modality': modality, 'file': path.name, **info}
        if forward:
            with torch.inference_mode():
                if modality == 'multimodal':
                    inputs = [torch.zeros(1,3,224,224,device=device),
                              torch.zeros(1,model.snp_enc.proj.in_features,device=device),
                              torch.zeros(1,model.soma_enc.num_tokens,device=device),
                              torch.zeros(1,1000,12,device=device)]
                    score = model(*inputs)
                    del inputs
                elif modality == 'snp':
                    projection = model.proj if hasattr(model, 'proj') else model.enc.proj
                    score = model(torch.zeros(1,projection.in_features,device=device))
                elif modality == 'soma':
                    score = model(torch.zeros(1,model.encoder.num_tokens,device=device))
                elif modality == 'ecg':
                    score = model(torch.zeros(1,1000,12,device=device))
                else:
                    score = model(torch.zeros(1,3,224,224,device=device))
                if score.shape != (1,) or not torch.isfinite(score).all():
                    raise ValueError('Invalid forward output: ' + path.name)
                record['finite_forward_output'] = True
                del score
        records.append(record)
        print(modality + ': ' + path.name + ' OK', flush=True)
        del model
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-root', required=True)
    parser.add_argument('--fusion-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--skip-forward', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    records = check(args.single_root, args.fusion_dir, not args.skip_forward, args.device)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({'checkpoints':records}, indent=2) + '\n')
    print('Verified checkpoints:', len(records))


if __name__ == '__main__':
    main()
