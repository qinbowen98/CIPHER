"""Train on uni-BD, select on uni-SG, and evaluate once on multi-SG."""
import argparse
import importlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
legacy = importlib.import_module('pipelines.1_single_modality.train')
architectures = importlib.import_module('pipelines.1_single_modality.checkpoint_models')
from pipelines.checkpoint_io import endpoint_checkpoints, load_matching
from pipelines.metrics import event_auroc
from pipelines.timebase import attach_followup
from pipelines.torch_data import add_input_arguments, load_assets, FusionDataset, outcome_rows, soma_tools


class SingleInput(Dataset):
    def __init__(self, frame, assets, modality, normalization):
        self.frame = frame.reset_index(drop=True)
        self.dataset = FusionDataset(self.frame, assets, [modality], normalization)
        self.index = ['fundus', 'snp', 'soma', 'ecg'].index(modality)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        batch = self.dataset[index]
        return batch[self.index], batch[4], batch[5], batch[6]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cohort_frames(assets, modality, endpoint, origin):
    available = assets['available'][modality]
    groups = {'uni_bd': available - assets['sg'] - assets['complete'],
              'uni_sg': (available & assets['sg']) - assets['complete'],
              'multi_sg': available & assets['sg'] & assets['complete']}
    labels = outcome_rows(assets['labels'], endpoint)
    raw = pd.concat([assets['train'], assets['valid']], ignore_index=True) if modality == 'soma' else labels
    if modality == 'soma':
        if raw.duplicated(['pid', 'visit']).any():
            raise ValueError('Duplicate protein participant/visit rows.')
        raw = raw.merge(labels, on='pid', validate='many_to_one')
    result = {}
    for group, ids in groups.items():
        frame = raw.loc[raw.pid.isin(ids)].copy()
        frame = attach_followup(frame, origin, modality).reset_index(drop=True)
        if not np.isfinite(frame.adjusted_time).all() or frame.empty:
            raise ValueError('Empty or nonfinite follow-up cohort: ' + group)
        if frame.event.nunique() != 2 or not frame.event.sum():
            raise ValueError('Both event classes are required in ' + group)
        if frame.adjusted_time.nunique() < 2:
            raise ValueError('Insufficient follow-up variation in ' + group)
        result[group] = frame
    sets = [set(frame.pid) for frame in result.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Training, validation and test participants overlap.')
    return result


def preprocess(assets, frames, modality):
    current = dict(assets)
    scaler = None
    if modality == 'snp':
        source = assets['snp'].loc[frames['uni_bd'].pid]
        scaler = StandardScaler().fit(source.to_numpy(float))
        ids = sorted(set().union(*(set(frame.pid) for frame in frames.values())))
        current['snp'] = pd.DataFrame(scaler.transform(assets['snp'].loc[ids].to_numpy(float)),
                                      index=ids, columns=assets['snp'].columns)
    elif modality == 'soma':
        features = assets['features']
        scaler = soma_tools.GlobalRankNormalizer().fit(frames['uni_bd'][features].to_numpy(float))
        for group in frames:
            frames[group] = frames[group].copy()
            frames[group].loc[:, features] = scaler.transform(frames[group][features].to_numpy(float))
    return current, frames, scaler


def predict(model, dataset, batch_size, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, time, event, pid in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            risk = model(x.to(device)).detach().cpu().numpy().reshape(-1)
            rows.extend({'pid': str(p), 'time': float(t), 'event': int(e), 'risk': float(r)}
                        for p, t, e, r in zip(pid, time, event, risk))
    result = pd.DataFrame(rows)
    if not np.isfinite(result.risk).all():
        raise ValueError('Nonfinite model predictions.')
    result['visit'] = dataset.frame.visit.to_numpy()
    return result


def metrics(frame):
    return {'n_rows': len(frame), 'n_participants': int(frame.pid.nunique()),
            'event_rows': int(frame.event.sum()),
            'event_participants': int(frame.groupby('pid').event.max().sum()),
            'cindex': float(concordance_index(frame.time, -frame.risk, frame.event)),
            'event_auroc': float(event_auroc(frame.event, frame.risk))}


def export_ig(model, dataset, directory, device, steps, internal_batch_size):
    directory.mkdir(parents=True, exist_ok=False)
    model.eval()
    ig = IntegratedGradients(model)
    rows = []
    for index, (x, time, event, pid) in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
        x = x.to(device)
        attr, delta = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=steps,
                                  internal_batch_size=internal_batch_size,
                                  return_convergence_delta=True)
        path = directory / f'sample_{index:06d}.npz'
        np.savez_compressed(path, attribution=attr.detach().cpu().numpy(),
                            inputs=x.detach().cpu().numpy(), pid=np.asarray(list(pid), dtype=str),
                            visit=np.asarray([str(dataset.frame.iloc[index].visit)], dtype=str),
                            time=time.numpy(), event=event.numpy(), delta=delta.detach().cpu().numpy())
        rows.append({'pid': str(pid[0]), 'visit': str(dataset.frame.iloc[index].visit),
                     'file': path.name, 'convergence_delta': float(delta.item())})
    pd.DataFrame(rows).to_csv(directory / 'index.csv', index=False)


def run(args, assets=None):
    if args.epochs < 1 or min(args.grid_bs) < 1 or min(args.grid_lr) <= 0 or args.ig_steps < 2 or args.ig_internal_batch_size < 1:
        raise ValueError('Invalid epochs, optimization grid or IG settings.')
    output = Path(args.outdir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty.')
    output.mkdir(parents=True, exist_ok=True)
    if assets is None and not args.cmcs_standard_csv:
        raise ValueError("Provide --cmcs_standard_csv to identify every SG participant across modalities.")
    assets = load_assets(args) if assets is None else assets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summaries = []
    for modality in args.modalities:
        for endpoint in args.endpoints:
            folder = output / modality / endpoint
            folder.mkdir(parents=True)
            frames = cohort_frames(assets, modality, endpoint, args.time_origin_year)
            current, frames, scaler = preprocess(assets, frames, modality)
            datasets = {key: SingleInput(frame, current, modality, args.fundus_normalization)
                        for key, frame in frames.items()}
            for key, frame in frames.items():
                frame[['pid', 'visit', 'adjusted_time', 'event']].to_csv(folder / (key + '_cohort.csv'), index=False)
            records, best_score, best = [], -np.inf, None
            selected_checkpoint = endpoint_checkpoints(output / 'checkpoints', endpoint)[modality]
            selected_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            for lr in args.grid_lr:
                for batch_size in args.grid_bs:
                    seed_everything(args.seed)
                    model = architectures.build_checkpoint_model(modality, assets['snp'].shape[1], assets['features']).to(device)
                    if modality == 'fundus':
                        if not args.fundus_weights:
                            raise ValueError('Provide the pretrained ViT backbone with --fundus_weights.')
                        load_matching(model.vit, args.fundus_weights, encoder=True)
                    curve, score, state, _ = legacy.fit_surv_model(
                        model, DataLoader(datasets['uni_bd'], batch_size=batch_size, shuffle=True),
                        DataLoader(datasets['uni_sg'], batch_size=batch_size, shuffle=False),
                        device, epochs=args.epochs, lr=lr)
                    if state is None or not np.isfinite(score):
                        raise ValueError('No finite validation checkpoint.')
                    tag = f'lr{lr}_bs{batch_size}'
                    curve.to_csv(folder / (tag + '_curve.csv'), index=False)
                    selected_epoch = int(curve.loc[curve.valid_ci.idxmax(), 'epoch'])
                    records.append({'lr': lr, 'batch_size': batch_size, 'best_epoch': selected_epoch, 'valid_cindex': score})
                    if score > best_score:
                        best_score = float(score)
                        best = dict(records[-1])
                        torch.save(state, selected_checkpoint)
                    del model, state
            pd.DataFrame(records).to_csv(folder / 'validation_search.csv', index=False)
            model = architectures.build_checkpoint_model(modality, assets['snp'].shape[1], assets['features']).to(device)
            load_matching(model, selected_checkpoint)
            if scaler is not None:
                transform_path = output / 'scalers' / modality / (endpoint + '.joblib')
                transform_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(scaler, transform_path)
            for group in ['uni_sg', 'multi_sg']:
                predictions = predict(model, datasets[group], best['batch_size'], device)
                predictions.to_csv(folder / (group + '_predictions.csv'), index=False)
                summaries.append({'modality': modality, 'endpoint': endpoint, 'cohort': group, **metrics(predictions)})
            spec = {'architecture': {'source': 'checkpoint_models', 'snp_dim': assets['snp'].shape[1],
                                      'soma_features': list(assets['features']), 'snp_features': list(assets['snp'].columns)},
                    'selected': best, 'selection_metric': 'uni_sg_cindex', 'seed': args.seed,
                    'cohort_counts': {key: {'rows': len(frame), 'participants': int(frame.pid.nunique())} for key, frame in frames.items()},
                    'time_origin_year': args.time_origin_year, 'fundus_normalization': args.fundus_normalization,
                    'ig': {'baseline': 'zero_in_model_input_space', 'steps': args.ig_steps,
                           'internal_batch_size': args.ig_internal_batch_size, 'cohorts': args.ig_cohorts}}
            (folder / 'model_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
            for group in args.ig_cohorts:
                export_ig(model, datasets[group], folder / (group + '_ig'), device,
                          args.ig_steps, args.ig_internal_batch_size)
            pd.DataFrame(summaries).to_csv(output / 'metrics.csv', index=False)
            print(f'{modality} {endpoint}: selected on uni-SG; evaluated on multi-SG.', flush=True)
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument('--modalities', nargs='+', choices=['snp', 'soma', 'ecg', 'fundus'], default=['snp', 'soma', 'ecg', 'fundus'])
    parser.add_argument('--endpoints', nargs='+', choices=['CHD', 'istroke', 'ICVD'], default=['CHD', 'istroke', 'ICVD'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--grid_lr', type=float, nargs='+', default=[1e-4, 5e-4, 1e-3])
    parser.add_argument('--grid_bs', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--fundus_weights')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--ig_steps', type=int, default=50)
    parser.add_argument('--ig_internal_batch_size', type=int, default=1)
    parser.add_argument('--ig_cohorts', nargs='*', choices=['uni_sg', 'multi_sg'], default=['multi_sg'])
    parser.add_argument('--outdir', default='private_runs/single_validation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
