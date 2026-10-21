"""Evaluate frozen unimodal checkpoints in explicitly defined SG cohorts."""
import argparse
import importlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from pipelines.checkpoint_io import endpoint_checkpoints, load_matching
from pipelines.gradcam import export_maps
from pipelines.metrics import event_auroc
from pipelines.timebase import attach_followup
from pipelines.torch_data import add_input_arguments, load_assets, FusionDataset, outcome_rows, soma_tools
build_model = importlib.import_module('pipelines.1_single_modality.checkpoint_models').build_checkpoint_model


def cohort_ids(assets, modality, cohort):
    available = assets['sg'] & assets['available'][modality]
    if cohort == 'multi_sg':
        return available & assets['complete']
    if cohort == 'uni_sg':
        return available - assets['complete']
    return available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument('--modality', choices=['snp', 'ecg', 'soma', 'fundus'], required=True)
    parser.add_argument('--cohort', choices=['multi_sg', 'uni_sg', 'all_sg'], default='multi_sg')
    parser.add_argument('--ckpt_root', default='data/private/ckpt_final')
    parser.add_argument('--scaler_root', help='Source-fitted transforms named <modality>/<endpoint>.joblib.')
    parser.add_argument('--outdir', default='private_runs/single_checkpoint_evaluation')
    parser.add_argument('--endpoints', nargs='+', choices=['CHD', 'istroke', 'ICVD'], default=['CHD', 'istroke', 'ICVD'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_gradcam', action='store_true')
    parser.add_argument('--gradcam_method', choices=['historical', 'spatial'], default='historical')
    parser.add_argument('--gradcam_mode', choices=['signed', 'positive'], default='signed')
    args = parser.parse_args()
    assets = load_assets(args)
    selected = cohort_ids(assets, args.modality, args.cohort)
    if not selected:
        parser.error('No participants match the requested modality and SG cohort.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output = Path(args.outdir) / args.modality / args.cohort
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for endpoint in args.endpoints:
        labels = outcome_rows(assets['labels'], endpoint)
        source_ids = assets['available'][args.modality] - assets['sg']
        if args.modality == 'soma':
            raw = pd.concat([assets['train'], assets['valid']], ignore_index=True)
            train = raw.loc[raw.pid.isin(source_ids)].merge(labels, on='pid', validate='many_to_one')
            evaluation = raw.loc[raw.pid.isin(selected)].merge(labels, on='pid', validate='many_to_one')
        else:
            train = labels.loc[labels.pid.isin(source_ids)].copy()
            evaluation = labels.loc[labels.pid.isin(selected)].copy()
        train = attach_followup(train, args.time_origin_year, args.modality)
        evaluation = attach_followup(evaluation, args.time_origin_year, args.modality)
        if evaluation.empty or set(train.pid) & set(evaluation.pid):
            raise ValueError('Empty evaluation cohort or overlapping source/evaluation participants.')
        current = dict(assets)
        transform_scope = 'none'
        if args.modality in {'snp', 'soma'}:
            if args.scaler_root:
                scaler = joblib.load(Path(args.scaler_root) / args.modality / (endpoint + '.joblib'))
                transform_scope = 'stored_source_transform'
            else:
                values = assets['snp'].loc[train.pid].to_numpy(float) if args.modality == 'snp' else train[assets['features']].to_numpy(float)
                scaler = (StandardScaler() if args.modality == 'snp' else soma_tools.GlobalRankNormalizer()).fit(values)
                transform_scope = 'all_non_sg_available_source_rows'
            if args.modality == 'snp':
                current['snp'] = pd.DataFrame(scaler.transform(assets['snp'].to_numpy(float)),
                                               index=assets['snp'].index, columns=assets['snp'].columns)
            else:
                evaluation.loc[:, assets['features']] = scaler.transform(evaluation[assets['features']].to_numpy(float))
            joblib.dump(scaler, output / (endpoint + '_source_scaler.joblib'))
        model = build_model(args.modality, assets['snp'].shape[1], assets['features']).to(device)
        loading = load_matching(model, endpoint_checkpoints(args.ckpt_root, endpoint)[args.modality])
        dataset = FusionDataset(evaluation, current, [args.modality], args.fundus_normalization)
        records = []
        model.eval()
        index = ['fundus', 'snp', 'soma', 'ecg'].index(args.modality)
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=args.batch_size):
                risk = model(batch[index].to(device)).cpu().numpy()
                records.extend({'ID': pid, 'time': float(t), 'event': int(e), 'risk': float(r)}
                               for pid, t, e, r in zip(batch[-1], batch[4], batch[5], risk))
        predictions = pd.DataFrame(records)
        if args.modality == 'soma':
            predictions['visit'] = evaluation.visit.to_numpy()
        predictions.to_csv(output / (endpoint + '_pred.csv'), index=False)
        summaries.append({'endpoint': endpoint, 'modality': args.modality, 'cohort': args.cohort,
                          'n': len(predictions), 'n_participants': int(evaluation.pid.nunique()),
                          'events': int(predictions.event.sum()),
                          'cindex': float(concordance_index(predictions.time, -predictions.risk, predictions.event)),
                          'event_auroc': float(event_auroc(predictions.event.to_numpy(), predictions.risk.to_numpy()))})
        manifest = {'checkpoint': loading, 'cohort': args.cohort, 'time_origin_year': args.time_origin_year,
                    'transform_scope': transform_scope, 'fundus_normalization': args.fundus_normalization,
                    'soma_features': assets['features'], 'snp_features': list(assets['snp'].columns),
                    'participant_outputs': True}
        (output / (endpoint + '_evaluation.json')).write_text(json.dumps(manifest, indent=2) + '\n')
        if args.save_gradcam:
            if args.modality != 'fundus':
                parser.error('Grad-CAM requires a fundus checkpoint.')
            export_maps(model, model.vit, DataLoader(dataset, batch_size=1), output / endpoint / 'gradcam',
                        device, fused=False, method=args.gradcam_method, mode=args.gradcam_mode, endpoint=endpoint)
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
