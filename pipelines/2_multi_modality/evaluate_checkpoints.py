"""Evaluate complete saved fusion state dictionaries on protected SG inputs."""
import argparse
import importlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pipelines.neural_checkpoints import load_fusion_model
from pipelines.timebase import attach_followup
from pipelines.torch_data import add_input_arguments, load_assets, outcome_rows, FusionDataset, soma_tools

helpers = importlib.import_module('pipelines.2_multi_modality.train')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--soma_scaler', help='Original fitted source IRN; otherwise reconstruct from BD visit 02.')
    parser.add_argument('--snp_scaler', help='Apply only if the saved fusion model used standardized SNP inputs.')
    parser.add_argument('--evaluation_visits', nargs='+')
    parser.add_argument('--endpoints', nargs='+', choices=['CHD','istroke','ICVD'], default=['CHD','istroke','ICVD'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_ig', action='store_true')
    parser.add_argument('--outdir', default='private_runs/fusion_checkpoint_evaluation')
    args = parser.parse_args()
    if not args.cmcs_standard_csv:
        parser.error('Provide --cmcs_standard_csv for SG participant membership.')
    assets = load_assets(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output = Path(args.outdir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty.')
    output.mkdir(parents=True, exist_ok=True)
    if args.soma_scaler:
        scaler = joblib.load(args.soma_scaler)
        scope = 'stored_source_IRN'
    else:
        source = assets['train'].loc[~assets['train'].pid.isin(assets['sg']) & assets['train'].visit.eq('02')]
        if source.empty:
            raise ValueError('No BD visit-02 source rows for protein normalization.')
        scaler = soma_tools.GlobalRankNormalizer().fit(source[assets['features']].to_numpy(float))
        scope = 'reconstructed_BD_visit02_IRN'
    joblib.dump(scaler, output / 'soma_scaler.joblib')
    if args.snp_scaler:
        snp_scaler = joblib.load(args.snp_scaler)
        assets['snp'] = pd.DataFrame(snp_scaler.transform(assets['snp'].to_numpy(float)), index=assets['snp'].index, columns=assets['snp'].columns)
        joblib.dump(snp_scaler, output / 'snp_scaler.joblib')
    summaries = []
    modalities = ['fundus','snp','soma','ecg']
    for endpoint in args.endpoints:
        folder = output / endpoint
        folder.mkdir()
        labels = outcome_rows(assets['labels'], endpoint)
        frame = assets['valid'].merge(labels, on='pid', validate='many_to_one')
        frame = frame.loc[frame.pid.isin(assets['sg'] & assets['complete'])].copy()
        if args.evaluation_visits:
            frame = frame.loc[frame.visit.isin(args.evaluation_visits)].copy()
        frame = attach_followup(frame, args.time_origin_year).reset_index(drop=True)
        if frame.empty:
            raise ValueError('Empty multi-SG cohort.')
        frame.loc[:, assets['features']] = scaler.transform(frame[assets['features']].to_numpy(float))
        model, loading = load_fusion_model(Path(args.checkpoint_dir) / (endpoint + '_model.pt'),
                                           snp_dim=assets['snp'].shape[1], features=assets['features'], device=device)
        dataset = FusionDataset(frame, assets, modalities, args.fundus_normalization)
        predictions = helpers.predict(model, DataLoader(dataset, batch_size=args.batch_size), device)
        if not np.isfinite(predictions.risk).all():
            raise ValueError('Nonfinite fusion predictions.')
        predictions['visit'] = frame.visit.to_numpy()
        predictions.to_csv(folder / 'multi_sg_predictions.csv', index=False)
        summaries.append({'endpoint':endpoint,'cohort':'multi_sg','n_participants':int(frame.pid.nunique()), **helpers.metrics(predictions)})
        spec = {'loading':loading, 'soma_transform':scope,
                'snp_transform':'stored_source_scaler' if args.snp_scaler else 'raw_dosage',
                'snp_features':list(assets['snp'].columns),'soma_features':assets['features'],
                'time_origin_year':args.time_origin_year,'fundus_normalization':args.fundus_normalization,
                'evaluation_visits':args.evaluation_visits,'participant_outputs':True}
        (folder / 'evaluation.json').write_text(json.dumps(spec,indent=2)+'\n')
        if args.save_ig:
            helpers.export_ig(model,DataLoader(dataset,batch_size=1),folder/'ig',device,modalities)
    pd.DataFrame(summaries).to_csv(output/'metrics.csv',index=False)


if __name__ == '__main__':
    main()
