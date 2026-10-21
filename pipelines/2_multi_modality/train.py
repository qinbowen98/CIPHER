"""Train initialized fusion encoders and export held-out risk explanations."""
import argparse
import importlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from pipelines.checkpoint_io import endpoint_checkpoints, load_matching
from pipelines.gradcam import export_maps
from pipelines.metrics import event_auroc
from pipelines.timebase import attach_followup
from pipelines.torch_data import add_input_arguments, load_assets, FusionDataset, outcome_rows, soma_tools
FusionModel = importlib.import_module('pipelines.2_multi_modality.model').FusionModel


def predict(model, loader, device):
    records = []
    model.eval()
    with torch.no_grad():
        for image, snp, soma, ecg, time, event, pids in loader:
            risk = model(image.to(device), snp.to(device), soma.to(device), ecg.to(device)).cpu().numpy()
            records.extend({'ID': pid, 'time': float(t), 'event': int(e), 'risk': float(r)}
                           for pid, t, e, r in zip(pids, time, event, risk))
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError('Empty fusion evaluation set.')
    return result


def metrics(frame):
    return {'n': len(frame), 'events': int(frame.event.sum()),
            'cindex': float(concordance_index(frame.time, -frame.risk, frame.event)),
            'event_auroc': float(event_auroc(frame.event.to_numpy(), frame.risk.to_numpy()))}


def cox_loss(risk, time, event):
    order = torch.argsort(-time)
    r, e = risk[order], event[order].float()
    return -((r - torch.logcumsumexp(r, 0)) * e).sum() / e.sum().clamp_min(1)


def export_ig(model, loader, output, device, modalities):
    from captum.attr import IntegratedGradients
    output.mkdir(parents=True, exist_ok=True)
    names = ['fundus', 'snp', 'soma', 'ecg']
    active = [i for i, name in enumerate(names) if name in modalities]
    def forward(*inputs):
        batch = inputs[0].shape[0]
        values = [torch.empty(batch, 0, device=device) for _ in names]
        for index, value in zip(active, inputs):
            values[index] = value
        return model(*values)
    explainer = IntegratedGradients(forward)
    model.eval()
    for number, batch in enumerate(loader):
        inputs = tuple(batch[i].to(device) for i in active)
        attrs = explainer.attribute(inputs, baselines=tuple(torch.zeros_like(x) for x in inputs),
                                    n_steps=32, internal_batch_size=max(1, len(batch[-1])))
        np.savez_compressed(output / ('batch_' + str(number).zfill(6) + '.npz'),
                            pid=np.asarray(batch[-1]), time=batch[4].numpy(), event=batch[5].numpy(),
                            visit=loader.dataset.df.visit.iloc[number * loader.batch_size:number * loader.batch_size + len(batch[-1])].to_numpy(dtype=str),
                            **{names[i] + '_attribution': value.detach().cpu().numpy() for i, value in zip(active, attrs)},
                            **{names[i] + '_input': value.detach().cpu().numpy() for i, value in zip(active, inputs)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    for modality in ('fundus', 'snp', 'soma', 'ecg'):
        parser.add_argument('--' + modality, action='store_true')
    parser.add_argument('--outdir', default='private_runs/training/multi')
    parser.add_argument('--ckpt_root', default='data/private/ckpt_final_1229')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=555)
    parser.add_argument('--encoder_lr', type=float, default=1e-6)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--longitudinal', action='store_true')
    parser.add_argument('--train_visit', default='12')
    parser.add_argument('--evaluation_visits', nargs='+')
    parser.add_argument('--selection_scope', choices=['sg'], default='sg')
    parser.add_argument('--endpoints', nargs='+', choices=['CHD', 'istroke', 'ICVD'], default=['CHD', 'istroke', 'ICVD'])
    parser.add_argument('--mode', choices=['train', 'evaluate', 'explain'], default='train')
    parser.add_argument('--save_ig', action='store_true')
    parser.add_argument('--save_gradcam', action='store_true')
    parser.add_argument('--gradcam_method', choices=['historical', 'spatial'], default='historical')
    parser.add_argument('--gradcam_mode', choices=['signed', 'positive'], default='signed')
    args = parser.parse_args()
    modalities = [name for name in ('fundus', 'snp', 'soma', 'ecg') if getattr(args, name)]
    if len(modalities) < 2 or args.epochs < 2 or args.batch_size < 1:
        parser.error('Select at least two modalities, at least two epochs, and a positive batch size.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assets = load_assets(args)
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for endpoint in args.endpoints:
        directory = output / endpoint
        directory.mkdir(exist_ok=True)
        labels = outcome_rows(assets['labels'], endpoint)
        bd = assets['train'].merge(labels, on='pid', validate='many_to_one')
        sg = assets['valid'].merge(labels, on='pid', validate='many_to_one')
        bd = bd.loc[bd.pid.isin(assets['complete']) & ~bd.pid.isin(assets['sg'])].copy()
        sg = sg.loc[sg.pid.isin(assets['complete']) & sg.pid.isin(assets['sg'])].copy()
        if set(bd.pid) & set(sg.pid):
            raise ValueError('BD and SG participant sets overlap.')
        if not args.longitudinal:
            bd = bd.loc[bd.visit.astype(str).str.zfill(2).eq(args.train_visit)].copy()
        if args.evaluation_visits:
            sg = sg.loc[sg.visit.isin(args.evaluation_visits)].copy()
        bd = attach_followup(bd, args.time_origin_year)
        sg = attach_followup(sg, args.time_origin_year)
        if bd.empty or sg.empty:
            raise ValueError('Empty BD/SG fusion cohort after visit and outcome alignment.')
        train, valid = bd, sg
        if args.mode == 'train':
            fit_source = assets['train'].loc[~assets['train'].pid.isin(assets['sg']) & assets['train'].visit.eq('02')]
            if fit_source.empty:
                raise ValueError('No training visit-02 rows for fitting the Soma transform.')
            scaler = soma_tools.GlobalRankNormalizer().fit(fit_source[assets['features']].to_numpy(float))
            joblib.dump(scaler, directory / 'soma_scaler.joblib')
        else:
            scaler = joblib.load(directory / 'soma_scaler.joblib')
            metadata = json.loads((directory / 'model_spec.json').read_text())
            if metadata['soma_features'] != assets['features'] or metadata['snp_features'] != list(assets['snp'].columns):
                raise ValueError('Checkpoint and input feature orders differ.')
            for name in ('time_origin_year', 'fundus_normalization', 'train_visit', 'evaluation_visits', 'longitudinal', 'selection_scope'):
                if metadata[name] != getattr(args, name):
                    raise ValueError('Training/evaluation setting mismatch: ' + name)
        transformed = set()
        for frame in (train, valid, sg):
            if id(frame) not in transformed:
                frame.loc[:, assets['features']] = scaler.transform(frame[assets['features']].to_numpy(float))
                transformed.add(id(frame))
        dataset = lambda frame: FusionDataset(frame, assets, modalities, args.fundus_normalization)
        valid_loader = DataLoader(dataset(valid), batch_size=args.batch_size)
        model = FusionModel(*(name in modalities for name in ('fundus', 'snp', 'soma', 'ecg')),
                            snp_dim=assets['snp'].shape[1], feature_cols=assets['features'],
                            init_ckpts=endpoint_checkpoints(args.ckpt_root, endpoint),
                            use_pretrained_ckpt=args.mode == 'train').to(device)
        if args.mode == 'train':
            groups = model.parameter_groups(args.encoder_lr, args.head_lr)
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
            training_loader = DataLoader(dataset(train), batch_size=args.batch_size, shuffle=True)
            best, best_state, patience, curves = -np.inf, None, 0, []
            for epoch in range(args.epochs):
                model.train()
                for image, snp, soma, ecg, time, event, _ in training_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = cox_loss(model(image.to(device), snp.to(device), soma.to(device), ecg.to(device)), time.to(device), event.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                value = metrics(predict(model, valid_loader, device))
                curves.append({'epoch': epoch, **value})
                if epoch > 0 and np.isfinite(value['cindex']) and value['cindex'] > best:
                    best, patience = value['cindex'], 0
                    best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
                else:
                    patience += 1
                if patience > 10:
                    break
            if best_state is None:
                raise ValueError('No finite validation checkpoint was produced.')
            torch.save(best_state, directory / (endpoint + '_model.pt'))
            pd.DataFrame(curves).to_csv(directory / 'training_curve.csv', index=False)
            metadata = {'architecture': {'hidden': 256, 'snp_layers': 4, 'soma_layers': 8, 'ecg_layers': 3},
                        'modalities': modalities, 'selection_scope': args.selection_scope,
                        'soma_features': assets['features'], 'snp_features': list(assets['snp'].columns),
                        'initialization': model.initialization, 'seed': args.seed,
                        'parameter_groups': [{'name': g['name'], 'lr': g['lr']} for g in groups],
                        'fixed_modules': ['fundus_proj', 'ln'], 'time_origin_year': args.time_origin_year,
                        'fundus_normalization': args.fundus_normalization, 'train_visit': args.train_visit,
                        'evaluation_visits': args.evaluation_visits, 'longitudinal': args.longitudinal,
                        'n_bd_participants': int(bd.pid.nunique()), 'n_sg_participants': int(sg.pid.nunique())}
            (directory / 'model_spec.json').write_text(json.dumps(metadata, indent=2) + '\n')
            model.load_state_dict(best_state)
        else:
            load_matching(model, directory / (endpoint + '_model.pt'))
        loader = DataLoader(dataset(sg), batch_size=args.batch_size)
        predictions = predict(model, loader, device)
        predictions['visit'] = sg.visit.to_numpy()
        predictions.to_csv(directory / (endpoint + '_pred.csv'), index=False)
        rows.append({'endpoint': endpoint, 'selection_scope': args.selection_scope, **metrics(predictions)})
        if args.save_ig:
            export_ig(model, loader, directory / 'ig', device, modalities)
        if args.save_gradcam:
            if 'fundus' not in modalities:
                parser.error('Grad-CAM requires the fundus modality.')
            export_maps(model, model.fundus_enc, DataLoader(dataset(sg), batch_size=1),
                        directory / 'gradcam', device, mode=args.gradcam_mode, method=args.gradcam_method, endpoint=endpoint)
    pd.DataFrame(rows).to_csv(output / 'all_endpoints_metrics.csv', index=False)


if __name__ == '__main__':
    main()
