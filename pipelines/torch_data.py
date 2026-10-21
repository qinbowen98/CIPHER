"""Aligned protected inputs for frozen unimodal and fusion models."""
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

soma_tools = importlib.import_module('pipelines.0_preprocess.soma')
signal_tools = importlib.import_module('pipelines.0_preprocess.imaging_ecg')


def add_input_arguments(parser):
    for name in ('soma_train', 'soma_valid', 'snp_prefix', 'fundus_dir', 'ecg_dir', 'label_csv', 'soma_gene_csv'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cmcs_standard_csv')
    parser.add_argument('--time_origin_year', type=int, choices=[1992, 2002], default=2002)
    parser.add_argument('--fundus_normalization', choices=['imagenet', 'half', 'none'], default='half')


def load_assets(args):
    from pandas_plink import read_plink
    train = soma_tools.load_soma_table(args.soma_train)
    valid = soma_tools.load_soma_table(args.soma_valid)
    train['pid'], valid['pid'] = train.pid.astype(str), valid.pid.astype(str)
    features = soma_tools.select_features(train.columns, args.soma_gene_csv)
    bim, fam, bed = read_plink(args.snp_prefix)
    snp = pd.DataFrame(bed.compute().T, index=fam.iid.astype(str).str.split('_').str[-1],
                       columns=bim.snp.astype(str)).fillna(0)
    if snp.index.duplicated().any():
        raise ValueError('Duplicate SNP participant identifiers.')
    images = {p.stem.split('.')[0]: p for p in sorted(Path(args.fundus_dir).iterdir())
              if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    ecgs = {p.stem: p for p in Path(args.ecg_dir).glob('*.xml')}
    labels = pd.read_csv(args.label_csv, dtype={'ID': str}).rename(columns={'ID': 'pid'})
    if labels.pid.duplicated().any():
        raise ValueError('Duplicate participant outcomes.')
    available = {'fundus': set(images), 'ecg': set(ecgs), 'snp': set(snp.index),
                 'soma': set(train.pid) | set(valid.pid)}
    complete = set.intersection(*available.values())
    if args.cmcs_standard_csv:
        metadata = pd.read_csv(args.cmcs_standard_csv)
        column = next((c for c in metadata if c.lower() in {'id', 'pid', 'cmcs_id', 'subject_id'}), None)
        if column is None or 'subcohort' not in metadata:
            raise ValueError('Cohort table requires an identifier and subcohort.')
        sg = set(metadata.loc[metadata.subcohort.astype(str).str.replace(r'\.0$', '', regex=True).eq('102'), column].astype(str))
    else:
        sg = set(valid.pid)
        if set(train.pid) & sg:
            raise ValueError('BD and SG source files overlap; provide a cohort metadata table.')
    return {'train': train, 'valid': valid, 'features': features, 'snp': snp, 'images': images,
            'ecgs': ecgs, 'labels': labels, 'available': available, 'complete': complete, 'sg': sg}


class FusionDataset(Dataset):
    def __init__(self, frame, assets, modalities, normalization='half'):
        self.df = frame.reset_index(drop=True)
        self.assets, self.modalities = assets, set(modalities)
        self.transform = signal_tools.fundus_transform(224, normalization)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pid = str(row.pid)
        image, snp, soma, ecg = [torch.empty(0) for _ in range(4)]
        if 'fundus' in self.modalities:
            with Image.open(self.assets['images'][pid]) as raw:
                image = self.transform(raw.convert('RGB'))
        if 'snp' in self.modalities:
            snp = torch.tensor(self.assets['snp'].loc[pid].to_numpy(dtype=np.float32))
        if 'soma' in self.modalities:
            soma = torch.tensor(row[self.assets['features']].to_numpy(dtype=np.float32))
        if 'ecg' in self.modalities:
            ecg = torch.tensor(signal_tools.read_ecg(self.assets['ecgs'][pid], normalization='none'))
        return image, snp, soma, ecg, torch.tensor(float(row.adjusted_time)), torch.tensor(int(row.event)), pid


def outcome_rows(labels, endpoint):
    event = endpoint if endpoint in labels else 'f' + endpoint
    duration = {'CHD': 'fCHD_y', 'istroke': 'fistroke_y', 'ICVD': 'fICVD_y'}[endpoint]
    frame = labels[['pid', event, duration]].dropna().copy()
    frame.columns = ['pid', 'event', 'time']
    if not frame.event.isin([0, 1]).all():
        raise ValueError('Expected observed binary event status.')
    return frame
