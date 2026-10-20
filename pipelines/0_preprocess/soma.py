"""Soma feature panels and fold-fitted normalization used by the training pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


def read_panel(path):
    frame = pd.read_csv(path, dtype=str).fillna('')
    column = next((c for c in ['Feature', 'Genes', 'Gene'] if c in frame), None)
    if column is None:
        raise ValueError('Panel CSV requires Feature, Genes, or Gene.')
    entries = list(dict.fromkeys(v.strip() for v in frame[column] if v.strip()))
    if not entries:
        raise ValueError('Empty feature panel.')
    return entries


def select_features(columns, panel=None, match='auto', prefix='seq.'):
    features = [c for c in columns if c.startswith(prefix)]
    if panel is None:
        return features
    entries = read_panel(panel)
    if match == 'auto':
        match = 'feature' if all(v.startswith(prefix) for v in entries) else 'gene'
    selected = set(entries)
    # Keep source column order: token embeddings depend on feature positions.
    if match == 'feature':
        features = [c for c in features if c in selected]
    elif match == 'gene':
        features = [c for c in features if c[len(prefix):].split('.')[0] in selected]
    else:
        raise ValueError('Panel matching must be auto, feature, or gene.')
    if not features:
        raise ValueError('No panel entries match the Soma feature columns.')
    return features


class Log1pTransformer:
    def fit(self, X):
        self.transform(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all() or (X < 0).any():
            raise ValueError("Raw protein values must be finite and nonnegative for log1p.")
        return np.log1p(X)


class GlobalRankNormalizer:
    """Historical empirical-CDF transform fitted on a training fold only."""
    def fit(self, X):
        X = np.asarray(X, float)
        self.n = X.shape[0]
        if not self.n:
            raise ValueError('Cannot fit rank normalization on an empty training fold.')
        self.sorted_values = [np.sort(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        result = np.zeros_like(X)
        for j in range(X.shape[1]):
            base = self.sorted_values[j]
            ranks = np.searchsorted(base, X[:, j], side='right')
            u = (ranks - 0.5) / len(base)
            result[:, j] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        return result


def apply_soma_norm_train(df, feature_cols, method):
    result = df.copy()
    if method == 'none':
        return result, None
    X = df[feature_cols].to_numpy(dtype=float)
    if method == 'log1p':
        scaler = Log1pTransformer().fit(X)
    elif method == 'minmax':
        scaler = MinMaxScaler().fit(X)
    elif method == 'ranknorm':
        scaler = GlobalRankNormalizer().fit(X)
    else:
        raise ValueError('Normalization must be ranknorm, log1p, minmax, or none.')
    result[feature_cols] = scaler.transform(X)
    return result, scaler


def apply_soma_norm_test(df, feature_cols, method, scaler):
    result = df.copy()
    if method != 'none':
        result[feature_cols] = scaler.transform(df[feature_cols].to_numpy(dtype=float))
    return result


def load_soma_table(path):
    frame = pd.read_csv(path)
    if 'SampleId' not in frame or frame.SampleId.isna().any():
        raise ValueError('Soma table requires nonmissing SampleId values.')
    def parse(value):
        parts = str(value).split('-')
        return (parts[-1], parts[1]) if len(parts) >= 3 else (parts[-1], '02')
    pairs = frame.SampleId.map(parse)
    frame['pid'] = [p[0] for p in pairs]
    frame['visit'] = [p[1] for p in pairs]
    if frame.duplicated(['pid', 'visit']).any():
        raise ValueError('Duplicate participant/visit rows in Soma input.')
    return frame


def prepare(train_path, valid_path, output_dir, panel=None, match='auto'):
    train, valid = load_soma_table(train_path), load_soma_table(valid_path)
    features = select_features(train.columns, panel, match)
    if not features or set(features) - set(valid):
        raise ValueError('Training and validation tables must contain the selected features.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [('train', train), ('valid', valid)]:
        values = frame[features].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError('Soma features contain missing or nonfinite values; resolve upstream.')
        frame[['SampleId', 'pid', 'visit', *features]].to_csv(output / (name + '.csv'), index=False)
    (output / 'features.json').write_text(json.dumps(features, indent=2) + '\n')
    (output / 'preprocessing.json').write_text(json.dumps({
        'features': len(features), 'panel_match': match, 'normalization': 'deferred_to_training_fold',
        'sample_id_rule': 'last hyphen-delimited token; visit is middle token for 3+ tokens, otherwise 02',
        'participant_outputs': True}, indent=2) + '\n')
    print('Prepared protected Soma tables; normalization remains inside training folds.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--valid', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--panel')
    parser.add_argument('--match', choices=['auto', 'gene', 'feature'], default='auto')
    args = parser.parse_args()
    prepare(args.train, args.valid, args.output_dir, args.panel, args.match)


if __name__ == '__main__':
    main()
