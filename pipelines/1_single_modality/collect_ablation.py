"""Collect all selected-setting protein ablation validation folds."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def collect_cv(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    rows, files = [], []
    for source in sorted(root.rglob('cv_folds.csv')):
        frame = pd.read_csv(source)
        required = {'endpoint', 'model', 'seed', 'fold', 'valid_cindex', 'normalization'}
        if not required <= set(frame):
            raise ValueError('Validation-fold table lacks required columns.')
        frame['condition'] = source.relative_to(root).parts[0]
        if frame.duplicated(['endpoint', 'model', 'seed', 'fold']).any():
            raise ValueError('Duplicate validation folds in a run.')
        for _, group in frame.groupby(['endpoint', 'model', 'seed']):
            if len(group) != 5 or group.fold.nunique() != 5:
                raise ValueError('Each selected setting requires all five validation folds.')
        rows.append(frame)
        files.append({'file': source.relative_to(root).as_posix(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    if not rows:
        raise FileNotFoundError('No cv_folds.csv found.')
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(['condition', 'endpoint', 'model', 'seed', 'fold']).any():
        raise ValueError('Multiple runs share the same condition/seed/fold keys.')
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / 'ablation_folds.csv', index=False)
    (output / 'collection.json').write_text(json.dumps({'selection': 'all supplied validation-fold tables', 'files': files}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    collect_cv(args.root, args.output)
