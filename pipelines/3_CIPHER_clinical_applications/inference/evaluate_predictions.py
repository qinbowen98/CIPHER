"""Recalculate aggregate event-status metrics from authorized prediction tables."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from pipelines.metrics import event_auroc


def evaluate(manifest_path, output, bootstrap=500, seed=42):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    specifications = json.loads(manifest_path.read_text())
    if not specifications or bootstrap < 0:
        raise ValueError('Provide prediction specifications and a nonnegative bootstrap count.')
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rows = []
    reserved = {'prediction_file', 'event_column', 'risk_column', 'time_column'}
    for spec in specifications:
        relative = Path(spec['prediction_file'])
        path = relative if relative.is_absolute() else manifest_path.parent / relative
        frame = pd.read_csv(path)
        event = frame[spec.get('event_column', 'event')].to_numpy()
        risk = frame[spec.get('risk_column', 'risk')].to_numpy(dtype=float)
        time = frame[spec.get('time_column', 'time')].to_numpy(dtype=float)
        event_auroc(event, risk)
        if not np.isfinite(time).all() or (time < 0).any() or not len(event):
            raise ValueError('Follow-up must be nonnegative and finite; prediction tables must be nonempty.')
        metadata = {k: spec[k] for k in ['endpoint', 'modality', 'model', 'strategy', 'split', 'cohort', 'seed', 'fold', 'target'] if k in spec}
        for repeat in range(-1, bootstrap):
            indices = np.arange(len(frame)) if repeat < 0 else rng.integers(0, len(frame), len(frame))
            e, t, score = event[indices], time[indices], risk[indices]
            try:
                ci = concordance_index(t, -score, e)
            except ZeroDivisionError:
                ci = float('nan')
            rows.append({**metadata, 'bootstrap': repeat, 'estimate': 'point' if repeat < 0 else 'bootstrap',
                         'n': len(e), 'events': int(e.sum()), 'cindex': ci,
                         'event_auroc': event_auroc(e, score)})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'event_metrics_long.csv', index=False)
    table[table.estimate.eq('point')].to_csv(output / 'event_metrics_summary.csv', index=False)
    (output / 'metric_definition.json').write_text(json.dumps({
        'event_auroc': 'Binary AUROC of observed event status versus frozen risk score; higher means higher risk.',
        'cindex': 'Harrell concordance using observed follow-up and event status.',
        'bootstrap': bootstrap, 'seed': seed, 'participant_outputs': False}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    evaluate(args.manifest, args.output, args.bootstrap, args.seed)


if __name__ == '__main__':
    main()
