#!/usr/bin/env python3
"""Run Soma normalization and feature-panel ablations with the shared trainer."""
import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PANELS = {'cardio': 'cardio.csv', 'replicate_1': 'replicate_1.csv',
          'replicate_2': 'replicate_2.csv', 'svm': 'svm.csv'}


def commands(cfg, config_path):
    runner = importlib.import_module('pipelines.run')
    spec = cfg['training']['ablation']
    def resolve(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (config_path.parent / path).resolve()
    cwd = resolve(cfg['training'].get('working_dir', '..'))
    data = {key: str(resolve(value)) for key, value in cfg['training']['data'].items()}
    data.pop('soma_gene_csv', None)
    grid = spec.get('arguments', cfg['training']['single'].get('arguments', {})).copy()
    for key in ['soma_norm', 'soma_gene_csv', 'soma_gene_norm', 'soma_gene_match', 'soma_gene_prefix', 'seed', 'outdir']:
        grid.pop(key, None)
    conditions = []
    for norm in spec.get('normalizations', ['ranknorm', 'minmax']):
        if norm not in {'ranknorm', 'log1p', 'minmax', 'none'}:
            raise ValueError('Invalid Soma normalization.')
        conditions.append(('all_' + norm, {'soma_norm': norm}))
    for panel in spec.get('panels', list(PANELS)):
        if panel not in PANELS:
            raise ValueError('Unknown Soma panel: ' + panel)
        for norm in spec.get('panel_normalizations', ['ranknorm']):
            if norm not in {'ranknorm', 'log1p', 'minmax', 'none'}:
                raise ValueError('Invalid panel normalization.')
            conditions.append((panel + '_' + norm, {
                'soma_norm': 'geneset', 'soma_gene_norm': norm,
                'soma_gene_match': 'feature' if panel == 'svm' else 'gene',
                'soma_gene_csv': str(ROOT / 'pipelines/0_preprocess/soma_panels' / PANELS[panel])}))
    if not conditions or len({name for name, _ in conditions}) != len(conditions):
        raise ValueError('Ablation conditions must be nonempty and unique.')
    seeds = spec.get('seeds', cfg['training']['single']['seeds'])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique.')
    for seed in seeds:
        for name, arguments in conditions:
            output = resolve(cfg.get('training_output', '../private_runs/training')) / 'soma_ablation' / name / ('seed' + str(seed))
            flags = {**data, **grid, **arguments, 'seed': seed, 'outdir': str(output)}
            command = [sys.executable, str(Path(__file__).with_name('train.py')), '--soma', *runner.flags(flags)]
            yield command, cwd, name, seed


def run(config_path, dry_run=False):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    for command, cwd, condition, seed in commands(cfg, config_path):
        print(('Planned: ' if dry_run else 'Running: ') + condition + ' seed=' + str(seed), flush=True)
        if not dry_run:
            if not cwd.is_dir():
                raise ValueError('Configure a valid protected training working directory.')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg['training'].get('gpu', '0')))
            # Sequential execution keeps failures visible and avoids GPU oversubscription.
            subprocess.run(command, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'pipeline.local.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run(args.config, args.dry_run)


if __name__ == '__main__':
    main()
