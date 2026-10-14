#!/usr/bin/env python3
"""Training and final two-model external validation CLI."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ['snp', 'soma', 'ecg', 'fundus']


def flags(mapping):
    result = []
    for key, value in mapping.items():
        if value is None or value is False:
            continue
        result.append('--' + key)
        if value is not True:
            result.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return result


def training_commands(stage, cfg, config_path):
    spec = cfg['training'][stage]
    def path(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (config_path.parent / p).resolve())
    common = {key: path(value) for key, value in cfg['training']['data'].items()}
    if stage == 'multi':
        common.pop('cmcs_standard_csv', None)
        common.pop('fundus_embed_dir', None)
    for seed in spec['seeds']:
        combos = [[m] for m in spec['modalities']] if stage == 'single' else spec['combinations']
        for mods in combos:
            if not mods or set(mods)-set(MODALITIES) or len(set(mods)) != len(mods):
                raise ValueError('Invalid modality combination')
            if stage == 'multi' and len(mods) < 2:
                raise ValueError('Multimodal training requires at least two modalities')
            output = Path(path(cfg.get('training_output', '../private_runs/training'))) / stage / ('_'.join(mods) + '_seed' + str(seed))
            script = ROOT / 'pipelines' / ('single_modal' if stage == 'single' else 'multi_modal') / 'train.py'
            command = [sys.executable, str(script)] + ['--' + mod for mod in mods] + flags({**common, **spec.get('arguments', {}), 'seed': seed, 'outdir': str(output)})
            yield command, path(cfg['training'].get('working_dir', '..')), mods, seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['single', 'multi', 'ukb', 'much', 'infer', 'publish-results', 'check'])
    p.add_argument('--config', type=Path, default=ROOT / 'config/pipeline.local.json')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.stage == 'check':
        subprocess.run([sys.executable, str(ROOT/'scripts/check_public_release.py')], cwd=ROOT, check=True)
        return
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    if args.stage in {'single', 'multi'}:
        for command, cwd, modalities, seed in training_commands(args.stage, cfg, config_path):
            print(('Planned: ' if args.dry_run else 'Running: ') + args.stage + ' ' + ','.join(modalities) + ' seed=' + str(seed), flush=True)
            if not args.dry_run:
                if not Path(cwd).is_dir():
                    raise ValueError('Configure a valid protected training working directory')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg['training'].get('gpu', '0')))
                subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    for cohort in (['UKB', 'MUCH'] if args.stage == 'infer' else [args.stage.upper()] if args.stage in {'ukb', 'much'} else []):
        print(('Planned: ' if args.dry_run else 'Running: ') + cohort + ' final frozen pair', flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, '-m', 'pipelines.inference.final_pair', '--config', str(config_path), '--cohort', cohort], cwd=ROOT, check=True)
    if args.stage == 'publish-results':
        if args.dry_run:
            print('Planned: publish validated aggregate UKB/MUCH metrics only')
        else:
            subprocess.run([sys.executable, '-m', 'pipelines.models.publish_results', '--config', str(config_path)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
