#!/usr/bin/env python3
"""Preprocessing, unimodal training, multimodal training, and clinical validation."""
import argparse
import importlib
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


def resolve(value, config_path):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def training_commands(stage, cfg, config_path):
    spec = cfg['training'][stage]
    common = {key: str(resolve(value, config_path)) for key, value in cfg['training']['data'].items()}
    if stage == 'multi':
        common.pop('cmcs_standard_csv', None)
        common.pop('fundus_embed_dir', None)
    for seed in spec['seeds']:
        combinations = [[m] for m in spec['modalities']] if stage == 'single' else spec['combinations']
        for mods in combinations:
            if not mods or set(mods) - set(MODALITIES) or len(set(mods)) != len(mods):
                raise ValueError('Invalid modality combination.')
            if stage == 'multi' and len(mods) < 2:
                raise ValueError('Multimodal training requires at least two modalities.')
            output = resolve(cfg.get('training_output', 'private_runs/training'), config_path) / stage / ('_'.join(mods) + '_seed' + str(seed))
            folder = '1_single_modality' if stage == 'single' else '2_multi_modality'
            command = [sys.executable, str(ROOT / 'pipelines' / folder / 'train.py')]
            command += ['--' + mod for mod in mods] + flags({**common, **spec.get('arguments', {}), 'seed': seed, 'outdir': str(output)})
            yield command, resolve(cfg['training'].get('working_dir', '..'), config_path), mods, seed


def preprocessing_commands(cfg, config_path, target):
    for name in (list(cfg.get('preprocessing', {})) if target == 'all' else [target]):
        spec = cfg.get('preprocessing', {}).get(name)
        if spec is None:
            raise ValueError('Define this preprocessing target in the local configuration.')
        arguments = dict(spec.get('arguments', {}))
        for key, value in spec['paths'].items():
            if value is not None:
                arguments[key] = str(resolve(value, config_path))
        module = 'imaging_ecg' if name in {'fundus', 'ecg'} else name
        if name in {'fundus', 'ecg'}:
            arguments['kind'] = name
        yield [sys.executable, '-m', 'pipelines.0_preprocess.' + module, *flags(arguments)], name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['init', 'preprocess', 'ablation', 'single', 'multi', 'ukb', 'much', 'infer'])
    parser.add_argument('--target', choices=['all', 'snp', 'soma', 'fundus', 'ecg'], default='all', help='Preprocessing target.')
    parser.add_argument('--config', type=Path, default=ROOT / 'pipeline.local.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.stage == 'init':
        sys.path.insert(0, str(ROOT))
        defaults = importlib.import_module('pipelines.settings').DEFAULT_CONFIG
        with config_path.open('x') as handle:
            json.dump(defaults, handle, indent=2)
            handle.write('\n')
        print('Created local pipeline configuration.')
        return
    if not config_path.is_file():
        parser.error('Configuration not found. Run python pipelines/run.py init first.')
    cfg = json.loads(config_path.read_text())
    if args.stage == 'preprocess':
        for command, name in preprocessing_commands(cfg, config_path, args.target):
            print(('Planned: ' if args.dry_run else 'Running: ') + 'preprocess ' + name, flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
        return
    if args.stage == 'ablation':
        sys.path.insert(0, str(ROOT))
        importlib.import_module('pipelines.1_single_modality.soma_ablation').run(config_path, args.dry_run)
        return
    if args.stage in {'single', 'multi'}:
        for command, cwd, modalities, seed in training_commands(args.stage, cfg, config_path):
            print(('Planned: ' if args.dry_run else 'Running: ') + args.stage + ' ' + ','.join(modalities) + ' seed=' + str(seed), flush=True)
            if not args.dry_run:
                if not cwd.is_dir():
                    raise ValueError('Configure a valid protected training working directory.')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg['training'].get('gpu', '0')))
                subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    for cohort in (['UKB', 'MUCH'] if args.stage == 'infer' else [args.stage.upper()]):
        print(('Planned: ' if args.dry_run else 'Running: ') + cohort + ' final frozen pair', flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, '-m', 'pipelines.3_CIPHER_clinical_applications.inference.final_pair',
                            '--config', str(config_path), '--cohort', cohort], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
