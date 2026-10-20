"""Evaluate six clinical prediction strategies with frozen source-cohort models."""
import argparse
from pathlib import Path
from .prepared import run

STRATEGIES = ['risk_regression', 'risk_cox', 'top_risk_label_classification',
              'direct_survival_regression', 'direct_any_event_classification', 'clinical_cox']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=['UKB', 'MUCH'], required=True)
    parser.add_argument('--table', required=True)
    parser.add_argument('--models-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--endpoint', default='ASCVD')
    parser.add_argument('--id-column', required=True)
    parser.add_argument('--event-column', required=True)
    parser.add_argument('--time-column', required=True)
    parser.add_argument('--age-range', nargs=2, type=float)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=66)
    args = parser.parse_args()
    spec = {'models_dir': str(Path(args.models_dir).resolve()), 'table': str(Path(args.table).resolve()),
            'id_column': args.id_column, 'endpoints': [args.endpoint],
            'outcomes': {args.endpoint: {'event': args.event_column, 'time': args.time_column}},
            'selection': {'strategy': STRATEGIES}, 'required_strategies': STRATEGIES,
            'complete_case': True, 'bootstrap': args.bootstrap, 'seed': args.seed}
    if args.age_range:
        spec['age_range'] = args.age_range
    run({'private_output': args.output, 'inference': {args.cohort.lower(): spec}}, args.cohort)


if __name__ == '__main__':
    main()
