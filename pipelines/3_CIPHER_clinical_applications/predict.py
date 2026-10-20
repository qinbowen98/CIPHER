"""Apply the frozen clinical models to a cohort feature table."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .models.final_pair import FEATURES, load_pair, score

ROOT = Path(__file__).resolve().parents[2]


def predict(input_path, output_path, id_column=None, model_dir=None):
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output or output.exists():
        raise ValueError('Choose a new output file distinct from the input.')
    frame = pd.read_parquet(source) if source.suffix.lower() == '.parquet' else pd.read_csv(source, dtype={id_column: str} if id_column else None)
    missing = [name for name in FEATURES if name not in frame]
    if missing:
        raise ValueError('Missing model features: ' + ', '.join(missing))
    values = frame[FEATURES].apply(pd.to_numeric, errors='raise')
    if values.empty or values.isna().all(axis=1).any() or np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError('Provide finite numeric features; each row must have observed features.')
    if id_column:
        if id_column not in frame or frame[id_column].isna().any() or frame[id_column].duplicated().any():
            raise ValueError('Provide a unique, nonmissing identifier column.')
        result = frame[[id_column]].copy()
    else:
        result = pd.DataFrame({'row_index': np.arange(len(frame))})
    pair, _ = load_pair(model_dir or ROOT / 'release')
    result['risk_label_probability'] = score(pair['risk'], values)
    result['event_risk_score'] = score(pair['event'], values)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print('Scored rows:', len(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='CSV or Parquet feature table.')
    parser.add_argument('--output', required=True, help='New CSV prediction file.')
    parser.add_argument('--id-column', help='Optional unique identifier column retained in output.')
    parser.add_argument('--model-dir', help='Directory containing public_models/risk.joblib and event.joblib.')
    args = parser.parse_args()
    predict(args.input, args.output, args.id_column, args.model_dir)


if __name__ == '__main__':
    main()
