"""The two frozen ASCVD predictors used by the final UKB/MUCH analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
from .sanitize import public_artifact

IDENTITIES = {
    'risk': {'endpoint': 'ASCVD', 'strategy': 'top_risk_label_classification', 'target': 'top50', 'model_name': 'Logistic'},
    'event': {'endpoint': 'ASCVD', 'strategy': 'direct_survival_regression', 'target': 'event_time', 'model_name': 'RF'},
}
RELATIVE_SOURCES = {
    'risk': 'ASCVD/top_risk_label_classification/top50/Logistic/final.joblib',
    'event': 'ASCVD/direct_survival_regression/event_time/RF/final.joblib',
}
FEATURES = ['age_feat', 'sex', 'SBP', 'DBP', 'TC', 'TG', 'HDLC', 'LDLC', 'BMI', 'height', 'weight', 'waist', 'smoking']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_identity(artifact, role):
    for key, value in IDENTITIES[role].items():
        if artifact.get(key) != value:
            raise ValueError('Wrong frozen ' + role + ' model: ' + key)
    if list(artifact['feature_cols']) != FEATURES:
        raise ValueError('Unexpected final model feature contract')
    if artifact.get('source_training_wave') != 1.0:
        raise ValueError('Expected clinical wave 1.0; do not substitute a wave-3 model')
    if artifact['predictor']['kind'] != 'estimator':
        raise ValueError('Unexpected final predictor representation')


def score(artifact, frame):
    # Historical final deployment applies the fitted pipeline directly, without
    # clipping, cohort-based scaling, refitting, or threshold selection.
    p = artifact['predictor']
    x = frame[artifact['feature_cols']]
    if hasattr(p['pipeline'], 'predict_proba'):
        raw = np.asarray(p['pipeline'].predict_proba(x))[:, 1]
    else:
        raw = np.asarray(p['pipeline'].predict(x)).reshape(-1)
    return raw * float(p.get('risk_direction', 1.0))


def export_pair(risk_path, event_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('The release directory must be empty')
    payloads, rows = [], []
    for role, path in [('risk', Path(risk_path)), ('event', Path(event_path))]:
        original = joblib.load(path)  # trusted local weights, in their training environment
        verify_identity(original, role)
        clean = public_artifact(original)
        clean['source_training_wave'] = 1.0
        verify_identity(clean, role)
        rng = np.random.RandomState(66)
        probes = pd.DataFrame(rng.normal(50, 15, (64, len(FEATURES))), columns=FEATURES)
        probes.iloc[0, 0] = np.nan
        reference = score(original, probes)
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp) / 'model.joblib'
            joblib.dump(clean, tmp, compress=3)
            restored = joblib.load(tmp)
            np.testing.assert_allclose(score(restored, probes), reference, rtol=1e-9, atol=1e-11)
            payloads.append((role, tmp.read_bytes()))
        rows.append({'role': role, **IDENTITIES[role], 'source_training_wave': 1.0,
                     'source_sha256': sha256(path), 'source_relative_name': RELATIVE_SOURCES[role],
                     'file': 'public_models/' + role + '.joblib', 'parity': 'generated_inputs_passed'})
    for role, content in payloads:
        target = output / 'public_models' / (role + '.joblib')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    files = []
    for row in rows:
        row['sha256'] = sha256(output / row['file'])
        files.append({'file': row['file'], 'sha256': row['sha256']})
    manifest = output / 'model_manifest.json'
    manifest.write_text(json.dumps(rows, indent=2) + '\n')
    files.append({'file': manifest.name, 'sha256': sha256(manifest)})
    (output / 'release_manifest.json').write_text(json.dumps({'schema': 'cipher-final-pair-v1', 'sample_rows': False, 'files': files}, indent=2) + '\n')
    print('Exported the two frozen ASCVD models. No participant records were copied.')


def load_pair(directory):
    directory = Path(directory)
    rows = json.loads((directory / 'model_manifest.json').read_text())
    if len(rows) != 2 or {row['role'] for row in rows} != {'risk', 'event'}:
        raise ValueError('Expected exactly two frozen models')
    pair = {}
    for row in rows:
        role = row['role']
        if row['file'] != 'public_models/' + role + '.joblib':
            raise ValueError('Unexpected model path in manifest')
        path = directory / row['file']
        if sha256(path) != row['sha256']:
            raise ValueError('Frozen model checksum mismatch')
        pair[role] = joblib.load(path)
        verify_identity(pair[role], role)
    return pair, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--risk', type=Path, required=True)
    parser.add_argument('--event', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('release'))
    args = parser.parse_args()
    export_pair(args.risk, args.event, args.output)
