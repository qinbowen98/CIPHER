"""The two frozen ASCVD predictors used by the final UKB/MUCH analysis."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np

IDENTITIES = {
    'risk': {'endpoint': 'ASCVD', 'strategy': 'top_risk_label_classification', 'target': 'top50', 'model_name': 'Logistic'},
    'event': {'endpoint': 'ASCVD', 'strategy': 'direct_survival_regression', 'target': 'event_time', 'model_name': 'RF'},
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
