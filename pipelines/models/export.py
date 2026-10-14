"""Export vetted clinical predictors and aggregate metrics; no recursive data copy."""
import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .sanitize import public_artifact, PrivateStateError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines/clinical"))
from clinical_cipher.common import clip_frame
from pipelines.inference.prepared import predict


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


TEXT = {
    "endpoint": {"ASCVD", "CHD", "istroke", "stroke", "hstroke", "ICVD", "ICVDr", "UA", "CVDdeath"},
    "strategy": {"risk_regression", "risk_cox", "top_risk_label_classification", "direct_survival_regression", "direct_event_classification", "direct_any_event_classification", "clinical_cox"},
    "target": {"multimodal_normalized_risk", "clinical_fitted_risk_then_cox", "event_time", "event_by_5y", "any_event", "top5", "top10", "top20", "top25", "top50"},
    "model": {"Ridge", "ElasticNet", "RF", "ExtraTrees", "GBR", "HistGBR", "GBC", "HistGBC", "SVR_RBF", "KNN", "XGB", "Logistic", "CoxPH_penalized", "RiskCox_after_ElasticNet", "RiskCox_after_SVR_RBF", "RiskCox_after_Ridge", "RiskCox_after_RF", "RiskCox_after_ExtraTrees", "RiskCox_after_GBR", "RiskCox_after_HistGBR", "RiskCox_after_KNN", "RiskCox_after_XGB"},
    "setting": {"Clinical to normalized risk", "Clinical to risk to Cox", "Clinical to top-risk label", "Clinical to time-to-event", "Clinical Cox", "Clinical to event"},
    "split": {"train", "valid", "test", "external", "UKB", "MUCH"},
    "cohort": {"UKB", "MUCH", "Shougang", "BD", "CMCS"},
    "age_band": {"40-74", "50-74", "60-74", "all"},
    "label": {"Framingham", "China-PAR", "CHINA-PAR", "CKM2S2-BAG", "Best risk-based", "Best event-based", "Best risk-based CIPHER", "Best event-based CIPHER", "Framingham2008", "ChinaPAR", "CKM2S2_BAG", "CIPHER_risk", "CIPHER_event"},
}
METRICS = {"event_auroc", "cindex", "point_event_auroc", "point_cindex", "horizon_auroc", "auc_5y", "roc_auc_5y_binary", "valid_cindex", "test_cindex", "best_valid_cindex", "risk_quantile", "train_top_rate"}
COUNTS = {"n", "n_before_complete_case", "events", "n_events", "n_event", "n_event_5y", "train_n_top", "setting_order", "bootstrap", "seed", "replicate"}
STATS = {"risk_cutoff"}
TEXT["selected_risk_model"] = TEXT["model"]
ALIASES = {"model_name": "model", "algorithm": "model", "age_group": "age_band"}


def clean_metrics(path):
    # Only model/replicate-level tables with recognized metric schemas are eligible.
    frame = pd.read_csv(path, dtype=str).rename(columns=ALIASES)
    columns = set(frame.columns)
    allowed = set(TEXT) | METRICS | COUNTS | STATS | {"observation"}
    if columns - allowed:
        raise ValueError("Non-aggregate/unreviewed columns: " + ", ".join(sorted(columns - allowed)))
    if not columns & METRICS or not columns & {"model", "setting", "label"}:
        raise ValueError("Expected model/setting-level metrics")
    if frame.empty or len(frame) > 200000:
        raise ValueError("Empty or unexpected aggregate table size")
    for col in columns & set(TEXT):
        if not frame[col].isin(TEXT[col]).all():
            raise ValueError("Unreviewed categorical values in " + col)
    if "observation" in columns:
        if not frame.observation.str.fullmatch(r"model|bootstrap_[0-9]+").all():
            raise ValueError("Unreviewed observation labels")
    for col in columns & (METRICS | COUNTS | STATS):
        if col == "bootstrap":
            frame[col] = frame[col].replace({"final": "-1"})
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        finite = frame[col].dropna()
        if not np.isfinite(finite).all():
            raise ValueError("Non-finite numeric values in " + col)
        if col in METRICS and not finite.between(0, 1).all():
            raise ValueError("Metric outside [0,1]")
        if col in COUNTS and not (finite == np.floor(finite)).all():
            raise ValueError("Noninteger count/index")
    group = [c for c in frame if c in TEXT or c in {"bootstrap", "seed", "replicate", "observation"}]
    if frame.duplicated(group).any():
        raise ValueError("Duplicate aggregate keys; input may contain participant rows")
    return frame


def export_models(source, output):
    files = sorted(source.glob("**/final.joblib"))
    if not files:
        raise FileNotFoundError("No final.joblib found in model directory")
    report = []
    for path in files:
        row = {"source_sha256": sha256(path)}
        try:
            original = joblib.load(path)  # trusted local training artifacts only
            clean = public_artifact(original)
            for key in ("endpoint", "strategy", "target", "model_name"):
                value = clean[key]
                if value not in TEXT[{"model_name": "model"}.get(key, key)]:
                    raise PrivateStateError("Unreviewed " + key)
                row[key] = value
            features = clean["feature_cols"]
            if not features or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", f) for f in features):
                raise PrivateStateError("Unreviewed feature names")
            # Parity uses generated feature vectors; no participant input enters this test.
            rng = np.random.RandomState(20261013)
            x = pd.DataFrame(rng.normal(50, 15, (64, len(features))), columns=features)
            x = clip_frame(x, clean.get("clip_limits", {}))
            x.iloc[0, 0] = np.nan
            expected = predict(original, x)
            if not np.isfinite(expected).all():
                raise PrivateStateError("Original prediction nonfinite on generated probe")
            relative = Path(*[row[k] for k in ("endpoint", "strategy", "target", "model_name")]) / "final.joblib"
            target = output / "public_models" / relative
            if target.exists():
                raise PrivateStateError("Duplicate model key; use one frozen final collection")
            with tempfile.TemporaryDirectory() as temp:
                tmp = Path(temp) / "clean.joblib"
                joblib.dump(clean, tmp, compress=3)
                restored = joblib.load(tmp)
                np.testing.assert_allclose(predict(restored, x), expected, rtol=1e-9, atol=1e-11)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tmp.read_bytes())
            row.update(status="exported", file="public_models/" + relative.as_posix(), sha256=sha256(target), parity="generated_inputs_passed")
        except Exception as error:
            # Avoid exposing source paths, sample values or arbitrary exception text.
            reason = str(error) if isinstance(error, PrivateStateError) else type(error).__name__
            row.update(status="blocked", reason=reason)
        report.append(row)
    dump_json(output / "model_manifest.json", report)
    return report
