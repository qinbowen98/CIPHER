"""Frozen clinical inference for prepared UKB/MUCH tables; aggregate output by default."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..clinical_cipher.common import clip_frame, minmax_apply, predict_estimator_score, read_table, safe_cindex, safe_event_auc


def predict(artifact, frame):
    p = artifact["predictor"]
    features = artifact["feature_cols"]
    x = clip_frame(frame[features], artifact.get("clip_limits", {}))
    if p["kind"] == "estimator":
        return np.asarray(predict_estimator_score(p["pipeline"], x), float) * float(p.get("risk_direction", 1))
    if p["kind"] == "clinical_cox":
        x = pd.DataFrame(p["scaler"].transform(p["imputer"].transform(x)), columns=features)
    elif p["kind"] == "risk_cox":
        risk = predict_estimator_score(p["risk_pipeline"], x)
        x = pd.DataFrame({"risk": minmax_apply(risk, p["risk_output_transform"])})
    else:
        raise ValueError("Unknown predictor kind")
    return p["cox"].predict_partial_hazard(x).to_numpy().reshape(-1)


def run(cfg, cohort, save_private=False):
    spec = cfg["inference"][cohort.lower()]
    paths = sorted(Path(spec["models_dir"]).glob("**/final.joblib"))
    models = []
    for path in paths:
        a = joblib.load(path)
        if a["endpoint"] not in spec["endpoints"]:
            continue
        selection = spec.get("selection", {})
        if any(a.get(k) not in values for k, values in selection.items()):
            continue
        models.append((path, a))
    if not models:
        raise ValueError("No frozen models match configured endpoints/selection")
    table = read_table(spec["table"])
    id_col = spec["id_column"]
    if table[id_col].isna().any() or table[id_col].astype(str).duplicated().any():
        raise ValueError("Missing or duplicate participant IDs in prepared table")
    output = Path(cfg["private_output"]) / cohort.lower()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for endpoint in spec["endpoints"]:
        chosen = [(path, a) for path, a in models if a["endpoint"] == endpoint]
        if not chosen:
            raise ValueError("Missing endpoint models: " + endpoint)
        required_strategies = set(spec.get("required_strategies", []))
        available = {artifact["strategy"] for _, artifact in chosen}
        if required_strategies - available:
            raise ValueError("Missing frozen strategies: " + ", ".join(sorted(required_strategies - available)))
        cols = sorted({col for _, a in chosen for col in a["feature_cols"]})
        outcomes = spec["outcomes"][endpoint]
        data = table.copy()
        t = pd.to_numeric(data[outcomes["time"]], errors="coerce")
        e = pd.to_numeric(data[outcomes["event"]], errors="coerce")
        keep = np.isfinite(t) & (t > 0) & e.isin([0, 1])
        for col in cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        if spec.get("complete_case", True):
            keep &= np.isfinite(data[cols]).all(axis=1)
        if spec.get("age_range"):
            keep &= pd.to_numeric(data["age_feat"], errors="coerce").between(*spec["age_range"])
        data = data.loc[keep].reset_index(drop=True)
        time, event = t.loc[keep].to_numpy(), e.loc[keep].to_numpy(dtype=int)
        if len(data) < 5 or len(np.unique(event)) < 2:
            raise ValueError("Insufficient evaluable outcome data")
        for name, actual in [("expected_n", len(data)), ("expected_events", int(event.sum()))]:
            expected = spec.get(name)
            if expected is not None and expected != actual:
                raise ValueError(name + " mismatch: verify final cohort/version")
        scores = [(path, a, predict(a, data)) for path, a in chosen]
        if any(len(score) != len(data) or not np.isfinite(score).all() for _, _, score in scores):
            raise ValueError("Nonfinite model predictions")
        def record(a, score, indices, replicate):
            return {"cohort": cohort, "endpoint": endpoint, "strategy": a["strategy"], "target": a["target"], "model": a["model_name"], "replicate": replicate, "bootstrap": replicate, "split": "test",
                    "n": len(indices), "events": int(event[indices].sum()),
                    "event_auroc": safe_event_auc(event[indices], score[indices]),
                    "cindex": safe_cindex(time[indices], event[indices], score[indices])}
        all_indices = np.arange(len(data))
        rows.extend(record(a, score, all_indices, 0) for _, a, score in scores)
        # Paired, event-stratified participant bootstrap: identical draws per model.
        rng = np.random.RandomState(spec.get("seed", 66))
        cases, controls = all_indices[event == 1], all_indices[event == 0]
        for rep in range(1, spec.get("bootstrap", 0) + 1):
            indices = np.concatenate([rng.choice(cases, len(cases), replace=True), rng.choice(controls, len(controls), replace=True)])
            rows.extend(record(a, score, indices, rep) for _, a, score in scores)
        if save_private:
            predictions = pd.DataFrame({id_col: data[id_col], "time": time, "event": event})
            for _, a, score in scores:
                key = "/".join(a[k] for k in ["strategy", "target", "model_name"])
                predictions[key] = score
            predictions.to_parquet(output / (endpoint + "_PRIVATE_predictions.parquet"), index=False)
    metrics = output / "aggregate_metrics.csv"
    result = pd.DataFrame(rows)
    result.to_csv(metrics, index=False)
    print("Saved aggregate model metrics.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", choices=["UKB", "MUCH"], required=True)
    parser.add_argument("--save-private-predictions", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.cohort, args.save_private_predictions)
