#!/usr/bin/env python3
"""Train six clinical prediction strategies and save source-cohort models."""
from __future__ import annotations

import argparse
import json
import platform
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
try:
    from lifelines import CoxPHFitter
except ImportError:  # Non-Cox strategies remain usable in lightweight environments.
    CoxPHFitter = None
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .clinical_cipher.common import (
    available_model_names,
    build_classifier_pipeline,
    build_pipeline,
    clip_frame,
    dump_artifact,
    legacy_split_minmax,
    load_yaml,
    minmax_apply,
    minmax_fit,
    normalize_eid,
    predict_estimator_score,
    safe_cindex,
    safe_event_auc,
    save_json,
)


ALL_STRATEGIES = [
    "risk_regression", "direct_survival_regression",
    "direct_any_event_classification",
    "clinical_cox", "risk_cox", "top_risk_label_classification",
]


def _normalize_visit_token(value) -> str:
    token = str(value).strip()
    return token[:-2] if token.endswith(".0") else token


def _parse_risk_key(key, visit_to_wave: Dict[str, float]) -> Tuple[str, float, str]:
    """Parse common eid/visit encodings without guessing a missing visit."""
    visit_map = {_normalize_visit_token(k): float(v) for k, v in visit_to_wave.items()}
    for wave in sorted(set(visit_map.values())):
        visit_map[str(int(wave))] = wave
    if isinstance(key, (tuple, list)) and len(key) >= 2:
        participant, visit = key[0], _normalize_visit_token(key[-1])
        if visit in visit_map:
            return str(int(float(participant))), visit_map[visit], "tuple"
    text = str(key).strip()
    for separator in ["-", "_", ":", "|", "/"]:
        if separator not in text:
            continue
        participant, visit = text.rsplit(separator, 1)
        visit = _normalize_visit_token(visit)
        if visit in visit_map:
            return str(int(float(participant))), visit_map[visit], f"separator:{separator}"
    # Handles strings such as "(123456, '12')".
    match = re.match(r"^\(?\s*([0-9]+(?:\.0)?)\s*,\s*['\"]?([0-9]+(?:\.0)?)['\"]?\s*\)?$", text)
    if match and _normalize_visit_token(match.group(2)) in visit_map:
        visit = _normalize_visit_token(match.group(2))
        return str(int(float(match.group(1)))), visit_map[visit], "tuple_string"
    raise ValueError("unrecognized eid/visit key")


def _extract_first_risk_scalar(value) -> Tuple[float, str]:
    """Extract the first numeric risk while avoiding ragged np.asarray calls."""
    if isinstance(value, Mapping):
        priority = ["risk", "score", "prediction", "pred", "logit", "hazard"]
        lower = {str(k).lower(): k for k in value}
        for name in priority:
            if name in lower:
                scalar, mode = _extract_first_risk_scalar(value[lower[name]])
                return scalar, f"dict:{name}>{mode}"
        if len(value) == 1:
            scalar, mode = _extract_first_risk_scalar(next(iter(value.values())))
            return scalar, f"dict:single>{mode}"
        raise ValueError(f"ambiguous dict value keys={list(value)[:8]}")
    if np.isscalar(value) and not isinstance(value, (str, bytes)):
        scalar = float(value)
        if np.isfinite(scalar):
            return scalar, "scalar"
        raise ValueError("non-finite scalar")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _extract_first_risk_scalar(value.item())
        if value.dtype != object:
            flat = value.reshape(-1)
            if flat.size:
                return _extract_first_risk_scalar(flat[0])
        else:
            for item in value.flat:
                try:
                    scalar, mode = _extract_first_risk_scalar(item)
                    return scalar, f"object_array>{mode}"
                except (TypeError, ValueError):
                    continue
        raise ValueError("numeric ndarray is empty or non-finite")
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                scalar, mode = _extract_first_risk_scalar(item)
                return scalar, f"sequence>{mode}"
            except (TypeError, ValueError):
                continue
        raise ValueError("sequence contains no numeric scalar")
    raise TypeError(f"unsupported value type {type(value).__name__}")


def load_cipher_risk(path: Path, visit_to_wave: Dict[str, float]) -> Tuple[pd.DataFrame, dict]:
    loaded = np.load(path, allow_pickle=True)
    raw = loaded.item() if (
        isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.size == 1
    ) else loaded
    if not isinstance(raw, Mapping):
        raise ValueError(f"top-level object is {type(raw).__name__}, expected dict-like mapping")
    rows, bad_examples = [], []
    key_modes, value_modes = Counter(), Counter()
    for key, value in raw.items():
        try:
            participant, wave, key_mode = _parse_risk_key(key, visit_to_wave)
            scalar, value_mode = _extract_first_risk_scalar(value)
            rows.append({"ID": participant, "wave": wave, "cipher_risk_raw": scalar})
            key_modes[key_mode] += 1
            value_modes[value_mode] += 1
        except Exception as exc:
            if len(bad_examples) < 10:
                bad_examples.append({"key": repr(key), "value_type": type(value).__name__,
                                     "value_preview": repr(value)[:180], "reason": str(exc)})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(
            f"No valid CIPHER-risk entries parsed from {path}; top-level={type(raw).__name__}, "
            f"n_items={len(raw)}, examples={json.dumps(bad_examples, ensure_ascii=False)}"
        )
    result = result.sort_values(["ID", "wave"]).drop_duplicates(["ID", "wave"], keep="last")
    transform = minmax_fit(result["cipher_risk_raw"])
    result["cipher_risk"] = minmax_apply(result["cipher_risk_raw"], transform)
    print(f"[risk] {path}: rows={len(result)}, IDs={result.ID.nunique()}, "
          f"bad_items={len(raw) - sum(key_modes.values())}", flush=True)
    print(f"[risk] key modes={dict(key_modes)}; value modes={dict(value_modes)}", flush=True)
    if bad_examples:
        print("[risk] first rejected items=" + json.dumps(bad_examples, ensure_ascii=False, indent=2), flush=True)
    return result.reset_index(drop=True), transform


def load_first_valid_risk(paths: Sequence[str], visit_to_wave: Dict[str, float], required_wave: float):
    failures, found = [], []
    for value in paths:
        path = Path(value)
        if not path.exists():
            continue
        found.append(path)
        try:
            risk, transform = load_cipher_risk(path, visit_to_wave)
            if not risk["wave"].eq(float(required_wave)).any():
                raise ValueError(
                    f"parsed successfully but has no required wave={required_wave}; "
                    f"wave_counts={risk['wave'].value_counts().sort_index().to_dict()}"
                )
            return risk, transform, path
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            print(f"[risk] candidate rejected: {path}\n  {exc}", flush=True)
    if not found:
        raise FileNotFoundError("None of these CIPHER-risk files exists:\n" + "\n".join(paths))
    raise ValueError("Every existing CIPHER-risk candidate failed:\n- " + "\n- ".join(failures))


def configured_source_waves(source: dict) -> Tuple[float, float]:
    """Return the source-training and evaluation waves."""
    legacy_wave = source.get("selected_wave")
    if "training_wave" not in source and legacy_wave is None:
        raise ValueError("source.training_wave is required")
    if "evaluation_wave" not in source and legacy_wave is None:
        raise ValueError("source.evaluation_wave is required")
    training_wave = float(source.get("training_wave", legacy_wave))
    evaluation_wave = float(source.get("evaluation_wave", legacy_wave))
    mapped_waves = {float(value) for value in source["visit_to_wave"].values()}
    if training_wave not in mapped_waves:
        raise ValueError(
            f"source.training_wave={training_wave} is absent from visit_to_wave; "
            f"mapped waves={sorted(mapped_waves)}"
        )
    return training_wave, evaluation_wave


def choose_present(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"No {label} column found. Tried {list(candidates)}")


def load_source_clinical(cfg: dict) -> pd.DataFrame:
    source = cfg["source"]
    path = Path(cfg["paths"]["longitudinal_tsv"])
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame = frame.loc[:, ~frame.columns.duplicated()].rename(columns=source.get("source_column_rename", {})).copy()
    id_column, wave_column = source["id_column"], source["wave_column"]
    frame[id_column] = normalize_eid(frame[id_column])
    frame[wave_column] = pd.to_numeric(frame[wave_column], errors="coerce")
    offsets = {float(k): float(v) for k, v in source["wave_time_offsets_years"].items()}
    offset = frame[wave_column].map(offsets).fillna(0.0)
    time_columns = {x for ep in cfg["endpoints"].values() for x in ep["time_candidates"]}
    for column in sorted(time_columns):
        if column in frame:
            frame[column] = (pd.to_numeric(frame[column], errors="coerce") - offset).clip(
                lower=float(source["minimum_followup_years"])
            )
    training_wave, evaluation_wave = configured_source_waves(source)
    retained_waves = sorted({training_wave, evaluation_wave})
    frame = frame[frame[wave_column].isin(retained_waves)].copy()
    before = len(frame)
    frame = (
        frame.sort_values([id_column, wave_column])
        .drop_duplicates([id_column, wave_column], keep="last")
    )
    wave_counts = (
        frame.groupby(wave_column)[id_column].nunique().sort_index().to_dict()
    )
    print(
        f"[clinical] training_wave={training_wave}, evaluation_wave={evaluation_wave}: "
        f"{before} raw rows -> {len(frame)} unique ID-wave rows; "
        f"unique_IDs_by_wave={wave_counts}",
        flush=True,
    )
    return frame.rename(columns={id_column: "ID", wave_column: "wave"})


def load_cohort_map(cfg: dict) -> pd.DataFrame:
    source = cfg["source"]
    columns = [source["baseline_id_column"], source["subcohort_column"]]
    frame = pd.read_csv(cfg["paths"]["baseline_tsv"], sep="\t", usecols=columns, low_memory=False)
    frame[columns[0]] = normalize_eid(frame[columns[0]])
    return frame.drop_duplicates(columns[0], keep="last").rename(columns={columns[0]: "ID"})


def feature_summary(frame: pd.DataFrame, features: Sequence[str]) -> Dict[str, dict]:
    output = {}
    for feature in features:
        value = pd.to_numeric(frame[feature], errors="coerce")
        output[feature] = {
            "n": int(value.notna().sum()), "missing_fraction": float(value.isna().mean()),
            "mean": float(value.mean()), "std": float(value.std()), "min": float(value.min()),
            "q01": float(value.quantile(.01)), "median": float(value.median()),
            "q99": float(value.quantile(.99)), "max": float(value.max()),
        }
    return output


def prepare_endpoint(cfg: dict, endpoint_key: str, clinical: pd.DataFrame,
                     cohort_map: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    spec = cfg["endpoints"][endpoint_key]
    source = cfg["source"]
    training_wave, evaluation_wave = configured_source_waves(source)
    endpoint = spec["display"]
    event_col = choose_present(clinical, spec["event_candidates"], f"{endpoint} event")
    time_col = choose_present(clinical, spec["time_candidates"], f"{endpoint} time")
    risk, _, risk_path = load_first_valid_risk(
        spec["risk_candidates"], source["visit_to_wave"], training_wave
    )
    risk_train = risk[risk["wave"].eq(training_wave)].copy()
    target_transform = minmax_fit(risk_train["cipher_risk_raw"])
    risk_train["cipher_risk"] = minmax_apply(
        risk_train["cipher_risk_raw"], target_transform
    )
    clinical_train = clinical[clinical["wave"].eq(training_wave)].copy()
    train = risk_train.merge(
        clinical_train, on=["ID", "wave"], how="inner", validate="one_to_one"
    )
    clinical_evaluation = clinical[clinical["wave"].eq(evaluation_wave)].copy()
    # Validation/test must remain independent of every participant represented
    # in the CIPHER-risk source, including source participants who did not have
    # a complete clinical row at the configured training wave.
    all_cipher_risk_ids = set(risk["ID"])
    external = clinical_evaluation[
        ~clinical_evaluation.ID.isin(all_cipher_risk_ids)
    ].copy()
    label_col = source["subcohort_column"]
    external = external.merge(cohort_map[["ID", label_col]], on="ID", how="left", validate="one_to_one")
    test_label = str(source["external_test_subcohort"])
    test = external[external[label_col].astype(str).eq(test_label)].copy()
    valid = external[~external[label_col].astype(str).eq(test_label)].copy()
    print(
        f"[{endpoint}] source split before complete-case filtering: "
        f"risk_wave_{training_wave:g}={risk_train.ID.nunique()}, "
        f"all_cipher_risk_IDs={len(all_cipher_risk_ids)}, "
        f"clinical_wave_{training_wave:g}={clinical_train.ID.nunique()}, "
        f"train_intersection={train.ID.nunique()}, "
        f"evaluation_wave_{evaluation_wave:g}={clinical_evaluation.ID.nunique()}, "
        f"valid={len(valid)}, test={len(test)}",
        flush=True,
    )
    features = source["feature_cols"]
    missing = [x for x in features if x not in clinical]
    if missing:
        raise ValueError(f"Source clinical features missing: {missing}")
    train = clip_frame(train, cfg["source"]["clip_limits"]).dropna(
        subset=list(features) + ["cipher_risk", event_col, time_col]
    ).reset_index(drop=True)
    valid = clip_frame(valid, cfg["source"]["clip_limits"]).dropna(
        subset=list(features) + [event_col, time_col]
    ).reset_index(drop=True)
    test = clip_frame(test, cfg["source"]["clip_limits"]).dropna(
        subset=list(features) + [event_col, time_col]
    ).reset_index(drop=True)
    for frame in (train, valid, test):
        frame[event_col] = pd.to_numeric(frame[event_col], errors="raise").astype(int)
        frame[time_col] = pd.to_numeric(frame[time_col], errors="raise").astype(float)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"{endpoint}: an empty split was produced")
    info = {
        "endpoint": endpoint, "event_column": event_col, "time_column": time_col,
        "risk_path": risk_path, "target_transform": target_transform,
        "training_wave": training_wave, "evaluation_wave": evaluation_wave,
        "all_cipher_risk_n": int(len(all_cipher_risk_ids)),
        "risk_training_wave_n": int(risk_train.ID.nunique()),
        "train_n": len(train), "train_events": int(train[event_col].sum()),
        "valid_n": len(valid), "valid_events": int(valid[event_col].sum()),
        "test_n": len(test), "test_events": int(test[event_col].sum()),
    }
    print(json.dumps({**info, "risk_path": str(risk_path)}, indent=2), flush=True)
    return train, valid, test, info


def metric_row(split: str, endpoint: str, bootstrap, strategy: str, model: str,
               target: str, frame: pd.DataFrame, time_col: str, event_col: str,
               score: np.ndarray) -> dict:
    return {
        "split": split, "endpoint": endpoint, "bootstrap": bootstrap,
        "strategy": strategy, "model": model, "target": target,
        "n": len(frame), "n_event": int(frame[event_col].sum()),
        "cindex": safe_cindex(frame[time_col], frame[event_col], score),
        "event_auroc": safe_event_auc(frame[event_col], score),
    }


def base_artifact(cfg: dict, info: dict, strategy: str, target: str, model_name: str,
                  bootstrap, train: pd.DataFrame, predictor: dict, source_performance: dict) -> dict:
    return {
        "schema_version": "clinical-cipher-artifact-v2",
        "strategy": strategy, "target": target, "endpoint": info["endpoint"],
        "model_name": model_name, "bootstrap": bootstrap,
        "predictor": predictor, "feature_cols": list(cfg["source"]["feature_cols"]),
        "clip_limits": cfg["source"]["clip_limits"],
        "cipher_target_transform": {"kind": "source_training_wave_minmax", **info["target_transform"]},
        "source_risk_path": str(info["risk_path"]),
        "source_training_wave": info["training_wave"],
        "source_evaluation_wave": info["evaluation_wave"],
        "source_event_column": info["event_column"], "source_time_column": info["time_column"],
        "source_training_n": len(train), "source_training_events": int(train[info["event_column"]].sum()),
        "source_feature_summary": feature_summary(train, cfg["source"]["feature_cols"]),
        "source_performance": source_performance, "feature_contract": cfg.get("feature_contract", {}),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "joblib": joblib.__version__},
    }


def fit_clinical_cox(train: pd.DataFrame, features: Sequence[str], time_col: str,
                     event_col: str, penalizer: float) -> dict:
    if CoxPHFitter is None:
        raise RuntimeError("clinical_cox requires lifelines from environment.yml")
    imputer = SimpleImputer(strategy="median").fit(train[features])
    scaler = StandardScaler().fit(imputer.transform(train[features]))
    fit = pd.DataFrame(scaler.transform(imputer.transform(train[features])), columns=features)
    fit["time"], fit["event"] = train[time_col].to_numpy(), train[event_col].astype(int).to_numpy()
    cox = CoxPHFitter(penalizer=penalizer).fit(fit, duration_col="time", event_col="event")
    return {"kind": "clinical_cox", "imputer": imputer, "scaler": scaler, "cox": cox}


def fit_risk_cox(train: pd.DataFrame, penalizer: float) -> CoxPHFitter:
    if CoxPHFitter is None:
        raise RuntimeError("risk_cox requires lifelines from environment.yml")
    fit = pd.DataFrame({"risk": train["cipher_risk"], "time": train["_time"], "event": train["_event"]})
    return CoxPHFitter(penalizer=penalizer).fit(fit, duration_col="time", event_col="event")


def predict_artifact(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    predictor, features = artifact["predictor"], artifact["feature_cols"]
    kind = predictor["kind"]
    if kind == "estimator":
        score = predict_estimator_score(predictor["pipeline"], frame[features])
        return score * float(predictor.get("risk_direction", 1.0))
    if kind == "clinical_cox":
        values = predictor["scaler"].transform(predictor["imputer"].transform(frame[features]))
        return predictor["cox"].predict_partial_hazard(pd.DataFrame(values, columns=features)).to_numpy().reshape(-1)
    if kind == "risk_cox":
        risk = predict_estimator_score(predictor["risk_pipeline"], frame[features])
        risk = minmax_apply(risk, predictor["risk_output_transform"])
        return predictor["cox"].predict_partial_hazard(pd.DataFrame({"risk": risk})).to_numpy().reshape(-1)
    raise ValueError(f"Unknown predictor kind: {kind}")


def artifact_path(root: Path, endpoint: str, strategy: str, target: str,
                  model: str, bootstrap) -> Path:
    safe_target = target.replace(".", "p")
    name = "final.joblib" if bootstrap == "final" else f"bootstrap_{int(bootstrap):03d}.joblib"
    return root / "models" / endpoint / strategy / safe_target / model / name


def add_and_save(cfg: dict, info: dict, strategy: str, target: str, model_name: str,
                 bootstrap, train: pd.DataFrame, predictor: dict,
                 valid: pd.DataFrame, test: pd.DataFrame, output_dir: Path,
                 metrics: list, model_index: list, predictions: list) -> dict:
    provisional = base_artifact(cfg, info, strategy, target, model_name, bootstrap, train, predictor, {})
    performance = {}
    for split, frame in [("valid", valid), ("test", test)]:
        score = predict_artifact(provisional, frame)
        evaluated = legacy_split_minmax(score) if strategy in {
            "risk_regression", "top_risk_label_classification"
        } else score
        row = metric_row(split, info["endpoint"], bootstrap, strategy, model_name, target,
                         frame, info["time_column"], info["event_column"], evaluated)
        if "risk_quantile" in predictor:
            row["risk_quantile"] = predictor["risk_quantile"]
            row["risk_cutoff"] = predictor["risk_cutoff"]
            row["train_n_top"] = int(predictor["train_n_top"])
            row["train_top_rate"] = float(predictor["train_n_top"] / len(train))
        if "selected_risk_model" in predictor:
            row["selected_risk_model"] = predictor["selected_risk_model"]
        metrics.append(row)
        performance[split] = row
        if bootstrap == 0:
            block = frame[["ID", "wave", info["time_column"], info["event_column"]]].copy()
            block.columns = ["eid", "wave", "time", "event"]
            block["split"], block["endpoint"], block["bootstrap"] = split, info["endpoint"], bootstrap
            block["strategy"], block["target"], block["model"], block["score"] = strategy, target, model_name, evaluated
            predictions.append(block)
    artifact = base_artifact(cfg, info, strategy, target, model_name, bootstrap, train, predictor, performance)
    path = artifact_path(output_dir, info["endpoint"], strategy, target, model_name, bootstrap)
    model_index.append({"endpoint": info["endpoint"], "strategy": strategy, "target": target,
                        "model": model_name, "kind": "final" if bootstrap == "final" else "bootstrap",
                        "bootstrap": None if bootstrap == "final" else bootstrap,
                        "source_training_wave": info["training_wave"],
                        "source_evaluation_wave": info["evaluation_wave"],
                        **dump_artifact(artifact, path)})
    return performance


def fit_all_for_sample(cfg: dict, info: dict, train: pd.DataFrame, valid: pd.DataFrame,
                       test: pd.DataFrame, bootstrap, output_dir: Path,
                       reg_models: Sequence[str], clf_models: Sequence[str], strategies: set,
                       metrics: list, model_index: list, predictions: list,
                       selected_risk_model: Optional[str] = None) -> Optional[str]:
    features = cfg["source"]["feature_cols"]
    event_col, time_col = info["event_column"], info["time_column"]
    seed = int(cfg["experiment"]["seed"])
    penalizer = float(cfg["experiment"].get("cox_penalizer", .1))
    risk_valid_scores = {}

    if "risk_regression" in strategies or "risk_cox" in strategies:
        for name in reg_models:
            pipe = build_pipeline(name, seed).fit(train[features], train["cipher_risk"])
            transform = minmax_fit(predict_estimator_score(pipe, train[features]))
            predictor = {"kind": "estimator", "pipeline": pipe, "risk_direction": 1.0,
                         "output_transform": transform}
            if "risk_regression" in strategies:
                perf = add_and_save(cfg, info, "risk_regression", "multimodal_normalized_risk", name,
                                    bootstrap, train, predictor, valid, test, output_dir,
                                    metrics, model_index, predictions)
                risk_valid_scores[name] = perf["valid"]["cindex"]

    if selected_risk_model is None and risk_valid_scores:
        finite = {k: v for k, v in risk_valid_scores.items() if np.isfinite(v)}
        selected_risk_model = max(finite, key=finite.get) if finite else next(iter(risk_valid_scores))
    if selected_risk_model is None and "risk_cox" in strategies and reg_models:
        selected_risk_model = reg_models[0]

    if "direct_survival_regression" in strategies:
        for name in reg_models:
            pipe = build_pipeline(name, seed).fit(train[features], train[time_col])
            train_pred = predict_estimator_score(pipe, train[features])
            corr = np.corrcoef(train[time_col].to_numpy(), train_pred)[0, 1]
            direction = -1.0 if np.isfinite(corr) and corr > 0 else 1.0
            predictor = {"kind": "estimator", "pipeline": pipe, "risk_direction": direction}
            add_and_save(cfg, info, "direct_survival_regression", "event_time", name, bootstrap,
                         train, predictor, valid, test, output_dir, metrics, model_index, predictions)

    classification_targets = []
    if "direct_any_event_classification" in strategies:
        classification_targets.append(("direct_any_event_classification", "any_event", train[event_col].astype(int)))
    for strategy, target, labels in classification_targets:
        if labels.nunique() < 2:
            continue
        for name in clf_models:
            pipe = build_classifier_pipeline(name, seed).fit(train[features], labels)
            add_and_save(cfg, info, strategy, target, name, bootstrap, train,
                         {"kind": "estimator", "pipeline": pipe, "risk_direction": 1.0},
                         valid, test, output_dir, metrics, model_index, predictions)

    if "clinical_cox" in strategies:
        predictor = fit_clinical_cox(train, features, time_col, event_col, penalizer)
        add_and_save(cfg, info, "clinical_cox", "event_time", "CoxPH_penalized", bootstrap,
                     train, predictor, valid, test, output_dir, metrics, model_index, predictions)

    if "risk_cox" in strategies and selected_risk_model is not None:
        risk_pipe = build_pipeline(selected_risk_model, seed).fit(train[features], train["cipher_risk"])
        risk_transform = minmax_fit(predict_estimator_score(risk_pipe, train[features]))
        fit = train.copy()
        fit["cipher_risk"] = minmax_apply(
            predict_estimator_score(risk_pipe, train[features]), risk_transform
        )
        fit["_time"], fit["_event"] = train[time_col], train[event_col]
        predictor = {"kind": "risk_cox", "risk_pipeline": risk_pipe,
                     "risk_output_transform": risk_transform, "cox": fit_risk_cox(fit, penalizer),
                     "selected_risk_model": selected_risk_model}
        add_and_save(cfg, info, "risk_cox", "clinical_fitted_risk_then_cox",
                     f"RiskCox_after_{selected_risk_model}", bootstrap, train, predictor,
                     valid, test, output_dir, metrics, model_index, predictions)

    if "top_risk_label_classification" in strategies:
        for label, quantile in cfg["experiment"]["top_risk_quantiles"].items():
            cutoff = float(train["cipher_risk"].quantile(float(quantile)))
            labels = (train["cipher_risk"] >= cutoff).astype(int)
            if labels.nunique() < 2:
                continue
            for name in clf_models:
                pipe = build_classifier_pipeline(name, seed).fit(train[features], labels)
                predictor = {"kind": "estimator", "pipeline": pipe, "risk_direction": 1.0,
                             "risk_quantile": float(quantile), "risk_cutoff": cutoff,
                             "train_n_top": int(labels.sum())}
                add_and_save(cfg, info, "top_risk_label_classification", label, name, bootstrap,
                             train, predictor, valid, test, output_dir, metrics, model_index, predictions)
    return selected_risk_model


def run_endpoint(cfg: dict, endpoint_key: str, clinical: pd.DataFrame, cohort_map: pd.DataFrame,
                 output_dir: Path, reg_models: Sequence[str], clf_models: Sequence[str],
                 strategies: set) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[dict]]:
    train, valid, test, info = prepare_endpoint(cfg, endpoint_key, clinical, cohort_map)
    metrics, predictions, model_index = [], [], []
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
    for bootstrap in range(int(cfg["experiment"]["n_bootstrap"])):
        sample = train.iloc[rng.choice(len(train), len(train), replace=True)].reset_index(drop=True)
        print(f"[{info['endpoint']}] bootstrap {bootstrap}", flush=True)
        fit_all_for_sample(cfg, info, sample, valid, test, bootstrap, output_dir,
                           reg_models, clf_models, strategies, metrics, model_index, predictions)
    risk_rows = pd.DataFrame(metrics)
    risk_rows = risk_rows[(risk_rows.strategy == "risk_regression") & (risk_rows.split == "valid")]
    mean_valid = risk_rows.groupby("model").cindex.mean().dropna() if not risk_rows.empty else pd.Series(dtype=float)
    selected_final = mean_valid.idxmax() if not mean_valid.empty else (reg_models[0] if reg_models else None)
    fit_all_for_sample(cfg, info, train, valid, test, "final", output_dir,
                       reg_models, clf_models, strategies, metrics, model_index, predictions, selected_final)
    metric_frame = pd.DataFrame(metrics)
    bootstrap_metrics = metric_frame[metric_frame.bootstrap != "final"].copy()
    summary = bootstrap_metrics.groupby(["split", "endpoint", "strategy", "model", "target"], dropna=False).agg(
        n=("n", "median"), n_event=("n_event", "median"),
        cindex_mean=("cindex", "mean"), cindex_std=("cindex", "std"),
        event_auroc_mean=("event_auroc", "mean"), event_auroc_std=("event_auroc", "std"),
    ).reset_index()
    pred_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    endpoint = info["endpoint"]
    metric_frame.to_csv(output_dir / f"{endpoint}_all_model_results.csv", index=False)
    summary.to_csv(output_dir / f"{endpoint}_model_results_summary.csv", index=False)
    pred_frame.to_csv(output_dir / f"{endpoint}_bootstrap0_predictions.csv", index=False)
    return metric_frame, summary, pred_frame, model_index



def write_config(path):
    defaults = json.loads(r'''{
  "paths": {
    "longitudinal_tsv": "data/private/clinical_longitudinal.tsv",
    "baseline_tsv": "data/private/clinical_baseline.tsv",
    "output_dir": "private_runs/clinical_six_strategies"
  },
  "experiment": {
    "seed": 42,
    "n_bootstrap": 20,
    "cox_penalizer": 0.1,
    "endpoints": [
      "istroke",
      "ASCVD"
    ],
    "strategies": [
      "risk_regression",
      "direct_survival_regression",
      "direct_any_event_classification",
      "clinical_cox",
      "risk_cox",
      "top_risk_label_classification"
    ],
    "regression_models": [
      "Ridge",
      "ElasticNet",
      "RF",
      "ExtraTrees",
      "GBR",
      "HistGBR",
      "SVR_RBF",
      "KNN",
      "XGB"
    ],
    "classification_models": [
      "Logistic",
      "RF",
      "ExtraTrees",
      "GBC",
      "HistGBC",
      "KNN",
      "XGB"
    ],
    "top_risk_quantiles": {
      "top5": 0.95,
      "top10": 0.9,
      "top25": 0.75,
      "top50": 0.5
    },
    "allow_missing_xgboost": true
  },
  "source": {
    "id_column": "ID",
    "wave_column": "wave",
    "baseline_id_column": "ID",
    "subcohort_column": "subcohort",
    "external_test_subcohort": "\u9996\u94a2",
    "training_wave": 1.0,
    "evaluation_wave": 3.0,
    "visit_to_wave": {
      "02": 1.0,
      "07": 2.0,
      "12": 3.0
    },
    "wave_time_offsets_years": {
      "1.0": 10.0,
      "2.0": 15.0,
      "3.0": 20.0
    },
    "minimum_followup_years": 0.1,
    "source_column_rename": {
      "ageExam": "age_feat"
    },
    "feature_cols": [
      "age_feat",
      "sex",
      "SBP",
      "DBP",
      "TC",
      "TG",
      "HDLC",
      "LDLC",
      "BMI",
      "height",
      "weight",
      "waist",
      "smoking"
    ],
    "clip_limits": {
      "age_feat": [
        18,
        100
      ],
      "SBP": [
        70,
        250
      ],
      "DBP": [
        40,
        150
      ],
      "BMI": [
        10,
        60
      ]
    }
  },
  "endpoints": {
    "CHD": {
      "display": "CHD",
      "risk_candidates": [
        "data/private/sample_risk_CHD.npy"
      ],
      "event_candidates": [
        "fCHD"
      ],
      "time_candidates": [
        "fCHD_y"
      ]
    },
    "istroke": {
      "display": "istroke",
      "risk_candidates": [
        "data/private/sample_risk_istroke.npy"
      ],
      "event_candidates": [
        "fistroke"
      ],
      "time_candidates": [
        "fistroke_y"
      ]
    },
    "ASCVD": {
      "display": "ASCVD",
      "risk_candidates": [
        "data/private/sample_risk_ICVD.npy"
      ],
      "event_candidates": [
        "fICVD",
        "fASCVD"
      ],
      "time_candidates": [
        "fICVD_y",
        "fASCVD_y"
      ]
    }
  },
  "feature_contract": {}
}''')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(defaults, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print("Created local six-strategy training configuration.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/clinical_six_strategies.local.json"))
    parser.add_argument("--write-config", action="store_true", help="Create a local training configuration and exit.")
    parser.add_argument("--endpoints", nargs="*")
    parser.add_argument("--strategies", nargs="*", choices=ALL_STRATEGIES)
    parser.add_argument(
        "--audit-splits-only", action="store_true",
        help="Load data and report train/validation/test counts without fitting models.",
    )
    args = parser.parse_args()
    if args.write_config:
        write_config(args.config)
        return
    cfg = load_yaml(args.config)
    if int(cfg["experiment"]["n_bootstrap"]) < 1:
        parser.error("n_bootstrap must be positive")
    configured = set(args.strategies or cfg["experiment"]["strategies"])
    if configured - set(ALL_STRATEGIES):
        parser.error("Unknown clinical training strategy")
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clinical, cohort_map = load_source_clinical(cfg), load_cohort_map(cfg)
    endpoints = args.endpoints or cfg["experiment"]["endpoints"]
    if args.audit_splits_only:
        audit = []
        for endpoint in endpoints:
            _, _, _, info = prepare_endpoint(cfg, endpoint, clinical, cohort_map)
            audit.append({
                key: str(value) if isinstance(value, Path) else value
                for key, value in info.items()
            })
        save_json(audit, output_dir / "source_split_audit.json")
        print(f"Saved split audit to {output_dir / 'source_split_audit.json'}", flush=True)
        return
    allow = cfg["experiment"].get("allow_missing_xgboost", True)
    reg_models = available_model_names(cfg["experiment"]["regression_models"], allow)
    clf_models = available_model_names(cfg["experiment"]["classification_models"], allow)
    strategies = set(args.strategies or cfg["experiment"]["strategies"])
    all_metrics, all_summaries, all_predictions, model_index = [], [], [], []
    for endpoint in endpoints:
        metrics, summary, predictions, index = run_endpoint(
            cfg, endpoint, clinical, cohort_map, output_dir, reg_models, clf_models, strategies
        )
        all_metrics.append(metrics)
        all_summaries.append(summary)
        model_index.extend(index)
        if not predictions.empty:
            all_predictions.append(predictions)
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(output_dir / "ALL_model_results_long.csv", index=False)
    combined_metrics[combined_metrics.strategy == "top_risk_label_classification"].to_csv(
        output_dir / "ALL_top_risk_label_results_long.csv", index=False
    )
    pd.concat(all_summaries, ignore_index=True).to_csv(output_dir / "ALL_model_results_summary.csv", index=False)
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "ALL_bootstrap0_predictions_long.csv", index=False)
    pd.DataFrame(model_index).to_csv(output_dir / "model_manifest.csv", index=False)
    training_wave, evaluation_wave = configured_source_waves(cfg["source"])
    save_json({"config": str(Path(args.config).resolve()), "endpoints": endpoints,
               "strategies": sorted(strategies), "regression_models": reg_models,
               "classification_models": clf_models, "n_bootstrap": cfg["experiment"]["n_bootstrap"],
               "source_training_wave": training_wave,
               "source_evaluation_wave": evaluation_wave,
               "n_saved_artifacts": len(model_index), "output_dir": str(output_dir)},
              output_dir / "run_manifest.json")
    print(f"Saved {len(model_index)} source artifacts under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
