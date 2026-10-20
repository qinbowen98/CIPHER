from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    return cfg


def save_json(value: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def normalize_eid(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def minmax_fit(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("Cannot fit min-max transform: no finite values")
    low = float(np.nanmin(x))
    high = float(np.nanmax(x))
    if high <= low:
        raise ValueError(f"Cannot fit min-max transform: low={low}, high={high}")
    return {"low": low, "high": high}


def minmax_apply(values: Sequence[float], params: Mapping[str, float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return (x - float(params["low"])) / (float(params["high"]) - float(params["low"]))


def legacy_split_minmax(values: Sequence[float]) -> np.ndarray:
    """Reproduce the original notebook's per-split min-max scaling."""
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, np.nan, dtype=float)
    low = np.nanmin(x)
    high = np.nanmax(x)
    if high <= low:
        return np.zeros_like(x, dtype=float)
    return (x - low) / (high - low)


def _harrell_cindex_risk(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell C for a score where larger means earlier event, in O(n log n)."""
    order = np.argsort(-time, kind="mergesort")
    time, event, risk = time[order], event[order], risk[order]
    unique_risk = np.unique(risk)
    rank = np.searchsorted(unique_risk, risk) + 1
    tree = np.zeros(len(unique_risk) + 1, dtype=np.int64)

    def add(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    comparable = 0
    concordant = 0.0
    inserted = 0
    start = 0
    while start < len(time):
        stop = start + 1
        while stop < len(time) and time[stop] == time[start]:
            stop += 1
        for index in range(start, stop):
            if event[index] == 1:
                lower = prefix(int(rank[index]) - 1)
                equal = prefix(int(rank[index])) - lower
                comparable += inserted
                concordant += lower + 0.5 * equal
        for index in range(start, stop):
            add(int(rank[index]))
            inserted += 1
        start = stop
    return float(concordant / comparable) if comparable else float("nan")


def safe_cindex(time: Sequence[float], event: Sequence[int], risk: Sequence[float]) -> float:
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    keep = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    if keep.sum() < 5 or e[keep].sum() == 0:
        return float("nan")
    try:
        return _harrell_cindex_risk(t[keep], e[keep], r[keep])
    except Exception:
        return float("nan")


def safe_event_auc(event: Sequence[int], risk: Sequence[float]) -> float:
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    keep = np.isfinite(e) & np.isfinite(r)
    if keep.sum() < 2 or np.unique(e[keep]).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(e[keep], r[keep]))
    except Exception:
        return float("nan")


def horizon_labels(time: Sequence[float], event: Sequence[int], horizon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cases: event by horizon. Controls: observed event-free beyond horizon."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    case = (e == 1) & (t <= horizon)
    control = t > horizon
    keep = np.isfinite(t) & np.isfinite(e) & (case | control)
    label = np.zeros(len(t), dtype=int)
    label[case] = 1
    return label, keep


def safe_horizon_auc(time: Sequence[float], event: Sequence[int], risk: Sequence[float], horizon: float) -> float:
    label, keep = horizon_labels(time, event, horizon)
    r = np.asarray(risk, dtype=float)
    keep &= np.isfinite(r)
    return safe_event_auc(label[keep], r[keep])


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_frame(frame: pd.DataFrame, clip_limits: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    out = frame.copy()
    for column, limits in clip_limits.items():
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").clip(float(limits[0]), float(limits[1]))
    return out


def build_regressor(name: str, seed: int):
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "ElasticNet":
        return ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=20000, random_state=seed)
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
    if name == "GBR":
        return GradientBoostingRegressor(random_state=seed)
    if name == "HistGBR":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, random_state=seed)
    if name == "SVR_RBF":
        return SVR(C=1.0, gamma="scale")
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=20, weights="distance")
    if name == "XGB":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError("XGB requested but xgboost is unavailable") from exc
        return XGBRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", random_state=seed, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def build_classifier(name: str, seed: int):
    if name == "Logistic":
        return LogisticRegression(
            penalty="l2", C=1.0, max_iter=10000,
            class_weight="balanced", random_state=seed,
        )
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    if name == "GBC":
        return GradientBoostingClassifier(random_state=seed)
    if name == "HistGBC":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, random_state=seed,
        )
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=20, weights="distance")
    if name == "XGB":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("XGB requested but xgboost is unavailable") from exc
        return XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
        )
    raise ValueError(f"Unknown classifier: {name}")


def available_model_names(requested: Iterable[str], allow_missing_xgboost: bool = True) -> List[str]:
    names = []
    for name in requested:
        if name != "XGB":
            names.append(name)
            continue
        try:
            import xgboost  # noqa: F401
            names.append(name)
        except Exception:
            if not allow_missing_xgboost:
                raise
            print("[INFO] xgboost unavailable; XGB is skipped.", flush=True)
    return names


def build_pipeline(name: str, seed: int) -> Pipeline:
    # Imputation has no effect on the original complete-case rows, but makes the
    # frozen artifact usable for sporadically missing UKB values.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", build_regressor(name, seed)),
    ])


def build_classifier_pipeline(name: str, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", build_classifier(name, seed)),
    ])


def predict_estimator_score(estimator, values) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probability = np.asarray(estimator.predict_proba(values))
        return probability[:, 1] if probability.ndim == 2 and probability.shape[1] > 1 else probability.reshape(-1)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(values), dtype=float).reshape(-1)
    return np.asarray(estimator.predict(values), dtype=float).reshape(-1)


def read_table(path: Union[str, Path], columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, columns=list(columns) if columns else None,
                             convert_categoricals=False, preserve_dtypes=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", usecols=list(columns) if columns else None, low_memory=False)
    return pd.read_csv(path, usecols=list(columns) if columns else None, low_memory=False)


def write_table(frame: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() in {".tsv", ".txt"}:
        frame.to_csv(path, sep="\t", index=False)
    else:
        frame.to_csv(path, index=False)


def table_columns(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dta":
        with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
            labels = reader.variable_labels()
            if labels:
                return list(labels.keys())
            return list(reader.read(1).columns)
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(path).schema.names)
        except Exception:
            return list(pd.read_parquet(path).columns)
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return list(pd.read_csv(path, sep=separator, nrows=0).columns)


def find_column(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    exact = set(columns)
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in exact:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Cannot find {label}; candidates={list(candidates)}")


def dump_artifact(artifact: Dict[str, Any], path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
