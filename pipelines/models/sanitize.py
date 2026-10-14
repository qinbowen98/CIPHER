"""Rebuild vetted predictor state; never export an original fitted object wholesale."""
import copy
import numpy as np
from .public_artifact import FrozenCox


class PrivateStateError(ValueError):
    pass


COMMON = {"n_features_in_", "feature_names_in_", "_sklearn_version"}
STATES = {
    "Pipeline": {"steps", "memory", "verbose"},
    "SimpleImputer": {"statistics_", "indicator_", "_fit_dtype", "_fill_dtype"},
    "StandardScaler": {"mean_", "var_", "scale_", "n_samples_seen_"},
    "MinMaxScaler": {"scale_", "min_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"},
    "Ridge": {"coef_", "intercept_", "n_iter_", "solver_"},
    "ElasticNet": {"coef_", "intercept_", "n_iter_", "dual_gap_"},
    "LogisticRegression": {"coef_", "intercept_", "n_iter_", "classes_"},
    "DecisionTreeRegressor": {"tree_", "n_outputs_", "max_features_"},
    "DecisionTreeClassifier": {"tree_", "n_outputs_", "max_features_", "classes_", "n_classes_"},
    "ExtraTreeRegressor": {"tree_", "n_outputs_", "max_features_"},
    "ExtraTreeClassifier": {"tree_", "n_outputs_", "max_features_", "classes_", "n_classes_"},
    "RandomForestRegressor": {"estimators_", "estimator_", "base_estimator_", "n_outputs_"},
    "ExtraTreesRegressor": {"estimators_", "estimator_", "base_estimator_", "n_outputs_"},
    "RandomForestClassifier": {"estimators_", "estimator_", "base_estimator_", "n_outputs_", "classes_", "n_classes_"},
    "ExtraTreesClassifier": {"estimators_", "estimator_", "base_estimator_", "n_outputs_", "classes_", "n_classes_"},
    "DummyRegressor": {"constant_", "n_outputs_"},
    "DummyClassifier": {"classes_", "n_classes_", "n_outputs_", "class_prior_", "sparse_output_", "_strategy"},
    "GradientBoostingRegressor": {"estimators_", "init_", "_loss", "loss_", "n_estimators_", "n_trees_per_iteration_", "max_features_", "train_score_", "_rng"},
    "GradientBoostingClassifier": {"estimators_", "init_", "_loss", "loss_", "n_estimators_", "n_trees_per_iteration_", "max_features_", "train_score_", "_rng", "classes_", "n_classes_", "_n_classes"},
    "HistGradientBoostingRegressor": {"_predictors", "_baseline_prediction", "_loss", "_bin_mapper", "_scorer", "_random_seed", "_is_fitted", "n_trees_per_iteration_", "do_early_stopping_", "is_categorical_", "_is_categorical_remapped", "_preprocessor", "_feature_names_in", "_n_features", "_n_features_in", "train_score_", "validation_score_"},
    "HistGradientBoostingClassifier": {"_predictors", "_baseline_prediction", "_loss", "_bin_mapper", "_scorer", "_random_seed", "_is_fitted", "n_trees_per_iteration_", "do_early_stopping_", "is_categorical_", "_is_categorical_remapped", "_preprocessor", "_feature_names_in", "_n_features", "_n_features_in", "train_score_", "validation_score_", "classes_"},
    "_BinMapper": {"bin_thresholds_", "n_bins_non_missing_", "missing_values_bin_idx_", "is_categorical_", "_known_categories"},
    "TreePredictor": {"nodes", "binned_left_cat_bitsets", "raw_left_cat_bitsets"},
}
# Loss implementations carry scalars/links, not fitted participant records.
LOSSES = {"HalfSquaredError", "HalfBinomialLoss", "HalfMultinomialLoss", "SquaredError", "LeastSquaresError", "LeastSquares", "BinaryCrossEntropy", "CategoricalCrossEntropy", "BinomialDeviance", "MultinomialDeviance", "IdentityLink", "LogitLink", "MultinomialLogit", "Interval"}
LOSS_KEYS = {"closs", "link", "approx_hessian", "constant_hessian", "need_update_leaves_values", "interval_y_true", "interval_y_pred", "n_classes", "low", "high", "low_inclusive", "high_inclusive", "K", "hessians_are_constant", "n_threads"}


def sanitize(value, location="predictor"):
    if isinstance(value, FrozenCox):
        if set(vars(value)) != {"columns", "coefficients", "center"}:
            raise PrivateStateError("Extra Cox state")
        return FrozenCox(value.columns, value.coefficients, value.center)
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return copy.deepcopy(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            clean = [sanitize(v, location) for v in value.flat]
            return np.asarray(clean, dtype=object).reshape(value.shape)
        return value.copy()
    if isinstance(value, np.dtype):
        return value
    if isinstance(value, np.random.RandomState):
        return np.random.RandomState(0)  # inference never needs original RNG state
    if isinstance(value, list):
        return [sanitize(v, location) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize(v, location) for v in value)
    if isinstance(value, dict):
        return {k: sanitize(v, location + "." + str(k)) for k, v in value.items()}
    name, module = type(value).__name__, type(value).__module__
    if name.startswith(("KNeighbors", "SVR", "SVC", "NuSV")):
        raise PrivateStateError(location + ": " + name + " retains training rows/support vectors")
    if module == "sklearn.tree._tree" and name == "Tree":
        return copy.deepcopy(value)  # node thresholds, counts and fitted outputs
    if module == "sklearn._loss._loss" and name in {"CyHalfSquaredError", "CyHalfBinomialLoss", "CyHalfMultinomialLoss"}:
        return type(value)()
    if not module.startswith("sklearn.") or name not in set(STATES) | LOSSES:
        raise PrivateStateError(location + ": unreviewed class " + module + "." + name)
    params = value.get_params(deep=False) if hasattr(value, "get_params") else {}
    allowed = set(params) | COMMON | STATES.get(name, LOSS_KEYS)
    state = value.__getstate__() if hasattr(value, "__getstate__") else vars(value)
    if not isinstance(state, dict):
        raise PrivateStateError(location + ": unreviewed state format")
    # Explicit allowlist: custom attributes and per-row OOB diagnostics are discarded.
    clean = {k: sanitize(v, location + "." + k) for k, v in state.items() if k in allowed}
    if name == "Pipeline":
        clean["memory"] = None
    result = type(value).__new__(type(value))
    if hasattr(result, "__setstate__"):
        result.__setstate__(clean)
    else:
        result.__dict__.update(clean)
    return result


def strip_cox(cox):
    if type(cox).__module__.split(".")[0] != "lifelines":
        raise PrivateStateError("Unreviewed Cox implementation")
    columns = list(cox.params_.index)
    return FrozenCox(columns, cox.params_.to_numpy(), cox._norm_mean.loc[columns].to_numpy())


def public_artifact(artifact):
    predictor = artifact["predictor"]
    kind = predictor["kind"]
    clean = {"kind": kind}
    if kind == "estimator":
        keys = ["pipeline", "risk_direction", "output_transform"]
    elif kind == "clinical_cox":
        keys = ["imputer", "scaler"]
        clean["cox"] = strip_cox(predictor["cox"])
    elif kind == "risk_cox":
        keys = ["risk_pipeline", "risk_output_transform"]
        clean["cox"] = strip_cox(predictor["cox"])
    else:
        raise PrivateStateError("Unreviewed predictor kind " + str(kind))
    for key in keys:
        if key in predictor:
            clean[key] = sanitize(predictor[key], key)
    allowed_meta = ["schema_version", "strategy", "target", "endpoint", "model_name", "source_training_wave", "feature_cols", "clip_limits", "feature_contract", "software"]
    result = {key: sanitize(artifact[key], key) for key in allowed_meta if key in artifact}
    result["predictor"] = clean
    result["public_inference_only"] = True
    return result
