"""Validate public model state independently of manifest hashes."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from .public_artifact import FrozenCox
from .sanitize import sanitize, PrivateStateError


def check(value):
    if isinstance(value, FrozenCox):
        if set(vars(value)) != {"columns", "coefficients", "center"}:
            raise PrivateStateError("Extra Cox state")
        if value.coefficients.shape != (len(value.columns),) or value.center.shape != (len(value.columns),):
            raise PrivateStateError("Invalid Cox coefficient shape")
        return
    cleaned = sanitize(value)
    if isinstance(value, dict):
        for item in value.values():
            check(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check(item)
    elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
        for item in value.flat:
            check(item)
    elif hasattr(value, "__dict__"):
        if set(vars(value)) != set(vars(cleaned)):
            raise PrivateStateError("Unreviewed fitted attributes present")
        for item in vars(value).values():
            check(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.release / "release_manifest.json").read_text())
    for entry in manifest["files"]:
        path = args.release / entry["file"]
        if path.suffix == ".joblib":
            a = joblib.load(path)
            if a.get("public_inference_only") is not True:
                raise PrivateStateError("Original artifact cannot be published")
            allowed = {"schema_version", "strategy", "target", "endpoint", "model_name", "source_training_wave", "feature_cols", "clip_limits", "feature_contract", "software", "predictor", "public_inference_only"}
            if set(a) - allowed:
                raise PrivateStateError("Unreviewed model metadata")
            for value in a.values():
                check(value)
        elif path.suffix == ".csv":
            from .export import clean_metrics
            clean_metrics(path)
    print("Public model states and aggregate table schemas passed.")


if __name__ == "__main__":
    main()
