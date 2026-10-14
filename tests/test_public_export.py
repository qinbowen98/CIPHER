import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from pipelines.models.sanitize import public_artifact, PrivateStateError
from pipelines.models.export import clean_metrics, export_models
from pipelines.models.audit_public import check
from pipelines.inference.prepared import predict


class ExportPrivacyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(11)
        self.x = pd.DataFrame(rng.normal(size=(50, 3)), columns=["age_feat", "SBP", "BMI"])
        self.y = self.x.age_feat.to_numpy() + .2 * self.x.SBP.to_numpy()

    def artifact(self, estimator):
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("estimator", estimator)])
        y = (self.y > 0).astype(int) if isinstance(estimator, (LogisticRegression, GradientBoostingClassifier, HistGradientBoostingClassifier)) else self.y
        pipe.fit(self.x, y)
        pipe.private_sample_ids = ["NOT_FOR_PUBLICATION"]
        return {"schema_version": "clinical-cipher-artifact-v2", "endpoint": "ASCVD", "strategy": "risk_regression", "target": "multimodal_normalized_risk", "model_name": "Ridge", "feature_cols": list(self.x.columns), "clip_limits": {}, "source_risk_path": "/private/subject_123", "private_samples": self.x, "predictor": {"kind": "estimator", "pipeline": pipe, "risk_direction": 1, "private_ids": ["NOT_FOR_PUBLICATION"]}}

    def test_supported_models_keep_scores_and_strip_extra_sample_state(self):
        for model in [Ridge(), LogisticRegression(), RandomForestRegressor(n_estimators=3), ExtraTreesRegressor(n_estimators=3), GradientBoostingClassifier(n_estimators=3)]:
            with self.subTest(model=type(model).__name__):
                original = self.artifact(model)
                clean = public_artifact(original)
                self.assertNotIn("source_risk_path", clean)
                self.assertNotIn("private_samples", clean)
                self.assertNotIn("private_ids", clean["predictor"])
                self.assertFalse(hasattr(clean["predictor"]["pipeline"], "private_sample_ids"))
                np.testing.assert_allclose(predict(original, self.x), predict(clean, self.x), rtol=1e-9, atol=1e-11)
                check(clean)

    def test_memory_based_models_are_blocked(self):
        for model in [KNeighborsRegressor(), SVR()]:
            with self.assertRaises(PrivateStateError):
                public_artifact(self.artifact(model))

    def test_histogram_boosting_prediction_parity(self):
        for model in [HistGradientBoostingRegressor(max_iter=3, early_stopping=False), HistGradientBoostingClassifier(max_iter=3, early_stopping=False)]:
            original = self.artifact(model)
            clean = public_artifact(original)
            np.testing.assert_allclose(predict(original, self.x), predict(clean, self.x), rtol=1e-9, atol=1e-11)
            check(clean)

    def test_cox_export_removes_training_records_and_preserves_scores(self):
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            self.skipTest("lifelines is not installed in this environment")
        rng = np.random.RandomState(20)
        fit = self.x.copy()
        fit["duration"] = rng.exponential(5, len(fit)) + .1
        fit["event"] = rng.binomial(1, .7, len(fit))
        cox = CoxPHFitter(penalizer=.1).fit(fit, duration_col="duration", event_col="event")
        original = self.artifact(Ridge())
        original["predictor"] = {"kind": "risk_cox", "risk_pipeline": original["predictor"]["pipeline"], "risk_output_transform": {"low": -3., "high": 3.}, "cox": cox}
        # Clinical Cox branch uses standardized columns, as the original trainer does.
        imputer = SimpleImputer().fit(self.x)
        scaler = StandardScaler().fit(imputer.transform(self.x))
        original["predictor"] = {"kind": "clinical_cox", "imputer": imputer, "scaler": scaler, "cox": cox}
        clean = public_artifact(original)
        self.assertEqual(set(vars(clean["predictor"]["cox"])), {"columns", "coefficients", "center"})
        np.testing.assert_allclose(predict(original, self.x), predict(clean, self.x), rtol=1e-9, atol=1e-11)
        check(clean)

    def test_identifier_and_row_prediction_tables_are_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.csv"
            pd.DataFrame({"eid": ["123"], "model": ["RF"], "event_auroc": [.8]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                clean_metrics(path)
            pd.DataFrame({"model": ["RF", "RF"], "event_auroc": [.6, .7]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                clean_metrics(path)

    def test_aggregate_metrics_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.csv"
            pd.DataFrame({"cohort": ["UKB"], "endpoint": ["ASCVD"], "model": ["RF"], "replicate": [0], "n": [200], "events": [25], "event_auroc": [.7], "cindex": [.65]}).to_csv(path, index=False)
            self.assertEqual(len(clean_metrics(path)), 1)

    def test_export_receipt_excludes_blocked_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, model in [("Ridge", Ridge()), ("KNN", KNeighborsRegressor())]:
                artifact = self.artifact(model)
                artifact["model_name"] = name
                target = root / "private_models" / name / "final.joblib"
                target.parent.mkdir(parents=True)
                joblib.dump(artifact, target)
            report = export_models(root / "private_models", root / "public")
            self.assertEqual(sum(row["status"] == "exported" for row in report), 1)
            self.assertEqual(sum(row["status"] == "blocked" for row in report), 1)
            self.assertEqual(len(list((root / "public").rglob("*.joblib"))), 1)
            clean = joblib.load(next((root / "public").rglob("*.joblib")))
            check(clean)


if __name__ == "__main__":
    unittest.main()
