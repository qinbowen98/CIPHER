import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pipelines.inference.prepared import run


class PreparedInferenceTests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.RandomState(10)
        x = pd.DataFrame({"age_feat": rng.normal(60, 5, 40), "SBP": rng.normal(130, 15, 40)})
        for name, cols in [("Ridge", ["age_feat"]), ("ElasticNet", ["age_feat", "SBP"])]:
            pipe = Pipeline([("imputer", SimpleImputer()), ("estimator", Ridge())]).fit(x[cols], x.age_feat)
            a = {"endpoint": "ASCVD", "strategy": "risk_regression", "target": "multimodal_normalized_risk", "model_name": name, "feature_cols": cols, "clip_limits": {}, "predictor": {"kind": "estimator", "pipeline": pipe}}
            path = root / "models" / name / "final.joblib"
            path.parent.mkdir(parents=True)
            joblib.dump(a, path)
        x["eid"] = np.arange(1000, 1040)
        x["event"] = np.arange(40) % 2
        x["time"] = np.arange(40) / 4 + 1
        x.loc[0, "SBP"] = np.nan
        table = root / "private_input.csv"
        x.to_csv(table, index=False)
        cfg = {"private_output": str(root / "private_outputs"), "inference": {"ukb": {"models_dir": str(root / "models"), "table": str(table), "id_column": "eid", "endpoints": ["ASCVD"], "outcomes": {"ASCVD": {"time": "time", "event": "event"}}, "complete_case": True, "bootstrap": 2}}}
        return cfg, table

    def test_models_share_cohort_and_no_sample_files_are_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, _ = self.fixture(root)
            run(cfg, "UKB")
            output = root / "private_outputs/ukb"
            self.assertEqual([p.name for p in output.iterdir()], ["aggregate_metrics.csv"])
            frame = pd.read_csv(output / "aggregate_metrics.csv")
            self.assertEqual(len(frame), 6)
            self.assertTrue(frame.n.eq(39).all())
            self.assertEqual(frame.groupby("replicate").events.nunique().max(), 1)
            self.assertFalse(set(frame.columns) & {"eid", "risk", "score", "time", "event"})

    def test_duplicate_participant_ids_stop_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, table = self.fixture(Path(temp))
            frame = pd.read_csv(table)
            frame.loc[1, "eid"] = frame.loc[0, "eid"]
            frame.to_csv(table, index=False)
            with self.assertRaises(ValueError):
                run(cfg, "UKB")


if __name__ == "__main__":
    unittest.main()
