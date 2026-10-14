"""Inference-only Cox representation: no training rows, events or durations."""
import numpy as np
import pandas as pd


class FrozenCox:
    def __init__(self, columns, coefficients, center):
        self.columns = list(columns)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.center = np.asarray(center, dtype=float)

    def predict_partial_hazard(self, frame):
        values = frame[self.columns].to_numpy(dtype=float)
        return pd.Series(np.exp((values - self.center) @ self.coefficients), index=frame.index)
