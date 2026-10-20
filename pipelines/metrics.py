"""Observed event-status discrimination for fixed risk scores."""
import numpy as np
from sklearn.metrics import roc_auc_score


def event_auroc(event, risk):
    event, risk = np.asarray(event), np.asarray(risk, dtype=float)
    if event.ndim != 1 or risk.ndim != 1 or event.shape != risk.shape:
        raise ValueError('Event and risk must be aligned one-dimensional arrays.')
    if not np.isin(event, [0, 1]).all() or not np.isfinite(risk).all():
        raise ValueError('Event must be binary and risk must be finite.')
    return float(roc_auc_score(event, risk)) if np.unique(event).size == 2 else float('nan')
