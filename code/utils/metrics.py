"""Regression metrics with support for masking missing values.

Following the DCRNN reference implementation, we ignore observations
that are exactly zero because they mark missing loop-detector readings.
Callers that need a different sentinel (e.g. ``NaN``) can pass an
explicit ``mask`` array.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()  # torch.Tensor
    except AttributeError:
        return np.asarray(x)


def _mask(preds: np.ndarray, targets: np.ndarray,
          mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        mask = (targets != 0.0)
    return mask.astype(np.float32)


def masked_mae(preds, targets, mask: Optional[np.ndarray] = None) -> float:
    preds = _to_numpy(preds).astype(np.float32)
    targets = _to_numpy(targets).astype(np.float32)
    m = _mask(preds, targets, mask)
    diff = np.abs(preds - targets) * m
    return float(diff.sum() / max(m.sum(), 1.0))


def masked_rmse(preds, targets, mask: Optional[np.ndarray] = None) -> float:
    preds = _to_numpy(preds).astype(np.float32)
    targets = _to_numpy(targets).astype(np.float32)
    m = _mask(preds, targets, mask)
    diff = ((preds - targets) ** 2) * m
    return float(np.sqrt(diff.sum() / max(m.sum(), 1.0)))


def masked_mape(preds, targets, mask: Optional[np.ndarray] = None,
                small: float = 1e-3) -> float:
    preds = _to_numpy(preds).astype(np.float32)
    targets = _to_numpy(targets).astype(np.float32)
    m = _mask(preds, targets, mask)
    denom = np.maximum(np.abs(targets), small)
    diff = np.abs(preds - targets) / denom * m
    return float(diff.sum() / max(m.sum(), 1.0) * 100.0)


def horizon_report(preds, targets, horizons=(3, 6, 12),
                   mask: Optional[np.ndarray] = None):
    """Return a dict with per-horizon MAE / RMSE / MAPE.

    Parameters
    ----------
    preds, targets : array-like
        Shape ``(num_windows, horizon, N, 1)``.
    horizons : tuple of int
        Step indices (1-based) at which to report metrics.  For example,
        ``(3, 6, 12)`` corresponds to 15, 30 and 60 minutes at a
        5-minute sampling interval.
    """
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)
    m = None if mask is None else _to_numpy(mask)
    out = {}
    for h in horizons:
        idx = h - 1
        p = preds[:, idx]
        t = targets[:, idx]
        mm = m[:, idx] if m is not None else None
        out[h] = {
            "mae": masked_mae(p, t, mm),
            "rmse": masked_rmse(p, t, mm),
            "mape": masked_mape(p, t, mm),
        }
    return out
