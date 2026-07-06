"""Safety-oriented alert layer on top of speed forecasts.

Predicted speed trajectories are transformed into a rolling congestion
/ incident-risk index

.. math::

    R_i(t) = w_1 \\cdot \\mathrm{norm}(\\Delta V_i(t))
           + w_2 \\cdot \\mathrm{norm}(\\mathrm{CV}_i(t))
           + w_3 \\cdot \\rho_i(t),

where

* :math:`\\Delta V_i(t)` is the predicted speed drop over the horizon,
* :math:`\\mathrm{CV}_i(t)` is the coefficient of variation of the
  predicted trajectory,
* :math:`\\rho_i(t)` is the fraction of one-hop neighbours whose
  forecasts also fall below the free-flow speed threshold.

An alert is emitted when ``R_i(t)`` exceeds ``alert_threshold``.  Using
both magnitude and spatial coherence prevents isolated sensor noise from
triggering false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass
class SafetyConfig:
    w_speed_drop: float = 0.5
    w_variability: float = 0.2
    w_neighbourhood: float = 0.3
    free_flow_mph: float = 55.0
    alert_threshold: float = 0.6
    lookahead_steps: int = 3


def _normalise(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def compute_risk_index(
    current_speed: np.ndarray,      # (N,)
    predicted_speed: np.ndarray,    # (horizon, N)
    adjacency: np.ndarray,          # (N, N)
    cfg: SafetyConfig,
) -> np.ndarray:
    """Return the rolling risk index for every sensor.

    The three sub-terms are normalised to ``[0, 1]`` using
    interpretable ranges, so ``R_i(t)`` sits in ``[0, 1]`` too.
    """
    horizon, num_sensors = predicted_speed.shape
    if adjacency.shape != (num_sensors, num_sensors):
        raise ValueError("Adjacency matrix shape mismatch.")

    # 1. Predicted speed drop over the lookahead window (mph).
    end = min(cfg.lookahead_steps, horizon)
    window = predicted_speed[:end]
    delta_v = current_speed - window.min(axis=0)  # (N,)
    term_drop = _normalise(delta_v, lo=5.0, hi=35.0)

    # 2. Coefficient of variation of the predicted trajectory.
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    cv = std / np.maximum(mean, 1.0)
    term_cv = _normalise(cv, lo=0.02, hi=0.30)

    # 3. Neighbour congestion rate.
    below = (window < cfg.free_flow_mph).any(axis=0).astype(np.float32)  # (N,)
    neigh = (adjacency > 0).astype(np.float32)  # binary neighbour mask
    neigh_count = neigh.sum(axis=1)
    neigh_count = np.where(neigh_count == 0.0, 1.0, neigh_count)
    rho = neigh @ below / neigh_count
    term_neigh = np.clip(rho, 0.0, 1.0)

    risk = (
        cfg.w_speed_drop * term_drop
        + cfg.w_variability * term_cv
        + cfg.w_neighbourhood * term_neigh
    )
    return risk.astype(np.float32)


def emit_alerts(
    risk_index: np.ndarray,
    cfg: SafetyConfig,
    sensor_ids: Iterable[int] | None = None,
) -> Dict[int, float]:
    """Return sensor_id -> risk_score for sensors above the threshold."""
    if sensor_ids is None:
        sensor_ids = range(risk_index.shape[0])
    alerts: Dict[int, float] = {}
    for i, s in enumerate(sensor_ids):
        if risk_index[i] >= cfg.alert_threshold:
            alerts[int(s)] = float(risk_index[i])
    return alerts


def evaluate_alerts(
    triggered: Dict[int, float],
    ground_truth_sensors: Iterable[int],
    all_sensors: Iterable[int],
) -> Dict[str, float]:
    """Compute precision / recall / false-alarm / miss rate.

    Parameters
    ----------
    triggered : dict
        Sensors above the alert threshold and their risk score.
    ground_truth_sensors : iterable of int
        Sensors that later experienced a real congestion event.
    all_sensors : iterable of int
        The full set of sensor identifiers under consideration.
    """
    tp_set = set(triggered.keys()) & set(ground_truth_sensors)
    fp_set = set(triggered.keys()) - set(ground_truth_sensors)
    fn_set = set(ground_truth_sensors) - set(triggered.keys())
    tn_count = len(set(all_sensors)) - len(tp_set) - len(fp_set) - len(fn_set)

    precision = len(tp_set) / max(1, len(triggered))
    recall = len(tp_set) / max(1, len(list(ground_truth_sensors)))
    false_alarm = len(fp_set) / max(1, tn_count + len(fp_set))
    miss_rate = len(fn_set) / max(1, len(list(ground_truth_sensors)))

    return {
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm,
        "miss_rate": miss_rate,
        "num_alerts": float(len(triggered)),
    }
