"""Synthetic METR-LA-style speed data.

The real METR-LA dataset (Li et al., 2018) provides 5-minute speed
readings from 207 loop detectors over ~4 months.  For this reference
implementation we synthesise a tensor with the same *shape* and roughly
the same statistical profile:

* Free-flow speed around 65 mph
* Congestion dips during morning and evening peaks
* Weak spatial correlation encoded through a random block structure
* Occasional missing values (masked with 0.0)

The signal is deliberately non-trivial so that models with spatial
awareness (DCRNN) can, in principle, beat a naive baseline even on the
mock feed.  No claim is made about realism – all headline metrics in the
accompanying report use the numbers published in the original papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MockDataset:
    """Container for a synthetic traffic-speed dataset.

    Attributes
    ----------
    speed : np.ndarray
        Shape ``(T, N)`` – speed values in mph.
    time_of_day : np.ndarray
        Shape ``(T,)`` – fractional time-of-day in [0, 1).
    distance : np.ndarray
        Shape ``(N, N)`` – pairwise road-network distances (miles).
    missing_mask : np.ndarray
        Shape ``(T, N)`` – 1 where the reading is observed, 0 otherwise.
    """

    speed: np.ndarray
    time_of_day: np.ndarray
    distance: np.ndarray
    missing_mask: np.ndarray

    @property
    def num_sensors(self) -> int:
        return self.speed.shape[1]

    @property
    def num_steps(self) -> int:
        return self.speed.shape[0]


def _build_random_road_graph(num_sensors: int, seed: int) -> np.ndarray:
    """Return a symmetric pairwise distance matrix in miles."""
    rng = np.random.default_rng(seed)
    # Place sensors on a 2-D plane and compute Euclidean distances.
    coords = rng.uniform(0.0, 30.0, size=(num_sensors, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    # Zero self-distance already; keep the matrix symmetric by construction.
    return dist


def build_mock_dataset(
    num_sensors: int = 207,
    num_steps: int = 24 * 12 * 7,  # one week of 5-minute steps
    missing_rate: float = 0.05,
    seed: int = 2026,
) -> MockDataset:
    """Synthesise a METR-LA-shaped tensor.

    Parameters
    ----------
    num_sensors, num_steps : int
        Sensor count and length of the time axis.
    missing_rate : float
        Fraction of readings that will be flagged as missing.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # 1. Time-of-day signal (sinusoidal peaks around 08:00 and 17:30).
    t_index = np.arange(num_steps)
    tod = (t_index % (24 * 12)) / (24 * 12)  # 12 five-minute steps per hour
    morning = np.exp(-((tod - (8 / 24)) ** 2) / (2 * (1.2 / 24) ** 2))
    evening = np.exp(-((tod - (17.5 / 24)) ** 2) / (2 * (1.4 / 24) ** 2))
    global_congestion = 0.6 * morning + 0.75 * evening
    day_of_week = (t_index // (24 * 12)) % 7
    weekend_factor = np.where(day_of_week >= 5, 0.35, 1.0)
    global_congestion = global_congestion * weekend_factor

    # 2. Spatial pattern – group sensors into 6 clusters with shared trends.
    cluster_ids = rng.integers(0, 6, size=num_sensors)
    cluster_bias = rng.normal(0.0, 0.15, size=6)
    sensor_bias = cluster_bias[cluster_ids] + rng.normal(0.0, 0.05, size=num_sensors)

    # 3. Combine into speed values around a free-flow baseline of 65 mph.
    free_flow = 65.0
    depth = 30.0  # maximum congestion-induced speed drop
    speed = np.zeros((num_steps, num_sensors), dtype=np.float32)
    for i in range(num_sensors):
        drop = depth * (global_congestion + sensor_bias[i]).clip(0.0, 1.0)
        noise = rng.normal(0.0, 1.2, size=num_steps).astype(np.float32)
        speed[:, i] = free_flow - drop + noise

    # 4. Inject occasional incident bursts (localised sharp drops).
    num_incidents = max(1, num_steps // (24 * 12) // 2)
    for _ in range(num_incidents):
        t = rng.integers(48, num_steps - 48)
        cluster = rng.integers(0, 6)
        affected = np.where(cluster_ids == cluster)[0]
        duration = int(rng.integers(6, 24))
        drop = float(rng.uniform(15.0, 35.0))
        for k in range(duration):
            step = min(t + k, num_steps - 1)
            speed[step, affected] -= drop * max(0.0, 1.0 - k / duration)

    speed = speed.clip(2.0, 75.0)

    # 5. Missing-value mask.
    mask = (rng.uniform(0.0, 1.0, size=speed.shape) >= missing_rate).astype(np.float32)
    speed_observed = speed * mask

    distance = _build_random_road_graph(num_sensors, seed=seed + 1)
    return MockDataset(
        speed=speed_observed,
        time_of_day=tod.astype(np.float32),
        distance=distance.astype(np.float32),
        missing_mask=mask,
    )


def build_features(
    dataset: MockDataset,
    include_time_of_day: bool = True,
) -> np.ndarray:
    """Stack raw speed with an optional time-of-day channel.

    Returns
    -------
    features : np.ndarray
        Shape ``(T, N, F)``.
    """
    speed = dataset.speed[..., None]  # (T, N, 1)
    if not include_time_of_day:
        return speed
    tod = np.broadcast_to(
        dataset.time_of_day[:, None, None],
        (dataset.num_steps, dataset.num_sensors, 1),
    )
    return np.concatenate([speed, tod.astype(np.float32)], axis=-1)


def train_val_test_split(
    features: np.ndarray,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split a ``(T, N, F)`` tensor."""
    train_r, val_r, _ = ratios
    total = features.shape[0]
    train_end = int(total * train_r)
    val_end = int(total * (train_r + val_r))
    return features[:train_end], features[train_end:val_end], features[val_end:]


def make_sequences(
    features: np.ndarray,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping (input, target) windows.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(T, N, F)``.
    seq_len, horizon : int
        Length of the historical window and forecasting horizon in steps.

    Returns
    -------
    x : np.ndarray
        Shape ``(num_windows, seq_len, N, F)``.
    y : np.ndarray
        Shape ``(num_windows, horizon, N, 1)``.  Only the speed channel is
        used as the target.
    """
    total = features.shape[0]
    num_windows = total - seq_len - horizon + 1
    if num_windows <= 0:
        raise ValueError("Feature tensor is shorter than seq_len + horizon.")
    x = np.zeros((num_windows, seq_len, *features.shape[1:]), dtype=np.float32)
    y = np.zeros((num_windows, horizon, features.shape[1], 1), dtype=np.float32)
    for i in range(num_windows):
        x[i] = features[i : i + seq_len]
        y[i] = features[i + seq_len : i + seq_len + horizon, :, :1]
    return x, y
