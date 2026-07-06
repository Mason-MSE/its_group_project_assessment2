"""Pure-NumPy sanity check for the MSE806 A2 reference implementation.

This script exercises the data loader, adjacency-matrix builder,
Historical Average baseline, metrics module and safety-alert layer
without importing PyTorch, so it can run in restricted environments.

Run it with ``python sanity_check.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.mock_data_loader import (
    build_mock_dataset,
    build_features,
    make_sequences,
    train_val_test_split,
)
from data.preprocessing import (
    build_adjacency_matrix,
    dual_random_walk_matrices,
    fit_scaler,
)
from models.baselines import HistoricalAverage
from utils.metrics import horizon_report, masked_mae, masked_rmse, masked_mape
from utils.safety_alert import (
    SafetyConfig,
    compute_risk_index,
    emit_alerts,
    evaluate_alerts,
)


def main() -> None:
    print("[sanity] 1. Building mock dataset ...")
    ds = build_mock_dataset(num_sensors=30, num_steps=24 * 12 * 3, seed=7)
    print(f"          speed shape = {ds.speed.shape} "
          f"(mean={ds.speed.mean():.2f}, std={ds.speed.std():.2f})")

    print("[sanity] 2. Building adjacency matrix ...")
    adj = build_adjacency_matrix(ds.distance, sigma=10.0, threshold=0.1)
    fwd, bwd = dual_random_walk_matrices(adj)
    print(f"          adj shape = {adj.shape}, "
          f"non-zero entries = {(adj > 0).sum()}, "
          f"forward row sum mean = {fwd.sum(axis=1).mean():.3f}")

    print("[sanity] 3. Preparing sequences (seq_len=12, horizon=12) ...")
    features = build_features(ds)
    scaler = fit_scaler(features)
    features_scaled = features.copy()
    features_scaled[..., 0] = scaler.transform(features[..., 0])
    train_f, val_f, test_f = train_val_test_split(features_scaled)
    _, _ = make_sequences(train_f, 12, 12)  # smoke test only
    _, y_test = make_sequences(test_f, 12, 12)
    print(f"          y_test shape = {y_test.shape}")

    print("[sanity] 4. Historical Average baseline ...")
    ha = HistoricalAverage(period=24 * 12).fit(ds.speed[: train_f.shape[0]])
    # Use raw speed for HA (baseline does not need scaling).
    _, y_true_raw = make_sequences(build_features(ds)[test_f.shape[0] * 0:], 12, 12)
    preds_ha = ha.predict(start_step=train_f.shape[0] + 12,
                           horizon=12, num_windows=y_test.shape[0])
    y_true_raw = build_features(ds)[
        train_f.shape[0] + val_f.shape[0]:
    ]
    _, y_true = make_sequences(y_true_raw, 12, 12)  # (W, 12, N, 1)
    horizon_metrics = horizon_report(preds_ha, y_true, horizons=(3, 6, 12))
    print("          HA metrics per horizon (illustrative on mock data):")
    for h, m in horizon_metrics.items():
        print(f"            horizon={h} steps -> MAE={m['mae']:.3f}  "
              f"RMSE={m['rmse']:.3f}  MAPE={m['mape']:.2f}%")

    print("[sanity] 5. Metrics helpers on random tensors ...")
    rng = np.random.default_rng(0)
    a = rng.normal(60, 5, size=(10, 20)).astype(np.float32)
    b = a + rng.normal(0, 1, size=a.shape).astype(np.float32)
    print(f"          MAE={masked_mae(a, b):.3f}  "
          f"RMSE={masked_rmse(a, b):.3f}  MAPE={masked_mape(a, b):.2f}%")

    print("[sanity] 6. Safety alert layer ...")
    cfg = SafetyConfig()
    # Simulate a rapid speed collapse over the lookahead window
    # (drop from 60 mph to 20 mph within the first 3 steps).
    horizon = 12
    forecast = np.full((horizon, ds.num_sensors), 60.0, dtype=np.float32)
    fast_drop = np.linspace(60.0, 15.0, cfg.lookahead_steps)
    forecast[: cfg.lookahead_steps, 0] = fast_drop
    forecast[: cfg.lookahead_steps, 1] = np.linspace(60.0, 25.0, cfg.lookahead_steps)
    # Extend the low speed for the rest of the horizon to reflect a
    # sustained congestion pattern.
    forecast[cfg.lookahead_steps:, 0] = 18.0
    forecast[cfg.lookahead_steps:, 1] = 28.0
    current = np.full(ds.num_sensors, 60.0, dtype=np.float32)
    risk = compute_risk_index(current, forecast, adj, cfg)
    alerts = emit_alerts(risk, cfg)
    evaluation = evaluate_alerts(
        triggered=alerts,
        ground_truth_sensors=[0, 1],
        all_sensors=range(ds.num_sensors),
    )
    top = sorted(alerts.items(), key=lambda x: -x[1])[:5]
    print(f"          Number of triggered sensors = {len(alerts)}")
    print(f"          Top-5 risk scores           = {top}")
    print(f"          Alert evaluation vs. synthetic ground truth: {evaluation}")

    print("[sanity] Done.")


if __name__ == "__main__":
    main()
