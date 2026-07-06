"""Evaluate a trained DCRNN checkpoint on the mock (or real) test split.

Usage
-----

.. code-block:: bash

    python evaluate.py --config configs/dcrnn_metr_la.yaml \
                      --checkpoint ckpt/best.pt

The evaluation reports MAE / RMSE / MAPE at the standard 15 / 30 / 60-minute
horizons and, in addition, a small safety diagnostic that counts the
number of triggered alerts in the test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    print("[evaluate] PyTorch is not installed; run sanity_check.py instead.")
    sys.exit(0)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.mock_data_loader import (
    build_mock_dataset,
    build_features,
    make_sequences,
    train_val_test_split,
)
from data.preprocessing import (
    StandardScaler,
    build_adjacency_matrix,
    dual_random_walk_matrices,
)
from models.dcrnn import DCRNN
from utils.metrics import horizon_report
from utils.safety_alert import SafetyConfig, compute_risk_index, emit_alerts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dcrnn_metr_la.yaml")
    parser.add_argument("--checkpoint", type=str, default="ckpt/best.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_mock_dataset(
        num_sensors=cfg["data"]["num_sensors"],
        seed=cfg["training"]["seed"],
    )
    features = build_features(ds)
    _, _, test_f = train_val_test_split(
        features,
        (cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"]),
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    scaler = StandardScaler(mean=ckpt["scaler_mean"], std=ckpt["scaler_std"])
    test_scaled = test_f.copy(); test_scaled[..., 0] = scaler.transform(test_f[..., 0])

    x, y = make_sequences(test_scaled, cfg["data"]["seq_len"], cfg["data"]["horizon"])
    adj = build_adjacency_matrix(ds.distance, sigma=cfg["data"]["adj_sigma"],
                                  threshold=cfg["data"]["adj_threshold"])
    fwd, bwd = dual_random_walk_matrices(adj)
    supports = [torch.from_numpy(fwd).to(device), torch.from_numpy(bwd).to(device)]

    model = DCRNN(
        supports=supports,
        input_dim=cfg["model"]["input_dim"],
        output_dim=cfg["model"]["output_dim"],
        hidden_dim=cfg["model"]["rnn_units"],
        num_layers=cfg["model"]["num_rnn_layers"],
        max_diffusion_step=cfg["model"]["max_diffusion_step"],
        horizon=cfg["data"]["horizon"],
        cl_decay_steps=cfg["model"]["cl_decay_steps"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(x).to(device)).cpu().numpy()
    preds_speed = scaler.inverse_transform(preds[..., 0])
    y_speed = scaler.inverse_transform(y[..., 0])

    report = horizon_report(preds_speed[..., None], y_speed[..., None])
    print("[evaluate] Accuracy report:")
    for h, metrics in report.items():
        print(f"  horizon={h:>2d} steps -> MAE={metrics['mae']:.3f}  "
              f"RMSE={metrics['rmse']:.3f}  MAPE={metrics['mape']:.2f}%")

    # --- safety diagnostic --------------------------------------------------
    safety_cfg = SafetyConfig(**cfg["safety"])
    triggered_windows = 0
    for w in range(preds_speed.shape[0]):
        current = preds_speed[w, 0]                  # (N,)
        future = preds_speed[w]                      # (horizon, N)
        risk = compute_risk_index(current, future, adj, safety_cfg)
        if emit_alerts(risk, safety_cfg):
            triggered_windows += 1
    print(f"[evaluate] Safety layer triggered alerts on "
          f"{triggered_windows}/{preds_speed.shape[0]} windows.")


if __name__ == "__main__":
    main()
