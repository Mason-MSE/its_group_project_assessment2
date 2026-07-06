"""End-to-end demonstration script.

Runs a short DCRNN training loop (2 epochs) on the synthetic dataset,
prints per-horizon accuracy on a hold-out split and emits an example
safety alert.  When PyTorch is unavailable, the script falls back to
``sanity_check.py`` automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch  # noqa: F401
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False


def _run_torch_demo() -> None:
    import numpy as np
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

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
    from models.dcrnn import DCRNN
    from utils.metrics import horizon_report
    from utils.safety_alert import SafetyConfig, compute_risk_index, emit_alerts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Device: {device}")

    ds = build_mock_dataset(num_sensors=30, num_steps=24 * 12 * 3, seed=7)
    features = build_features(ds)
    scaler = fit_scaler(features)
    features_scaled = features.copy()
    features_scaled[..., 0] = scaler.transform(features[..., 0])
    train_f, val_f, test_f = train_val_test_split(features_scaled)

    seq_len, horizon = 12, 12
    x_train, y_train = make_sequences(train_f, seq_len, horizon)
    x_test, y_test = make_sequences(test_f, seq_len, horizon)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=16,
        shuffle=True,
    )

    adj = build_adjacency_matrix(ds.distance, sigma=10.0, threshold=0.1)
    fwd, bwd = dual_random_walk_matrices(adj)
    supports = [torch.from_numpy(fwd).to(device), torch.from_numpy(bwd).to(device)]

    model = DCRNN(
        supports=supports,
        input_dim=2,
        output_dim=1,
        hidden_dim=16,
        num_layers=1,
        max_diffusion_step=2,
        horizon=horizon,
        cl_decay_steps=1000,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    def masked_mae_loss(preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = (y != 0.0).float()
        diff = torch.abs(preds - y) * mask
        return diff.sum() / mask.sum().clamp(min=1.0)

    print("[demo] Training for 2 epochs on synthetic data ...")
    global_step = 0
    for epoch in range(2):
        losses = []
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            preds = model(x, targets=y, global_step=global_step)
            loss = masked_mae_loss(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item())); global_step += 1
        print(f"[demo] Epoch {epoch + 1}/2  train_loss={sum(losses) / len(losses):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
    preds_speed = scaler.inverse_transform(preds[..., 0])
    y_speed = scaler.inverse_transform(y_test[..., 0])
    report = horizon_report(preds_speed[..., None], y_speed[..., None])
    print("[demo] Test-set accuracy:")
    for h, metrics in report.items():
        print(f"  horizon={h} steps -> MAE={metrics['mae']:.3f}  "
              f"RMSE={metrics['rmse']:.3f}  MAPE={metrics['mape']:.2f}%")

    # --- safety alert demonstration ----------------------------------------
    safety_cfg = SafetyConfig()
    example = preds_speed[0]           # (horizon, N)
    current = example[0]
    risk = compute_risk_index(current, example, adj, safety_cfg)
    alerts = emit_alerts(risk, safety_cfg)
    print(f"[demo] Safety layer triggered {len(alerts)} alerts on the "
          f"first test window (sensors: {sorted(alerts.keys())[:5]}...).")


def _run_numpy_fallback() -> None:
    from sanity_check import main as sc_main
    print("[demo] PyTorch not available – running NumPy sanity_check.")
    sc_main()


def main() -> None:
    if _TORCH_OK:
        _run_torch_demo()
    else:  # pragma: no cover
        _run_numpy_fallback()


if __name__ == "__main__":
    main()
