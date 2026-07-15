"""Training entry point for DCRNN on the mock (or real) dataset.

Usage
-----

.. code-block:: bash

    python train.py --config configs/dcrnn_metr_la.yaml

The script assumes PyTorch is installed.  When run without ``torch`` it
prints a friendly message and exits.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    print("[train] PyTorch is not installed – run sanity_check.py instead.")
    sys.exit(0)

# Make sure sibling packages resolve correctly when the script is run
# from the parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.mock_data_loader import (
    build_mock_dataset,
    build_features as build_mock_features,
    make_sequences,
    train_val_test_split,
)
from data.real_data_loader import (
    load_real_dataset,
    build_features as build_real_features,
    train_val_test_split as real_split,
    make_sequences as real_sequences,
)
from data.preprocessing import (
    build_adjacency_matrix,
    dual_random_walk_matrices,
    fit_scaler,
)
from models.dcrnn import DCRNN
from utils.metrics import masked_mae, horizon_report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_mae_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Differentiable masked MAE."""
    mask = (targets != 0.0).float()
    diff = torch.abs(preds - targets) * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dcrnn_metr_la.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the number of epochs.")
    parser.add_argument("--dataset", type=str, default="mock",
                        choices=["mock", "metr-la", "pems-bay"],
                        help="Dataset to use (mock=synthetic, metr-la, pems-bay).")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Path to save training log CSV (epoch, train_loss, val_mae).")
    parser.add_argument("--dataset-dir", type=str, default="dataset",
                        help="Directory containing the dataset files.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # --- data ---------------------------------------------------------------
    seq_len = cfg["data"]["seq_len"]; horizon = cfg["data"]["horizon"]

    if args.dataset == "mock":
        ds = build_mock_dataset(
            num_sensors=cfg["data"]["num_sensors"],
            num_steps=24 * 12 * 7,
            seed=cfg["training"]["seed"],
        )
        features = build_mock_features(ds)
        train_f, val_f, test_f = train_val_test_split(
            features,
            (cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"]),
        )
        scaler = fit_scaler(train_f)
        train_scaled = train_f.copy(); train_scaled[..., 0] = scaler.transform(train_f[..., 0])
        val_scaled = val_f.copy(); val_scaled[..., 0] = scaler.transform(val_f[..., 0])
        test_scaled = test_f.copy(); test_scaled[..., 0] = scaler.transform(test_f[..., 0])

        x_train, y_train = make_sequences(train_scaled, seq_len, horizon)
        x_val, y_val = make_sequences(val_scaled, seq_len, horizon)
        x_test, y_test = make_sequences(test_scaled, seq_len, horizon)

        adj = build_adjacency_matrix(
            ds.distance,
            sigma=cfg["data"]["adj_sigma"],
            threshold=cfg["data"]["adj_threshold"],
        )
    else:
        ds = load_real_dataset(args.dataset, args.dataset_dir)
        features = build_real_features(ds.speed, ds.time_of_day)
        train_f, val_f, test_f = real_split(
            features,
            (cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"]),
        )
        scaler = fit_scaler(train_f)
        train_scaled = train_f.copy(); train_scaled[..., 0] = scaler.transform(train_f[..., 0])
        val_scaled = val_f.copy(); val_scaled[..., 0] = scaler.transform(val_f[..., 0])
        test_scaled = test_f.copy(); test_scaled[..., 0] = scaler.transform(test_f[..., 0])

        x_train, y_train = real_sequences(train_scaled, seq_len, horizon)
        x_val, y_val = real_sequences(val_scaled, seq_len, horizon)
        x_test, y_test = real_sequences(test_scaled, seq_len, horizon)

        adj = ds.adjacency

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=cfg["training"]["batch_size"],
    )

    # --- adjacency ----------------------------------------------------------
    fwd, bwd = dual_random_walk_matrices(adj)
    supports = [torch.from_numpy(fwd).to(device), torch.from_numpy(bwd).to(device)]

    # --- model --------------------------------------------------------------
    model = DCRNN(
        supports=supports,
        input_dim=cfg["model"]["input_dim"],
        output_dim=cfg["model"]["output_dim"],
        hidden_dim=cfg["model"]["rnn_units"],
        num_layers=cfg["model"]["num_rnn_layers"],
        max_diffusion_step=cfg["model"]["max_diffusion_step"],
        horizon=horizon,
        cl_decay_steps=cfg["model"]["cl_decay_steps"],
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    epochs = args.epochs or cfg["training"]["epochs"]
    global_step = 0
    best_val = float("inf")

    log_rows = []
    os.makedirs("ckpt", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            preds = model(x, targets=y, global_step=global_step)
            loss = masked_mae_loss(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg["training"]["clip_grad_norm"])
            opt.step()
            total_loss += float(loss.item()); n_batches += 1; global_step += 1
        avg = total_loss / max(1, n_batches)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0; n_val = 0
            for x, y in val_loader:
                preds = model(x.to(device))
                val_loss += masked_mae(preds.cpu().numpy(), y.numpy())
                n_val += 1
        val_avg = val_loss / max(1, n_val)
        print(f"[train] Epoch {epoch + 1:03d}/{epochs}  train_loss={avg:.4f}  val_mae={val_avg:.4f}")
        log_rows.append((epoch + 1, avg, val_avg))

        if val_avg < best_val:
            best_val = val_avg
            torch.save({"model_state": model.state_dict(),
                         "scaler_mean": scaler.mean, "scaler_std": scaler.std},
                        "ckpt/best.pt")

    print(f"[train] Best val MAE: {best_val:.4f}")

    if args.log_csv:
        with open(args.log_csv, "w") as f:
            f.write("epoch,train_loss,val_mae\n")
            for ep, tl, vm in log_rows:
                f.write(f"{ep},{tl:.6f},{vm:.6f}\n")
        print(f"[train] Training log saved to {args.log_csv}")

    # --- final evaluation --------------------------------------------------
    x_test_t = torch.from_numpy(x_test).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(x_test_t).cpu().numpy()
    preds_speed = scaler.inverse_transform(preds[..., 0])
    y_speed = scaler.inverse_transform(y_test[..., 0])
    report = horizon_report(preds_speed[..., None], y_speed[..., None])
    for h, metrics in report.items():
        print(f"[train] horizon={h} steps -> MAE={metrics['mae']:.3f}  "
              f"RMSE={metrics['rmse']:.3f}  MAPE={metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
