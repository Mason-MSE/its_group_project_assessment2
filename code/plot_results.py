"""Generate report-quality figures for the DCRNN traffic prediction project.

Usage
-----
.. code-block:: bash

    python plot_results.py --config configs/dcrnn_metr_la_cpu.yaml \
                           --dataset metr-la \
                           --checkpoint ckpt/best.pt

Outputs are saved to ``figures/`` (created automatically).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

try:
    import torch
except ImportError:
    print("[plot_results] PyTorch is not installed.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.mock_data_loader import (
    build_mock_dataset,
    build_features as build_mock_features,
    train_val_test_split,
    make_sequences,
)
from data.preprocessing import (
    StandardScaler,
    build_adjacency_matrix,
    dual_random_walk_matrices,
)
from models.dcrnn import DCRNN
from utils.metrics import horizon_report, masked_mae, masked_rmse, masked_mape
from utils.safety_alert import SafetyConfig, compute_risk_index, emit_alerts

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

COLORS = {
    "actual": "#2c3e50",
    "predicted": "#e74c3c",
    "train": "#3498db",
    "val": "#e67e22",
    "mae": "#3498db",
    "rmse": "#e74c3c",
    "mape": "#2ecc71",
}


def load_data_and_model(cfg, checkpoint_path, device, num_sensors=None):
    ds = build_mock_dataset(
        num_sensors=num_sensors or cfg["data"]["num_sensors"],
        seed=cfg["training"]["seed"],
    )
    features = build_mock_features(ds)
    _, _, test_f = train_val_test_split(
        features,
        (cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"]),
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    scaler = StandardScaler(mean=ckpt["scaler_mean"], std=ckpt["scaler_std"])
    test_scaled = test_f.copy()
    test_scaled[..., 0] = scaler.transform(test_f[..., 0])
    x, y = make_sequences(test_scaled, cfg["data"]["seq_len"], cfg["data"]["horizon"])

    adj = build_adjacency_matrix(
        ds.distance,
        sigma=cfg["data"]["adj_sigma"],
        threshold=cfg["data"]["adj_threshold"],
    )
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

    return ds, preds_speed, y_speed, scaler, test_f


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ===== Figure 1: Training & Validation Loss Curve =====
def plot_training_curve(save_dir, log_csv="train_log.csv"):
    csv_path = Path(log_csv)
    if not csv_path.exists():
        print(f"[plot] {log_csv} not found – skip training curve. "
              f"Run train.py with --log-csv first.")
        return

    data = np.genfromtxt(str(csv_path), delimiter=",", names=True)
    epochs = data["epoch"]
    train_loss = data["train_loss"]
    val_mae = data["val_mae"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, train_loss, "o-", color=COLORS["train"],
             linewidth=2, markersize=4, label="Train Loss (MAE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (Masked MAE)", color=COLORS["train"])
    ax1.tick_params(axis="y", labelcolor=COLORS["train"])

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_mae, "s--", color=COLORS["val"],
             linewidth=2, markersize=4, label="Validation MAE")
    ax2.set_ylabel("Validation MAE", color=COLORS["val"])
    ax2.tick_params(axis="y", labelcolor=COLORS["val"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Training & Validation Loss Curve")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_training_curve.png"))
    plt.close(fig)
    print(f"[plot] Saved fig1_training_curve.png")


# ===== Figure 2: Predicted vs Actual Time Series =====
def plot_prediction_timeseries(preds_speed, y_speed, save_dir,
                               sensor_indices=None, num_windows=3):
    if sensor_indices is None:
        n = preds_speed.shape[1]
        sensor_indices = [0, min(50, n-1), min(100, n-1)]
        sensor_indices = list(dict.fromkeys(sensor_indices))  # deduplicate

    T_total = preds_speed.shape[0]
    window_len = min(50, T_total // num_windows)
    step = T_total // num_windows

    fig, axes = plt.subplots(len(sensor_indices), num_windows,
                             figsize=(14, 3 * len(sensor_indices)),
                             sharex=False)
    if len(sensor_indices) == 1:
        axes = axes[np.newaxis, :]

    for row, si in enumerate(sensor_indices):
        for col in range(num_windows):
            start = col * step
            end = start + window_len
            t = np.arange(start, end)
            ax = axes[row, col]
            ax.plot(t, y_speed[start:end, si], color=COLORS["actual"],
                    linewidth=1.2, label="Actual")
            ax.plot(t, preds_speed[start:end, si], color=COLORS["predicted"],
                    linewidth=1.2, linestyle="--", label="Predicted")
            if row == 0:
                ax.set_title(f"Window {col + 1}")
            if row == len(sensor_indices) - 1:
                ax.set_xlabel("Time Step")
            if col == 0:
                ax.set_ylabel(f"Sensor {si}\nSpeed (mph)")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    fig.suptitle("Predicted vs Actual Speed – Selected Sensors & Time Windows",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig2_prediction_timeseries.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved fig2_prediction_timeseries.png")


# ===== Figure 3: Horizon Performance Metrics (Grouped Bar Chart) =====
def plot_horizon_metrics(preds_speed, y_speed, save_dir):
    report = horizon_report(preds_speed[..., None], y_speed[..., None],
                            horizons=(1, 3, 6, 9, 12))
    horizons = sorted(report.keys())
    mae_vals = [report[h]["mae"] for h in horizons]
    rmse_vals = [report[h]["rmse"] for h in horizons]
    mape_vals = [report[h]["mape"] for h in horizons]

    x = np.arange(len(horizons))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(8, 5))
    bars1 = ax1.bar(x - width, mae_vals, width, label="MAE (mph)",
                    color=COLORS["mae"], alpha=0.85)
    bars2 = ax1.bar(x, rmse_vals, width, label="RMSE (mph)",
                    color=COLORS["rmse"], alpha=0.85)
    ax1.set_xlabel("Prediction Horizon (steps × 5 min)")
    ax1.set_ylabel("Error (mph)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h}\n({h * 5} min)" for h in horizons])
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    bars3 = ax2.bar(x + width, mape_vals, width, label="MAPE (%)",
                    color=COLORS["mape"], alpha=0.85)
    ax2.set_ylabel("MAPE (%)")
    ax2.legend(loc="upper right")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax = bar.axes
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax1.set_title("Prediction Accuracy by Horizon")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_horizon_metrics.png"))
    plt.close(fig)
    print(f"[plot] Saved fig3_horizon_metrics.png")


# ===== Figure 4: Predicted vs Actual Scatter Plot =====
def plot_scatter(preds_speed, y_speed, save_dir):
    flat_pred = preds_speed.flatten()
    flat_true = y_speed.flatten()

    mask = flat_true > 0
    flat_pred = flat_pred[mask]
    flat_true = flat_true[mask]

    max_val = max(flat_true.max(), flat_pred.max()) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(flat_true, flat_pred, s=0.3, alpha=0.15, color=COLORS["predicted"],
               rasterized=True)
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.5, label="Ideal (y=x)")

    mae = masked_mae(flat_pred, flat_true)
    rmse = masked_rmse(flat_pred, flat_true)
    mape = masked_mape(flat_pred, flat_true)
    corr = np.corrcoef(flat_true, flat_pred)[0, 1]

    textstr = (f"MAE = {mae:.2f} mph\n"
               f"RMSE = {rmse:.2f} mph\n"
               f"MAPE = {mape:.1f}%\n"
               f"r = {corr:.4f}")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    ax.set_xlabel("Actual Speed (mph)")
    ax.set_ylabel("Predicted Speed (mph)")
    ax.set_title("Predicted vs Actual Speed")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig4_scatter_plot.png"))
    plt.close(fig)
    print(f"[plot] Saved fig4_scatter_plot.png")


# ===== Figure 5: Residual Distribution =====
def plot_residuals(preds_speed, y_speed, save_dir):
    residuals = preds_speed.flatten() - y_speed.flatten()
    mask = y_speed.flatten() > 0
    residuals = residuals[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(residuals, bins=100, color=COLORS["predicted"], alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(residuals.mean(), color="blue", linestyle="-", linewidth=1,
               label=f"Mean = {residuals.mean():.2f}")
    ax.set_xlabel("Prediction Error (mph)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    abs_res = np.abs(residuals)
    ax.hist(abs_res, bins=100, color=COLORS["mae"], alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(abs_res.mean(), color="red", linestyle="-", linewidth=1,
               label=f"Mean |error| = {abs_res.mean():.2f}")
    ax.set_xlabel("Absolute Error (mph)")
    ax.set_ylabel("Frequency")
    ax.set_title("Absolute Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Residual Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_residual_distribution.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved fig5_residual_distribution.png")


# ===== Figure 6: Spatial Risk Heatmap =====
def plot_risk_heatmap(preds_speed, ds, save_dir, cfg, max_windows=200):
    adj = build_adjacency_matrix(ds.distance, sigma=cfg["data"]["adj_sigma"], threshold=cfg["data"]["adj_threshold"])
    safety_cfg = SafetyConfig(**cfg["safety"])
    num_windows = min(preds_speed.shape[0], max_windows)
    num_sensors = preds_speed.shape[1]
    risk_matrix = np.zeros((num_windows, num_sensors), dtype=np.float32)

    for w in range(num_windows):
        current = preds_speed[w, 0]
        future = np.zeros((1, num_sensors), dtype=np.float32)
        future[0] = preds_speed[w]
        risk_matrix[w] = compute_risk_index(current, future, adj, safety_cfg)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(risk_matrix.T, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("Test Window Index")
    ax.set_ylabel("Sensor Index")
    ax.set_title("Spatial-Temporal Risk Index Heatmap")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Risk Index")

    threshold = safety_cfg.alert_threshold
    ax.axhline(y=-0.5, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig6_risk_heatmap.png"))
    plt.close(fig)
    print(f"[plot] Saved fig6_risk_heatmap.png")

    num_alerted = int((risk_matrix >= threshold).any(axis=1).sum())
    print(f"  -> {num_alerted}/{num_windows} windows triggered alerts "
          f"(threshold={threshold})")


# ===== Figure 7: Speed Distribution by Sensor =====
def plot_speed_distribution(ds, save_dir):
    speed = ds.speed
    speed_valid = speed[speed > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(speed_valid, bins=80, color=COLORS["train"], alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(speed_valid), color="red", linestyle="--",
               label=f"Mean = {np.mean(speed_valid):.1f} mph")
    ax.set_xlabel("Speed (mph)")
    ax.set_ylabel("Frequency")
    ax.set_title("Overall Speed Distribution (All Sensors)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sensor_means = [np.mean(speed[:, i][speed[:, i] > 0]) for i in range(speed.shape[1])]
    ax.bar(range(len(sensor_means)), sensor_means, color=COLORS["train"],
           alpha=0.75, width=1.0)
    ax.axvline(np.mean(sensor_means), color="red", linestyle="--",
               label=f"Avg = {np.mean(sensor_means):.1f} mph")
    ax.set_xlabel("Sensor Index")
    ax.set_ylabel("Mean Speed (mph)")
    ax.set_title("Mean Speed per Sensor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("METR-LA Dataset Speed Statistics", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig7_speed_distribution.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved fig7_speed_distribution.png")


# ===== Main =====
def main():
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--config", default="configs/dcrnn_metr_la_cpu.yaml")
    parser.add_argument("--checkpoint", default="ckpt/best.pt")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--log-csv", default="train_log.csv",
                        help="Training log CSV from train.py --log-csv")
    parser.add_argument("--num-sensors", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("[plot] Loading data and model …")
    ds, preds_speed, y_speed, scaler, test_f = load_data_and_model(
        cfg, args.checkpoint, device, num_sensors=args.num_sensors
    )
    print(f"[plot] Test set: {preds_speed.shape[0]} windows, "
          f"{preds_speed.shape[2]} sensors, "
          f"horizon={preds_speed.shape[1]} steps")

    # Use the last horizon step for 2D visualizations
    ps_2d = preds_speed[:, -1, :]  # (W, N) - farthest prediction
    ys_2d = y_speed[:, -1, :]

    plot_training_curve(args.output_dir, args.log_csv)
    n_sensors = ps_2d.shape[1]
    sensor_samples = list(range(0, n_sensors, max(1, n_sensors // 3)))[:3]
    plot_prediction_timeseries(ps_2d, ys_2d, args.output_dir,
                                sensor_indices=sensor_samples)
    plot_horizon_metrics(preds_speed, y_speed, args.output_dir)
    plot_scatter(ps_2d, ys_2d, args.output_dir)
    plot_residuals(ps_2d, ys_2d, args.output_dir)
    plot_risk_heatmap(ps_2d, ds, args.output_dir, cfg)
    plot_speed_distribution(ds, args.output_dir)

    print(f"\n[plot] All figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
