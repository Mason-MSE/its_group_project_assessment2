#!/usr/bin/env python3
"""Generate comprehensive, report-quality figures for MSE806 A2."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    import torch
except ImportError:
    print("[generate] PyTorch not installed."); sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.mock_data_loader import (build_mock_dataset, build_features,
                                    make_sequences, train_val_test_split)
from data.preprocessing import (StandardScaler, build_adjacency_matrix,
                                dual_random_walk_matrices, fit_scaler)
from models.dcrnn import DCRNN
from models.baselines import HistoricalAverage, LSTMBaseline
from utils.metrics import horizon_report, masked_mae, masked_rmse, masked_mape
from utils.safety_alert import SafetyConfig, compute_risk_index, emit_alerts

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif",
})
C = {"act":"#2c3e50","pred":"#e74c3c","train":"#3498db","val":"#e67e22",
     "test":"#9b59b6","mae":"#3498db","rmse":"#e74c3c","mape":"#2ecc71",
     "ha":"#95a5a6","lstm":"#f39c12","dcrnn":"#e74c3c"}

def ensure(p):
    os.makedirs(p, exist_ok=True)

def load_all(cfg, ckpt_path, device, ns=None):
    ds = build_mock_dataset(num_sensors=ns or cfg["data"]["num_sensors"], seed=cfg["training"]["seed"])
    features = build_features(ds)
    tr, va, te = train_val_test_split(features, (0.7, 0.1, 0.2))
    ck = torch.load(ckpt_path, map_location=device)
    sc = StandardScaler(mean=ck["scaler_mean"], std=ck["scaler_std"])
    tes = te.copy(); tes[...,0] = sc.transform(te[...,0])
    x, y = make_sequences(tes, cfg["data"]["seq_len"], cfg["data"]["horizon"])
    adj = build_adjacency_matrix(ds.distance, sigma=cfg["data"]["adj_sigma"], threshold=cfg["data"]["adj_threshold"])
    fw, bw = dual_random_walk_matrices(adj)
    sp = [torch.from_numpy(fw).to(device), torch.from_numpy(bw).to(device)]
    m = DCRNN(sp, cfg["model"]["input_dim"], cfg["model"]["output_dim"],
              cfg["model"]["rnn_units"], cfg["model"]["num_rnn_layers"],
              cfg["model"]["max_diffusion_step"], cfg["data"]["horizon"],
              cfg["model"]["cl_decay_steps"]).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    with torch.no_grad():
        pr = m(torch.from_numpy(x).to(device)).cpu().numpy()
    ps = sc.inverse_transform(pr[...,0]); ys = sc.inverse_transform(y[...,0])
    return ds, ps, ys, adj, sc, te, tr, va

def plot_univariate_analysis(ds, save_dir):
    """Univariate statistical analysis of the speed dataset."""
    speed = ds.speed; speed_valid = speed[speed > 0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0,0]
    ax.hist(speed_valid, bins=100, color=C["train"], alpha=0.75, edgecolor="white", linewidth=0.3, density=True)
    mv, sv = np.mean(speed_valid), np.std(speed_valid)
    ax.axvline(mv, color="red", linestyle="--", linewidth=2, label=f"Mean = {mv:.1f} mph")
    ax.axvline(mv-sv, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.axvline(mv+sv, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Speed (mph)"); ax.set_ylabel("Density")
    ax.set_title("(a) Speed Distribution"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    sample_sensors = list(range(0, speed.shape[1], 20))[:10]
    data_to_plot = [speed[:,s][speed[:,s]>0] for s in sample_sensors]
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6)
    for patch, col in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.2,0.8,len(sample_sensors)))):
        patch.set_facecolor(col)
    ax.set_xticklabels([f"S{s}" for s in sample_sensors], rotation=45)
    ax.set_ylabel("Speed (mph)"); ax.set_title("(b) Speed Distribution per Sensor")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0,2]
    tod_hours = ds.time_of_day * 24
    ms_by_hod = np.array([np.mean(speed[(tod_hours>=h)&(tod_hours<h+1)][speed[(tod_hours>=h)&(tod_hours<h+1)]>0]) for h in range(24)])
    ax.plot(range(24), ms_by_hod, "o-", color=C["pred"], linewidth=2, markersize=6)
    ax.axvline(8, color="orange", linestyle="--", alpha=0.7, label="Morning peak")
    ax.axvline(17.5, color="purple", linestyle="--", alpha=0.7, label="Evening peak")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Mean Speed (mph)")
    ax.set_title("(c) Diurnal Speed Pattern"); ax.legend(fontsize=9)
    ax.set_xticks(range(0,24,2)); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    n_steps = min(500, speed.shape[0])
    ax.plot(np.arange(n_steps), speed[:n_steps,0], linewidth=0.8, color=C["act"], alpha=0.8)
    ax.set_xlabel("Time Step (5-min)"); ax.set_ylabel("Speed (mph)")
    ax.set_title("(d) Speed Time Series (Sensor 0)"); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    mr = (1 - ds.missing_mask.mean(axis=0)) * 100
    ax.bar(range(len(mr)), mr, color=C["val"], alpha=0.75, width=1.0)
    ax.axhline(np.mean(mr), color="red", linestyle="--", label=f"Avg = {np.mean(mr):.1f}%")
    ax.set_xlabel("Sensor Index"); ax.set_ylabel("Missing Rate (%)")
    ax.set_title("(e) Missing Data Rate per Sensor"); ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1,2]
    acorrs = []
    for s in range(min(10, speed.shape[1])):
        valid = speed[:,s][speed[:,s]>0]
        if len(valid) > 200:
            ac = np.correlate(valid-valid.mean(), valid-valid.mean(), mode="full")
            ac = ac[ac.size//2:] / ac.max()
            acorrs.append(ac[:200])
    if acorrs:
        am = np.mean(acorrs, axis=0)
        ax.plot(np.arange(len(am)), am, linewidth=1.5, color=C["train"])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(np.exp(-1), color="gray", linestyle="--", label="1/e decorrelation")
        ax.set_xlabel("Lag (5-min steps)"); ax.set_ylabel("Autocorrelation")
        ax.set_title("(f) Temporal Autocorrelation"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("Univariate Analysis of Traffic Speed Data", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "univariate_analysis.png"), bbox_inches="tight")
    plt.close(fig)
    print("[generate] Saved univariate_analysis.png")

def plot_bivariate_analysis(ds, save_dir):
    """Bivariate analysis: correlation matrix, scatter plots, sensor relationships."""
    speed = ds.speed; N = min(speed.shape[1], 30)
    fig = plt.figure(figsize=(18, 10))

    ax = fig.add_subplot(2,3,1)
    corr = np.corrcoef(speed[:,:N].T)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    im = ax.imshow(np.ma.masked_where(mask, corr), cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_title("(a) Sensor Correlation Matrix"); ax.set_xlabel("Sensor Index"); ax.set_ylabel("Sensor Index")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    ax = fig.add_subplot(2,3,2)
    triu_vals = corr[np.triu_indices_from(corr, k=1)]
    ax.hist(triu_vals, bins=50, color=C["train"], alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(triu_vals), color="red", linestyle="--", label=f"Mean r = {np.mean(triu_vals):.3f}")
    ax.set_xlabel("Pearson Correlation Coefficient"); ax.set_ylabel("Frequency")
    ax.set_title("(b) Distribution of Pairwise Correlations"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2,3,3)
    sidx = np.random.choice(speed.shape[0], min(5000, speed.shape[0]), replace=False)
    ax.scatter(ds.time_of_day[sidx]*24, speed[sidx,0], s=1, alpha=0.3, color=C["pred"])
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Speed (mph)"); ax.set_title("(c) Speed vs Time-of-Day")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2,3,4)
    ax.scatter(speed[:2000,0], speed[:2000,1], s=2, alpha=0.4, color=C["train"])
    r01 = np.corrcoef(speed[:2000,0], speed[:2000,1])[0,1]
    ax.set_xlabel("Sensor 0 Speed (mph)"); ax.set_ylabel("Sensor 1 Speed (mph)")
    ax.set_title(f"(d) Sensor 0 vs Sensor 1 (r={r01:.3f})"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2,3,5)
    n_steps = min(500, speed.shape[0]); sa, sb = 0, N//2
    ax.plot(np.arange(n_steps), speed[:n_steps,sa], linewidth=1, label=f"Sensor {sa}", color=C["act"])
    ax.plot(np.arange(n_steps), speed[:n_steps,sb], linewidth=1, label=f"Sensor {sb}", color=C["pred"])
    diff = np.mean(np.abs(speed[:n_steps,sa] - speed[:n_steps,sb]))
    ax.set_xlabel("Time Step"); ax.set_ylabel("Speed (mph)")
    ax.set_title(f"(e) Distant Sensors (mean |diff|={diff:.1f} mph)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2,3,6)
    a = build_adjacency_matrix(ds.distance, sigma=10.0, threshold=0.1)
    ax.spy(a[:50,:50], markersize=2, color=C["train"])
    ax.set_title("(f) Adjacency Matrix (first 50 sensors)"); ax.set_xlabel("Sensor Index"); ax.set_ylabel("Sensor Index")

    fig.suptitle("Bivariate & Spatial Analysis of Traffic Network Data", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "bivariate_analysis.png"), bbox_inches="tight")
    plt.close(fig)
    print("[generate] Saved bivariate_analysis.png")

def plot_elbow_silhouette(ds, save_dir):
    """K-means clustering analysis: elbow curve and silhouette scores."""
    import sys
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    speed = ds.speed; N = min(speed.shape[1], 80); tod_hours = ds.time_of_day * 24
    speed = speed[:, :N]
    print(f"[dbg] Using {N} sensors...", flush=True)
    hour_bins = np.floor(tod_hours).astype(int)
    hp = np.zeros((N, 24))
    for h in range(24):
        mask_h = (hour_bins == h)
        if mask_h.any():
            for s in range(N):
                mask = mask_h & (speed[:,s] > 0)
                hp[s,h] = np.mean(speed[mask,s]) if mask.any() else 0
    print(f"[dbg] hp done", flush=True)
    ss = np.zeros((N, 3))
    for s in range(N):
        v = speed[:,s][speed[:,s]>0]
        if len(v) > 0:
            ss[s,0] = float(np.mean(v)); ss[s,1] = float(np.std(v)); ss[s,2] = float(np.percentile(v, 5))
    print(f"[dbg] ss done", flush=True)
    X = np.column_stack([hp, ss])
    print(f"[dbg] X shape: {X.shape}", flush=True)
    print("[dbg] Creating subplots...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    print("[dbg] Running KMeans...", flush=True)
    K_range = range(2, 11); inertias = []; sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=3).fit(X)
        inertias.append(km.inertia_)
        lab = km.labels_
        sil_scores.append(silhouette_score(X, lab) if len(set(lab)) > 1 else 0)

    best_k = K_range[np.argmax(sil_scores)]
    ax = axes[0,0]
    ax.plot(K_range, inertias, "o-", color=C["train"], linewidth=2, markersize=6)
    diffs2 = np.diff(np.diff(inertias)); elbow = np.argmax(diffs2) + 3
    ax.axvline(elbow, color="red", linestyle="--", linewidth=2, label=f"Elbow at k={elbow}")
    ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia")
    ax.set_title("(a) Elbow Method for Optimal k"); ax.legend(fontsize=9)
    ax.set_xticks(list(K_range)); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(K_range, sil_scores, "s-", color=C["pred"], linewidth=2, markersize=6)
    ax.axvline(best_k, color="red", linestyle="--", linewidth=2, label=f"Best k={best_k} (score={max(sil_scores):.3f})")
    ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Silhouette Score")
    ax.set_title("(b) Silhouette Analysis"); ax.legend(fontsize=9)
    ax.set_xticks(list(K_range)); ax.grid(True, alpha=0.3)

    ax = axes[0,2]
    inertias_a = np.array(inertias); sil_scores_a = np.array(sil_scores)
    ni = (inertias_a-inertias_a.min())/(inertias_a.max()-inertias_a.min()+1e-8)
    ns = (sil_scores_a-sil_scores_a.min())/(sil_scores_a.max()-sil_scores_a.min()+1e-8)
    combined = 1-ni+ns
    ax.plot(K_range, combined, "D-", color=C["mape"], linewidth=2, markersize=6)
    ax.axvline(K_range[np.argmax(combined)], color="red", linestyle="--", linewidth=2, label=f"Best k={K_range[np.argmax(combined)]}")
    ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Combined Score")
    ax.set_title("(c) Combined Elbow + Silhouette"); ax.legend(fontsize=9)
    ax.set_xticks(list(K_range)); ax.grid(True, alpha=0.3)

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=3).fit(X)
    labels = km_best.labels_
    sil_vals = silhouette_samples(X, labels)
    ax = axes[1,0]; y_lower = 0
    for i in range(best_k):
        cs = np.sort(sil_vals[labels==i]); sz = len(cs)
        ax.fill_betweenx(np.arange(y_lower, y_lower+sz), 0, cs, facecolor=plt.cm.tab10(i), alpha=0.7, label=f"Cluster {i}")
        y_lower += sz + 5
    ax.axvline(silhouette_score(X, labels), color="red", linestyle="--", label=f"Avg = {silhouette_score(X, labels):.3f}")
    ax.set_xlabel("Silhouette Coefficient"); ax.set_ylabel("Cluster")
    ax.set_title(f"(d) Silhouette Plot (k={best_k})"); ax.legend(fontsize=8); ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1,1]
    unique, counts = np.unique(labels, return_counts=True)
    bars = ax.bar(unique, counts, color=[plt.cm.tab10(i) for i in unique], alpha=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, str(count), ha="center", fontsize=10)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Number of Sensors")
    ax.set_title(f"(e) Cluster Sizes (k={best_k})"); ax.set_xticks(unique); ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1,2]
    hr = np.arange(24)
    for i in range(best_k):
        members = np.where(labels==i)[0]
        profile = hp[members].mean(axis=0)
        ax.plot(hr, profile, linewidth=2, color=plt.cm.tab10(i), label=f"Cluster {i} (n={len(members)})")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Mean Speed (mph)")
    ax.set_title("(f) Hourly Speed Profile by Cluster"); ax.legend(fontsize=8, loc="lower left")
    ax.set_xticks(range(0,24,2)); ax.grid(True, alpha=0.3)

    fig.suptitle("K-Means Clustering of Traffic Sensors: Optimal k Selection", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "elbow_silhouette.png"), bbox_inches="tight")
    plt.close(fig)
    print("[generate] Saved elbow_silhouette.png")
    return X, labels

def plot_cluster_pca(X, labels, save_dir):
    """PCA visualization of sensor clusters."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2); X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    n_clusters = len(set(labels))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0,0]
    for i in range(n_clusters):
        mask = labels==i
        ax.scatter(X_pca[mask,0], X_pca[mask,1], color=plt.cm.tab10(i), label=f"Cluster {i}",
                   s=30, alpha=0.7, edgecolors="black", linewidth=0.3)
    ax.set_xlabel(f"PC 1 ({explained[0]:.1%})"); ax.set_ylabel(f"PC 2 ({explained[1]:.1%})")
    ax.set_title("(a) PCA Projection of Sensor Clusters"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    for i in range(n_clusters):
        pts = X_pca[labels==i]
        ax.scatter(pts[:,0], pts[:,1], color=plt.cm.tab10(i), label=f"Cluster {i}", s=15, alpha=0.5)
    ax.set_xlabel(f"PC 1 ({explained[0]:.1%})"); ax.set_ylabel(f"PC 2 ({explained[1]:.1%})")
    ax.set_title("(b) Cluster Density in PC Space"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0,2]
    pca_full = PCA().fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    nc = np.arange(1, len(cum_var)+1)
    ax.bar(nc[:10], pca_full.explained_variance_ratio_[:10], color=C["train"], alpha=0.75, label="Individual")
    ax.plot(nc[:10], cum_var[:10], "o-", color=C["pred"], linewidth=2, markersize=6, label="Cumulative")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.7, label="95% threshold")
    ax.set_xlabel("Number of Components"); ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("(c) PCA Explained Variance"); ax.legend(fontsize=9)
    ax.set_xticks(nc[:10]); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    for i in range(n_clusters):
        cent = X_pca[labels==i].mean(axis=0)
        ax.scatter(cent[0], cent[1], color=plt.cm.tab10(i), s=200, marker="X", edgecolors="black", linewidth=1, zorder=5)
        ax.annotate(f"C{i}", xy=cent, xytext=(5,5), textcoords="offset points", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({explained[0]:.1%})"); ax.set_ylabel(f"PC 2 ({explained[1]:.1%})")
    ax.set_title("(d) Cluster Centroids"); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    fnames = [f"H{h}" for h in range(24)] + ["Mean","Std","P5"]
    top_f = np.argsort(np.abs(pca.components_[0]))[::-1][:12]
    ax.barh(np.arange(12), pca.components_[0, top_f], color=C["train"], alpha=0.8)
    ax.set_yticks(np.arange(12)); ax.set_yticklabels([fnames[i] for i in top_f])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("PC 1 Loading"); ax.set_title("(e) Top PC 1 Feature Loadings")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1,2]
    from sklearn.metrics import pairwise_distances
    cents = np.array([X[labels==i].mean(axis=0) for i in range(n_clusters)])
    dm = pairwise_distances(cents)
    im = ax.imshow(dm, cmap="YlOrRd", interpolation="nearest")
    for i in range(n_clusters):
        for j in range(n_clusters):
            c = "black" if dm[i,j] < dm.max()/2 else "white"
            ax.text(j, i, f"{dm[i,j]:.1f}", ha="center", va="center", fontsize=8, color=c)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Cluster")
    ax.set_title("(f) Inter-Cluster Distance Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Euclidean Distance")

    fig.suptitle("PCA-Based Visualization of Traffic Sensor Clusters", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cluster_pca.png"), bbox_inches="tight")
    plt.close(fig)
    print("[generate] Saved cluster_pca.png")

def plot_model_comparison(ds_full, ps_2d, ys_2d, cfg, save_dir):
    """Compare DCRNN with baselines (HA, LSTM) at multiple horizons.
    Builds everything from scratch with fewer sensors."""
    n_sub = min(30, ds_full.num_sensors)
    raw = ds_full.speed[:, :n_sub]
    tod_broadcast = np.broadcast_to(ds_full.time_of_day[:,None,None], (ds_full.num_steps, n_sub, 1)).astype(np.float32)
    features = np.concatenate([raw[..., None], tod_broadcast], axis=-1)
    tr, va, te = train_val_test_split(features, (0.7, 0.1, 0.2))
    Ttr = tr.shape[0]
    ha = HistoricalAverage(period=24*12).fit(raw[:Ttr])

    sc = fit_scaler(tr)
    trs = tr.copy(); trs[...,0] = sc.transform(tr[...,0])
    tes = te.copy(); tes[...,0] = sc.transform(te[...,0])
    xt, yt = make_sequences(trs, cfg["data"]["seq_len"], cfg["data"]["horizon"])
    xte, yte = make_sequences(tes, cfg["data"]["seq_len"], cfg["data"]["horizon"])
    n_windows_sub = xte.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm = LSTMBaseline(cfg["model"]["input_dim"], cfg["model"]["output_dim"], 32, cfg["data"]["horizon"]).to(device)
    opt = torch.optim.Adam(lstm.parameters(), lr=0.001)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.from_numpy(xt), torch.from_numpy(yt)), batch_size=128, shuffle=True)
    for _ in range(1):
        lstm.train()
        for xb, yb in loader:
            opt.zero_grad(); pr = lstm(xb.to(device))
            torch.abs(pr - yb.to(device)).mean().backward(); opt.step()
    lstm.eval()
    with torch.no_grad():
        pl = lstm(torch.from_numpy(xte).to(device)).cpu().numpy()
    pls = sc.inverse_transform(pl[...,0]); yls = sc.inverse_transform(yte[...,0])
    preds_ha = ha.predict(start_step=Ttr+cfg["data"]["seq_len"], horizon=cfg["data"]["horizon"], num_windows=n_windows_sub)[...,0]

    horizons = [1,3,6,9,12]
    def met(p,t):
        r = horizon_report(p[...,None], t[...,None], horizons=horizons)
        return {h: r[h] for h in horizons}
    # Use the first n_windows_sub windows of the 2D DCRNN data (the last horizon step)
    dm = met(ps_2d[:n_windows_sub, :n_sub, None, None], ys_2d[:n_windows_sub, :n_sub, None, None])
    hm = met(preds_ha[...,None], yls[...,None])
    lm = met(pls[...,None], yls[...,None])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    x = np.arange(len(horizons)); w = 0.25

    for idx, (metric, ylab, title) in enumerate([("mae","MAE (mph)","(a) MAE"), ("rmse","RMSE (mph)","(b) RMSE"), ("mape","MAPE (%)","(c) MAPE")]):
        ax = axes[0, idx]
        md = [dm[h][metric] for h in horizons]; mh = [hm[h][metric] for h in horizons]; ml = [lm[h][metric] for h in horizons]
        ax.bar(x-w, mh, w, color=C["ha"], alpha=0.85, label="HA", hatch="//")
        ax.bar(x, ml, w, color=C["lstm"], alpha=0.85, label="LSTM", hatch="..")
        ax.bar(x+w, md, w, color=C["dcrnn"], alpha=0.85, label="DCRNN")
        ax.set_xticks(x); ax.set_xticklabels([f"{h}\\n({h*5}m)" for h in horizons])
        ax.set_ylabel(ylab); ax.set_title(title + " Comparison Across Horizons")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1,0]; ax.axis("off")
    col_lab = ["Horizon","HA MAE","LSTM MAE","DCRNN MAE","vs HA"]
    cell_text = [[f"{h} ({h*5}m)", f"{hm[h]['mae']:.2f}", f"{lm[h]['mae']:.2f}", f"{dm[h]['mae']:.2f}", f"{(hm[h]['mae']-dm[h]['mae'])/hm[h]['mae']*100:+.1f}%"] for h in horizons]
    tbl = ax.table(cellText=cell_text, colLabels=col_lab, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.1, 1.5)
    ax.set_title("(d) Performance Summary", fontweight="bold")

    ax = axes[1,1]
    for i, (k, c) in enumerate([("mae",C["mae"]),("rmse",C["rmse"]),("mape",C["mape"])]):
        vals = [(hm[h][k]-dm[h][k])/hm[h][k]*100 for h in horizons]
        ax.bar(x+i*w-w, vals, w, color=c, alpha=0.85, label=k.upper())
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{h}\\n({h*5}m)" for h in horizons])
    ax.set_ylabel("Improvement vs HA (%)"); ax.set_title("(e) DCRNN Improvement")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1,2]; si = 0; nsp = min(100, n_windows_sub)
    ax.plot(np.arange(nsp), yls[:nsp,si], linewidth=1.5, color=C["act"], label="Actual")
    ax.plot(np.arange(nsp), preds_ha[:nsp,si], linewidth=1, color=C["ha"], label="HA", linestyle="--")
    ax.plot(np.arange(nsp), pls[:nsp,si], linewidth=1, color=C["lstm"], label="LSTM", linestyle="-.")
    ax.plot(np.arange(nsp), ps_2d[:nsp, si], linewidth=1.5, color=C["dcrnn"], label="DCRNN", linestyle=":")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Speed (mph)")
    ax.set_title(f"(f) Model Predictions (Sensor {si})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Model Performance Comparison: DCRNN vs Baselines", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "model_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("[generate] Saved model_comparison.png")

def plot_residual_analysis(preds_speed, y_speed, save_dir):
    """Comprehensive residual analysis with statistical tests."""
    from scipy import stats
    residuals = preds_speed.flatten() - y_speed.flatten()
    mask = y_speed.flatten() > 0; residuals = residuals[mask]
    flat_pred = preds_speed.flatten()[mask]; flat_true = y_speed.flatten()[mask]
    abs_res = np.abs(residuals); mean_r = residuals.mean(); std_r = residuals.std()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0,0]
    ax.hist(residuals, bins=120, color=C["pred"], alpha=0.75, edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(mean_r, color="blue", linestyle="-", linewidth=2, label=f"Mean = {mean_r:.2f}")
    ax.axvline(mean_r-std_r, color="blue", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.axvline(mean_r+std_r, color="blue", linestyle=":", linewidth=1.5, alpha=0.5)
    xn = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xn, stats.norm.pdf(xn, mean_r, std_r), "r-", linewidth=2, label="Normal fit")
    _, pv = stats.normaltest(residuals[:min(10000,len(residuals))])
    ax.set_xlabel("Prediction Error (mph)"); ax.set_ylabel("Density")
    ax.set_title(f"(a) Residual Distribution (p={pv:.2e})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    sample = residuals[::max(1,len(residuals)//5000)]
    stats.probplot(sample, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(C["pred"]); ax.get_lines()[0].set_markeredgecolor(C["pred"])
    ax.get_lines()[0].set_markersize(2); ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[1].set_color("black")
    ax.set_title("(b) Q-Q Plot vs Normal Distribution"); ax.grid(True, alpha=0.3)

    ax = axes[0,2]
    ax.hist(abs_res, bins=100, color=C["mae"], alpha=0.75, edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(abs_res.mean(), color="red", linestyle="-", linewidth=2, label=f"Mean = {abs_res.mean():.2f}")
    ax.axvline(np.median(abs_res), color="orange", linestyle="--", linewidth=2, label=f"Median = {np.median(abs_res):.2f}")
    p90 = np.percentile(abs_res, 90)
    ax.axvline(p90, color="purple", linestyle=":", linewidth=2, label=f"P90 = {p90:.2f}")
    ax.set_xlabel("Absolute Error (mph)"); ax.set_ylabel("Density")
    ax.set_title("(c) Absolute Error Distribution"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    sidx = np.random.choice(len(residuals), min(10000,len(residuals)), replace=False)
    ax.scatter(flat_pred[sidx], residuals[sidx], s=1, alpha=0.3, color=C["train"])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Speed (mph)"); ax.set_ylabel("Residual (mph)")
    ax.set_title("(d) Residuals vs Predicted"); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    ax.scatter(flat_true[sidx], residuals[sidx], s=1, alpha=0.3, color=C["pred"])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual Speed (mph)"); ax.set_ylabel("Residual (mph)")
    ax.set_title("(e) Residuals vs Actual"); ax.grid(True, alpha=0.3)

    ax = axes[1,2]
    s_idx = np.argsort(flat_true)
    cum = np.cumsum(residuals[s_idx]) / (np.arange(len(residuals)) + 1)
    ax.plot(cum[::100], linewidth=1.5, color=C["mape"])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Sorted Observation Index (x100)"); ax.set_ylabel("Cumulative Mean Residual")
    ax.set_title("(f) Cumulative Mean Residual (Bias Trend)"); ax.grid(True, alpha=0.3)

    stxt = (f"Residual Statistics:\n  Mean = {mean_r:.3f}\n  Std = {std_r:.3f}\n"
            f"  RMSE = {np.sqrt((residuals**2).mean()):.3f}\n  MAE = {abs_res.mean():.3f}\n"
            f"  MAPE = {(abs_res/flat_true*100).mean():.2f}%\n"
            f"  Skew = {stats.skew(residuals[:10000]):.3f}\n"
            f"  Kurt = {stats.kurtosis(residuals[:10000]):.3f}")
    fig.text(0.91, 0.5, stxt, fontsize=9, fontfamily="monospace", verticalalignment="center",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle("Residual Analysis of DCRNN Traffic Speed Predictions", fontsize=16, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(os.path.join(save_dir, "residual_analysis.png"), bbox_inches="tight")
    plt.close(fig)
    print("[generate] Saved residual_analysis.png")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive report figures")
    parser.add_argument("--config", default="configs/dcrnn_metr_la_cpu.yaml")
    parser.add_argument("--checkpoint", default="ckpt/best.pt"); parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--num-sensors", type=int, default=207)
    args = parser.parse_args()

    ensure(args.output_dir); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] Device: {device}")

    with open(args.config) as f: cfg = yaml.safe_load(f)

    print("[generate] Loading data and model ...")
    ds, ps, ys, adj, sc, te, tr, va = load_all(cfg, args.checkpoint, device, ns=args.num_sensors)
    # ps shape: (W, H, N) = (windows, horizon, sensors)
    # Use last horizon step for 2D plots
    ps_2d = ps[:, -1, :]
    ys_2d = ys[:, -1, :]
    print(f"[generate] Test set: {ps.shape[0]} windows, {ps.shape[2]} sensors, horizon={ps.shape[1]}")

    print("\n[generate] === Figure 1: Univariate Analysis ===")
    plot_univariate_analysis(ds, args.output_dir)

    print("\n[generate] === Figure 2: Bivariate Analysis ===")
    plot_bivariate_analysis(ds, args.output_dir)

    print("\n[generate] === Figure 3: Elbow & Silhouette ===")
    X, labels = plot_elbow_silhouette(ds, args.output_dir)

    print("\n[generate] === Figure 4: Cluster PCA ===")
    plot_cluster_pca(X, labels, args.output_dir)

    print("\n[generate] === Figure 5: Model Comparison ===")
    plot_model_comparison(ds, ps_2d, ys_2d, cfg, args.output_dir)

    print("\n[generate] === Figure 6: Residual Analysis ===")
    plot_residual_analysis(ps_2d, ys_2d, args.output_dir)

    print(f"\n[generate] All figures saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
