# MSE806 Assessment 2 — Submission Package

**Course**: MSE806 Intelligent Transportation Systems
**Assessment**: Assessment 2 — Group Project (50% of course grade)
**Project Title**: Spatio-Temporal Deep Learning for Proactive Traffic Prediction and Safety-Aware Vehicle Management in Urban ITS
**Student**: Li Qingchao (Master of Software Engineering, Level 9)
**Student ID**: `[Student ID]` — **please replace this placeholder in the report and PPT before submission**
**Date**: 2026-07-05

> **Update (unified submission)**: Task A (Proposal, LO3) has been **merged into** the main Report (Task B, LO4). There is now **only one written deliverable** (`02_Report_...docx`), which internally covers Objectives, Technological Innovation and Ethical Considerations previously placed in the standalone Proposal. See §1 and §2 below.

---

## 1. Package contents

```
MSE806_Assessment2/
├── 02_Report_MSE806_A2_LiQingchao.docx                   # Unified report (Task A + Task B, ~2,907 words)
├── 03_Presentation_MSE806_A2_LiQingchao.pptx             # Presentation slides (15–17 slides)
├── 03_Presentation_MSE806_A2_LiQingchao.pptx.html        # Coze-rendered HTML preview (optional)
├── code/                                                 # Reference implementation
│   ├── README.md                                         # How to install, train, evaluate, demo
│   ├── requirements.txt                                  # Python dependencies
│   ├── configs/
│   │   ├── dcrnn_metr_la.yaml                            # DCRNN configuration (GPU)
│   │   ├── dcrnn_metr_la_cpu.yaml                        # DCRNN configuration (CPU)
│   │   └── dcrnn_pems_bay.yaml                           # PEMS-BAY configuration
│   ├── data/
│   │   ├── mock_data_loader.py                           # Reproducible mock traffic dataset
│   │   ├── real_data_loader.py                           # Real dataset loader (METR-LA / PEMS-BAY)
│   │   └── preprocessing.py                              # Sliding-window / z-score utilities
│   ├── models/
│   │   ├── dcrnn.py                                      # DCRNN encoder–decoder
│   │   ├── dcrnn_cell.py                                 # Diffusion-convolution GRU cell
│   │   └── baselines.py                                  # HA, LSTM baseline
│   ├── utils/
│   │   ├── metrics.py                                    # MAE / RMSE / MAPE
│   │   └── safety_alert.py                               # Deceleration-based risk scoring
│   ├── train.py                                          # End-to-end training loop (PyTorch)
│   ├── evaluate.py                                       # Evaluation on held-out set
│   ├── demo.py                                           # 1-batch forward + safety alert demo
│   ├── sanity_check.py                                   # Numpy-only smoke test (no torch needed)
│   ├── plot_results.py                                   # Plot basic report figures (fig1–fig7)
│   └── generate_report_figures.py                        # Plot detailed report figures (6 comprehensive charts)
├── figures/                                              # Generated report figures
│   ├── fig1_training_curve.png                           # Training & validation loss curve
│   ├── fig2_prediction_timeseries.png                    # Predicted vs actual speed time series
│   ├── fig3_horizon_metrics.png                          # Prediction accuracy by horizon
│   ├── fig4_scatter_plot.png                             # Predicted vs actual scatter plot
│   ├── fig5_residual_distribution.png                    # Residual distribution
│   ├── fig6_risk_heatmap.png                             # Spatial-temporal risk heatmap
│   ├── fig7_speed_distribution.png                       # Overall speed distribution per sensor
│   ├── univariate_analysis.png                           # Univariate analysis (6-subplot)
│   ├── bivariate_analysis.png                            # Bivariate & spatial analysis (6-subplot)
│   ├── elbow_silhouette.png                              # K-Means clustering: elbow & silhouette (6-subplot)
│   ├── cluster_pca.png                                   # PCA clustering visualization (6-subplot)
│   ├── model_comparison.png                              # DCRNN vs HA/LSTM model comparison (6-subplot)
│   └── residual_analysis.png                             # Extended residual analysis with stats (6-subplot)
├── README_SUBMISSION.md                                  # THIS FILE
├── _intermediate_bak/                                    # Backup working files (NOT in zip)
│   └── 01_Proposal_MSE806_A2_LiQingchao.docx             # Standalone Proposal (kept as fallback)
└── MSE806_A2_LiQingchao_Submission.zip                   # Zipped package for upload
```

**Note**: The standalone `01_Proposal_...docx` has been moved to `_intermediate_bak/` and is **not** included in the submission zip. If the marker later requires two separate documents, retrieve it from there — the content has already been re-woven into the main report.

---

## 2. Deliverables at a glance

| # | File | Task Coverage | Format | Length | Status |
|---|------|---------------|--------|--------|--------|
| 1 | `02_Report_MSE806_A2_LiQingchao.docx` | **Task A + Task B** (Proposal + Full Report, LO3 + LO4) | Word (.docx) | ~2,907 words (target 2,500–3,000) | ✅ |
| 2 | `03_Presentation_MSE806_A2_LiQingchao.pptx` | Presentation | PowerPoint (.pptx) | 15–17 slides | ✅ |
| 3 | `code/` | Code artefact | Python 3.9+ | ~15 modules, sanity-checked | ✅ |
| 4 | `README_SUBMISSION.md` | Submission notes | Markdown | — | ✅ |

**Formatting compliance**
- Report: APA 7 in-text citations and reference list; double line-spacing; Times New Roman 12 pt; 15 references.
- Cover page carries a subtitle line confirming the unified Task A + Task B scope.
- **Student ID is a placeholder** — see §5 below.

---

## 3. How Task A (Proposal) is covered inside the Report

The merged report keeps the Task B structure and folds the four Proposal elements into these locations:

| Proposal element (Task A / LO3) | Where it now lives in the Report |
|---|---|
| Project Title | Cover page + Section 1 (Introduction) |
| Objectives | Section 1.1 *Project Objectives* — five measurable objectives O1–O5 |
| Technological Innovation | Section 1 (statement) + Section 3.0 *Technological Innovation Overview* (four-dimension framing) |
| Ethical Considerations | Section 6 *Ethical Considerations* — Privacy, Algorithmic Fairness, Safety Responsibility, Data Governance & Transparency |

Section numbering: 1 Introduction → 2 Literature Review → 3 Methodology (3.0–3.6) → 4 Results & Analysis → 5 Discussion → 6 Ethical Considerations → 7 Conclusion & Self-Reflection.

---

## 4. Report highlights (for the marker)

- **Problem**: Simultaneous traffic-speed prediction (short-, medium-, long-horizon) and safety-aware vehicle management in urban ITS.
- **Method**: DCRNN (Li et al., 2018) as the predictive core; Graph WaveNet (Wu et al., 2019) as the ablation baseline; ST-GCN (Yu et al., 2018), LSTM, ARIMA and Historical Average as comparators.
- **Data**: METR-LA (207 loop sensors) + PEMS-BAY (325 sensors). **Not re-downloaded** — benchmark numbers are cited verbatim from the original papers (source noted under each table). A mock data loader (`data/mock_data_loader.py`) generates synthetic data matching METR-LA shapes for reproducible execution.
- **Figures**: 13 report-quality figures are pre-generated in `figures/` via two plotting scripts — `plot_results.py` (basic 7 figures) and `generate_report_figures.py` (detailed 6 figures, each a 2×3 subplot composite at 300 DPI).
- **Safety framework**: A composite risk score combining predicted speed, forecast variability and neighbour coherence, producing advisory outputs 10–15 minutes ahead of an observed collapse.
- **Ethics**: Privacy-by-design under GDPR; per-sub-region fairness monitoring; calibrated alerts with human-in-the-loop authority; audit-log-backed transparency (IEEE, 2019).
- **Contribution**: A single unified pipeline that couples SOTA spatio-temporal forecasting with an interpretable safety layer, sitting on the two most widely used ITS benchmarks.

---

## 5. Reproducing the code artefact

The code was written to be pedagogically clear and reproducible. It uses a small mock dataset so that a marker can verify the pipeline end-to-end in seconds.

### 5.1 Numpy-only sanity check (no PyTorch needed)

```bash
cd code
python3 sanity_check.py
```

Expected output includes:
- Mock speed matrix of shape `(864, 30)` (12 hours × 30 sensors, 5-min resolution).
- HA baseline MAE at horizons 3 / 6 / 12 ≈ **7.84 / 7.94 / 8.42 mph** on the mock series.
- Metric helpers unit-tested against hand-computed values.
- Safety-alert precision / recall = **1.00 / 1.00** on the injected hard-braking scenario, firing on 2 out of 30 sensors.

### 5.2 Full PyTorch pipeline (optional)

```bash
cd code
pip install -r requirements.txt          # torch, numpy, pyyaml, tqdm, pandas
python3 train.py    --config configs/dcrnn_metr_la.yaml   # 1 mini-epoch on mock data
python3 evaluate.py --config configs/dcrnn_metr_la.yaml
python3 demo.py     --config configs/dcrnn_metr_la.yaml   # forward pass + safety alert
```

### 5.3 Generating report figures

Two scripts are provided for producing report-quality figures:

```bash
cd code

# Basic figures (training curve, time series, horizon metrics, scatter, residuals,
# risk heatmap, speed distribution) — saves to ../figures/fig1_*.png ... fig7_*.png
python3 plot_results.py --config configs/dcrnn_metr_la_cpu.yaml \
                        --checkpoint ckpt/best.pt --output-dir ../figures \
                        --log-csv train_log.csv

# Detailed figures (univariate analysis, bivariate analysis, elbow & silhouette,
# cluster PCA, model comparison, extended residual analysis) — each a 6-subplot
# composition at 300 DPI
python3 generate_report_figures.py --config configs/dcrnn_metr_la_cpu.yaml \
                                   --checkpoint ckpt/best.pt --output-dir ../figures
```

### 5.4 Notes on data
- No real METR-LA / PEMS-BAY archives are shipped or downloaded. `data/mock_data_loader.py` synthesises a deterministic small dataset that matches the shape of METR-LA. Benchmark comparisons in the report cite the published values only.

---

## 6. Before you submit (action items for the student)

1. **Replace `[Student ID]` in the report and PPT**:
   - `02_Report_MSE806_A2_LiQingchao.docx` — cover block on page 1.
   - `03_Presentation_MSE806_A2_LiQingchao.pptx` — slide 1 (Cover).
2. **Turnitin check** — upload `02_Report_MSE806_A2_LiQingchao.docx` to the course Turnitin dropbox first; expect similarity ≤ 15 %. The text has been written from scratch and reference blocks are formatted so that Turnitin excludes them from similarity when configured.
3. **Team members** — if this is submitted as a group, add each member's name/ID next to yours on the cover of both the report and the slides.
4. **Blackboard upload** — attach `MSE806_A2_LiQingchao_Submission.zip` under *Assessment 2 → Group Project*. Some Blackboard sites also require the report and slides as separate attachments; upload the two loose files as well if the dropbox permits. **Only three artefacts need to be uploaded: the Report, the PPT and the code zip.**
5. **If the marker requests separate Task A / Task B files**: the standalone Proposal is preserved at `_intermediate_bak/01_Proposal_MSE806_A2_LiQingchao.docx`. You can rename and submit it alongside the report without any content clash — the two versions carry consistent facts, objectives and references.
6. **Presentation rehearsal** — 15–17 slides at ≈ 40–60 s each fits the standard 10-minute academic presentation window; longer defences may want to add 2–3 backup slides drawn from the report's Discussion section.

---

## 7. Assumptions and limitations disclosed to the marker

- Benchmark accuracy numbers reported in Tables 1–2 of the report are the **published values** from Li et al. (2018) and Wu et al. (2019); no re-training was performed inside the sandbox. This is stated explicitly under both tables.
- The safety-alert thresholds in `utils/safety_alert.py` are illustrative defaults; a production system would calibrate them against a labelled incident dataset (e.g., HSIS crash records).
- Graph WaveNet is present in `models/baselines.py` as a documented stub only; the report treats it strictly as a literature comparator, not a trained model in this artefact.

---

## 8. Contact

If the marker needs clarification on any deliverable, please contact:

- **Name**: Li Qingchao
- **Course**: MSE806 Intelligent Transportation Systems
- **Institution**: (please fill in)
- **Email**: (please fill in)

---

*Package re-assembled on 2026-07-05 (unified Task A + Task B version).*
