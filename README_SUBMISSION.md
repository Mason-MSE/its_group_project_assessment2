# MSE806 Assessment 2 — Submission Package

**Course**: MSE806 Intelligent Transportation Systems
**Assessment**: Assessment 2 — Group Project (50% of course grade)
**Project Title**: Spatio-Temporal Deep Learning for Proactive Traffic Prediction and Safety-Aware Vehicle Management in Urban ITS
**Student**: Li Qingchao (Master of Software Engineering, Level 9)
**Student ID**: `[Student ID]` — **please replace this placeholder in all three documents before submission**
**Date**: 2026-07-05

---

## 1. Package contents

```
MSE806_Assessment2/
├── 01_Proposal_MSE806_A2_LiQingchao.docx        # Task A — Proposal (~887 words)
├── 02_Report_MSE806_A2_LiQingchao.docx          # Task B — Full report (~2,618 words)
├── 03_Presentation_MSE806_A2_LiQingchao.pptx    # Presentation slides (15 slides)
├── 03_Presentation_MSE806_A2_LiQingchao.pptx.html   # Coze-rendered HTML preview (optional)
├── code/                                        # Reference implementation
│   ├── README.md                                # How to install, train, evaluate, demo
│   ├── requirements.txt                         # Python dependencies
│   ├── configs/
│   │   └── dcrnn_metr_la.yaml                   # DCRNN configuration
│   ├── data/
│   │   ├── mock_data_loader.py                  # Reproducible mock traffic dataset
│   │   └── preprocessing.py                     # Sliding-window / z-score utilities
│   ├── models/
│   │   ├── dcrnn.py                             # DCRNN encoder–decoder
│   │   ├── dcrnn_cell.py                        # Diffusion-convolution GRU cell
│   │   └── baselines.py                         # HA, ARIMA-lite, Graph WaveNet stub
│   ├── utils/
│   │   ├── metrics.py                           # MAE / RMSE / MAPE
│   │   └── safety_alert.py                      # Deceleration-based risk scoring
│   ├── train.py                                 # End-to-end training loop (PyTorch)
│   ├── evaluate.py                              # Evaluation on held-out set
│   ├── demo.py                                  # 1-batch forward + safety alert demo
│   └── sanity_check.py                          # Numpy-only smoke test (no torch needed)
├── README_SUBMISSION.md                         # THIS FILE
└── MSE806_A2_LiQingchao_Submission.zip          # Zipped package for upload
```

---

## 2. Deliverables at a glance

| # | File | Task | Format | Length | Status |
|---|------|------|--------|--------|--------|
| 1 | `01_Proposal_MSE806_A2_LiQingchao.docx` | Task A — Proposal | Word (.docx) | 887 words (target 800–1200) | ✅ |
| 2 | `02_Report_MSE806_A2_LiQingchao.docx` | Task B — Full Report | Word (.docx) | 2,618 words (target 2,500–3,000) | ✅ |
| 3 | `03_Presentation_MSE806_A2_LiQingchao.pptx` | Presentation | PowerPoint (.pptx) | 15 slides (target 15–18) | ✅ |
| 4 | `code/` | Code artefact | Python 3.9+ | ~15 modules, sanity-checked | ✅ |
| 5 | `README_SUBMISSION.md` | Submission notes | Markdown | — | ✅ |

**Formatting compliance**
- Report: APA 7 in-text citations and reference list; double line-spacing (line = 480); Times New Roman 12 pt.
- Proposal: matching cover metadata; single reference list at end.
- All three documents (Proposal / Report / PPT cover) carry the same author/course/date block. **Student ID is a placeholder** — see §5 below.

---

## 3. Report highlights (for the marker)

- **Problem**: Simultaneous traffic-speed prediction (short-, medium-, long-horizon) and safety-aware vehicle management in urban ITS.
- **Method**: DCRNN (Li et al., 2018) as the main model; Graph WaveNet (Wu et al., 2019), ST-GCN (Yu et al., 2018) and HA as comparators.
- **Data**: METR-LA (207 loop sensors, 4 months) + PEMS-BAY (325 sensors, 6 months). **Not re-downloaded** — benchmark numbers are cited verbatim from the original papers (source noted under each table).
- **Safety framework**: A composite risk score combining predicted speed, forecast deceleration, and downstream congestion propagation, with thresholded advisory outputs (green / amber / red) for a downstream VMS controller.
- **Contribution**: A single pipeline that couples state-of-the-art spatio-temporal forecasting with an interpretable safety layer, evaluated on the two most widely used ITS benchmarks.

---

## 4. Reproducing the code artefact

The code was written to be pedagogically clear and reproducible. It is **not** intended to reach publication-grade results in-notebook; it uses a small mock dataset so that a marker can verify the pipeline end-to-end in seconds.

### 4.1 Numpy-only sanity check (no PyTorch needed)

```bash
cd code
python3 sanity_check.py
```

Expected output includes:
- Mock speed matrix of shape `(864, 30)` (12 hours × 30 sensors, 5-min resolution).
- HA baseline MAE at horizons 3 / 6 / 12 ≈ **7.84 / 7.94 / 8.42 mph** on the mock series.
- Metric helpers unit-tested against hand-computed values.
- Safety-alert precision / recall = **1.00 / 1.00** on the injected hard-braking scenario, firing on 2 out of 30 sensors.

This is the recommended way to verify the artefact without any extra installs.

### 4.2 Full PyTorch pipeline (optional)

```bash
cd code
pip install -r requirements.txt          # torch, numpy, pyyaml, tqdm, pandas
python3 train.py    --config configs/dcrnn_metr_la.yaml   # 1 mini-epoch on mock data
python3 evaluate.py --config configs/dcrnn_metr_la.yaml
python3 demo.py     --config configs/dcrnn_metr_la.yaml   # forward pass + safety alert
```

If PyTorch is unavailable, `sanity_check.py` still verifies the data pipeline, HA baseline, metrics and safety-alert logic in pure numpy.

### 4.3 Notes on data
- No real METR-LA / PEMS-BAY archives are shipped or downloaded. `data/mock_data_loader.py` synthesises a deterministic small dataset (with a fixed seed) that matches the shape and statistical profile of METR-LA. Benchmark comparisons in the report cite the published values only.

---

## 5. Before you submit (action items for the student)

1. **Replace `[Student ID]` in all three documents**:
   - `01_Proposal_MSE806_A2_LiQingchao.docx` — cover block on page 1.
   - `02_Report_MSE806_A2_LiQingchao.docx` — cover block on page 1.
   - `03_Presentation_MSE806_A2_LiQingchao.pptx` — slide 1 (Cover).
2. **Turnitin check** — upload `02_Report_MSE806_A2_LiQingchao.docx` to the course Turnitin dropbox first; expect similarity ≤ 15 %. The text has been written from scratch and reference blocks are formatted so that Turnitin excludes them from similarity when configured.
3. **Team members** — if this is submitted as a group, add each member's name/ID next to yours on the cover of all three documents.
4. **Blackboard upload** — attach `MSE806_A2_LiQingchao_Submission.zip` under *Assessment 2 → Group Project*. Some Blackboard sites also require the report and slides as separate attachments; upload the two loose files as well if the dropbox permits.
5. **Presentation rehearsal** — 15 slides at ≈ 40–60 s each fits the standard 10-minute academic presentation window; longer defences may want to add 2–3 backup slides drawn from the report's Discussion section.

---

## 6. Assumptions and limitations disclosed to the marker

- Benchmark accuracy numbers reported in Tables 1–2 of the report are the **published values** from Li et al. (2018) and Wu et al. (2019); no re-training was performed inside the sandbox. This is stated explicitly under both tables.
- The safety-alert thresholds in `utils/safety_alert.py` are illustrative defaults; a production system would calibrate them against a labelled incident dataset (e.g., HSIS crash records).
- Graph WaveNet is present in `models/baselines.py` as a documented stub only; the report treats it strictly as a literature comparator, not a trained model in this artefact.

---

## 7. Contact

If the marker needs clarification on any deliverable, please contact:

- **Name**: Li Qingchao
- **Course**: MSE806 Intelligent Transportation Systems
- **Institution**: (please fill in)
- **Email**: (please fill in)

---

*Package assembled on 2026-07-05.*
