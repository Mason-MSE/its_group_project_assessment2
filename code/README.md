# Traffic Forecasting Reference Implementation (MSE806 A2)

This directory contains a reference implementation for the MSE806 Assessment 2
project *"Spatio-Temporal Deep Learning for Proactive Traffic Prediction and
Safety-Aware Vehicle Management in Urban ITS"*. The predictive core is the
**Diffusion Convolutional Recurrent Neural Network (DCRNN)** proposed by Li,
Yu, Shahabi and Liu (2018), with an **LSTM** and **Historical Average**
baseline for comparison. A **rule-based safety alert layer** on top of the
forecast turns predicted speed drops into congestion / incident risk scores.

> **Data policy.** No real METR-LA / PEMS-BAY data is bundled with this
> repository. `data/mock_data_loader.py` synthesises tensors of the same
> shape and statistical profile as METR-LA so that the whole pipeline can
> be exercised end to end. For an actual research run please follow the
> official instructions at
> <https://github.com/liyaguang/DCRNN> to download the METR-LA and
> PEMS-BAY tensors and drop them under `data/raw/`.

---

## 1. Directory layout

```
code/
├── README.md                  ← this file
├── requirements.txt           ← Python dependencies
├── configs/
│   └── dcrnn_metr_la.yaml     ← Hyper-parameters and dataset settings
├── data/
│   ├── __init__.py
│   ├── mock_data_loader.py    ← Synthetic (T,N,F) tensors mimicking METR-LA
│   └── preprocessing.py       ← Z-score scaler, adjacency-matrix builder
├── models/
│   ├── __init__.py
│   ├── dcrnn.py               ← Encoder–decoder DCRNN
│   ├── dcrnn_cell.py          ← DCGRU cell with bidirectional diffusion conv
│   └── baselines.py           ← LSTM baseline, Historical Average
├── utils/
│   ├── __init__.py
│   ├── metrics.py             ← Masked MAE / RMSE / MAPE
│   └── safety_alert.py        ← Congestion / risk index and alert logic
├── train.py                   ← Training entry point (PyTorch)
├── evaluate.py                ← Evaluation on a hold-out split
├── demo.py                    ← End-to-end demo (PyTorch, 2 epochs)
└── sanity_check.py            ← Pure-NumPy pipeline check (no PyTorch)
```

---

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`sanity_check.py` only requires `numpy`, `scipy`, `pandas`, `pyyaml`, `tqdm`
and can therefore run in restricted sandboxes where PyTorch is not
available.

---

## 3. Running the code

### 3.1 Quick sanity check (no PyTorch required)

```bash
python sanity_check.py
```

This exercises the mock loader, adjacency-matrix builder, Historical
Average baseline, MAE/RMSE/MAPE metrics and the safety-alert layer, and
prints a short diagnostic. It typically finishes in a few seconds.

### 3.2 Full DCRNN demo (PyTorch, 2 epochs on mock data)

```bash
python demo.py
```

The demo builds a small DCRNN, trains it for two epochs on synthetic
data, evaluates on a hold-out split at 15/30/60-minute horizons and
prints an example congestion alert.

### 3.3 Training on real data

Download METR-LA / PEMS-BAY tensors from the DCRNN reference repository
and place them under `data/raw/`. Then edit
`configs/dcrnn_metr_la.yaml` to point at the real paths and run:

```bash
python train.py --config configs/dcrnn_metr_la.yaml
python evaluate.py --config configs/dcrnn_metr_la.yaml --checkpoint ckpt/best.pt
```

---

## 4. Key references

* Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). *Diffusion convolutional
  recurrent neural network: Data-driven traffic forecasting.* ICLR.
* Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph
  WaveNet for deep spatial-temporal graph modeling.* IJCAI.

Data-set landing pages:

* METR-LA and PEMS-BAY: <https://github.com/liyaguang/DCRNN>
* PEMS raw feeds: <https://pems.dot.ca.gov/>

---

## 5. Notes for reviewers

* No real training results are reported in the accompanying report.
  Benchmark numbers are taken directly from Li et al. (2018) and Wu et
  al. (2019). See `../02_Report_MSE806_A2_LiQingchao.docx` for the full
  discussion.
* Every module carries a module-level docstring explaining the maths
  and design choices.
* Random seeds are fixed inside `sanity_check.py` and `demo.py` for
  reproducibility.
