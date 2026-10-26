<h1 align="center">🛰️ DriftRadar-Vision</h1>

<p align="center">
  <b>ML-Driven Drift Detection, Auto-Retraining, and Promotion System for Vision Models</b><br>
  <sub>Built with Evidently, Great Expectations, MLflow, Hydra, and GitHub Actions</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square"/>
  <img src="https://img.shields.io/badge/pytorch-2.3+-red.svg?style=flat-square"/>
  <img src="https://img.shields.io/github/actions/workflow/status/dhayanand458/driftradar-vision/ci.yml?label=CI&style=flat-square"/>
  <img src="https://img.shields.io/github/v/release/dhayanand458/driftradar-vision?style=flat-square"/>
  <img src="https://img.shields.io/badge/mlflow-tracking-orange.svg?style=flat-square"/>
  <img src="https://img.shields.io/badge/evidently-0.7.14-green.svg?style=flat-square"/>
</p>

---

### 🌌 Overview

**DriftRadar-Vision** is an advanced, production-ready **drift monitoring and adaptive retraining pipeline** for vision models.
It automatically detects **dataset, embedding, and prediction drifts** using Evidently and Great Expectations — then decides, retrains, stages, and promotes models via **GitHub Actions** and **MLflow**.

🚀 Designed for **autonomous ML lifecycle management**, **vision model governance**, and **drift accountability**.

---

### 🧱 Core Architecture

```
               ┌────────────────────────┐
               │    data_prod.py        │
               │   (Simulate batches)   │
               └──────────┬─────────────┘
                          │
               ┌──────────▼─────────────┐
               │    drift_check.py      │
               │ (Evidently + embedding)│
               └──────────┬─────────────┘
                          │ summary.json
               ┌──────────▼─────────────┐
               │    dq_validate.py      │
               │ (Great Expectations)   │
               └──────────┬─────────────┘
                          │
               ┌──────────▼─────────────┐
               │     retrain.py         │
               │  (Policy-based trigger)│
               └──────────┬─────────────┘
                          │
             ┌────────────▼───────────────┐
             │    train.py / eval.py      │
             │  (MLflow + ONNX export)    │
             └────────────┬───────────────┘
                          │
         ┌────────────────▼────────────────┐
         │   drift_cron.yml + retrain_smoke│
         │  (GitHub Actions automation)    │
         └─────────────────────────────────┘
```
---

### 🧩 Features

| Category | Description |
|-----------|-------------|
| 🧠 **Detection** | Dataset drift, embedding drift (PSI), and prediction confidence drift |
| 🔍 **Validation** | Great Expectations quality checks on input batches |
| ⚙️ **Retraining** | Automatic retrain trigger from policy thresholds |
| 🧾 **Reporting** | Evidently + GE HTML reports stored under `artifacts/reports/` |
| 🚀 **Automation** | Full CI/CD loop via GitHub Actions: Drift → PR → Smoke → Promote |
| 🧮 **Backtesting** | Historical drift replay simulation with `backtest.py` |
| 📊 **Dashboard** | Streamlit dashboard with 5 pages for QA, drift, metrics, backtests |
| 🧱 **MLOps** | MLflow registry, Hydra configs, ONNX export, reproducible runs |

---

### ⚙️ Quick Start

#### 1️⃣ Create environment

```
conda create -n driftradar python=3.11
conda activate driftradar
pip install -r requirements.txt
````

#### 2️⃣ Run initial training

```
python -m driftradar_vision.train --config-path configs --config-name model.yaml
```

#### 3️⃣ Evaluate and export ONNX

```
python -m driftradar_vision.eval --config-path configs --config-name eval.yaml --model-config-path configs --model-config-name model.yaml
```

#### 4️⃣ Drift check (Evidently)

```
python -m driftradar_vision.drift_check --config-path configs --config-name drift.yaml --data-config-path configs --data-config-name data.yaml
```

#### 5️⃣ Policy-based retrain

```
python -m driftradar_vision.retrain --config-path configs --config-name policy.yaml
```

#### 6️⃣ Backtest thresholds

```
python -m driftradar_vision.backtest --mode simulate --days 10 --clean-days 3 --drift-days 7
```

#### 7️⃣ Compare metrics (PR body)

```
python -m driftradar_vision.artifacts --what metrics-delta
```

---

### 🧠 Example Decision Output

```
{
  "retrain_triggered": true,
  "reason": "conf_drop(56.1%)>=thr(10.0%)",
  "embedding_top5_psi_mean": 15.29,
  "psi_threshold": 0.2,
  "run_id": "63a1a44c58b14e55b3e8e0b2d56f3bff",
  "staging_version": 7,
  "onnx_staging_path": "artifacts/runs/onnx/staging.onnx"
}
```

---

### 🔄 GitHub Actions CI/CD

| Workflow            | Trigger            | Purpose                        |
| ------------------- | ------------------ | ------------------------------ |
| `ci.yml`            | Push / PR          | Lint, build, pytest            |
| `drift_cron.yml`    | Schedule           | Run drift_check + auto PR      |
| `retrain_smoke.yml` | Pull Request       | Mini retrain + GE + ONNX check |
| `promote.yml`       | `/promote` comment | Promote Staging → Production   |

---

### 📈 Dashboard Pages

| Page              | Description                                         |
| ----------------- | --------------------------------------------------- |
| **Overview**      | Summary of latest production metrics and versions   |
| **Data Quality**  | Great Expectations blockers and warnings            |
| **Drift Monitor** | Evidently HTML + PSI statistics                     |
| **Model Metrics** | MLflow metrics, latency, ECE, size                  |
| **Backtest**      | Historical FPR/FNR, replayed days, threshold tuning |

---

### 🧪 Backtest Result Example

```
{
  "mode": "simulate",
  "n_days": 10,
  "counts": {"tp": 6, "fp": 1, "tn": 3, "fn": 0},
  "rates": {"precision": 0.857, "recall": 1.0, "fpr": 0.25, "fnr": 0.0, "accuracy": 0.9}
}
```

---

### 🧰 Stack

* 🧠 **PyTorch 2.3** — vision model backbone (ResNet-18)
* ⚙️ **Hydra** — configuration management
* 📈 **Evidently** — dataset & embedding drift
* 🧮 **Great Expectations** — data quality validation
* 📊 **MLflow** — experiment tracking & model registry
* 🧰 **ONNX Runtime** — inference validation
* 💾 **GitHub Actions** — CI/CD orchestration
* 🌐 **Streamlit** — dashboard visualization

---

### 🧪 MLflow Registry Example

| Stage            | Version | Accuracy | ECE   | Size (MB) |
| ---------------- | ------- | -------- | ----- | --------- |
| Production       | 6       | 92.8%    | 0.031 | 88.7      |
| Staging          | 7       | 93.1%    | 0.028 | 89.1      |
| Δ (Staging–Prod) | +0.3    | −0.003   | +0.4  |           |

---

### 🧭 Configuration Files

| File                    | Purpose                                |
| ----------------------- | -------------------------------------- |
| `configs/model.yaml`    | Main training hyperparameters          |
| `configs/data.yaml`     | Dataset paths and augmentations        |
| `configs/drift.yaml`    | Evidently and embedding drift settings |
| `configs/policy.yaml`   | Retraining & threshold policy          |
| `configs/eval.yaml`     | Evaluation and ONNX export             |
| `configs/model-ci.yaml` | Fast smoke test variant                |

---

### 🤖 Automated Promotion

Once `drift_cron.yml` detects drift:

1. A **Pull Request** opens → with `needs-review` + reports attached.
2. `retrain_smoke.yml` validates ONNX + GE expectations.
3. Maintainer comments `/promote` to approve.
4. `promote.yml` transitions model to **Production** and updates dashboard.

---

### 📜 Citation

If you use DriftRadar-Vision in your work, please cite:

```
@software{dhayanand2025driftradar,
  author = {Dhayanand},
  title  = {DriftRadar-Vision: Autonomous Drift Detection and Retraining Pipeline for Vision Models},
  year   = {2025},
  url    = {https://github.com/dhayanand458/driftradar-vision}
}
```

---

### 👤 Author

**Dhayanand**  
[🐙 GitHub](https://github.com/dhayanand458) · [💼 LinkedIn](https://www.linkedin.com/in/m-b-dhayanand)

---

### ⭐ Contributing

```
git clone https://github.com/dhayanand458/driftradar-vision
cd driftradar-vision
pip install -r requirements.txt
pytest -v
```

Contributions, ideas, and benchmarks are always welcome — feel free to open a pull request!

---

### 🪪 License

Released under the **MIT License** — see [LICENSE](LICENSE).
