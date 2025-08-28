## **Mini-Prod ML Challenge (3–4 hours)**

**Goal:** Ship a small but production-minded binary classifier on a provided tabular dataset, with training, serving, monitoring, and a light “agentic” check. Keep it simple; correctness \> fanciness.

### **What we provide**

* A starter repo skeleton ([link](https://github.com/gabriel-caffaratti-at-job/zubale-ml-test)).  
* Data files in `data/`: `customer_churn_synth.csv` (main dataset), `churn_ref_sample.csv`, `churn_shifted_sample.csv`, `metrics_history.jsonl`, `drift_latest.json`.

### **Dataset schema (inputs only; target \= `churned`)**

* Categorical: `plan_type {Basic,Standard,Pro}`, `contract_type {Monthly,Annual}`, `autopay {Yes,No}`, `is_promo_user {Yes,No}`  
* Numeric: `add_on_count`, `tenure_months`, `monthly_usage_gb`, `avg_latency_ms`, `support_tickets_30d`, `discount_pct`, `payment_failures_90d`, `downtime_hours_30d`  
* Target: `churned ∈ {0,1}`

### **Your deliverables**

* Reproducible training script that saves model \+ metrics.  
* FastAPI service with `/health` and `/predict`.  
* Minimal drift check CLI.  
* Agentic Monitor CLI (LLM-optional; rules are fine).  
* Basic tests, Dockerfile, CI skeleton, and a 1-page GCP design note.  
* A 15 min video showing your work.

---

## **Tasks**

### **Part A — Train (core ML)**

Implement `python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/` that:

* Splits train/val with fixed seed.  
* Preprocesses (impute, encode categoricals, scale as needed).  
* Trains one strong model: XGBoost/LightGBM **or** a well-tuned logistic regression.  
* Logs metrics (ROC-AUC, PR-AUC, Accuracy) on **val** into `artifacts/metrics.json`.  
* Saves `model.pkl`, `feature_pipeline.pkl`, `feature_importances.csv` (or SHAP summary).  
* (Optional) 10–20 trial randomized HPO.

**Acceptance:** ROC-AUC ≥ **0.83** on our split; artifacts saved.

---

### **Part B — Serve (FastAPI)**

`uvicorn src.app:app` with:

* `GET /health` → `{"status":"ok"}`  
* `POST /predict` → accepts a JSON list of rows (Pydantic schema), returns probabilities \+ class.  
* Return 400 with a helpful message for missing fields / unknown categories.

---

### **Part C — MLOps bits**

* **Logging:** `artifacts/metrics.json` with metrics, timestamp, and git SHA (if available).  
* **Tests (pytest):**  
  * `tests/test_training.py` sanity-checks artifacts exist and ROC-AUC ≥ 0.83.  
  * `tests/test_inference.py` boots the API locally and checks `POST /predict` on 2 sample rows returns probs in \[0,1\].  
* **Docker:** image that can train and serve.  
* **CI (GitHub Actions):** install, run tests, build Docker on push (skeleton is provided; fill TODOs).

---

### **Part D — Monitoring (drift mini-check)**

`python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv`  
Outputs `artifacts/drift_report.json` with PSI/KS per feature and:

```json
{"threshold": 0.2, "overall_drift": true|false, "features": {"avg_latency_ms": 0.31, ...}}
```

**Part E — Agentic Monitor (LLM-optional)**

Implement a tiny “monitoring agent” that **observes → thinks → acts**:

**CLI:**

```shell
python -m src.agent_monitor \
  --metrics data/metrics_history.jsonl \
  --drift data/drift_latest.json \
  --out artifacts/agent_plan.yaml
```

**Behavior:**

* **Observe:** load metrics history \+ latest drift report.  
* **Think (rules or LLM-optional):** classify status: `healthy|warn|critical`. Suggested heuristics:  
  * `warn` if ROC-AUC drops ≥ 3% vs 7-day median **or** p95 latency \> 400ms for 2 consecutive points.  
  * `critical` if drop ≥ 6% **or** (`overall_drift` true **and** PR-AUC down ≥ 5%).  
* **Act:** emit an **action plan** (YAML) with `status`, `findings`, `actions` (subset of):  
  * `open_incident`, `trigger_retraining`, `roll_back_model`, `raise_thresholds`, `page_oncall=false`, `do_nothing`.  
* **Optional:** `POST /monitor` returns the same payload.

Example output:

```
status: warn
findings:
  - roc_auc_drop_pct: 3.8
  - latency_p95_ms: 412
  - drift_overall: false
actions:
  - trigger_retraining
  - raise_thresholds
rationale: >
  AUC fell 3.8% vs 7-day median; p95 latency > 400ms for two windows.
```

## **Deliverables checklist**

* `src/` code, `tests/`, `docker/Dockerfile`, `.github/workflows/ci.yml`, `requirements.txt`.  
* `artifacts/metrics.json` after training; `artifacts/drift_report.json`; `artifacts/agent_plan.yaml`.  
* `design_gcp.md` (≤1 page): how you’d run on GCP (BigQuery, training on Vertex AI or GKE, serving on Cloud Run/Vertex, metrics in Cloud Monitoring/Grafana), brief cost notes.  
* `README.md` with quickstart: `make train`, `make serve`, `make test`.

## **Constraints**

* Offline friendly (no internet required to train/eval).  
* Keep secrets out of code/CI.  
* Deterministic seeds.

## **Submission**

* GitHub repo link \+ brief README.  
* Include commands to reproduce in the README:

```
# Train
python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/

# Serve
uvicorn src.app:app --port 8000

# Drift
python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv

# Agent Monitor
python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml

# Tests
pytest -q
```
