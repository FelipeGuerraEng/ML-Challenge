# Mini-Prod ML Challenge — Churn Mini-Prod

Small, production-minded binary classifier on a tabular dataset with training, serving, drift check, and a light agent monitor. 

## Project structure
```
.
├── src/
│   ├── features.py
│   ├── models.py
│   ├── metrics.py
│   ├── train.py
│   ├── drift.py
│   └── agent_monitor.py
├── tests/
│   ├── test_training.py
│   └── test_inference.py
├── data/                      
├── artifacts/                 
├── docker/Dockerfile
└── .github/workflows/ci.yml
```

## Requirements
- Python 3.10+
- Install deps:  
  ```bash
  pip install -r requirements.txt
  ```
- Offline friendly: no internet required to train/eval.

`requirements.txt`:
```
scikit-learn
pandas
numpy
xgboost
lightgbm
fastapi
uvicorn
pydantic
pyyaml
pytest
joblib
optuna
imbalanced-learn
httpx
```

## Quickstart

### 1 Train (offline, reproducible)
```bash
python -m src.train --data data/customer_churn_synth.csv --outdir artifacts/
# artifacts/: model.pkl, feature_pipeline.pkl, feature_importances.csv, metrics.json
```

### 2 Serve (FastAPI)
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
# Health
curl -s http://localhost:8000/health
# Predict
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "items":[
    {"plan_type":"Pro","contract_type":"Annual","autopay":"Yes","is_promo_user":"No",
     "add_on_count":2,"tenure_months":18,"monthly_usage_gb":120.5,"avg_latency_ms":85,
     "support_tickets_30d":1,"discount_pct":5.0,"payment_failures_90d":0,"downtime_hours_30d":0.2},
    {"plan_type":"Basic","contract_type":"Monthly","autopay":"No","is_promo_user":"Yes",
     "add_on_count":0,"tenure_months":3,"monthly_usage_gb":12.1,"avg_latency_ms":220,
     "support_tickets_30d":3,"discount_pct":15.0,"payment_failures_90d":2,"downtime_hours_30d":1.5}
  ]
}'
```

### 3 Drift check (PSI/KS)
```bash
python -m src.drift --ref data/churn_ref_sample.csv --new data/churn_shifted_sample.csv --out artifacts/drift_report.json
cat artifacts/drift_report.json
```

### 4 Agentic Monitor (rules-based)
```bash
python -m src.agent_monitor   --metrics data/metrics_history.jsonl   --drift artifacts/drift_report.json   --out artifacts/agent_plan.yaml
cat artifacts/agent_plan.yaml
```

### 5 Tests
```bash
pytest -q
```

## Docker

> Best practice: images contain only code + deps. Data and artifacts are mounted at runtime.

**Build:**
```bash
docker build -t churn-mini -f docker/Dockerfile .
```

**Train (mount data/artifacts):**
```bash
docker run --rm   -v "$PWD/data:/app/data:ro"   -v "$PWD/artifacts:/app/artifacts"   churn-mini   python -m src.train --data /app/data/customer_churn_synth.csv --outdir /app/artifacts/
```

**Serve (mount artifacts):**
```bash
docker run --rm -p 8000:8000   -e ARTIFACTS_DIR=/app/artifacts   -v "$PWD/artifacts:/app/artifacts:ro"   churn-mini
```

## CI

GitHub Actions workflow in `.github/workflows/ci.yml` runs: install deps → tests → Docker build on every push/PR.

## Design notes
- **Training**: deterministic seeds, CV with L1 logistic regression; Optuna Bayesian search for `C`. Metrics logged: ROC-AUC, PR-AUC, Accuracy.
- **Serving**: FastAPI `/health`, `/predict`. Pydantic schema; 400 on missing fields/unknown categories.
- **Monitoring**: Drift via PSI/KS per feature; `overall_drift` when PSI ≥ 0.2. Agent monitor classifies `healthy|warn|critical` using rules over ROC/PR and p95 latency.
- **MLOps**: Tests (training/inference), minimal Docker image, CI (install → tests → docker build).

## GCP (one-pager)
See `design_gcp.md` for a simple, cost-aware deployment on BigQuery + Vertex AI + Cloud Run + Cloud Monitoring.

## Reproducibility
- Fixed seeds (`--seed`), no secrets in code/CI, offline-friendly training.
