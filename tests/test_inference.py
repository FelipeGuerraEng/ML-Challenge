import json
import os
import subprocess
import importlib
from pathlib import Path
from typing import Dict, Any

from fastapi.testclient import TestClient


def _run_training(data_path: str, outdir: Path, trials: int = 5) -> None:
    """
    Run the training CLI to produce artifacts for serving.
    """
    try:
        cmd = [
            "python",
            "-m",
            "src.train",
            "--data",
            data_path,
            "--outdir",
            str(outdir),
            "--trials",
            str(trials),
        ]
        subprocess.check_call(cmd)
    except Exception as e:
        raise RuntimeError(f"Training CLI failed: {e}")


def _start_client_with_artifacts(artifacts_dir: Path) -> TestClient:
    """
    Start a TestClient wired to the API using the given artifacts directory.
    """
    try:
        os.environ["ARTIFACTS_DIR"] = str(artifacts_dir)
        app_module = importlib.import_module("src.app")
        importlib.reload(app_module)
        return TestClient(app_module.app)
    except Exception as e:
        raise RuntimeError(f"Failed to start API client: {e}")


def _sample_payload() -> Dict[str, Any]:
    """
    Return a minimal valid payload with two rows.
    """
    return {
        "items": [
            {
                "plan_type": "Pro",
                "contract_type": "Annual",
                "autopay": "Yes",
                "is_promo_user": "No",
                "add_on_count": 2,
                "tenure_months": 18,
                "monthly_usage_gb": 120.5,
                "avg_latency_ms": 85,
                "support_tickets_30d": 1,
                "discount_pct": 5.0,
                "payment_failures_90d": 0,
                "downtime_hours_30d": 0.2,
            },
            {
                "plan_type": "Basic",
                "contract_type": "Monthly",
                "autopay": "No",
                "is_promo_user": "Yes",
                "add_on_count": 0,
                "tenure_months": 3,
                "monthly_usage_gb": 12.1,
                "avg_latency_ms": 220,
                "support_tickets_30d": 3,
                "discount_pct": 15.0,
                "payment_failures_90d": 2,
                "downtime_hours_30d": 1.5,
            },
        ]
    }


def test_api_health_and_predict(tmp_path: Path) -> None:
    """
    Ensure /health returns ok and /predict returns probabilities in [0,1] for two rows.
    """
    try:
        data = "data/customer_churn_synth.csv"
        outdir = tmp_path / "artifacts_api"
        _run_training(data, outdir, trials=5)
        client = _start_client_with_artifacts(outdir)

        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json() == {"status": "ok"}

        payload = _sample_payload()
        r_pred = client.post("/predict", json=payload)
        assert r_pred.status_code == 200
        body = r_pred.json()
        assert "predictions" in body
        preds = body["predictions"]
        assert isinstance(preds, list) and len(preds) == 2
        for p in preds:
            prob = float(p["probability"])
            cls = int(p["predicted_class"])
            assert 0.0 <= prob <= 1.0
            assert cls in (0, 1)
    except Exception as e:
        raise AssertionError(f"Inference test failed: {e}")
