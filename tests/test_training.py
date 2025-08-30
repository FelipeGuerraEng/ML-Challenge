import json
import subprocess
from pathlib import Path


def _run_training(data_path: str, outdir: Path, trials: int = 10) -> None:
    """
    Run the training CLI producing artifacts in the given directory.
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


def test_training_artifacts_and_metrics(tmp_path: Path) -> None:
    """
    Ensure training produces expected artifacts and meets the ROC-AUC threshold.
    """
    try:
        data = "data/customer_churn_synth.csv"
        outdir = tmp_path / "artifacts_test"
        _run_training(data, outdir, trials=10)
        required = ["model.pkl", "feature_pipeline.pkl", "feature_importances.csv", "metrics.json"]
        for name in required:
            p = outdir / name
            assert p.exists() and p.stat().st_size > 0
        with open(outdir / "metrics.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert "metrics" in payload
        assert float(payload["metrics"]["roc_auc"]) >= 0.83
    except Exception as e:
        raise AssertionError(f"Training test failed: {e}")
