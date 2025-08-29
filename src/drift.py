import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .features import get_feature_lists


def _load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV '{path}': {e}")


def _psi_numeric(ref: np.ndarray, new: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """
    Compute PSI for a numeric array using quantile-based bins from the reference.
    """
    try:
        ref = np.asarray(ref).astype(float)
        new = np.asarray(new).astype(float)
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(ref, qs))
        if edges.size < 2:
            edges = np.array([np.min(ref), np.max(ref)])
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_hist, _ = np.histogram(ref, bins=edges)
        new_hist, _ = np.histogram(new, bins=edges)
        ref_pct = ref_hist / max(ref.size, 1)
        new_pct = new_hist / max(new.size, 1)
        ref_pct = np.clip(ref_pct, eps, None)
        new_pct = np.clip(new_pct, eps, None)
        psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
        return psi
    except Exception as e:
        raise RuntimeError(f"Failed to compute numeric PSI: {e}")


def _psi_categorical(ref: np.ndarray, new: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute PSI for a categorical array by aligning category proportions.
    """
    try:
        ref = np.asarray(ref).astype(str)
        new = np.asarray(new).astype(str)
        categories = np.union1d(np.unique(ref), np.unique(new))
        ref_counts = np.array([(ref == c).sum() for c in categories], dtype=float)
        new_counts = np.array([(new == c).sum() for c in categories], dtype=float)
        ref_pct = ref_counts / max(ref.size, 1)
        new_pct = new_counts / max(new.size, 1)
        ref_pct = np.clip(ref_pct, eps, None)
        new_pct = np.clip(new_pct, eps, None)
        psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
        return psi
    except Exception as e:
        raise RuntimeError(f"Failed to compute categorical PSI: {e}")


def _ks_numeric(ref: np.ndarray, new: np.ndarray) -> float:
    """
    Compute KS statistic for two numeric samples using ECDFs.
    """
    try:
        ref = np.asarray(ref).astype(float)
        new = np.asarray(new).astype(float)
        all_vals = np.sort(np.unique(np.concatenate([ref, new], axis=0)))
        if all_vals.size == 0:
            return 0.0
        ref_cdf = np.searchsorted(np.sort(ref), all_vals, side="right") / max(ref.size, 1)
        new_cdf = np.searchsorted(np.sort(new), all_vals, side="right") / max(new.size, 1)
        ks = float(np.max(np.abs(ref_cdf - new_cdf)))
        return ks
    except Exception as e:
        raise RuntimeError(f"Failed to compute KS: {e}")


def _compute_drift(ref_df: pd.DataFrame, new_df: pd.DataFrame, psi_threshold: float) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    """
    Compute per-feature PSI (all features) and KS (numeric only), and overall_drift flag.
    """
    try:
        cat_cols, num_cols, _ = get_feature_lists()
        psi_by_feature: Dict[str, float] = {}
        ks_by_feature: Dict[str, float] = {}

        for col in cat_cols:
            if col in ref_df.columns and col in new_df.columns:
                psi_by_feature[col] = _psi_categorical(ref_df[col].values, new_df[col].values)

        for col in num_cols:
            if col in ref_df.columns and col in new_df.columns:
                psi_by_feature[col] = _psi_numeric(ref_df[col].values, new_df[col].values)
                ks_by_feature[col] = _ks_numeric(ref_df[col].values, new_df[col].values)

        overall = bool(any(v >= psi_threshold for v in psi_by_feature.values()))
        return psi_by_feature, ks_by_feature, overall
    except Exception as e:
        raise RuntimeError(f"Failed to compute drift: {e}")


def _save_json(path: str, payload: Dict) -> None:
    """
    Save a JSON payload to disk.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON: {e}")


def main() -> None:
    """
    CLI entrypoint for drift calculation with PSI (per feature) and KS (numeric).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--new", type=str, required=True)
    parser.add_argument("--out", type=str, default="artifacts/drift_report.json")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    ref_df = _load_csv(args.ref)
    new_df = _load_csv(args.new)

    psi_by_feature, ks_by_feature, overall = _compute_drift(ref_df, new_df, psi_threshold=float(args.threshold))

    payload = {
        "threshold": float(args.threshold),
        "overall_drift": bool(overall),
        "features": {k: float(v) for k, v in psi_by_feature.items()},
        "ks": {k: float(v) for k, v in ks_by_feature.items()},
    }
    _save_json(args.out, payload)


if __name__ == "__main__":
    main()
