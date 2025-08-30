import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from .features import (
    get_feature_lists,
    build_preprocessor,
    get_transformed_feature_names,
)
from .models import build_lasso_logreg, predict_proba, get_feature_importances
from .metrics import class_balance, compute_metrics


def _git_sha() -> str:
    """
    Return the short git SHA if available, otherwise an empty string.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return sha.decode().strip()
    except Exception:
        return ""


def _load_data(path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV dataset and split features and target.
    """
    try:
        df = pd.read_csv(path)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col])
        return X, y
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")


def _split_data(
    X: pd.DataFrame, y: pd.Series, seed: int, val_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/validation split with fixed seed.
    """
    try:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=val_size, stratify=y, random_state=seed
        )
        return X_tr, X_va, y_tr, y_va
    except Exception as e:
        raise RuntimeError(f"Failed to split data: {e}")


def _build_cv(n_splits: int, seed: int) -> StratifiedKFold:
    """
    Return a StratifiedKFold object with shuffling and fixed seed.
    """
    try:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except Exception as e:
        raise RuntimeError(f"Failed to build cross-validator: {e}")


def _make_pipeline_with_optional_smote(
    C: float,
    categorical,
    numeric,
    class_weight,
    use_smote: bool,
    seed: int,
):
    """
    Build a training pipeline with preprocessing, optional SMOTE, and logistic regression.
    """
    try:
        clf = build_lasso_logreg(C=C, class_weight=class_weight, random_state=seed)
        pre = build_preprocessor(categorical, numeric)
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.pipeline import Pipeline as ImbPipeline
            except Exception as e:
                raise RuntimeError(f"SMOTE components not available: {e}")
            sm = SMOTE(random_state=seed)
            pipe = ImbPipeline(steps=[("preprocessor", pre), ("smote", sm), ("model", clf)])
            return pipe
        pipe = Pipeline(steps=[("preprocessor", pre), ("model", clf)])
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to build training pipeline: {e}")


def _bayes_search_best_C(
    X: pd.DataFrame,
    y: pd.Series,
    categorical,
    numeric,
    class_weight,
    use_smote: bool,
    seed: int,
    n_splits: int = 5,
    n_trials: int = 15,
) -> Tuple[float, float]:
    """
    Run Bayesian optimization over C for L1 Logistic Regression and return best C and best CV score.
    """
    cv = _build_cv(n_splits=n_splits, seed=seed)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        pipe = _make_pipeline_with_optional_smote(
            C=C,
            categorical=categorical,
            numeric=numeric,
            class_weight=class_weight,
            use_smote=use_smote,
            seed=seed,
        )
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=None)
        return float(np.mean(scores))

    try:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_C = float(study.best_params["C"])
        best_score = float(study.best_value)
        return best_C, best_score
    except Exception as e:
        raise RuntimeError(f"Failed during Bayesian optimization: {e}")


def _ensure_dir(path: str) -> None:
    """
    Ensure the output directory exists.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to ensure output directory: {e}")


def _save_json(path: str, payload: Dict) -> None:
    """
    Save a JSON payload to disk.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON: {e}")


def _train_final_pipeline(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    categorical,
    numeric,
    best_C: float,
    class_weight,
    use_smote: bool,
    seed: int,
):
    """
    Fit the final pipeline on the training split.
    """
    try:
        pipe = _make_pipeline_with_optional_smote(
            C=best_C,
            categorical=categorical,
            numeric=numeric,
            class_weight=class_weight,
            use_smote=use_smote,
            seed=seed,
        )
        pipe.fit(X_tr, y_tr)
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to fit final pipeline: {e}")


def _evaluate_on_holdout(pipe, X_va: pd.DataFrame, y_va: pd.Series) -> Dict[str, float]:
    """
    Evaluate the fitted pipeline on the validation split and return metrics.
    """
    try:
        proba = pipe.predict_proba(X_va)[:, 1]
        return compute_metrics(y_va.to_numpy(), proba, threshold=0.5)
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate pipeline: {e}")


def _export_artifacts(
    pipe,
    outdir: str,
    categorical,
    numeric,
    y_tr: pd.Series,
    y_va: pd.Series,
    holdout_metrics: Dict[str, float],
    best_C: float,
    cv_best_score: float,
    sampling: str,
) -> None:
    """
    Persist model, preprocessor, importances, and metrics JSON.
    """
    _ensure_dir(outdir)
    try:
        pre = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["model"]
        dump(model, os.path.join(outdir, "model.pkl"))
        dump(pre, os.path.join(outdir, "feature_pipeline.pkl"))
        feat_names = get_transformed_feature_names(pre, categorical, numeric)
        imp_df = get_feature_importances(model, feat_names)
        imp_df.to_csv(os.path.join(outdir, "feature_importances.csv"), index=False)
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_sha": _git_sha(),
            "train": class_balance(y_tr.to_numpy()),
            "valid": class_balance(y_va.to_numpy()),
            "cv_best_roc_auc": float(cv_best_score),
            "best_params": {"C": float(best_C)},
            "metrics": holdout_metrics,
            "sampling": sampling,
        }
        _save_json(os.path.join(outdir, "metrics.json"), payload)
    except Exception as e:
        raise RuntimeError(f"Failed to export artifacts: {e}")


def main() -> None:
    """
    Entry point for model training with Bayesian search, CV, and optional SMOTE.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--imbalance_threshold", type=float, default=0.35)
    parser.add_argument("--smote_threshold", type=float, default=0.15)
    args = parser.parse_args()

    categorical, numeric, target = get_feature_lists()
    X, y = _load_data(args.data, target)
    X_tr, X_va, y_tr, y_va = _split_data(X, y, seed=args.seed, val_size=args.val_size)

    stats = class_balance(y_tr.to_numpy())
    p = float(stats["positive_rate"])
    minority = min(p, 1.0 - p)
    use_smote = bool(minority < float(args.smote_threshold))
    use_balanced = bool((minority < float(args.imbalance_threshold)) and not use_smote)
    class_weight = "balanced" if use_balanced else None
    sampling = "smote" if use_smote else "none"

    best_C, cv_best = _bayes_search_best_C(
        X_tr,
        y_tr,
        categorical,
        numeric,
        class_weight=class_weight,
        use_smote=use_smote,
        seed=args.seed,
        n_splits=args.cv_splits,
        n_trials=args.trials,
    )

    pipe = _train_final_pipeline(
        X_tr,
        y_tr,
        categorical,
        numeric,
        best_C,
        class_weight=class_weight,
        use_smote=use_smote,
        seed=args.seed,
    )

    holdout = _evaluate_on_holdout(pipe, X_va, y_va)

    _export_artifacts(
        pipe,
        args.outdir,
        categorical,
        numeric,
        y_tr,
        y_va,
        holdout,
        best_C,
        cv_best,
        sampling=sampling,
    )


if __name__ == "__main__":
    main()
