from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def build_lasso_logreg(
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = 42,
    max_iter: int = 2000,
) -> LogisticRegression:
    """
    Build an L1-regularized logistic regression suitable for CV-based selection of C.
    """
    try:
        return LogisticRegression(
            penalty="l1",
            C=float(C),
            solver="saga",
            max_iter=int(max_iter),
            random_state=int(random_state),
            class_weight=class_weight,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build logistic regression: {e}")


def fit_model(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Fit the provided model on features X and target y and return the fitted model.
    """
    try:
        model.fit(X, y)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to fit model: {e}")


def predict_proba(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """
    Return positive class probabilities for the given features X.
    """
    try:
        proba = model.predict_proba(X)[:, 1]
        return proba
    except Exception as e:
        raise RuntimeError(f"Failed to compute probabilities: {e}")


def get_feature_importances(model: LogisticRegression, feature_names: List[str]) -> pd.DataFrame:
    """
    Return feature importances as absolute coefficients for linear models or
    tree-style importances if available.
    """
    try:
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).ravel()
            imp = np.abs(coef)
            df = pd.DataFrame(
                {
                    "feature": feature_names[: len(coef)],
                    "importance": imp[: len(coef)],
                    "coefficient": coef[: len(coef)],
                }
            ).sort_values("importance", ascending=False)
            return df.reset_index(drop=True)
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).ravel()
            df = pd.DataFrame(
                {"feature": feature_names[: len(imp)], "importance": imp[: len(imp)]}
            ).sort_values("importance", ascending=False)
            return df.reset_index(drop=True)
        raise ValueError("Model does not expose coefficients or feature_importances_.")
    except Exception as e:
        raise RuntimeError(f"Failed to compute feature importances: {e}")
