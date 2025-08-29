from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES: List[str] = [
    "plan_type",
    "contract_type",
    "autopay",
    "is_promo_user",
]

NUMERIC_FEATURES: List[str] = [
    "add_on_count",
    "tenure_months",
    "monthly_usage_gb",
    "avg_latency_ms",
    "support_tickets_30d",
    "discount_pct",
    "payment_failures_90d",
    "downtime_hours_30d",
]

TARGET_COLUMN: str = "churned"


def get_feature_lists() -> Tuple[List[str], List[str], str]:
    """
    Return the categorical features, numeric features, and target column name.
    """
    return CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def _onehot_encoder() -> OneHotEncoder:
    """
    Return a OneHotEncoder compatible with different sklearn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes and encodes categorical features and
    imputes and scales numeric features.
    """
    try:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _onehot_encoder()),
            ]
        )
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, categorical),
                ("num", num_pipe, numeric),
            ]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build preprocessor: {e}")


def make_pipeline(estimator: Any, categorical: List[str], numeric: List[str]) -> Pipeline:
    """
    Return a training pipeline that includes preprocessing and the given estimator.
    """
    try:
        pre = build_preprocessor(categorical, numeric)
        return Pipeline(steps=[("preprocessor", pre), ("model", estimator)])
    except Exception as e:
        raise RuntimeError(f"Failed to build pipeline: {e}")


def get_transformed_feature_names(
    fitted_preprocessor: ColumnTransformer,
    categorical: List[str],
    numeric: List[str],
) -> List[str]:
    """
    Return the transformed feature names from a fitted ColumnTransformer.
    """
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception:
        try:
            ohe: OneHotEncoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = list(ohe.get_feature_names_out(categorical))
        except Exception:
            cat_names = [f"cat_{i}" for i in range(len(categorical))]
        num_names = list(numeric)
        return [*cat_names, *num_names]


def transform_dataframe(fitted_preprocessor: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    """
    Transform a DataFrame using a fitted preprocessor and return a dense array.
    """
    try:
        X = fitted_preprocessor.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X
    except Exception as e:
        raise RuntimeError(f"Failed to transform features: {e}")
