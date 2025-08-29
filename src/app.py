import os
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load

from .features import get_feature_lists
from .io_schemas import (
    PredictRequest,
    PredictResponse,
    PredictResponseRow,
    ALLOWED_PLAN_TYPES,
    ALLOWED_CONTRACT_TYPES,
    ALLOWED_YESNO,
)


app = FastAPI(title="Churn Mini-Prod API")


def _artifacts_dir() -> str:
    """
    Return the artifacts directory path, defaulting to ./artifacts.
    """
    d = os.getenv("ARTIFACTS_DIR", "artifacts")
    return d


def _load_artifacts():
    """
    Load model and preprocessor from disk.
    """
    try:
        d = _artifacts_dir()
        model = load(os.path.join(d, "model.pkl"))
        pre = load(os.path.join(d, "feature_pipeline.pkl"))
        return model, pre
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")


MODEL, PREPROCESSOR = _load_artifacts()
CAT_FEATURES, NUM_FEATURES, TARGET = get_feature_lists()


@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


def _validate_categories(payload: PredictRequest) -> None:
    """
    Validate categorical fields against allowed sets, raising 400 on errors.
    """
    try:
        for i, row in enumerate(payload.items):
            if row.plan_type not in ALLOWED_PLAN_TYPES:
                raise HTTPException(status_code=400, detail=f"Row {i}: invalid plan_type '{row.plan_type}'")
            if row.contract_type not in ALLOWED_CONTRACT_TYPES:
                raise HTTPException(status_code=400, detail=f"Row {i}: invalid contract_type '{row.contract_type}'")
            if row.autopay not in ALLOWED_YESNO:
                raise HTTPException(status_code=400, detail=f"Row {i}: invalid autopay '{row.autopay}'")
            if row.is_promo_user not in ALLOWED_YESNO:
                raise HTTPException(status_code=400, detail=f"Row {i}: invalid is_promo_user '{row.is_promo_user}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {e}")


def _rows_to_dataframe(payload: PredictRequest) -> pd.DataFrame:
    """
    Convert a PredictRequest into a pandas DataFrame with correct column order.
    """
    try:
        records: List[dict] = [r.dict() for r in payload.items]
        df = pd.DataFrame.from_records(records)
        expected_cols = CAT_FEATURES + NUM_FEATURES
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")
        df = df[expected_cols]
        return df
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse input: {e}")


def _predict_proba(df: pd.DataFrame) -> np.ndarray:
    """
    Transform features and return positive class probabilities.
    """
    try:
        X = PREPROCESSOR.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        proba = MODEL.predict_proba(X)[:, 1]
        proba = np.clip(proba.astype(float), 0.0, 1.0)
        return proba
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Score a list of rows and return probabilities and predicted classes.
    """
    _validate_categories(req)
    df = _rows_to_dataframe(req)
    proba = _predict_proba(df)
    preds = [PredictResponseRow(probability=float(p), predicted_class=int(p >= 0.5)) for p in proba]
    return PredictResponse(predictions=preds)
