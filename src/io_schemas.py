from typing import List
from pydantic import BaseModel, Field

ALLOWED_PLAN_TYPES = {"Basic", "Standard", "Pro"}
ALLOWED_CONTRACT_TYPES = {"Monthly", "Annual"}
ALLOWED_YESNO = {"Yes", "No"}


class ChurnRequestRow(BaseModel):
    """
    One input row for churn prediction.
    """
    plan_type: str = Field(...)
    contract_type: str = Field(...)
    autopay: str = Field(...)
    is_promo_user: str = Field(...)
    add_on_count: float = Field(...)
    tenure_months: float = Field(...)
    monthly_usage_gb: float = Field(...)
    avg_latency_ms: float = Field(...)
    support_tickets_30d: float = Field(...)
    discount_pct: float = Field(...)
    payment_failures_90d: float = Field(...)
    downtime_hours_30d: float = Field(...)


class PredictRequest(BaseModel):
    """
    List of rows to score.
    """
    items: List[ChurnRequestRow]


class PredictResponseRow(BaseModel):
    """
    One scored row with probability and predicted class.
    """
    probability: float
    predicted_class: int


class PredictResponse(BaseModel):
    """
    Scoring response containing a list of predictions.
    """
    predictions: List[PredictResponseRow]
