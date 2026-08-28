from typing import Any

from pydantic import BaseModel, Field


class Issue(BaseModel):
    type: str
    severity: str
    confidence: float


class Explanation(BaseModel):
    summary: str
    contributing_factors: list[str]
    feature_importances: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    analysis_id: str
    created_at: str | None = None
    filename: str
    image_data: str | None = None
    image_mime_type: str | None = None
    quality_score: int
    quality_label: str
    quality_confidence: float
    issues: list[Issue]
    statistics: dict[str, float]
    explanation: Explanation
    class_probabilities: dict[str, float]
    model_version: str
    anomaly_score: float | None = None


class AnalysisSummary(BaseModel):
    analysis_id: str
    created_at: str
    filename: str
    quality_score: int
    quality_label: str
    issues: list[Issue]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    database: str
