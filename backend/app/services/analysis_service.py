from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Analysis
from app.services.explainability import build_explanation
from ml.inference.predictor import QualityPredictor


class AnalysisService:
    def __init__(self, predictor: QualityPredictor) -> None:
        self.predictor = predictor

    @staticmethod
    def _to_data_url(data: bytes, mime_type: str | None = None) -> str:
        mime = mime_type or "image/png"
        if "/" not in mime:
            mime = f"image/{mime}"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def analyze(
        self,
        image_bgr,
        filename: str,
        db: Session,
        source_bytes: bytes | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        try:
            result = self.predictor.predict(image_bgr)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Model inference failed") from exc

        image_data = None
        image_mime = None
        if source_bytes:
            image_mime = mime_type or "image/png"
            image_data = self._to_data_url(source_bytes, image_mime)

        explanation = build_explanation(result)
        record = Analysis(
            filename=filename,
            image_data=image_data,
            image_mime_type=image_mime,
            quality_score=result["quality_score"],
            quality_label=result["quality_label"],
            quality_confidence=result["quality_confidence"],
            issues=result["issues"],
            statistics=result["statistics"],
            explanation=explanation,
            class_probabilities=result["class_probabilities"],
            model_version=result["model_version"],
            anomaly_score=result.get("anomaly_score"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return serialize_analysis(record)


def serialize_analysis(record: Analysis) -> dict[str, Any]:
    created = record.created_at
    if created is not None and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        "analysis_id": str(record.id),
        "created_at": created.isoformat() if created else None,
        "filename": record.filename,
        "image_data": record.image_data,
        "image_mime_type": record.image_mime_type,
        "quality_score": record.quality_score,
        "quality_label": record.quality_label,
        "quality_confidence": record.quality_confidence,
        "issues": record.issues,
        "statistics": record.statistics,
        "explanation": record.explanation,
        "class_probabilities": record.class_probabilities,
        "model_version": record.model_version,
        "anomaly_score": record.anomaly_score,
    }


def serialize_summary(record: Analysis) -> dict[str, Any]:
    created = record.created_at
    if created is not None and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        "analysis_id": str(record.id),
        "created_at": created.isoformat() if created else datetime.now(timezone.utc).isoformat(),
        "filename": record.filename,
        "quality_score": record.quality_score,
        "quality_label": record.quality_label,
        "issues": record.issues,
    }
