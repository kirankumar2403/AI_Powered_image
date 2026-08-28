from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_data: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_mime_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quality_score: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_label: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    issues: Mapped[list] = mapped_column(JSONB, nullable=False)
    statistics: Mapped[dict] = mapped_column(JSONB, nullable=False)
    explanation: Mapped[dict] = mapped_column(JSONB, nullable=False)
    class_probabilities: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
