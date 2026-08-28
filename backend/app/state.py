from app.config import settings
from ml.inference.predictor import QualityPredictor

_predictor: QualityPredictor | None = None
_service = None


def _resolve_model_path() -> str:
    from pathlib import Path

    path = Path(settings.model_path)
    if path.is_file():
        return str(path)

    candidates = [
        Path(__file__).resolve().parents[2] / settings.model_path,
        Path(__file__).resolve().parents[2] / "models" / "quality_pipeline.joblib",
        Path("/app/models/quality_pipeline.joblib"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(path)


def set_predictor(predictor: QualityPredictor | None) -> None:
    global _predictor, _service
    from app.services.analysis_service import AnalysisService

    _predictor = predictor
    _service = AnalysisService(predictor) if predictor else None


def predictor_loaded() -> bool:
    return _predictor is not None


def predictor_version() -> str | None:
    return None if _predictor is None else _predictor.model_version


def get_or_create_predictor() -> QualityPredictor:
    global _predictor
    if _predictor is not None:
        return _predictor
    predictor = QualityPredictor(_resolve_model_path())
    set_predictor(predictor)
    return predictor


def get_analysis_service():
    global _service
    if _service is None:
        get_or_create_predictor()
    if _service is None:
        raise RuntimeError("Model is not loaded")
    return _service
