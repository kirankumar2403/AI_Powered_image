from ml.inference.predictor import QualityPredictor

_predictor: QualityPredictor | None = None
_service = None


def set_predictor(predictor: QualityPredictor | None) -> None:
    global _predictor, _service
    from app.services.analysis_service import AnalysisService

    _predictor = predictor
    _service = AnalysisService(predictor) if predictor else None


def predictor_loaded() -> bool:
    return _predictor is not None


def predictor_version() -> str | None:
    return None if _predictor is None else _predictor.model_version


def get_analysis_service():
    if _service is None:
        raise RuntimeError("Model is not loaded")
    return _service
