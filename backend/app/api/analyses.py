from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Analysis
from app.db.session import get_db
from app.schemas.analysis import AnalysisResponse, AnalysisSummary
from app.services.analysis_service import serialize_analysis, serialize_summary
from app.services.image_validation import (
    ImageValidationError,
    decode_image,
    safe_filename,
    validate_upload,
)
from app.state import get_analysis_service

router = APIRouter(prefix="/api")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file is None:
        raise HTTPException(status_code=400, detail="Missing file")
    data = await file.read()
    try:
        validate_upload(file.filename or "", file.content_type, data, settings.max_upload_bytes)
        image = decode_image(data)
        payload = get_analysis_service().analyze(
            image,
            safe_filename(file.filename or "upload"),
            db,
            source_bytes=data,
            mime_type=file.content_type or "image/png",
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"message": exc.message, "code": exc.code}) from exc
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail={"message": str(exc), "code": "inference_error"}) from exc
    except Exception:  # noqa: BLE001
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={"message": "Could not complete analysis.", "code": "server_error"},
        )
    return payload


@router.get("/analyses", response_model=list[AnalysisSummary])
def list_analyses(limit: int = 50, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    try:
        rows = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
    except Exception:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={"message": "Could not load analysis history.", "code": "database_error"},
        )
    return [serialize_summary(r) for r in rows]


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    row = db.get(Analysis, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"message": "Analysis not found.", "code": "not_found"})
    return serialize_analysis(row)
