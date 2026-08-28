from fastapi import APIRouter

from app.state import predictor_loaded, predictor_version
from app.db.session import engine

router = APIRouter()


@router.get("/health")
def health():
    db_ok = "up"
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception:  # noqa: BLE001
        db_ok = "down"
    loaded = predictor_loaded()
    status = "ok" if loaded and db_ok == "up" else "degraded"
    return {
        "status": status,
        "model_loaded": loaded,
        "model_version": predictor_version(),
        "database": db_ok,
    }
