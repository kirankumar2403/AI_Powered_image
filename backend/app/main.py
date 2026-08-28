from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyses import router as analyses_router
from app.api.health import router as health_router
from app.config import settings
from app.db.session import Base, engine
from app.state import set_predictor
from ml.inference.predictor import QualityPredictor


def _resolve_model_path() -> Path:
    path = Path(settings.model_path)
    if path.is_file():
        return path
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / settings.model_path,
        here.parents[2] / "models" / "quality_pipeline.joblib",
        Path("/app/models/quality_pipeline.joblib"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    predictor = QualityPredictor(_resolve_model_path())
    set_predictor(predictor)
    yield
    set_predictor(None)


app = FastAPI(
    title="AI-Powered Image Quality & Defect Detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analyses_router)
