from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyses import router as analyses_router
from app.api.health import router as health_router
from app.config import settings
from app.db.session import Base, ensure_analysis_image_columns, engine
from app.state import get_or_create_predictor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_analysis_image_columns()
    get_or_create_predictor()
    yield


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
