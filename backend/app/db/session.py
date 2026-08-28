from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def ensure_analysis_image_columns() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("analyses"):
        return
    columns = {col["name"] for col in inspector.get_columns("analyses")}
    if "image_data" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE analyses ADD COLUMN image_data TEXT"))
    if "image_mime_type" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE analyses ADD COLUMN image_mime_type VARCHAR(64)"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
