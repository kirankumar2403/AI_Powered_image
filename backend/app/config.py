from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        extra="ignore",
        protected_namespaces=(),
    )

    database_url: str = "postgresql+psycopg://iqd:iqd@localhost:5432/image_quality"
    model_path: str = "models/quality_pipeline.joblib"
    model_version: str = "1.0.0"
    max_upload_bytes: int = 8 * 1024 * 1024
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
