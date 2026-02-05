from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings
from pydantic import Field


def tortoise_orm_factory() -> dict[str, Any]:
    return {
        "connections": {
            "conn_system": {
                "engine": "tortoise.backends.sqlite",
                "credentials": {"file_path": "app_system.sqlite3"}
            }
        },
        "apps": {
            "app_system": {"models": ["app.models.system", "aerich.models"], "default_connection": "conn_system"}
        },
        "use_tz": False,
        "timezone": "Asia/Shanghai"
    }


class Settings(BaseSettings):
    VERSION: str = "0.1.0"
    APP_TITLE: str = "FastSoyAdmin"
    APP_DESCRIPTION: str = "Description"

    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_HEADERS: list[str] = Field(default_factory=lambda: ["*"])

    ADD_LOG_ORIGINS_INCLUDE: list[str] = Field(default_factory=lambda: ["*"])
    ADD_LOG_ORIGINS_DECLUDE: list[str] = Field(default_factory=lambda: ["/system-manage", "/redoc", "/doc", "/openapi.json"])

    DEBUG: bool = False

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    BASE_DIR: Path = PROJECT_ROOT.parent
    LOGS_ROOT: Path = BASE_DIR / "app/logs/"
    STATIC_ROOT: Path = BASE_DIR / "static/"
    SECRET_KEY: str = "015a42020f023ac2c3eda3d45fe5ca3fef8921ce63589f6d4fcdef9814cd7fa7"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12  # 12 hours
    JWT_REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    TORTOISE_ORM: dict[str, Any] = Field(default_factory=tortoise_orm_factory)

    DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    REDIS_URL: str = "redis://redis:6379/0"  # "redis://:password@233.233.233.233:33333/0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
