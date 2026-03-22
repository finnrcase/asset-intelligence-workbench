"""
Application configuration utilities.

This module centralizes environment loading and path resolution so the rest of
the codebase can depend on a single, explicit source of truth for runtime
settings.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "processed" / "asset_intelligence.db"
FALLBACK_SQLITE_DIRNAME = "asset-intelligence-workbench"


def _load_environment() -> None:
    """Load environment variables from the project-level `.env` file if present."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _is_path_writable(path: Path) -> bool:
    """Return True when the SQLite file path can be written by the current process."""

    if path.exists():
        return os.access(path, os.W_OK)
    return os.access(path.parent, os.W_OK)


def _fallback_sqlite_path() -> Path:
    """Return a writable SQLite path outside the potentially read-only repo mount."""

    return Path(tempfile.gettempdir()) / FALLBACK_SQLITE_DIRNAME / DEFAULT_SQLITE_PATH.name


def _resolve_sqlite_path() -> Path:
    """
    Resolve the SQLite path for the current environment.

    Local development keeps the project database under `data/processed`. When the
    repo mount is read-only, such as some hosted Streamlit environments, the app
    falls back to a temp-backed writable copy.
    """

    configured_path = Path(os.getenv("SQLITE_DB_PATH", str(DEFAULT_SQLITE_PATH))).expanduser()
    if os.getenv("SQLITE_DB_PATH"):
        return configured_path

    if _is_path_writable(configured_path):
        return configured_path

    fallback_path = _fallback_sqlite_path()
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    if not fallback_path.exists() and DEFAULT_SQLITE_PATH.exists():
        shutil.copy2(DEFAULT_SQLITE_PATH, fallback_path)
    return fallback_path


@dataclass(frozen=True)
class AppConfig:
    """
    Runtime configuration for the project.

    The database URL defaults to a local SQLite file for development, while the
    same interface can point to PostgreSQL or another SQLAlchemy-supported
    backend later without changing downstream modules.
    """

    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    reports_dir: Path
    database_url: str
    sqlalchemy_echo: bool
    sqlite_path: Path
    gnews_api_key: str | None
    finnhub_api_key: str | None
    newsapi_api_key: str | None

    @property
    def is_sqlite(self) -> bool:
        """Return True when the configured database backend is SQLite."""

        return self.database_url.startswith("sqlite")


def get_config() -> AppConfig:
    """Build and return the application configuration."""

    _load_environment()

    sqlite_path = _resolve_sqlite_path()
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{sqlite_path}")
    sqlalchemy_echo = os.getenv("SQLALCHEMY_ECHO", "false").strip().lower() == "true"

    return AppConfig(
        project_root=PROJECT_ROOT,
        data_dir=PROJECT_ROOT / "data",
        raw_data_dir=PROJECT_ROOT / "data" / "raw",
        processed_data_dir=PROJECT_ROOT / "data" / "processed",
        reports_dir=PROJECT_ROOT / "reports" / "generated",
        database_url=database_url,
        sqlalchemy_echo=sqlalchemy_echo,
        sqlite_path=sqlite_path,
        gnews_api_key=os.getenv("GNEWS_API_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
        newsapi_api_key=os.getenv("NEWSAPI_API_KEY") or os.getenv("NEWS_API_KEY"),
    )
