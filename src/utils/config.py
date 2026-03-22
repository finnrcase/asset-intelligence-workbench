"""
Application configuration utilities.

This module centralizes environment loading, SQLite path resolution, and
startup validation so the rest of the codebase can depend on a single,
explicit source of truth for runtime settings.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy.engine import make_url


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "app.db"


class DatabaseConfigurationError(RuntimeError):
    """Raised when the configured database path is invalid or not writable."""


def _load_environment() -> None:
    """Load environment variables from the project-level `.env` file if present."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _build_default_database_url(sqlite_path: Path) -> str:
    """Return the default SQLite database URL for a resolved filesystem path."""

    return f"sqlite:///{sqlite_path}"


def _resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative filesystem paths against the repository root."""

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _resolve_sqlite_path(database_url: str | None = None) -> Path:
    """Resolve the SQLite file path from env vars or the local default."""

    explicit_sqlite_path = os.getenv("SQLITE_DB_PATH")
    if explicit_sqlite_path:
        return _resolve_project_path(explicit_sqlite_path)

    if database_url and database_url.startswith("sqlite"):
        parsed_url = make_url(database_url)
        if parsed_url.database in (None, "", ":memory:"):
            raise DatabaseConfigurationError(
                f"SQLite DATABASE_URL must point to a filesystem path. Resolved DATABASE_URL: {database_url}"
            )
        return _resolve_project_path(parsed_url.database)

    return DEFAULT_SQLITE_PATH.resolve(strict=False)


def _ensure_directory_writable(directory: Path) -> None:
    """Raise when the parent directory does not exist and cannot be written."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatabaseConfigurationError(
            f"SQLite parent directory is not writable or cannot be created: {directory}"
        ) from exc

    probe_path = directory / f".sqlite-write-check-{os.getpid()}.tmp"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        raise DatabaseConfigurationError(
            f"SQLite parent directory is not writable: {directory}"
        ) from exc


def _ensure_sqlite_file_writable(sqlite_path: Path) -> None:
    """Raise when the SQLite file cannot be created or opened for writes."""

    try:
        connection = sqlite3.connect(sqlite_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise DatabaseConfigurationError(
            f"SQLite database file is not writable: {sqlite_path}"
        ) from exc


def _can_create_temp_file(directory: Path) -> bool:
    """Return True when a normal temporary file can be created and removed."""

    probe_path = directory / f".sqlite-write-check-{os.getpid()}.tmp"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
        return True
    except OSError:
        return False


def _can_write_sqlite_file(sqlite_path: Path) -> bool:
    """Return True when SQLite can open the file and start a write transaction."""

    try:
        connection = sqlite3.connect(sqlite_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
            return True
        finally:
            connection.close()
    except sqlite3.Error:
        return False


def _can_create_sqlite_sidefiles(directory: Path) -> bool:
    """Return True when SQLite can create journal/WAL side files in the same directory."""

    probe_db = directory / f".sqlite-sidefile-check-{os.getpid()}.db"
    try:
        connection = sqlite3.connect(probe_db)
        try:
            cursor = connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.fetchone()
            cursor.execute("CREATE TABLE IF NOT EXISTS sidefile_probe (id INTEGER PRIMARY KEY, note TEXT)")
            cursor.execute("INSERT INTO sidefile_probe(note) VALUES ('ok')")
            connection.commit()
        finally:
            connection.close()

        wal_path = Path(str(probe_db) + "-wal")
        shm_path = Path(str(probe_db) + "-shm")
        journal_path = Path(str(probe_db) + "-journal")
        sidefile_created = wal_path.exists() or shm_path.exists() or journal_path.exists() or probe_db.exists()

        for cleanup_path in (wal_path, shm_path, journal_path, probe_db):
            if cleanup_path.exists():
                cleanup_path.unlink()

        return sidefile_created
    except sqlite3.Error:
        return False


def get_sqlite_startup_diagnostics(sqlite_path: Path) -> dict[str, Any]:
    """Return a small diagnostic snapshot for the resolved SQLite location."""

    resolved_path = sqlite_path.expanduser().resolve(strict=False)
    parent_directory = resolved_path.parent

    return {
        "resolved_path": str(resolved_path),
        "parent_directory": str(parent_directory),
        "parent_exists": parent_directory.exists(),
        "parent_writable": os.access(parent_directory, os.W_OK) if parent_directory.exists() else False,
        "db_exists": resolved_path.exists(),
        "db_writable": os.access(resolved_path, os.W_OK) if resolved_path.exists() else None,
        "temp_file_create_delete": _can_create_temp_file(parent_directory) if parent_directory.exists() else False,
        "sqlite_write_probe": _can_write_sqlite_file(resolved_path) if parent_directory.exists() else False,
        "sqlite_sidefiles": _can_create_sqlite_sidefiles(parent_directory) if parent_directory.exists() else False,
    }


def _validate_sqlite_database_url(database_url: str) -> None:
    """Reject explicit SQLite URLs that request read-only access."""

    normalized_url = database_url.lower()
    if "mode=ro" in normalized_url or "immutable=1" in normalized_url:
        raise DatabaseConfigurationError(
            "SQLite DATABASE_URL requests read-only access, which prevents on-demand ingestion. "
            f"Resolved DATABASE_URL: {database_url}"
        )


def validate_sqlite_runtime(sqlite_path: Path, database_url: str) -> None:
    """
    Validate that the configured SQLite location can be used for read/write app operation.

    The checks are intentionally explicit so deployment issues fail at startup
    rather than later during ingestion.
    """

    _validate_sqlite_database_url(database_url)

    resolved_path = sqlite_path.expanduser().resolve(strict=False)
    parent_directory = resolved_path.parent

    _ensure_directory_writable(parent_directory)
    _ensure_sqlite_file_writable(resolved_path)


def get_resolved_sqlite_path(database_url: str | None = None) -> Path:
    """Return and log the fully resolved SQLite path used by the application."""

    resolved_path = _resolve_sqlite_path(database_url)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    LOGGER.info("Resolved SQLite database path: %s", resolved_path)
    return resolved_path


def log_sqlite_startup_diagnostics(database_url: str | None = None) -> dict[str, Any]:
    """Log the resolved SQLite path and basic filesystem/write diagnostics."""

    resolved_path = get_resolved_sqlite_path(database_url)
    diagnostics = get_sqlite_startup_diagnostics(resolved_path)
    LOGGER.info("SQLite database URL: %s", database_url or _build_default_database_url(resolved_path))
    LOGGER.info("SQLite database parent directory: %s", diagnostics["parent_directory"])
    LOGGER.info("SQLite startup diagnostics: %s", diagnostics)
    return diagnostics


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
    market_data_metadata_freshness_hours: int
    market_data_prices_freshness_hours: int
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

    configured_database_url = os.getenv("DATABASE_URL")
    sqlite_path = get_resolved_sqlite_path(configured_database_url)
    database_url = (
        configured_database_url
        if configured_database_url and not configured_database_url.startswith("sqlite")
        else _build_default_database_url(sqlite_path)
    )
    sqlalchemy_echo = os.getenv("SQLALCHEMY_ECHO", "false").strip().lower() == "true"
    metadata_freshness_hours = int(os.getenv("MARKET_DATA_METADATA_FRESHNESS_HOURS", "24"))
    prices_freshness_hours = int(os.getenv("MARKET_DATA_PRICES_FRESHNESS_HOURS", "6"))

    if database_url.startswith("sqlite"):
        log_sqlite_startup_diagnostics(database_url)

    config = AppConfig(
        project_root=PROJECT_ROOT,
        data_dir=PROJECT_ROOT / "data",
        raw_data_dir=PROJECT_ROOT / "data" / "raw",
        processed_data_dir=PROJECT_ROOT / "data" / "processed",
        reports_dir=PROJECT_ROOT / "reports" / "generated",
        database_url=database_url,
        sqlalchemy_echo=sqlalchemy_echo,
        sqlite_path=sqlite_path,
        market_data_metadata_freshness_hours=metadata_freshness_hours,
        market_data_prices_freshness_hours=prices_freshness_hours,
        gnews_api_key=os.getenv("GNEWS_API_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
        newsapi_api_key=os.getenv("NEWSAPI_API_KEY") or os.getenv("NEWS_API_KEY"),
    )

    if config.is_sqlite:
        validate_sqlite_runtime(config.sqlite_path, config.database_url)

    return config
