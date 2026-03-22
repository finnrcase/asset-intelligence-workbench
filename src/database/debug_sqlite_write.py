"""
Standalone SQLite write diagnostic for local debugging.
"""

from __future__ import annotations

from sqlalchemy import text

from src.database.connection import initialize_database
from src.database.connection import session_scope
from src.utils.config import get_config
from src.utils.config import get_sqlite_startup_diagnostics


def main() -> None:
    config = get_config()

    print(f"database_url={config.database_url}")
    print(f"sqlite_path={config.sqlite_path}")
    print(f"parent_directory={config.sqlite_path.parent}")
    print(f"diagnostics={get_sqlite_startup_diagnostics(config.sqlite_path)}")

    initialize_database()

    with session_scope() as session:
        session.execute(
            text(
                """
                INSERT INTO data_sources (source_name, source_type, source_url, created_at, updated_at)
                VALUES ('debug_probe', 'debug', 'local', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(source_name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                """
            )
        )

    print("sqlite_write_ok=true")


if __name__ == "__main__":
    main()
