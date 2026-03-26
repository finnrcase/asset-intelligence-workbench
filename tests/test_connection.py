"""Tests for shared database engine lifecycle behavior."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.database import connection


class ConnectionEngineTests(unittest.TestCase):
    """Validate that the shared engine only rotates when config actually changes."""

    def test_reset_database_engine_reuses_existing_engine_when_config_is_unchanged(self) -> None:
        with patch.object(connection, "get_config", return_value=connection.DATABASE_CONFIG):
            with patch.object(connection.ENGINE, "dispose") as dispose_mock:
                with patch.object(connection, "create_database_engine") as create_engine_mock:
                    with patch.object(connection, "create_session_factory") as create_session_factory_mock:
                        engine = connection.reset_database_engine()

        self.assertIs(engine, connection.ENGINE)
        dispose_mock.assert_not_called()
        create_engine_mock.assert_not_called()
        create_session_factory_mock.assert_not_called()

    def test_reset_database_engine_rebuilds_when_database_target_changes(self) -> None:
        original_config = connection.DATABASE_CONFIG
        original_engine = connection.ENGINE
        original_session_factory = connection.SessionLocal
        new_engine = object()
        new_session_factory = SimpleNamespace(name="session_factory")

        refreshed_config = connection.AppConfig(
            project_root=original_config.project_root,
            data_dir=original_config.data_dir,
            raw_data_dir=original_config.raw_data_dir,
            processed_data_dir=original_config.processed_data_dir,
            reports_dir=original_config.reports_dir,
            database_url="sqlite:///tmp/rebuilt-engine.db",
            sqlalchemy_echo=original_config.sqlalchemy_echo,
            sqlite_path=original_config.sqlite_path.parent / "rebuilt-engine.db",
            market_data_metadata_freshness_hours=original_config.market_data_metadata_freshness_hours,
            market_data_prices_freshness_hours=original_config.market_data_prices_freshness_hours,
            gnews_api_key=original_config.gnews_api_key,
            finnhub_api_key=original_config.finnhub_api_key,
            newsapi_api_key=original_config.newsapi_api_key,
        )

        try:
            with patch.object(connection, "get_config", return_value=refreshed_config):
                with patch.object(connection.ENGINE, "dispose") as dispose_mock:
                    with patch.object(connection, "create_database_engine", return_value=new_engine) as create_engine_mock:
                        with patch.object(connection, "create_session_factory", return_value=new_session_factory) as create_session_factory_mock:
                            engine = connection.reset_database_engine()

            self.assertIs(engine, new_engine)
            dispose_mock.assert_called_once()
            create_engine_mock.assert_called_once_with(refreshed_config)
            create_session_factory_mock.assert_called_once_with(new_engine)
            self.assertIs(connection.DATABASE_CONFIG, refreshed_config)
            self.assertIs(connection.ENGINE, new_engine)
            self.assertIs(connection.SessionLocal, new_session_factory)
        finally:
            connection.DATABASE_CONFIG = original_config
            connection.ENGINE = original_engine
            connection.SessionLocal = original_session_factory


if __name__ == "__main__":
    unittest.main()
