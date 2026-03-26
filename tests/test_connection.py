"""Tests for shared database engine lifecycle behavior."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.database import connection


class ConnectionEngineTests(unittest.TestCase):
    """Validate that the shared engine only rotates when config actually changes."""

    def setUp(self) -> None:
        self.original_config = connection.DATABASE_CONFIG
        self.original_engine = connection.ENGINE
        self.original_session_factory = connection.SessionLocal

    def tearDown(self) -> None:
        connection.DATABASE_CONFIG = self.original_config
        connection.ENGINE = self.original_engine
        connection.SessionLocal = self.original_session_factory

    def test_ensure_database_engine_reuses_existing_engine_when_config_is_unchanged(self) -> None:
        existing_engine = SimpleNamespace(url="sqlite:///tmp/current.db")
        existing_session_factory = SimpleNamespace(name="session_factory")
        connection.DATABASE_CONFIG = SimpleNamespace(
            database_url="sqlite:///tmp/current.db",
            sqlite_path="tmp/current.db",
            sqlalchemy_echo=False,
        )
        connection.ENGINE = existing_engine
        connection.SessionLocal = existing_session_factory

        with patch.object(connection, "get_config", return_value=connection.DATABASE_CONFIG):
            with patch.object(connection, "_set_database_runtime") as set_runtime_mock:
                engine = connection.ensure_database_engine()

        self.assertIs(engine, existing_engine)
        set_runtime_mock.assert_not_called()

    def test_reset_database_engine_rebuilds_when_database_target_changes(self) -> None:
        base_config = connection.get_config()
        base_engine = SimpleNamespace(url="sqlite:///tmp/current.db")
        base_session_factory = SimpleNamespace(name="base_session_factory")
        connection.DATABASE_CONFIG = base_config
        connection.ENGINE = base_engine
        connection.SessionLocal = base_session_factory
        new_engine = object()
        new_session_factory = SimpleNamespace(name="session_factory")

        refreshed_config = connection.AppConfig(
            project_root=base_config.project_root,
            data_dir=base_config.data_dir,
            raw_data_dir=base_config.raw_data_dir,
            processed_data_dir=base_config.processed_data_dir,
            reports_dir=base_config.reports_dir,
            database_url="sqlite:///tmp/rebuilt-engine.db",
            sqlalchemy_echo=base_config.sqlalchemy_echo,
            sqlite_path=base_config.sqlite_path.parent / "rebuilt-engine.db",
            market_data_metadata_freshness_hours=base_config.market_data_metadata_freshness_hours,
            market_data_prices_freshness_hours=base_config.market_data_prices_freshness_hours,
            gnews_api_key=base_config.gnews_api_key,
            finnhub_api_key=base_config.finnhub_api_key,
            newsapi_api_key=base_config.newsapi_api_key,
        )

        with patch.object(connection, "get_config", return_value=refreshed_config):
            with patch.object(connection, "_dispose_engine_if_present") as dispose_mock:
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


if __name__ == "__main__":
    unittest.main()
