import os
import sqlite3
import unittest
from importlib import reload
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from src.utils import config


TEST_ROOT = Path(__file__).resolve().parent / ".tmp"
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class ConfigTests(unittest.TestCase):
    def _cleanup_tree(self, root: Path) -> None:
        if not root.exists():
            return

        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()

    def test_default_sqlite_path_uses_project_data_directory(self) -> None:
        self.assertEqual(
            config.DEFAULT_SQLITE_PATH,
            (config.PROJECT_ROOT / "data" / "app.db").resolve(strict=False),
        )

    def test_resolve_sqlite_path_prefers_explicit_env_override(self) -> None:
        explicit_path = TEST_ROOT / "explicit" / "asset_intelligence.db"

        with patch.dict(os.environ, {"SQLITE_DB_PATH": str(explicit_path)}, clear=False):
            resolved = config._resolve_sqlite_path()

        self.assertEqual(resolved, explicit_path)

    def test_resolve_sqlite_path_uses_database_url_when_present(self) -> None:
        database_url = "sqlite:///tmp/custom_app.db"

        with patch.dict(os.environ, {"SQLITE_DB_PATH": ""}, clear=False):
            os.environ.pop("SQLITE_DB_PATH", None)
            resolved = config._resolve_sqlite_path(database_url)

        self.assertEqual(
            resolved,
            (config.PROJECT_ROOT / "tmp" / "custom_app.db").resolve(strict=False),
        )

    def test_get_resolved_sqlite_path_logs_absolute_path(self) -> None:
        expected_path = config._resolve_sqlite_path()

        with self.assertLogs("src.utils.config", level="INFO") as captured:
            resolved = config.get_resolved_sqlite_path()

        self.assertEqual(resolved, expected_path)
        self.assertTrue(any(str(expected_path) in message for message in captured.output))

    def test_relative_sqlite_env_path_is_resolved_from_project_root(self) -> None:
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "data/custom.db"}, clear=False):
            resolved = config._resolve_sqlite_path()

        self.assertEqual(
            resolved,
            (config.PROJECT_ROOT / "data" / "custom.db").resolve(strict=False),
        )

    def test_startup_diagnostics_report_expected_keys(self) -> None:
        diagnostics = config.get_sqlite_startup_diagnostics(
            (config.PROJECT_ROOT / "data" / "app.db").resolve(strict=False)
        )

        self.assertIn("resolved_path", diagnostics)
        self.assertIn("parent_directory", diagnostics)
        self.assertIn("parent_exists", diagnostics)
        self.assertIn("parent_writable", diagnostics)
        self.assertIn("db_exists", diagnostics)
        self.assertIn("db_writable", diagnostics)
        self.assertIn("temp_file_create_delete", diagnostics)
        self.assertIn("sqlite_write_probe", diagnostics)
        self.assertIn("sqlite_sidefiles", diagnostics)

    def test_validate_sqlite_runtime_rejects_readonly_database_url(self) -> None:
        sqlite_path = TEST_ROOT / "readonly_url.db"

        with self.assertRaises(config.DatabaseConfigurationError) as context:
            config.validate_sqlite_runtime(
                sqlite_path=sqlite_path,
                database_url=f"sqlite:///{sqlite_path}?mode=ro",
            )

        self.assertIn("read-only access", str(context.exception))
        self.assertIn("DATABASE_URL", str(context.exception))

    def test_validate_sqlite_runtime_creates_directory_and_file_when_writable(self) -> None:
        temp_root = TEST_ROOT / f"config_{uuid4().hex}"
        sqlite_path = temp_root / "nested" / "asset_intelligence.db"

        try:
            config.validate_sqlite_runtime(
                sqlite_path=sqlite_path,
                database_url=f"sqlite:///{sqlite_path}",
            )

            self.assertTrue(sqlite_path.parent.exists())
            self.assertTrue(sqlite_path.exists())
        finally:
            for path in sorted(temp_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            if temp_root.exists():
                temp_root.rmdir()

    def test_validate_sqlite_runtime_raises_clear_error_when_no_runtime_fallback_is_available(self) -> None:
        sqlite_path = TEST_ROOT / "write_probe.db"

        with patch("src.utils.config._build_runtime_sqlite_path", return_value=sqlite_path.resolve(strict=False)):
            with patch("src.utils.config._build_ephemeral_runtime_sqlite_path", return_value=sqlite_path.resolve(strict=False)):
                with patch("src.utils.config.sqlite3.connect", side_effect=sqlite3.OperationalError("readonly")):
                    with self.assertRaises(config.DatabaseConfigurationError) as context:
                        config.validate_sqlite_runtime(
                            sqlite_path=sqlite_path,
                            database_url=f"sqlite:///{sqlite_path}",
                        )

        self.assertIn(str(sqlite_path), str(context.exception))
        self.assertIn("not writable", str(context.exception))

    def test_validate_sqlite_runtime_handles_oserror_from_sqlite_probe(self) -> None:
        sqlite_path = TEST_ROOT / "write_probe_oserror.db"

        with patch("src.utils.config._build_runtime_sqlite_path", return_value=sqlite_path.resolve(strict=False)):
            with patch("src.utils.config._build_ephemeral_runtime_sqlite_path", return_value=sqlite_path.resolve(strict=False)):
                with patch("src.utils.config.sqlite3.connect", side_effect=PermissionError("access denied")):
                    with self.assertRaises(config.DatabaseConfigurationError) as context:
                        config.validate_sqlite_runtime(
                            sqlite_path=sqlite_path,
                            database_url=f"sqlite:///{sqlite_path}",
                        )

        self.assertIn(str(sqlite_path), str(context.exception))
        self.assertIn("not writable", str(context.exception))

    def test_validate_sqlite_runtime_uses_fallback_when_file_is_not_writable(self) -> None:
        temp_root = TEST_ROOT / f"validate_runtime_fallback_{uuid4().hex}"
        sqlite_path = temp_root / "readonly" / "write_probe.db"
        runtime_root = temp_root / "runtime"

        try:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            sqlite_path.write_text("", encoding="utf-8")

            original_ensure_writable = config._ensure_sqlite_file_writable

            def fake_ensure_writable(path: Path) -> None:
                if path == sqlite_path.resolve(strict=False):
                    raise config.DatabaseConfigurationError(
                        f"SQLite database file is not writable: {path}"
                    )
                original_ensure_writable(path)

            with patch.dict(os.environ, {"SQLITE_RUNTIME_DIR": str(runtime_root)}, clear=False):
                with patch("src.utils.config._ensure_sqlite_file_writable", side_effect=fake_ensure_writable):
                    config.validate_sqlite_runtime(
                        sqlite_path=sqlite_path,
                        database_url=f"sqlite:///{sqlite_path}",
                    )
        finally:
            self._cleanup_tree(temp_root)

    def test_prepare_sqlite_runtime_falls_back_to_runtime_copy_when_original_is_not_writable(self) -> None:
        temp_root = TEST_ROOT / f"runtime_fallback_{uuid4().hex}"
        source_path = temp_root / "readonly" / "asset_intelligence.db"
        runtime_root = temp_root / "runtime"

        try:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(source_path)
            try:
                connection.execute("CREATE TABLE assets (id INTEGER PRIMARY KEY, ticker TEXT NOT NULL)")
                connection.execute("INSERT INTO assets(ticker) VALUES ('MSFT')")
                connection.commit()
            finally:
                connection.close()

            original_ensure_writable = config._ensure_sqlite_file_writable

            def fake_ensure_writable(sqlite_path: Path) -> None:
                if sqlite_path == source_path.resolve(strict=False):
                    raise config.DatabaseConfigurationError(
                        f"SQLite database file is not writable: {sqlite_path}"
                    )
                original_ensure_writable(sqlite_path)

            with patch.dict(os.environ, {"SQLITE_RUNTIME_DIR": str(runtime_root)}, clear=False):
                with patch("src.utils.config._ensure_sqlite_file_writable", side_effect=fake_ensure_writable):
                    runtime_path = config.prepare_sqlite_runtime(
                        sqlite_path=source_path,
                        database_url=f"sqlite:///{source_path.resolve(strict=False)}",
                    )

            self.assertEqual(runtime_path, (runtime_root / source_path.name).resolve(strict=False))
            self.assertTrue(runtime_path.exists())

            copied_connection = sqlite3.connect(runtime_path)
            try:
                row = copied_connection.execute("SELECT ticker FROM assets").fetchone()
            finally:
                copied_connection.close()

            self.assertEqual(row[0], "MSFT")
        finally:
            self._cleanup_tree(temp_root)

    def test_prepare_sqlite_runtime_uses_fresh_runtime_db_when_copy_fails(self) -> None:
        temp_root = TEST_ROOT / f"runtime_copy_failure_{uuid4().hex}"
        source_path = temp_root / "readonly" / "asset_intelligence.db"
        runtime_root = temp_root / "runtime"

        try:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("placeholder", encoding="utf-8")

            original_ensure_writable = config._ensure_sqlite_file_writable

            def fake_ensure_writable(sqlite_path: Path) -> None:
                if sqlite_path == source_path.resolve(strict=False):
                    raise config.DatabaseConfigurationError(
                        f"SQLite database file is not writable: {sqlite_path}"
                    )
                original_ensure_writable(sqlite_path)

            with patch.dict(os.environ, {"SQLITE_RUNTIME_DIR": str(runtime_root)}, clear=False):
                with patch("src.utils.config._ensure_sqlite_file_writable", side_effect=fake_ensure_writable):
                    with patch("src.utils.config.shutil.copy2", side_effect=PermissionError("copy blocked")):
                        runtime_path = config.prepare_sqlite_runtime(
                            sqlite_path=source_path,
                            database_url=f"sqlite:///{source_path.resolve(strict=False)}",
                        )

            self.assertEqual(runtime_path, (runtime_root / source_path.name).resolve(strict=False))
            self.assertTrue(runtime_path.exists())

            connection = sqlite3.connect(runtime_path)
            try:
                connection.execute("CREATE TABLE IF NOT EXISTS fallback_probe (id INTEGER PRIMARY KEY)")
                connection.commit()
            finally:
                connection.close()
        finally:
            self._cleanup_tree(temp_root)

    def test_prepare_sqlite_runtime_uses_ephemeral_runtime_path_when_primary_runtime_fails(self) -> None:
        temp_root = TEST_ROOT / f"ephemeral_runtime_fallback_{uuid4().hex}"
        source_path = temp_root / "readonly" / "asset_intelligence.db"
        primary_runtime_root = temp_root / "runtime_primary"
        ephemeral_runtime_root = temp_root / "runtime_ephemeral"

        try:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("", encoding="utf-8")

            original_ensure_directory_writable = config._ensure_directory_writable
            original_ensure_writable = config._ensure_sqlite_file_writable

            def fake_ensure_directory_writable(directory: Path) -> None:
                if directory == primary_runtime_root.resolve(strict=False):
                    raise config.DatabaseConfigurationError(
                        f"SQLite parent directory is not writable: {directory}"
                    )
                original_ensure_directory_writable(directory)

            def fake_ensure_writable(sqlite_path: Path) -> None:
                if sqlite_path == source_path.resolve(strict=False):
                    raise config.DatabaseConfigurationError(
                        f"SQLite database file is not writable: {sqlite_path}"
                    )
                original_ensure_writable(sqlite_path)

            with patch("src.utils.config._build_runtime_sqlite_path", return_value=(primary_runtime_root / source_path.name).resolve(strict=False)):
                with patch("src.utils.config._build_ephemeral_runtime_sqlite_path", return_value=(ephemeral_runtime_root / source_path.name).resolve(strict=False)):
                    with patch("src.utils.config._ensure_directory_writable", side_effect=fake_ensure_directory_writable):
                        with patch("src.utils.config._ensure_sqlite_file_writable", side_effect=fake_ensure_writable):
                            runtime_path = config.prepare_sqlite_runtime(
                                sqlite_path=source_path,
                                database_url=f"sqlite:///{source_path.resolve(strict=False)}",
                            )

            self.assertEqual(runtime_path, (ephemeral_runtime_root / source_path.name).resolve(strict=False))
            self.assertTrue(runtime_path.exists())
        finally:
            self._cleanup_tree(temp_root)

    def test_get_config_uses_runtime_sqlite_fallback_when_configured_path_is_not_writable(self) -> None:
        temp_root = TEST_ROOT / f"config_runtime_fallback_{uuid4().hex}"
        source_path = temp_root / "readonly" / "engine_target.db"
        runtime_root = temp_root / "runtime"

        try:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("", encoding="utf-8")

            original_ensure_writable = config._ensure_sqlite_file_writable

            def fake_ensure_writable(sqlite_path: Path) -> None:
                if sqlite_path == source_path.resolve(strict=False):
                    raise config.DatabaseConfigurationError(
                        f"SQLite database file is not writable: {sqlite_path}"
                    )
                original_ensure_writable(sqlite_path)

            with patch.dict(
                os.environ,
                {
                    "SQLITE_DB_PATH": str(source_path),
                    "SQLITE_RUNTIME_DIR": str(runtime_root),
                },
                clear=False,
            ):
                with patch("src.utils.config._ensure_sqlite_file_writable", side_effect=fake_ensure_writable):
                    app_config = config.get_config()

            expected_runtime_path = (runtime_root / source_path.name).resolve(strict=False)
            self.assertEqual(app_config.sqlite_path, expected_runtime_path)
            self.assertEqual(app_config.database_url, f"sqlite:///{expected_runtime_path}")
        finally:
            self._cleanup_tree(temp_root)

    def test_get_config_falls_back_to_in_memory_sqlite_when_no_writable_file_path_exists(self) -> None:
        temp_root = TEST_ROOT / f"in_memory_runtime_fallback_{uuid4().hex}"
        source_path = temp_root / "readonly" / "engine_target.db"

        try:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "SQLITE_DB_PATH": str(source_path),
                },
                clear=False,
            ):
                with patch("src.utils.config.prepare_sqlite_runtime", side_effect=config.DatabaseConfigurationError("no writable sqlite path")):
                    app_config = config.get_config()

            self.assertEqual(app_config.database_url, config.SQLITE_IN_MEMORY_URL)
            self.assertEqual(
                app_config.sqlite_path,
                config.get_default_writable_db_path(source_path.name),
            )
        finally:
            self._cleanup_tree(temp_root)

    def test_get_config_uses_sqlite_db_path_for_sqlite_engine_target(self) -> None:
        explicit_path = TEST_ROOT / "explicit" / "engine_target.db"

        try:
            with patch.dict(
                os.environ,
                {
                    "SQLITE_DB_PATH": str(explicit_path),
                    "DATABASE_URL": "sqlite:///should/not/be/used.db",
                },
                clear=False,
            ):
                reloaded_config = reload(config)
                app_config = reloaded_config.get_config()

            self.assertEqual(
                app_config.sqlite_path,
                explicit_path.resolve(strict=False),
            )
            self.assertEqual(
                app_config.database_url,
                f"sqlite:///{explicit_path.resolve(strict=False)}",
            )
        finally:
            reload(config)


if __name__ == "__main__":
    unittest.main()
