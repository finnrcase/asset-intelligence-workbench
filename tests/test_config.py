import os
import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from src.utils import config


TEST_ROOT = Path(__file__).resolve().parent / ".tmp"
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class ConfigTests(unittest.TestCase):
    def test_resolve_sqlite_path_prefers_explicit_env_override(self) -> None:
        explicit_path = TEST_ROOT / "explicit" / "asset_intelligence.db"

        with patch.dict(os.environ, {"SQLITE_DB_PATH": str(explicit_path)}, clear=False):
            resolved = config._resolve_sqlite_path()

        self.assertEqual(resolved, explicit_path)

    def test_resolve_sqlite_path_uses_database_url_when_present(self) -> None:
        database_url = "sqlite:///tmp/custom_app.db"

        with patch.dict(os.environ, {}, clear=False):
            resolved = config._resolve_sqlite_path(database_url)

        self.assertEqual(resolved, Path("tmp/custom_app.db"))

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

    def test_validate_sqlite_runtime_raises_clear_error_when_file_is_not_writable(self) -> None:
        sqlite_path = TEST_ROOT / "write_probe.db"

        with patch("src.utils.config.sqlite3.connect", side_effect=sqlite3.OperationalError("readonly")):
            with self.assertRaises(config.DatabaseConfigurationError) as context:
                config.validate_sqlite_runtime(
                    sqlite_path=sqlite_path,
                    database_url=f"sqlite:///{sqlite_path}",
                )

        self.assertIn(str(sqlite_path), str(context.exception))
        self.assertIn("not writable", str(context.exception))


if __name__ == "__main__":
    unittest.main()
