import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from src.utils import config

TEST_ROOT = Path(__file__).resolve().parent / ".tmp"
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class ConfigTests(unittest.TestCase):
    def test_resolve_sqlite_path_prefers_explicit_env_override(self) -> None:
        explicit_path = Path(tempfile.gettempdir()) / "custom_asset_intelligence.db"

        with patch.dict(os.environ, {"SQLITE_DB_PATH": str(explicit_path)}, clear=False):
            resolved = config._resolve_sqlite_path()

        self.assertEqual(resolved, explicit_path)

    def test_resolve_sqlite_path_falls_back_and_copies_seed_db_when_default_is_not_writable(self) -> None:
        temp_root = TEST_ROOT / f"config_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=True)
        source_db = temp_root / "source.db"
        fallback_db = temp_root / "fallback" / "asset_intelligence.db"

        try:
            source_db.write_text("seed-db", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=False):
                with patch.object(config, "DEFAULT_SQLITE_PATH", source_db):
                    with patch("src.utils.config._is_path_writable", return_value=False):
                        with patch("src.utils.config._fallback_sqlite_path", return_value=fallback_db):
                            resolved = config._resolve_sqlite_path()

            self.assertEqual(resolved, fallback_db)
            self.assertTrue(fallback_db.exists())
            self.assertEqual(fallback_db.read_text(encoding="utf-8"), "seed-db")
        finally:
            for path in sorted(temp_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temp_root.rmdir()

    def test_existing_sqlite_file_that_fails_write_probe_is_treated_as_not_writable(self) -> None:
        temp_root = TEST_ROOT / f"config_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=True)
        db_path = temp_root / "seed.db"

        try:
            db_path.write_text("seed-db", encoding="utf-8")
            with patch("src.utils.config.sqlite3.connect", side_effect=OSError("readonly")):
                self.assertFalse(config._is_path_writable(db_path))
        finally:
            for path in sorted(temp_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temp_root.rmdir()


if __name__ == "__main__":
    unittest.main()
