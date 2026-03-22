"""
Tests for database loader behavior.
"""

from __future__ import annotations

import importlib
import os
import unittest
import uuid
from decimal import Decimal
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent / ".tmp"


def _reload_database_modules(sqlite_path: str):
    os.environ.pop("DATABASE_URL", None)
    os.environ["SQLITE_DB_PATH"] = sqlite_path

    import src.utils.config as config_module
    import src.database.connection as connection_module
    import src.database.loaders as loaders_module
    import src.database.queries as queries_module

    importlib.reload(config_module)
    connection_module = importlib.reload(connection_module)
    loaders_module = importlib.reload(loaders_module)
    queries_module = importlib.reload(queries_module)
    return connection_module, loaders_module, queries_module


class LoaderTests(unittest.TestCase):
    """Validate write-side asset and price loading behavior."""

    def setUp(self) -> None:
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = TEST_ROOT / f"loader_test_{uuid.uuid4().hex}.db"
        (
            self.connection_module,
            self.loaders_module,
            self.queries_module,
        ) = _reload_database_modules(
            sqlite_path=str(self.sqlite_path),
        )
        self.connection_module.initialize_database()

    def tearDown(self) -> None:
        self.connection_module.ENGINE.dispose()
        if self.sqlite_path.exists():
            self.sqlite_path.unlink()

    def test_upsert_asset_metadata_creates_asset_record(self) -> None:
        """Asset metadata loads should create a retrievable security master record."""

        with self.connection_module.session_scope() as session:
            self.loaders_module.upsert_asset_metadata(
                session=session,
                assets=[
                    {
                        "ticker": " aapl ",
                        "asset_name": "Apple Inc.",
                        "asset_class": "EQUITY",
                        "exchange": "NMS",
                        "currency": "USD",
                        "sector": "Technology",
                        "industry": "Consumer Electronics",
                        "country": "United States",
                    }
                ],
                source_name="unit_test_feed",
                source_type="market_data",
            )

        with self.connection_module.session_scope() as session:
            metadata = self.queries_module.get_asset_metadata(session, "AAPL")

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["ticker"], "AAPL")
        self.assertEqual(metadata["asset_name"], "Apple Inc.")
        self.assertEqual(metadata["currency"], "USD")
        self.assertEqual(metadata["primary_source"], "unit_test_feed")

    def test_load_historical_prices_upserts_duplicate_dates(self) -> None:
        """Duplicate daily rows should update existing observations, not multiply them."""

        with self.connection_module.session_scope() as session:
            self.loaders_module.upsert_asset_metadata(
                session=session,
                assets=[
                    {
                        "ticker": "MSFT",
                        "asset_name": "Microsoft Corporation",
                        "currency": "USD",
                    }
                ],
                source_name="unit_test_feed",
                source_type="market_data",
            )
            self.loaders_module.load_historical_prices(
                session=session,
                ticker="MSFT",
                source_name="unit_test_feed",
                source_type="market_data",
                price_rows=[
                    {
                        "price_date": "2026-01-02",
                        "open_price": Decimal("420.10"),
                        "high_price": Decimal("425.00"),
                        "low_price": Decimal("419.40"),
                        "close_price": Decimal("424.75"),
                        "adjusted_close": Decimal("424.75"),
                        "volume": 15000000,
                    }
                ],
            )
            self.loaders_module.load_historical_prices(
                session=session,
                ticker="msft",
                source_name="unit_test_feed",
                source_type="market_data",
                price_rows=[
                    {
                        "price_date": "2026-01-02",
                        "open_price": Decimal("421.00"),
                        "high_price": Decimal("426.50"),
                        "low_price": Decimal("420.00"),
                        "close_price": Decimal("425.95"),
                        "adjusted_close": Decimal("425.95"),
                        "volume": 15100000,
                    }
                ],
            )

        with self.connection_module.session_scope() as session:
            history = self.queries_module.get_price_history(session, "MSFT")

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["close_price"], Decimal("425.950000"))
        self.assertEqual(history[0]["volume"], 15100000)


if __name__ == "__main__":
    unittest.main()
