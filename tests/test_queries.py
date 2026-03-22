"""
Tests for database query helpers.
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


class QueryTests(unittest.TestCase):
    """Validate read-side query helpers against a realistic seed dataset."""

    def setUp(self) -> None:
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = TEST_ROOT / f"query_test_{uuid.uuid4().hex}.db"
        (
            self.connection_module,
            self.loaders_module,
            self.queries_module,
        ) = _reload_database_modules(
            sqlite_path=str(self.sqlite_path),
        )
        self.connection_module.initialize_database()

        with self.connection_module.session_scope() as session:
            self.loaders_module.upsert_asset_metadata(
                session=session,
                assets=[
                    {
                        "ticker": "AAPL",
                        "asset_name": "Apple Inc.",
                        "asset_class": "EQUITY",
                        "exchange": "NMS",
                        "currency": "USD",
                        "sector": "Technology",
                        "industry": "Consumer Electronics",
                        "country": "United States",
                    },
                    {
                        "ticker": "SPY",
                        "asset_name": "SPDR S&P 500 ETF Trust",
                        "asset_class": "ETF",
                        "exchange": "ARCX",
                        "currency": "USD",
                        "sector": None,
                        "industry": None,
                        "country": "United States",
                    },
                ],
                source_name="unit_test_feed",
                source_type="market_data",
            )
            self.loaders_module.load_historical_prices(
                session=session,
                ticker="AAPL",
                source_name="unit_test_feed",
                source_type="market_data",
                price_rows=[
                    {
                        "price_date": "2026-01-02",
                        "open_price": Decimal("248.10"),
                        "high_price": Decimal("250.20"),
                        "low_price": Decimal("246.90"),
                        "close_price": Decimal("249.75"),
                        "adjusted_close": Decimal("249.75"),
                        "volume": 52000000,
                    },
                    {
                        "price_date": "2026-01-03",
                        "open_price": Decimal("250.00"),
                        "high_price": Decimal("251.80"),
                        "low_price": Decimal("248.50"),
                        "close_price": Decimal("251.10"),
                        "adjusted_close": Decimal("251.10"),
                        "volume": 48750000,
                    },
                ],
            )

    def tearDown(self) -> None:
        self.connection_module.ENGINE.dispose()
        if self.sqlite_path.exists():
            self.sqlite_path.unlink()

    def test_get_asset_list_returns_loaded_assets(self) -> None:
        """Loaded assets should be available through the query layer."""

        with self.connection_module.session_scope() as session:
            assets = self.queries_module.get_asset_list(session)

        tickers = [asset["ticker"] for asset in assets]
        self.assertEqual(tickers, ["AAPL", "SPY"])

    def test_get_price_history_returns_expected_structure(self) -> None:
        """Price history query output should provide stable loader-facing fields."""

        with self.connection_module.session_scope() as session:
            history = self.queries_module.get_price_history(session, "aapl")

        self.assertEqual(len(history), 2)
        self.assertEqual(
            sorted(history[0].keys()),
            sorted(
                [
                    "price_date",
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                    "adjusted_close",
                    "volume",
                    "source_name",
                    "ingestion_timestamp",
                ]
            ),
        )
        self.assertEqual(history[0]["source_name"], "unit_test_feed")
        self.assertEqual(history[0]["close_price"], Decimal("249.750000"))


if __name__ == "__main__":
    unittest.main()
