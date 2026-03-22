import importlib
import os
import unittest
import uuid
from datetime import date
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from src.data.providers.market_data_provider import MarketDataPayload
from src.data.providers.market_data_provider import ProviderAssetMetadata
from src.data.providers.market_data_provider import ProviderPriceRow


TEST_ROOT = Path(__file__).resolve().parent / ".tmp"


def _reload_market_data_modules(sqlite_path: str):
    os.environ.pop("DATABASE_URL", None)
    os.environ["SQLITE_DB_PATH"] = sqlite_path

    import src.utils.config as config_module
    import src.database.connection as connection_module
    import src.data.storage.repository as repository_module
    import src.data.ingestion.service as service_module

    importlib.reload(config_module)
    connection_module = importlib.reload(connection_module)
    repository_module = importlib.reload(repository_module)
    service_module = importlib.reload(service_module)
    return connection_module, repository_module, service_module


class FakeProvider:
    source_name = "yfinance"
    source_type = "market_data_api"
    source_url = "https://example.test/provider"

    def __init__(self):
        self.call_count = 0

    def fetch_market_data(self, ticker: str, lookback_days: int = 365):
        self.call_count += 1
        fetched_at = datetime.utcnow()
        return MarketDataPayload(
            ticker=ticker,
            provider_name=self.source_name,
            provider_type=self.source_type,
            provider_url=self.source_url,
            metadata=ProviderAssetMetadata(
                ticker=ticker,
                asset_name="Vanguard 500 Index Fund ETF",
                asset_class="ETF",
                exchange="ARCX",
                currency="USD",
                sector=None,
                industry=None,
                country="United States",
            ),
            price_rows=[
                ProviderPriceRow(
                    price_date=date(2026, 3, 20),
                    open_price=Decimal("500.10"),
                    high_price=Decimal("502.10"),
                    low_price=Decimal("499.50"),
                    close_price=Decimal("501.25"),
                    adjusted_close=Decimal("501.25"),
                    volume=1000,
                    ingestion_timestamp=fetched_at,
                )
            ],
            fetched_at=fetched_at,
        )


class MarketDataServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = TEST_ROOT / f"market_data_service_{uuid.uuid4().hex}.db"
        (
            self.connection_module,
            self.repository_module,
            self.service_module,
        ) = _reload_market_data_modules(str(self.sqlite_path))
        self.connection_module.initialize_database(schema_path=Path(__file__).resolve().parents[1] / "sql" / "schema.sql")
        self.provider = FakeProvider()
        self.service = self.service_module.MarketDataIngestionService(
            provider=self.provider,
            repository=self.repository_module.MarketDataRepository(),
        )

    def tearDown(self) -> None:
        self.connection_module.ENGINE.dispose()
        for suffix in ("", "-wal", "-shm", "-journal"):
            path = Path(f"{self.sqlite_path}{suffix}")
            if path.exists():
                path.unlink()

    def test_ingestion_service_writes_and_then_hits_cache(self) -> None:
        first_result = self.service.ingest_ticker("VOO", lookback_days=30)
        second_result = self.service.ingest_ticker("VOO", lookback_days=30)

        self.assertTrue(first_result.success)
        self.assertEqual(first_result.status, "ingested")
        self.assertEqual(second_result.status, "cache_hit")
        self.assertEqual(self.provider.call_count, 1)

        repository = self.repository_module.MarketDataRepository()
        metadata = repository.get_asset_metadata("VOO")
        price_rows = repository.get_price_history("VOO")

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["ticker"], "VOO")
        self.assertEqual(len(price_rows), 1)


if __name__ == "__main__":
    unittest.main()
