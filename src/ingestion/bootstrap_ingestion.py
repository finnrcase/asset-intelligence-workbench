"""
Bootstrap script for loading a starter market-data universe into the database.
"""

from __future__ import annotations

from pathlib import Path

from src.data.ingestion.service import MarketDataIngestionService
from src.database.connection import initialize_database


DEFAULT_STARTER_UNIVERSE = ["AAPL", "MSFT", "SPY", "QQQ", "BTC-USD"]
DEFAULT_LOOKBACK_DAYS = 365
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def bootstrap_market_data_ingestion(
    tickers: list[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, int]:
    """
    Initialize the schema and ingest a starter universe of market data.

    Returns a concise summary dictionary that can be used by CLI wrappers,
    notebooks, or tests.
    """

    initialize_database(schema_path=SCHEMA_PATH)
    service = MarketDataIngestionService()
    universe = tickers or DEFAULT_STARTER_UNIVERSE

    assets_loaded = 0
    price_rows_loaded = 0

    for ticker in universe:
        result = service.ingest_ticker(ticker, lookback_days=lookback_days)
        if result.success:
            assets_loaded += 1
            price_rows_loaded += result.records_written

    summary = {
        "tickers_processed": len(universe),
        "assets_loaded": assets_loaded,
        "price_rows_loaded": price_rows_loaded,
        "lookback_days": lookback_days,
    }
    return summary


def main() -> None:
    """Run the starter ingestion workflow and print a concise summary."""

    summary = bootstrap_market_data_ingestion()
    print(
        "Loaded {assets_loaded} assets across {tickers_processed} tickers "
        "with {price_rows_loaded} daily price rows (lookback={lookback_days} days).".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
