"""
Bootstrap script for loading a starter market-data universe into the database.
"""

from __future__ import annotations

from pathlib import Path

from src.database.connection import initialize_database
from src.database.connection import session_scope
from src.database.loaders import load_historical_prices
from src.database.loaders import upsert_asset_metadata
from src.ingestion.market_data import YFINANCE_SOURCE_NAME
from src.ingestion.market_data import YFINANCE_SOURCE_TYPE
from src.ingestion.market_data import YFINANCE_SOURCE_URL
from src.ingestion.market_data import YFinanceMarketDataClient


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
    client = YFinanceMarketDataClient()
    universe = tickers or DEFAULT_STARTER_UNIVERSE

    assets_loaded = 0
    price_rows_loaded = 0

    with session_scope() as session:
        for ticker in universe:
            metadata = client.fetch_asset_metadata(ticker).as_dict()
            price_rows = client.fetch_normalized_price_rows(
                ticker=ticker,
                lookback_days=lookback_days,
            )

            upsert_asset_metadata(
                session=session,
                assets=[metadata],
                source_name=YFINANCE_SOURCE_NAME,
                source_type=YFINANCE_SOURCE_TYPE,
                source_url=YFINANCE_SOURCE_URL,
            )
            load_historical_prices(
                session=session,
                ticker=metadata["ticker"],
                price_rows=price_rows,
                source_name=YFINANCE_SOURCE_NAME,
                source_type=YFINANCE_SOURCE_TYPE,
                source_url=YFINANCE_SOURCE_URL,
            )

            assets_loaded += 1
            price_rows_loaded += len(price_rows)

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

