"""
Central market-data ingestion service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.providers.market_data_provider import EmptyProviderResponseError
from src.data.providers.market_data_provider import InvalidTickerError
from src.data.providers.market_data_provider import ProviderRateLimitError
from src.data.providers.market_data_provider import ProviderUnavailableError
from src.data.providers.market_data_provider import YFinanceMarketDataProvider
from src.data.storage.repository import MarketDataRepository
from src.database.connection import initialize_database


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    success: bool
    ticker: str
    status: str
    message: str
    source: str
    records_written: int = 0
    cache_status: str = "miss"


class MarketDataIngestionService:
    """Coordinates provider fetches, cache checks, normalization, and persistence."""

    def __init__(
        self,
        provider: YFinanceMarketDataProvider | None = None,
        repository: MarketDataRepository | None = None,
    ) -> None:
        self.provider = provider or YFinanceMarketDataProvider()
        self.repository = repository or MarketDataRepository()

    def ingest_ticker(
        self,
        ticker: str,
        lookback_days: int = 365,
    ) -> IngestionResult:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return IngestionResult(
                success=False,
                ticker="",
                status="invalid_ticker",
                message="Enter a ticker symbol before attempting to load an asset.",
                source="validation",
            )

        initialize_database()

        cache_snapshot = self.repository.get_cached_market_data_snapshot(normalized_ticker)
        if cache_snapshot.is_cache_hit:
            LOGGER.info("Cache hit for %s", normalized_ticker)
            return IngestionResult(
                success=True,
                ticker=normalized_ticker,
                status="cache_hit",
                message=f"{normalized_ticker} was loaded from the SQL cache.",
                source="sql_cache",
                records_written=0,
                cache_status="hit",
            )

        LOGGER.info("Cache miss for %s", normalized_ticker)

        try:
            payload = self.provider.fetch_market_data(
                ticker=normalized_ticker,
                lookback_days=lookback_days,
            )
        except InvalidTickerError as exc:
            self.repository.record_ingestion_failure(
                ticker=normalized_ticker,
                provider_name=self.provider.source_name,
                fetch_status="invalid_ticker",
                error_message=str(exc),
                cache_status="miss",
            )
            return IngestionResult(
                success=False,
                ticker=normalized_ticker,
                status="invalid_ticker",
                message=f"{normalized_ticker} is not a valid or supported ticker.",
                source="provider",
            )
        except EmptyProviderResponseError as exc:
            self.repository.record_ingestion_failure(
                ticker=normalized_ticker,
                provider_name=self.provider.source_name,
                fetch_status="empty_response",
                error_message=str(exc),
                cache_status="miss",
            )
            return IngestionResult(
                success=False,
                ticker=normalized_ticker,
                status="empty_response",
                message=str(exc),
                source="provider",
            )
        except ProviderRateLimitError as exc:
            self.repository.record_ingestion_failure(
                ticker=normalized_ticker,
                provider_name=self.provider.source_name,
                fetch_status="rate_limited",
                error_message=str(exc),
                cache_status="miss",
            )
            return IngestionResult(
                success=False,
                ticker=normalized_ticker,
                status="rate_limited",
                message=(
                    "Yahoo market data is temporarily rate limited for this ticker. "
                    f"Please wait a minute and try again. Ticker: {normalized_ticker}."
                ),
                source="provider",
            )
        except ProviderUnavailableError as exc:
            self.repository.record_ingestion_failure(
                ticker=normalized_ticker,
                provider_name=self.provider.source_name,
                fetch_status="provider_unavailable",
                error_message=str(exc),
                cache_status="miss",
            )
            return IngestionResult(
                success=False,
                ticker=normalized_ticker,
                status="provider_unavailable",
                message=f"Market-data provider unavailable for {normalized_ticker}. Detail: {exc}",
                source="provider",
            )
        except Exception as exc:
            self.repository.record_ingestion_failure(
                ticker=normalized_ticker,
                provider_name=self.provider.source_name,
                fetch_status="provider_error",
                error_message=str(exc),
                cache_status="miss",
            )
            return IngestionResult(
                success=False,
                ticker=normalized_ticker,
                status="provider_error",
                message=f"Unable to resolve or download market data for {normalized_ticker}. Detail: {exc}",
                source="provider",
            )

        try:
            persistence = self.repository.persist_market_data(payload, cache_status="miss")
        except Exception as exc:
            LOGGER.exception("SQL write failure while persisting %s", normalized_ticker)
            self.repository.record_ingestion_failure(
                ticker=normalized_ticker,
                provider_name=self.provider.source_name,
                fetch_status="sql_write_failure",
                error_message=str(exc),
                cache_status="miss",
            )
            return IngestionResult(
                success=False,
                ticker=normalized_ticker,
                status="sql_write_failure",
                message=(
                    f"Failed to write {normalized_ticker} into the local database. "
                    f"Detail: {exc}"
                ),
                source="storage",
            )

        return IngestionResult(
            success=True,
            ticker=normalized_ticker,
            status="ingested",
            message=(
                f"{normalized_ticker} was fetched from the provider and stored in the local database."
            ),
            source="provider",
            records_written=int(persistence["records_written"]),
            cache_status="miss",
        )
