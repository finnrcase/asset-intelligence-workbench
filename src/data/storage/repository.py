"""
Storage repository for market-data ingestion and SQL-backed reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy import select

from src.data.providers.market_data_provider import MarketDataPayload
from src.database.connection import Asset
from src.database.connection import DataSource
from src.database.connection import HistoricalPrice
from src.database.connection import IngestionRunLog
from src.database.connection import MarketDataIngestionState
from src.database.connection import session_scope
from src.utils.config import get_config


LOGGER = logging.getLogger(__name__)
CONFIG = get_config()


@dataclass(frozen=True)
class CachedMarketDataSnapshot:
    ticker: str
    metadata: dict[str, Any] | None
    price_rows: list[dict[str, Any]]
    metadata_fresh: bool
    price_fresh: bool
    last_successful_fetch_at: datetime | None

    @property
    def is_cache_hit(self) -> bool:
        return self.metadata is not None and bool(self.price_rows) and self.metadata_fresh and self.price_fresh


class MarketDataRepository:
    """Repository that owns market-data SQL persistence and cache reads."""

    def get_cached_market_data_snapshot(self, ticker: str) -> CachedMarketDataSnapshot:
        normalized_ticker = ticker.strip().upper()
        with session_scope() as session:
            metadata = self._get_asset_metadata(session, normalized_ticker)
            price_rows = self._get_price_history(session, normalized_ticker)
            state = session.scalar(
                select(MarketDataIngestionState).where(
                    MarketDataIngestionState.ticker == normalized_ticker,
                    MarketDataIngestionState.provider_name == "yfinance",
                )
            )
            metadata_fetched_at = state.metadata_fetched_at if state else None
            prices_fetched_at = state.prices_fetched_at if state else None
            last_successful_fetch_at = state.last_successful_fetch_at if state else None

        metadata_fresh = self._is_fresh(
            timestamp=metadata_fetched_at,
            freshness_hours=CONFIG.market_data_metadata_freshness_hours,
        )
        price_fresh = self._is_fresh(
            timestamp=prices_fetched_at,
            freshness_hours=CONFIG.market_data_prices_freshness_hours,
        )
        return CachedMarketDataSnapshot(
            ticker=normalized_ticker,
            metadata=metadata,
            price_rows=price_rows,
            metadata_fresh=metadata_fresh,
            price_fresh=price_fresh,
            last_successful_fetch_at=last_successful_fetch_at,
        )

    def persist_market_data(
        self,
        payload: MarketDataPayload,
        cache_status: str,
    ) -> dict[str, Any]:
        normalized_ticker = payload.ticker.strip().upper()
        LOGGER.info("Persisting %s market-data rows for %s", len(payload.price_rows), normalized_ticker)
        with session_scope() as session:
            data_source = self._get_or_create_data_source(
                session=session,
                source_name=payload.provider_name,
                source_type=payload.provider_type,
                source_url=payload.provider_url,
            )
            asset = self._upsert_asset_metadata(session, payload, data_source.id)
            records_written = self._upsert_price_history(session, asset.id, data_source.id, payload)
            self._upsert_ingestion_state(
                session=session,
                payload=payload,
                fetch_status="success",
                error_message=None,
            )
            self._record_ingestion_run(
                session=session,
                payload=payload,
                fetch_status="success",
                cache_status=cache_status,
                error_message=None,
                records_written=records_written,
            )

        LOGGER.info("Persisted %s records for %s", records_written, normalized_ticker)
        return {
            "ticker": normalized_ticker,
            "records_written": records_written,
            "coverage_start_date": payload.coverage_start_date,
            "coverage_end_date": payload.coverage_end_date,
        }

    def record_ingestion_failure(
        self,
        ticker: str,
        provider_name: str,
        fetch_status: str,
        error_message: str,
        cache_status: str,
    ) -> None:
        normalized_ticker = ticker.strip().upper()
        now = datetime.utcnow()
        with session_scope() as session:
            state = session.scalar(
                select(MarketDataIngestionState).where(
                    MarketDataIngestionState.ticker == normalized_ticker,
                    MarketDataIngestionState.provider_name == provider_name,
                )
            )
            if state is None:
                state = MarketDataIngestionState(
                    ticker=normalized_ticker,
                    provider_name=provider_name,
                )
                session.add(state)
            state.request_timestamp = now
            state.fetch_status = fetch_status
            state.error_message = error_message
            state.updated_at = now

            session.add(
                IngestionRunLog(
                    ticker=normalized_ticker,
                    provider_name=provider_name,
                    started_at=now,
                    completed_at=now,
                    fetch_status=fetch_status,
                    error_message=error_message,
                    cache_status=cache_status,
                )
            )

    def list_available_assets(self) -> list[dict[str, Any]]:
        with session_scope() as session:
            return [
                {
                    "ticker": asset.ticker,
                    "asset_name": asset.asset_name,
                    "asset_class": asset.asset_class,
                    "exchange": asset.exchange,
                    "currency": asset.currency,
                    "is_active": asset.is_active,
                }
                for asset in session.scalars(
                    select(Asset).where(Asset.is_active.is_(True)).order_by(Asset.ticker.asc())
                ).all()
            ]

    def get_asset_metadata(self, ticker: str) -> dict[str, Any] | None:
        with session_scope() as session:
            return self._get_asset_metadata(session, ticker.strip().upper())

    def get_price_history(self, ticker: str) -> list[dict[str, Any]]:
        with session_scope() as session:
            return self._get_price_history(session, ticker.strip().upper())

    def _get_asset_metadata(self, session, ticker: str) -> dict[str, Any] | None:
        row = session.execute(
            select(Asset, DataSource)
            .outerjoin(DataSource, Asset.primary_source_id == DataSource.id)
            .where(Asset.ticker == ticker)
        ).first()
        if row is None:
            return None
        asset, data_source = row
        return {
            "ticker": asset.ticker,
            "asset_name": asset.asset_name,
            "asset_class": asset.asset_class,
            "exchange": asset.exchange,
            "currency": asset.currency,
            "sector": asset.sector,
            "industry": asset.industry,
            "country": asset.country,
            "is_active": asset.is_active,
            "primary_source": data_source.source_name if data_source else None,
            "created_at": asset.created_at,
            "updated_at": asset.updated_at,
        }

    def _get_price_history(self, session, ticker: str) -> list[dict[str, Any]]:
        rows = session.execute(
            select(HistoricalPrice, DataSource)
            .join(Asset, HistoricalPrice.asset_id == Asset.id)
            .join(DataSource, HistoricalPrice.source_id == DataSource.id)
            .where(Asset.ticker == ticker)
            .order_by(HistoricalPrice.price_date.asc())
        ).all()
        return [
            {
                "price_date": price.price_date,
                "open_price": price.open_price,
                "high_price": price.high_price,
                "low_price": price.low_price,
                "close_price": price.close_price,
                "adjusted_close": price.adjusted_close,
                "volume": price.volume,
                "source_name": data_source.source_name,
                "ingestion_timestamp": price.ingestion_timestamp,
            }
            for price, data_source in rows
        ]

    def _get_or_create_data_source(
        self,
        session,
        source_name: str,
        source_type: str,
        source_url: str | None,
    ) -> DataSource:
        existing = session.scalar(
            select(DataSource).where(DataSource.source_name == source_name)
        )
        if existing is not None:
            existing.source_type = source_type
            existing.source_url = source_url
            existing.updated_at = datetime.utcnow()
            session.flush()
            return existing

        data_source = DataSource(
            source_name=source_name,
            source_type=source_type,
            source_url=source_url,
        )
        session.add(data_source)
        session.flush()
        return data_source

    def _upsert_asset_metadata(self, session, payload: MarketDataPayload, primary_source_id: int) -> Asset:
        asset = session.scalar(select(Asset).where(Asset.ticker == payload.ticker))
        if asset is None:
            asset = Asset(ticker=payload.ticker, asset_name=payload.metadata.asset_name)
            session.add(asset)

        asset.asset_name = payload.metadata.asset_name
        asset.asset_class = payload.metadata.asset_class
        asset.exchange = payload.metadata.exchange
        asset.currency = payload.metadata.currency
        asset.sector = payload.metadata.sector
        asset.industry = payload.metadata.industry
        asset.country = payload.metadata.country
        asset.is_active = payload.metadata.is_active
        asset.primary_source_id = primary_source_id
        asset.updated_at = payload.fetched_at
        session.flush()
        return asset

    def _upsert_price_history(
        self,
        session,
        asset_id: int,
        source_id: int,
        payload: MarketDataPayload,
    ) -> int:
        records_written = 0
        for row in payload.price_rows:
            existing = session.scalar(
                select(HistoricalPrice).where(
                    HistoricalPrice.asset_id == asset_id,
                    HistoricalPrice.source_id == source_id,
                    HistoricalPrice.price_date == row.price_date,
                )
            )
            price_record = existing or HistoricalPrice(
                asset_id=asset_id,
                source_id=source_id,
                price_date=row.price_date,
                close_price=row.close_price,
            )
            if existing is None:
                session.add(price_record)
            price_record.open_price = row.open_price
            price_record.high_price = row.high_price
            price_record.low_price = row.low_price
            price_record.close_price = row.close_price
            price_record.adjusted_close = row.adjusted_close
            price_record.volume = row.volume
            price_record.ingestion_timestamp = row.ingestion_timestamp
            records_written += 1
        session.flush()
        return records_written

    def _upsert_ingestion_state(
        self,
        session,
        payload: MarketDataPayload,
        fetch_status: str,
        error_message: str | None,
    ) -> None:
        state = session.scalar(
            select(MarketDataIngestionState).where(
                MarketDataIngestionState.ticker == payload.ticker,
                MarketDataIngestionState.provider_name == payload.provider_name,
            )
        )
        if state is None:
            state = MarketDataIngestionState(
                ticker=payload.ticker,
                provider_name=payload.provider_name,
            )
            session.add(state)

        state.asset_type = payload.metadata.asset_class
        state.request_timestamp = payload.fetched_at
        state.last_successful_fetch_at = payload.fetched_at
        state.fetch_status = fetch_status
        state.error_message = error_message
        state.record_count_fetched = len(payload.price_rows)
        state.coverage_start_date = payload.coverage_start_date
        state.coverage_end_date = payload.coverage_end_date
        state.metadata_fetched_at = payload.fetched_at
        state.prices_fetched_at = payload.fetched_at
        state.updated_at = payload.fetched_at

    def _record_ingestion_run(
        self,
        session,
        payload: MarketDataPayload,
        fetch_status: str,
        cache_status: str,
        error_message: str | None,
        records_written: int,
    ) -> None:
        session.add(
            IngestionRunLog(
                ticker=payload.ticker,
                provider_name=payload.provider_name,
                started_at=payload.fetched_at,
                completed_at=datetime.utcnow(),
                fetch_status=fetch_status,
                error_message=error_message,
                cache_status=cache_status,
                record_count_fetched=len(payload.price_rows),
                records_written=records_written,
                coverage_start_date=payload.coverage_start_date,
                coverage_end_date=payload.coverage_end_date,
            )
        )

    @staticmethod
    def _is_fresh(timestamp: datetime | None, freshness_hours: int) -> bool:
        if timestamp is None:
            return False
        return timestamp >= datetime.utcnow() - timedelta(hours=freshness_hours)
