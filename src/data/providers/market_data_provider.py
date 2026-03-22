"""
Centralized market-data provider access.

This module is the only place in the application that talks to the external
market-data API.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from decimal import Decimal
from typing import Any

import yfinance as yf


LOGGER = logging.getLogger(__name__)

YFINANCE_SOURCE_NAME = "yfinance"
YFINANCE_SOURCE_TYPE = "market_data_api"
YFINANCE_SOURCE_URL = "https://finance.yahoo.com/"


class MarketDataProviderError(RuntimeError):
    """Base provider error."""


class InvalidTickerError(MarketDataProviderError):
    """Raised when a ticker request is invalid."""


class ProviderRateLimitError(MarketDataProviderError):
    """Raised when the upstream provider rate limits requests."""


class ProviderUnavailableError(MarketDataProviderError):
    """Raised when the provider is unavailable or returns unusable data."""


class EmptyProviderResponseError(MarketDataProviderError):
    """Raised when the provider returns no usable market data."""


@dataclass(frozen=True)
class ProviderAssetMetadata:
    ticker: str
    asset_name: str
    asset_class: str | None
    exchange: str | None
    currency: str
    sector: str | None
    industry: str | None
    country: str | None
    is_active: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderPriceRow:
    price_date: date
    open_price: Decimal | None
    high_price: Decimal | None
    low_price: Decimal | None
    close_price: Decimal
    adjusted_close: Decimal | None
    volume: int | None
    ingestion_timestamp: datetime

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketDataPayload:
    ticker: str
    provider_name: str
    provider_type: str
    provider_url: str
    metadata: ProviderAssetMetadata
    price_rows: list[ProviderPriceRow]
    fetched_at: datetime

    @property
    def coverage_start_date(self) -> date | None:
        return self.price_rows[0].price_date if self.price_rows else None

    @property
    def coverage_end_date(self) -> date | None:
        return self.price_rows[-1].price_date if self.price_rows else None


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols to the canonical project format."""

    normalized = ticker.strip().upper()
    if not normalized:
        raise InvalidTickerError("Enter a ticker symbol before attempting ingestion.")
    return normalized


def _normalize_text(value: Any) -> str | None:
    if value in (None, "", "None", "nan"):
        return None
    return str(value).strip()


def _to_decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(round(float(value), 6)))
    except Exception:
        return None


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


class YFinanceMarketDataProvider:
    """Provider adapter that fetches and normalizes yfinance market data."""

    source_name = YFINANCE_SOURCE_NAME
    source_type = YFINANCE_SOURCE_TYPE
    source_url = YFINANCE_SOURCE_URL

    def fetch_market_data(
        self,
        ticker: str,
        lookback_days: int = 365,
        end_date: date | None = None,
    ) -> MarketDataPayload:
        normalized_ticker = normalize_ticker(ticker)
        LOGGER.info("Market data provider call started for %s", normalized_ticker)

        instrument = yf.Ticker(normalized_ticker)
        fetched_at = datetime.utcnow()

        try:
            metadata = self._fetch_metadata(instrument, normalized_ticker)
            price_rows = self._fetch_price_rows(
                instrument=instrument,
                ticker=normalized_ticker,
                lookback_days=lookback_days,
                end_date=end_date,
                fetched_at=fetched_at,
            )
        except MarketDataProviderError:
            raise
        except Exception as exc:
            raise self._map_provider_exception(exc) from exc

        LOGGER.info(
            "Market data provider call completed for %s with %s rows",
            normalized_ticker,
            len(price_rows),
        )
        return MarketDataPayload(
            ticker=normalized_ticker,
            provider_name=self.source_name,
            provider_type=self.source_type,
            provider_url=self.source_url,
            metadata=metadata,
            price_rows=price_rows,
            fetched_at=fetched_at,
        )

    def _fetch_metadata(self, instrument: Any, ticker: str) -> ProviderAssetMetadata:
        info = getattr(instrument, "info", {}) or {}
        fast_info = getattr(instrument, "fast_info", {}) or {}
        history_metadata = getattr(instrument, "history_metadata", {}) or {}

        asset_name = (
            info.get("shortName")
            or info.get("longName")
            or history_metadata.get("instrumentType")
            or ticker
        )
        if not asset_name:
            raise EmptyProviderResponseError(f"{ticker} did not return usable asset metadata.")

        return ProviderAssetMetadata(
            ticker=ticker,
            asset_name=str(asset_name),
            asset_class=_normalize_text(
                info.get("quoteType")
                or info.get("instrumentType")
                or history_metadata.get("instrumentType")
            ),
            exchange=_normalize_text(info.get("exchange") or fast_info.get("exchange")),
            currency=str(
                info.get("currency")
                or fast_info.get("currency")
                or history_metadata.get("currency")
                or "USD"
            ).upper(),
            sector=_normalize_text(info.get("sector")),
            industry=_normalize_text(info.get("industry")),
            country=_normalize_text(info.get("country")),
            is_active=not bool(info.get("exchangeDataDelayedBy") == -1),
        )

    def _fetch_price_rows(
        self,
        instrument: Any,
        ticker: str,
        lookback_days: int,
        end_date: date | None,
        fetched_at: datetime,
    ) -> list[ProviderPriceRow]:
        resolved_end_date = end_date or date.today()
        resolved_start_date = resolved_end_date.fromordinal(
            resolved_end_date.toordinal() - lookback_days
        )

        history = instrument.history(
            start=resolved_start_date.isoformat(),
            end=(resolved_end_date.fromordinal(resolved_end_date.toordinal() + 1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
        )
        if history.empty:
            raise EmptyProviderResponseError(
                f"No daily price history returned for ticker '{ticker}'."
            )

        normalized_rows: list[ProviderPriceRow] = []
        normalized_history = history.copy()
        normalized_history.index = normalized_history.index.tz_localize(None)

        for price_timestamp, row in normalized_history.iterrows():
            close_price = _to_decimal_or_none(row.get("Close"))
            if close_price is None:
                continue
            normalized_rows.append(
                ProviderPriceRow(
                    price_date=price_timestamp.date(),
                    open_price=_to_decimal_or_none(row.get("Open")),
                    high_price=_to_decimal_or_none(row.get("High")),
                    low_price=_to_decimal_or_none(row.get("Low")),
                    close_price=close_price,
                    adjusted_close=_to_decimal_or_none(row.get("Adj Close")),
                    volume=_to_int_or_none(row.get("Volume")),
                    ingestion_timestamp=fetched_at,
                )
            )

        if not normalized_rows:
            raise EmptyProviderResponseError(
                f"{ticker} did not return daily historical price data."
            )

        return normalized_rows

    @staticmethod
    def _map_provider_exception(exc: Exception) -> MarketDataProviderError:
        detail = str(exc).strip() or exc.__class__.__name__
        normalized = detail.lower()
        if any(token in normalized for token in ("too many requests", "rate limited", "429")):
            return ProviderRateLimitError(detail)
        if any(token in normalized for token in ("not found", "404", "invalid ticker")):
            return InvalidTickerError(detail)
        return ProviderUnavailableError(detail)
