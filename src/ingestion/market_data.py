"""
Market data ingestion helpers built around a provider abstraction.

The first implementation uses yfinance for local development, but the module is
structured so another upstream source can be added later without rewriting the
loader layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import yfinance as yf


YFINANCE_SOURCE_NAME = "yfinance"
YFINANCE_SOURCE_TYPE = "market_data_api"
YFINANCE_SOURCE_URL = "https://finance.yahoo.com/"


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols to a consistent storage format."""

    return ticker.strip().upper()


@dataclass(frozen=True)
class ProviderAssetMetadata:
    """Normalized asset metadata returned by a market data provider."""

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
        """Return the metadata as a loader-friendly dictionary."""

        return {
            "ticker": self.ticker,
            "asset_name": self.asset_name,
            "asset_class": self.asset_class,
            "exchange": self.exchange,
            "currency": self.currency,
            "sector": self.sector,
            "industry": self.industry,
            "country": self.country,
            "is_active": self.is_active,
        }


class YFinanceMarketDataClient:
    """
    Minimal market data client backed by yfinance.

    The interface intentionally returns normalized Python structures so the rest
    of the application does not depend directly on provider-specific response
    shapes.
    """

    source_name = YFINANCE_SOURCE_NAME
    source_type = YFINANCE_SOURCE_TYPE
    source_url = YFINANCE_SOURCE_URL

    def fetch_asset_metadata(self, ticker: str) -> ProviderAssetMetadata:
        """Fetch and normalize metadata for a single ticker."""

        normalized_ticker = normalize_ticker(ticker)
        instrument = yf.Ticker(normalized_ticker)

        info = self._safe_mapping(getattr(instrument, "info", {}))
        fast_info = self._safe_mapping(getattr(instrument, "fast_info", {}))
        history_metadata = self._safe_mapping(
            getattr(instrument, "history_metadata", {}) or {}
        )

        asset_name = (
            info.get("shortName")
            or info.get("longName")
            or history_metadata.get("instrumentType")
            or normalized_ticker
        )
        asset_class = (
            info.get("quoteType")
            or info.get("instrumentType")
            or history_metadata.get("instrumentType")
        )
        exchange = info.get("exchange") or fast_info.get("exchange")
        currency = (
            info.get("currency")
            or fast_info.get("currency")
            or history_metadata.get("currency")
            or "USD"
        )

        return ProviderAssetMetadata(
            ticker=normalized_ticker,
            asset_name=str(asset_name),
            asset_class=self._normalize_text(asset_class),
            exchange=self._normalize_text(exchange),
            currency=str(currency).upper(),
            sector=self._normalize_text(info.get("sector")),
            industry=self._normalize_text(info.get("industry")),
            country=self._normalize_text(info.get("country")),
            is_active=not bool(info.get("exchangeDataDelayedBy") == -1),
        )

    def fetch_daily_price_history(
        self,
        ticker: str,
        lookback_days: int = 365,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch daily price history for a single ticker.

        The returned DataFrame is normalized for downstream loading with these
        columns:
        `price_date`, `open_price`, `high_price`, `low_price`, `close_price`,
        `adjusted_close`, `volume`, `ingestion_timestamp`.
        """

        normalized_ticker = normalize_ticker(ticker)
        resolved_end_date = end_date or date.today()
        resolved_start_date = resolved_end_date - timedelta(days=lookback_days)

        history = yf.Ticker(normalized_ticker).history(
            start=resolved_start_date.isoformat(),
            end=(resolved_end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
        )

        if history.empty:
            raise ValueError(
                f"No daily price history returned for ticker '{normalized_ticker}'."
            )

        return self.normalize_price_history(history)

    @staticmethod
    def normalize_price_history(history: pd.DataFrame) -> pd.DataFrame:
        """Normalize a provider DataFrame into the canonical price schema."""

        normalized = history.copy()
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        normalized = normalized.reset_index().rename(columns={"Date": "price_timestamp"})

        normalized["price_date"] = normalized["price_timestamp"].dt.date
        normalized["open_price"] = normalized["Open"].apply(_to_decimal_or_none)
        normalized["high_price"] = normalized["High"].apply(_to_decimal_or_none)
        normalized["low_price"] = normalized["Low"].apply(_to_decimal_or_none)
        normalized["close_price"] = normalized["Close"].apply(_to_decimal_or_none)
        normalized["adjusted_close"] = normalized.get(
            "Adj Close",
            pd.Series([None] * len(normalized)),
        ).apply(_to_decimal_or_none)
        normalized["volume"] = normalized.get(
            "Volume",
            pd.Series([None] * len(normalized)),
        ).apply(_to_int_or_none)
        normalized["ingestion_timestamp"] = datetime.utcnow()

        normalized = normalized[
            [
                "price_date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "adjusted_close",
                "volume",
                "ingestion_timestamp",
            ]
        ]
        normalized = normalized.dropna(subset=["close_price"])
        normalized = normalized.sort_values("price_date").reset_index(drop=True)
        return normalized

    def fetch_normalized_price_rows(
        self,
        ticker: str,
        lookback_days: int = 365,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch daily price history and return loader-friendly row dictionaries."""

        history = self.fetch_daily_price_history(
            ticker=ticker,
            lookback_days=lookback_days,
            end_date=end_date,
        )
        return history.to_dict(orient="records")

    @staticmethod
    def _safe_mapping(value: Any) -> dict[str, Any]:
        """Convert provider responses into a dictionary when possible."""

        return value if isinstance(value, dict) else {}

    @staticmethod
    def _normalize_text(value: Any) -> str | None:
        """Convert provider string-like values into clean optional text."""

        if value in (None, "", "None", "nan"):
            return None
        return str(value).strip()


def fetch_asset_metadata(ticker: str) -> dict[str, Any]:
    """Fetch normalized asset metadata for a ticker using the default provider."""

    return YFinanceMarketDataClient().fetch_asset_metadata(ticker).as_dict()


def fetch_daily_historical_prices(
    ticker: str,
    lookback_days: int = 365,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Fetch normalized daily price history for a ticker using the default provider."""

    return YFinanceMarketDataClient().fetch_daily_price_history(
        ticker=ticker,
        lookback_days=lookback_days,
        end_date=end_date,
    )


def fetch_daily_historical_price_rows(
    ticker: str,
    lookback_days: int = 365,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    """Fetch loader-ready daily price rows for a ticker using the default provider."""

    return YFinanceMarketDataClient().fetch_normalized_price_rows(
        ticker=ticker,
        lookback_days=lookback_days,
        end_date=end_date,
    )


def _to_decimal_or_none(value: Any) -> Decimal | None:
    """Convert provider numeric values into database-friendly decimals."""

    if pd.isna(value):
        return None
    return Decimal(str(round(float(value), 6)))


def _to_int_or_none(value: Any) -> int | None:
    """Convert provider volume values into nullable integers."""

    if pd.isna(value):
        return None
    return int(value)

