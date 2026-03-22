"""
SQL-backed market-data query interface for the app and reporting layers.
"""

from __future__ import annotations

from src.data.storage.repository import MarketDataRepository


REPOSITORY = MarketDataRepository()


def list_available_assets() -> list[dict]:
    return REPOSITORY.list_available_assets()


def get_asset_metadata(ticker: str) -> dict | None:
    return REPOSITORY.get_asset_metadata(ticker)


def get_price_history(ticker: str) -> list[dict]:
    return REPOSITORY.get_price_history(ticker)
