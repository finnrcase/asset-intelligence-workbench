"""
Finance-oriented feature engineering utilities for return and downside-risk forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_VERSION = "v1"
TRADING_DAYS_PER_YEAR = 252



def _coerce_price_column(frame: pd.DataFrame) -> str:
    """Return the preferred analysis price column available in the frame."""

    for column in ("analysis_price", "adjusted_close", "close_price"):
        if column in frame.columns:
            return column
    raise KeyError(
        "A price column is required. Expected one of: analysis_price, adjusted_close, close_price."
    )



def _prepare_market_frame(price_history: pd.DataFrame) -> pd.DataFrame:
    """Normalize market history into a sorted per-asset time-series frame."""

    if price_history.empty:
        return pd.DataFrame()

    frame = price_history.copy()
    required_columns = {"asset_id", "ticker", "price_date"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required price-history columns: {sorted(missing)}")

    price_column = _coerce_price_column(frame)
    frame["price_date"] = pd.to_datetime(frame["price_date"])
    numeric_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close",
        "analysis_price",
        "volume",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
    frame = frame.dropna(subset=[price_column]).sort_values(["asset_id", "price_date"]).reset_index(drop=True)
    return frame



def build_technical_feature_frame(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Build a readable technical feature frame from stored historical market data.
    """

    frame = _prepare_market_frame(price_history)
    if frame.empty:
        return pd.DataFrame()

    price_column = _coerce_price_column(frame)
    feature_frames: list[pd.DataFrame] = []

    for (_, _), group in frame.groupby(["asset_id", "ticker"], sort=False):
        asset_frame = group.copy()
        prices = asset_frame[price_column].astype(float)
        daily_return = prices.pct_change()
        log_return = np.log(prices).diff()
        running_peak = prices.cummax()
        drawdown = prices / running_peak - 1.0

        asset_frame["feature_date"] = asset_frame["price_date"].dt.date
        asset_frame["analysis_price"] = prices
        asset_frame["daily_return"] = daily_return
        asset_frame["return_lag_1d"] = daily_return.shift(1)
        asset_frame["return_lag_5d"] = daily_return.shift(5)
        asset_frame["return_lag_10d"] = daily_return.shift(10)
        asset_frame["rolling_mean_return_5d"] = daily_return.rolling(5).mean()
        asset_frame["rolling_mean_return_20d"] = daily_return.rolling(20).mean()
        asset_frame["rolling_volatility_10d"] = daily_return.rolling(10).std(ddof=1)
        asset_frame["rolling_volatility_20d"] = daily_return.rolling(20).std(ddof=1)
        asset_frame["realized_volatility_20d"] = (
            log_return.rolling(20).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        asset_frame["recent_realized_volatility_5d"] = (
            log_return.rolling(5).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        asset_frame["momentum_5d"] = prices / prices.shift(5) - 1.0
        asset_frame["momentum_10d"] = prices / prices.shift(10) - 1.0
        asset_frame["momentum_20d"] = prices / prices.shift(20) - 1.0

        moving_average_10d = prices.rolling(10).mean()
        moving_average_20d = prices.rolling(20).mean()
        asset_frame["ma_distance_10d"] = prices / moving_average_10d - 1.0
        asset_frame["ma_distance_20d"] = prices / moving_average_20d - 1.0

        asset_frame["drawdown_from_peak"] = drawdown
        asset_frame["rolling_drawdown_20d"] = drawdown.rolling(20).min()
        asset_frame["downside_volatility_20d"] = (
            daily_return.clip(upper=0.0).rolling(20).std(ddof=1)
        )

        if {"high_price", "low_price"}.issubset(asset_frame.columns):
            asset_frame["intraday_range_pct"] = (
                (asset_frame["high_price"] - asset_frame["low_price"]) / prices
            )
        else:
            asset_frame["intraday_range_pct"] = np.nan

        if "volume" in asset_frame.columns:
            volume = asset_frame["volume"].astype(float)
            volume_20d_avg = volume.rolling(20).mean()
            volume_20d_std = volume.rolling(20).std(ddof=1)
            asset_frame["volume_change_1d"] = volume.pct_change()
            asset_frame["volume_ratio_20d"] = volume / volume_20d_avg
            asset_frame["volume_zscore_20d"] = (volume - volume_20d_avg) / volume_20d_std
        else:
            asset_frame["volume_change_1d"] = np.nan
            asset_frame["volume_ratio_20d"] = np.nan
            asset_frame["volume_zscore_20d"] = np.nan

        asset_frame["feature_version"] = FEATURE_VERSION
        feature_frames.append(asset_frame)

    combined = pd.concat(feature_frames, ignore_index=True)
    ordered_columns = [
        "asset_id",
        "ticker",
        "feature_date",
        "price_date",
        "analysis_price",
        "close_price",
        "adjusted_close",
        "volume",
        "daily_return",
        "return_lag_1d",
        "return_lag_5d",
        "return_lag_10d",
        "rolling_mean_return_5d",
        "rolling_mean_return_20d",
        "rolling_volatility_10d",
        "rolling_volatility_20d",
        "realized_volatility_20d",
        "recent_realized_volatility_5d",
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "ma_distance_10d",
        "ma_distance_20d",
        "drawdown_from_peak",
        "rolling_drawdown_20d",
        "downside_volatility_20d",
        "intraday_range_pct",
        "volume_change_1d",
        "volume_ratio_20d",
        "volume_zscore_20d",
        "feature_version",
    ]
    available_columns = [column for column in ordered_columns if column in combined.columns]
    return combined[available_columns]



def build_sentiment_feature_frame(news_history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stored article-level sentiment into daily and trailing features."""

    if news_history.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "ticker",
                "feature_date",
                "article_count_1d",
                "sentiment_mean_1d",
                "sentiment_mean_7d",
                "sentiment_std_7d",
                "negative_article_share_7d",
                "positive_article_share_7d",
                "article_count_7d",
                "source_count_7d",
                "source_sentiment_dispersion_7d",
            ]
        )

    required_columns = {"asset_id", "ticker", "published_at", "sentiment_score", "sentiment_label"}
    missing = required_columns.difference(news_history.columns)
    if missing:
        raise KeyError(f"Missing required sentiment columns: {sorted(missing)}")

    frame = news_history.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    frame["sentiment_score"] = pd.to_numeric(frame["sentiment_score"], errors="coerce")
    frame = frame.dropna(subset=["published_at", "sentiment_score"]).sort_values(
        ["asset_id", "published_at"]
    )
    frame["feature_date"] = frame["published_at"].dt.normalize()
    frame["is_negative"] = (frame["sentiment_label"].str.lower() == "negative").astype(float)
    frame["is_positive"] = (frame["sentiment_label"].str.lower() == "positive").astype(float)
    if "source_name" not in frame.columns:
        frame["source_name"] = "unknown_source"
    frame["source_name"] = frame["source_name"].fillna("unknown_source").astype(str)

    daily = (
        frame.groupby(["asset_id", "ticker", "feature_date"], as_index=False)
        .agg(
            article_count_1d=("sentiment_score", "count"),
            sentiment_mean_1d=("sentiment_score", "mean"),
            negative_article_share_1d=("is_negative", "mean"),
            positive_article_share_1d=("is_positive", "mean"),
            source_count_1d=("source_name", "nunique"),
        )
        .sort_values(["asset_id", "feature_date"])
    )
    source_daily = (
        frame.groupby(["asset_id", "ticker", "feature_date", "source_name"], as_index=False)
        .agg(source_sentiment_mean_1d=("sentiment_score", "mean"))
    )
    source_dispersion = (
        source_daily.groupby(["asset_id", "ticker", "feature_date"], as_index=False)
        .agg(source_sentiment_dispersion_1d=("source_sentiment_mean_1d", "std"))
    )
    daily = daily.merge(
        source_dispersion,
        on=["asset_id", "ticker", "feature_date"],
        how="left",
    )

    sentiment_frames: list[pd.DataFrame] = []
    for (asset_id, ticker), group in daily.groupby(["asset_id", "ticker"], sort=False):
        asset_frame = group.copy().set_index("feature_date")
        asset_frame["sentiment_mean_7d"] = asset_frame["sentiment_mean_1d"].rolling(7, min_periods=1).mean()
        asset_frame["sentiment_std_7d"] = asset_frame["sentiment_mean_1d"].rolling(7, min_periods=2).std(ddof=1)
        asset_frame["negative_article_share_7d"] = asset_frame["negative_article_share_1d"].rolling(
            7,
            min_periods=1,
        ).mean()
        asset_frame["positive_article_share_7d"] = asset_frame["positive_article_share_1d"].rolling(
            7,
            min_periods=1,
        ).mean()
        asset_frame["article_count_7d"] = asset_frame["article_count_1d"].rolling(7, min_periods=1).sum()
        asset_frame["source_count_7d"] = asset_frame["source_count_1d"].rolling(7, min_periods=1).max()
        asset_frame["source_sentiment_dispersion_7d"] = asset_frame["source_sentiment_dispersion_1d"].rolling(
            7,
            min_periods=1,
        ).mean()
        asset_frame = asset_frame.reset_index()
        asset_frame["feature_date"] = asset_frame["feature_date"].dt.date
        asset_frame["asset_id"] = asset_id
        asset_frame["ticker"] = ticker
        sentiment_frames.append(asset_frame)

    combined = pd.concat(sentiment_frames, ignore_index=True)
    return combined[
        [
            "asset_id",
            "ticker",
            "feature_date",
            "article_count_1d",
            "sentiment_mean_1d",
            "sentiment_mean_7d",
            "sentiment_std_7d",
            "negative_article_share_7d",
            "positive_article_share_7d",
            "article_count_7d",
            "source_count_7d",
            "source_sentiment_dispersion_7d",
        ]
    ]
