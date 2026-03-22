"""
Feature engineering for the machine learning signal-calibration layer.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.ml.targets import NEGATIVE_TARGET_COLUMN
from src.ml.targets import TARGET_COLUMN


PILLAR_COLUMNS = ["history_score", "risk_score", "sentiment_score"]


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _bounded(series: pd.Series, scale: float) -> pd.Series:
    return np.tanh(series.astype(float) / scale)


def _build_benchmark_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    benchmark_candidates = ["SPY", "VOO", "IVV", "^GSPC"]
    benchmark_rows = frame[frame["ticker"].isin(benchmark_candidates)].copy()

    if benchmark_rows.empty:
        benchmark_proxy = (
            frame.groupby("feature_date", as_index=False)
            .agg(
                benchmark_daily_return=("daily_return", "mean"),
                benchmark_momentum_20d=("momentum_20d", "mean"),
                benchmark_momentum_60d=("trailing_return_60d", "mean"),
            )
        )
    else:
        benchmark_proxy = (
            benchmark_rows.groupby("feature_date", as_index=False)
            .agg(
                benchmark_daily_return=("daily_return", "mean"),
                benchmark_momentum_20d=("momentum_20d", "mean"),
                benchmark_momentum_60d=("trailing_return_60d", "mean"),
            )
        )

    enriched = frame.merge(benchmark_proxy, on="feature_date", how="left")
    enriched["benchmark_relative_return_1d"] = (
        enriched["daily_return"] - enriched["benchmark_daily_return"]
    )
    enriched["benchmark_relative_momentum_20d"] = (
        enriched["momentum_20d"] - enriched["benchmark_momentum_20d"]
    )
    enriched["benchmark_relative_return_60d"] = (
        enriched["trailing_return_60d"] - enriched["benchmark_momentum_60d"]
    )
    return enriched


def build_ml_feature_frame(training_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build a richer ML feature frame from stored market and sentiment features."""

    if training_frame.empty:
        return training_frame.copy(), {"history": [], "risk": [], "sentiment": [], "all": []}

    frame = training_frame.copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    frame = frame.sort_values(["feature_date", "asset_id"]).reset_index(drop=True)
    numeric_candidates = [
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
        "article_count_1d",
        "sentiment_mean_1d",
        "sentiment_mean_7d",
        "sentiment_std_7d",
        "negative_article_share_7d",
        "positive_article_share_7d",
        "article_count_7d",
        "source_count_7d",
        "source_sentiment_dispersion_7d",
        TARGET_COLUMN,
        NEGATIVE_TARGET_COLUMN,
    ]
    frame = _safe_numeric(frame, numeric_candidates)

    grouped = frame.groupby("asset_id", sort=False)
    frame["trailing_return_20d"] = frame["momentum_20d"]
    frame["trailing_return_60d"] = grouped["analysis_price"].transform(lambda series: series / series.shift(60) - 1.0)
    frame["moving_average_spread_10_20"] = frame["ma_distance_10d"] - frame["ma_distance_20d"]
    frame["sharpe_like_ratio_20d"] = frame["rolling_mean_return_20d"] / frame["rolling_volatility_20d"].replace(0.0, np.nan)
    frame["trend_strength_20d"] = frame["momentum_20d"] / frame["realized_volatility_20d"].replace(0.0, np.nan)
    frame["volume_confirmation_20d"] = frame["momentum_20d"] * frame["volume_ratio_20d"]

    annualized_daily_vol_20d = frame["realized_volatility_20d"] / np.sqrt(252.0)
    annualized_daily_vol_5d = frame["recent_realized_volatility_5d"] / np.sqrt(252.0)
    frame["value_at_risk_20d"] = frame["rolling_mean_return_20d"] - (1.65 * frame["rolling_volatility_20d"])
    frame["expected_shortfall_20d"] = frame["rolling_mean_return_20d"] - (2.06 * frame["rolling_volatility_20d"])
    frame["volatility_regime_20d"] = annualized_daily_vol_5d / annualized_daily_vol_20d.replace(0.0, np.nan)
    frame["drawdown_regime_20d"] = frame["drawdown_from_peak"] - frame["rolling_drawdown_20d"]
    frame["stress_return_20d"] = frame["momentum_20d"] - (1.65 * frame["realized_volatility_20d"] * np.sqrt(20.0 / 252.0))
    frame["monte_carlo_downside_20d"] = (
        (frame["rolling_mean_return_20d"] * 20.0)
        - (1.65 * annualized_daily_vol_20d * np.sqrt(20.0))
    )
    frame["downside_ratio_20d"] = frame["downside_volatility_20d"] / frame["rolling_volatility_20d"].replace(0.0, np.nan)

    frame["article_count_1d"] = frame["article_count_1d"].fillna(0.0)
    frame["article_count_7d"] = frame["article_count_7d"].fillna(0.0)
    frame["source_count_7d"] = frame["source_count_7d"].fillna(0.0)
    frame["sentiment_mean_1d"] = frame["sentiment_mean_1d"].fillna(0.0)
    frame["sentiment_mean_7d"] = frame["sentiment_mean_7d"].fillna(0.0)
    frame["sentiment_std_7d"] = frame["sentiment_std_7d"].fillna(0.0)
    frame["negative_article_share_7d"] = frame["negative_article_share_7d"].fillna(0.0)
    frame["positive_article_share_7d"] = frame["positive_article_share_7d"].fillna(0.0)
    frame["source_sentiment_dispersion_7d"] = frame["source_sentiment_dispersion_7d"].fillna(0.0)

    frame["sentiment_trend_7d"] = frame["sentiment_mean_1d"] - frame["sentiment_mean_7d"]
    frame["sentiment_change_7d"] = grouped["sentiment_mean_7d"].transform(lambda series: series - series.shift(7))
    rolling_article_mean_30d = grouped["article_count_1d"].transform(lambda series: series.rolling(30, min_periods=5).mean())
    rolling_article_std_30d = grouped["article_count_1d"].transform(lambda series: series.rolling(30, min_periods=5).std(ddof=1))
    frame["article_volume_zscore_30d"] = (
        frame["article_count_1d"] - rolling_article_mean_30d
    ) / rolling_article_std_30d.replace(0.0, np.nan)
    frame["combined_sentiment_score_7d"] = (
        (0.55 * frame["sentiment_mean_7d"])
        + (0.20 * (frame["positive_article_share_7d"] - frame["negative_article_share_7d"]))
        + (0.15 * _bounded(frame["article_volume_zscore_30d"].fillna(0.0), 2.0))
        + (0.10 * _bounded(frame["source_count_7d"].fillna(0.0), 4.0))
    )

    frame = _build_benchmark_relative_features(frame)

    history_components = pd.concat(
        [
            _bounded(frame["momentum_5d"].fillna(0.0), 0.05),
            _bounded(frame["trailing_return_20d"].fillna(0.0), 0.10),
            _bounded(frame["trailing_return_60d"].fillna(0.0), 0.18),
            _bounded(frame["moving_average_spread_10_20"].fillna(0.0), 0.04),
            _bounded(frame["sharpe_like_ratio_20d"].fillna(0.0), 1.50),
            _bounded(frame["benchmark_relative_momentum_20d"].fillna(0.0), 0.08),
            _bounded(frame["volume_confirmation_20d"].fillna(0.0), 0.12),
        ],
        axis=1,
    )
    risk_components = pd.concat(
        [
            _bounded(-frame["realized_volatility_20d"].fillna(0.0), 0.45),
            _bounded(-frame["downside_volatility_20d"].fillna(0.0), 0.03),
            _bounded(frame["drawdown_from_peak"].fillna(0.0), 0.20),
            _bounded(frame["value_at_risk_20d"].fillna(0.0), 0.05),
            _bounded(frame["expected_shortfall_20d"].fillna(0.0), 0.07),
            _bounded(frame["stress_return_20d"].fillna(0.0), 0.10),
            _bounded(frame["monte_carlo_downside_20d"].fillna(0.0), 0.10),
            _bounded(-frame["volatility_regime_20d"].fillna(1.0) + 1.0, 0.60),
        ],
        axis=1,
    )
    sentiment_components = pd.concat(
        [
            _bounded(frame["sentiment_mean_7d"].fillna(0.0), 0.50),
            _bounded(frame["sentiment_trend_7d"].fillna(0.0), 0.25),
            _bounded(-frame["sentiment_std_7d"].fillna(0.0), 0.40),
            _bounded(frame["article_volume_zscore_30d"].fillna(0.0), 2.00),
            _bounded(frame["combined_sentiment_score_7d"].fillna(0.0), 0.50),
            _bounded(frame["positive_article_share_7d"].fillna(0.0), 0.60),
            _bounded(-frame["negative_article_share_7d"].fillna(0.0), 0.60),
            _bounded(frame["source_count_7d"].fillna(0.0), 4.00),
        ],
        axis=1,
    )

    frame["history_score"] = history_components.mean(axis=1)
    frame["risk_score"] = risk_components.mean(axis=1)
    frame["sentiment_score"] = sentiment_components.mean(axis=1)
    frame["pillar_composite_pre_ml"] = frame[PILLAR_COLUMNS].mean(axis=1)

    history_features = [
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "trailing_return_20d",
        "trailing_return_60d",
        "moving_average_spread_10_20",
        "ma_distance_10d",
        "ma_distance_20d",
        "rolling_mean_return_20d",
        "sharpe_like_ratio_20d",
        "trend_strength_20d",
        "volume_ratio_20d",
        "volume_confirmation_20d",
        "benchmark_relative_return_1d",
        "benchmark_relative_momentum_20d",
        "benchmark_relative_return_60d",
        "history_score",
    ]
    risk_features = [
        "rolling_volatility_20d",
        "realized_volatility_20d",
        "recent_realized_volatility_5d",
        "downside_volatility_20d",
        "drawdown_from_peak",
        "rolling_drawdown_20d",
        "value_at_risk_20d",
        "expected_shortfall_20d",
        "volatility_regime_20d",
        "drawdown_regime_20d",
        "stress_return_20d",
        "monte_carlo_downside_20d",
        "downside_ratio_20d",
        "risk_score",
    ]
    sentiment_features = [
        "article_count_1d",
        "article_count_7d",
        "source_count_7d",
        "sentiment_mean_1d",
        "sentiment_mean_7d",
        "sentiment_std_7d",
        "sentiment_trend_7d",
        "sentiment_change_7d",
        "negative_article_share_7d",
        "positive_article_share_7d",
        "article_volume_zscore_30d",
        "source_sentiment_dispersion_7d",
        "combined_sentiment_score_7d",
        "sentiment_score",
    ]
    all_features = [
        column
        for column in history_features + risk_features + sentiment_features + ["pillar_composite_pre_ml"]
        if column in frame.columns
    ]

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame, {
        "history": history_features,
        "risk": risk_features,
        "sentiment": sentiment_features,
        "pillar_scores": PILLAR_COLUMNS,
        "all": all_features,
    }


def latest_rows_by_asset(feature_frame: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return the latest engineered row per asset for scoring."""

    if feature_frame.empty:
        return feature_frame.copy()

    frame = feature_frame.copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    if as_of_date is not None:
        frame = frame[frame["feature_date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        return frame

    return (
        frame.sort_values(["asset_id", "feature_date"])
        .groupby("asset_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def feature_group_lookup(feature_groups: Mapping[str, list[str]]) -> dict[str, str]:
    """Return a reverse lookup of feature name to pillar group."""

    lookup: dict[str, str] = {}
    for group_name in ("history", "risk", "sentiment"):
        for column in feature_groups.get(group_name, []):
            lookup[column] = group_name
    return lookup
