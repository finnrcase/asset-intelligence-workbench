"""
Risk analytics utilities for stored market price data.

The functions in this module operate on daily simple returns, either provided
directly or derived from stored price series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.returns import DEFAULT_TRADING_DAYS
from src.analytics.returns import compute_daily_returns


def _coerce_return_series(data: pd.Series | pd.DataFrame, column: str | None = None) -> pd.Series:
    """
    Normalize return input into a clean float Series.

    If a DataFrame is supplied, `column` identifies the return series to use.
    """

    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("`column` is required when input data is a DataFrame.")
        if column not in data.columns:
            raise KeyError(f"Column '{column}' was not found in the DataFrame.")
        series = data[column]
    else:
        series = data

    return pd.to_numeric(series, errors="coerce").dropna().astype(float)


def compute_annualized_volatility(
    returns: pd.Series | pd.DataFrame,
    column: str | None = None,
    periods_per_year: int = DEFAULT_TRADING_DAYS,
) -> float:
    """
    Compute annualized volatility from daily simple returns.

    Convention:
    sample standard deviation scaled by the square root of trading periods per
    year.
    """

    series = _coerce_return_series(returns, column=column)
    if len(series) < 2:
        return float("nan")
    return float(series.std(ddof=1) * np.sqrt(periods_per_year))


def compute_rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int = 21,
    column: str | None = None,
    periods_per_year: int = DEFAULT_TRADING_DAYS,
) -> pd.Series:
    """
    Compute rolling annualized volatility from daily simple returns.

    A 21-day window is a practical one-month default for trading-day data.
    """

    series = _coerce_return_series(returns, column=column)
    rolling_volatility = series.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)
    return rolling_volatility


def compute_max_drawdown(
    data: pd.Series | pd.DataFrame,
    price_column: str | None = None,
) -> float:
    """
    Compute maximum drawdown from a price series.

    Formula:
    `drawdown_t = price_t / running_peak_t - 1`
    """

    if isinstance(data, pd.DataFrame):
        if price_column is None:
            raise ValueError("`price_column` is required when input data is a DataFrame.")
        if price_column not in data.columns:
            raise KeyError(f"Price column '{price_column}' was not found in the DataFrame.")
        prices = pd.to_numeric(data[price_column], errors="coerce").dropna().astype(float)
    else:
        prices = pd.to_numeric(data, errors="coerce").dropna().astype(float)

    if prices.empty:
        return float("nan")

    running_peak = prices.cummax()
    drawdowns = prices / running_peak - 1.0
    return float(drawdowns.min())


def compute_historical_var(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
    column: str | None = None,
) -> float:
    """
    Compute historical Value at Risk from daily returns.

    Output convention:
    positive loss magnitude at the requested confidence level.
    Example:
    a result of `0.02` corresponds to a 2% one-period VaR.
    """

    series = _coerce_return_series(returns, column=column)
    if series.empty:
        return float("nan")

    tail_quantile = np.quantile(series, 1.0 - confidence_level)
    return float(-tail_quantile)


def compute_expected_shortfall(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
    column: str | None = None,
) -> float:
    """
    Compute historical Expected Shortfall (CVaR) from daily returns.

    Output convention:
    positive average loss magnitude for observations at or below the VaR cutoff.
    """

    series = _coerce_return_series(returns, column=column)
    if series.empty:
        return float("nan")

    cutoff = np.quantile(series, 1.0 - confidence_level)
    tail_losses = series[series <= cutoff]
    if tail_losses.empty:
        return float("nan")
    return float(-tail_losses.mean())


def build_risk_summary(
    price_data: pd.Series | pd.DataFrame,
    price_column: str | None = None,
    confidence_level: float = 0.95,
    volatility_window: int = 21,
    periods_per_year: int = DEFAULT_TRADING_DAYS,
) -> dict[str, float]:
    """
    Compute a compact first-pass risk summary from price data.

    This is intended as a convenient bridge into future reporting and app-layer
    views.
    """

    daily_returns = compute_daily_returns(price_data, price_column=price_column)
    rolling_volatility = compute_rolling_volatility(
        daily_returns,
        window=volatility_window,
        periods_per_year=periods_per_year,
    )

    return {
        "annualized_volatility": compute_annualized_volatility(
            daily_returns,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": compute_max_drawdown(price_data, price_column=price_column),
        "historical_var": compute_historical_var(
            daily_returns,
            confidence_level=confidence_level,
        ),
        "expected_shortfall": compute_expected_shortfall(
            daily_returns,
            confidence_level=confidence_level,
        ),
        "latest_rolling_volatility": float(rolling_volatility.dropna().iloc[-1])
        if not rolling_volatility.dropna().empty
        else float("nan"),
    }

