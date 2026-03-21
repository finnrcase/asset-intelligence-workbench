"""
Structured report-data preparation for Asset Intelligence Workbench.

This module gathers the database, analytics, simulation, and narrative content
needed to render a polished asset briefing report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.exc import OperationalError

from src.analytics.returns import build_return_frame
from src.analytics.returns import compute_annualized_return
from src.analytics.returns import compute_total_return
from src.analytics.risk import build_risk_summary
from src.analytics.risk import compute_rolling_volatility
from src.analytics.simulation import run_comparative_monte_carlo_simulation
from src.utils import app_data


ROLLING_VOLATILITY_WINDOW = 21
VAR_CONFIDENCE_LEVEL = 0.95


def _format_percent(value: float) -> str:
    """Format a float as a percentage string."""

    return "N/A" if value is None or value != value else f"{value:.2%}"


def _format_number(value: float) -> str:
    """Format a float as a numeric string."""

    return "N/A" if value is None or value != value else f"{value:,.2f}"


def _build_executive_summary(
    metadata: dict[str, Any],
    total_return: float,
    annualized_return: float,
    risk_summary: dict[str, float],
) -> str:
    """Build a short institutional executive summary paragraph."""

    asset_name = metadata.get("asset_name") or metadata.get("ticker")
    asset_class = metadata.get("asset_class") or "asset"

    return (
        f"{asset_name} is presented as a {asset_class.lower()} briefing based on the locally stored "
        f"historical price record. Over the covered period, the asset delivered a total return of "
        f"{_format_percent(total_return)} and an annualized return of {_format_percent(annualized_return)}, "
        f"while realized annualized volatility measured {_format_percent(risk_summary['annualized_volatility'])}. "
        f"Historical downside metrics indicate a maximum drawdown of {_format_percent(risk_summary['max_drawdown'])} "
        f"and a one-day 95% historical VaR of {_format_percent(risk_summary['historical_var'])}."
    )


def _build_performance_commentary(total_return: float, annualized_return: float) -> str:
    """Build a concise performance interpretation paragraph."""

    direction = "positive" if total_return >= 0 else "negative"
    return (
        f"Stored-period performance was {direction}, with total return of "
        f"{_format_percent(total_return)} and annualized return of {_format_percent(annualized_return)}. "
        "The paired exhibits frame both price-path behavior and compounded outcome through time."
    )


def _build_risk_commentary(risk_summary: dict[str, float]) -> str:
    """Build a concise risk interpretation paragraph."""

    return (
        f"Realized annualized volatility was {_format_percent(risk_summary['annualized_volatility'])}, "
        f"with maximum drawdown reaching {_format_percent(risk_summary['max_drawdown'])}. "
        f"Historical one-day downside measures include 95% VaR of {_format_percent(risk_summary['historical_var'])} "
        f"and Expected Shortfall of {_format_percent(risk_summary['expected_shortfall'])}. "
        "These measures are presented as historical downside context, not forward guarantees."
    )


def _build_simulation_commentary(simulation_summary: dict[str, float], horizon_days: int) -> str:
    """Build a concise forward-view commentary paragraph."""

    return (
        f"Monte Carlo simulation projects a {horizon_days}-trading-day forward distribution using historical "
        "daily return characteristics as the calibration base. The median simulated terminal price is "
        f"{_format_number(simulation_summary['median_terminal_price'])}, with a 5th to 95th percentile "
        f"range of {_format_number(simulation_summary['p05_terminal_price'])} to "
        f"{_format_number(simulation_summary['p95_terminal_price'])}. "
        "The output is scenario-oriented and should be read as a range-based planning view."
    )


def _build_sentiment_commentary(sentiment_summary: dict[str, Any]) -> str:
    """Build a concise commentary paragraph for stored news sentiment context."""

    article_count = sentiment_summary["article_count"]
    if article_count == 0:
        return (
            "No stored news sentiment records were available for the covered period. "
            "The sentiment section should therefore be read as unavailable rather than neutral."
        )

    average_sentiment = sentiment_summary["average_sentiment"]
    if average_sentiment is None:
        tone = "mixed"
    elif average_sentiment >= 0.2:
        tone = "constructive"
    elif average_sentiment <= -0.2:
        tone = "cautious"
    else:
        tone = "mixed"

    return (
        f"Recent stored article coverage totals {article_count} items and implies a {tone} news tone, "
        f"with an average sentiment score of {_format_number(average_sentiment)}. "
        f"Positive, neutral, and negative article counts were {sentiment_summary['positive_count']}, "
        f"{sentiment_summary['neutral_count']}, and {sentiment_summary['negative_count']} respectively. "
        "This section is intended as directional context for the briefing rather than deep NLP inference."
    )


def _build_ml_commentary(ml_summary: dict[str, Any]) -> str:
    """Build a concise analyst-language paragraph for the ML forecast layer."""

    if not ml_summary["available"]:
        return (
            "No stored model-informed forecast was available at report generation time, so the forward outlook "
            "continues to rely on historical-input scenario analysis only."
        )

    snapshot = ml_summary["snapshot"]
    return (
        f"The latest model-implied expected {int(snapshot['prediction_horizon_days'])}-day return is "
        f"{_format_percent(snapshot['predicted_return_20d'])}, while the probability of a negative forward "
        f"return is {_format_percent(snapshot['downside_probability_20d'])}. "
        f"Recent realized volatility of {_format_percent(snapshot['predicted_volatility_20d'])} is used as a "
        "practical uncertainty proxy for the ML-informed scenario overlay. "
        f"Current regime context is summarized as {snapshot['regime_label'].lower()}."
    )


def _build_comparative_simulation_commentary(
    historical_terminal_summary: dict[str, float],
    ml_terminal_summary: dict[str, float],
) -> str:
    """Build a short comparison paragraph for historical and ML-informed scenarios."""

    return (
        f"The historical-input simulation produces a median terminal price of "
        f"{_format_number(historical_terminal_summary['median_terminal_price'])}, while the ML-informed "
        f"scenario produces {_format_number(ml_terminal_summary['median_terminal_price'])}. "
        "The comparison is intended as a decision-support overlay that incorporates the current return and "
        "downside-risk forecast into the scenario range rather than replacing historical context."
    )


def _prepare_recent_price_rows(recent_price_table) -> list[dict[str, Any]]:
    """Convert recent price data into report-table records with formatted values."""

    rows: list[dict[str, Any]] = []
    for _, row in recent_price_table.iterrows():
        rows.append(
            {
                "price_date": row["price_date"].strftime("%Y-%m-%d"),
                "open_price": _format_number(row["open_price"]),
                "high_price": _format_number(row["high_price"]),
                "low_price": _format_number(row["low_price"]),
                "close_price": _format_number(row["close_price"]),
                "adjusted_close": _format_number(row["adjusted_close"]),
                "volume": "N/A" if row["volume"] != row["volume"] else f"{int(row['volume']):,}",
            }
        )
    return rows


def _build_drawdown_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a drawdown series from the selected analysis price."""

    drawdown_frame = price_frame[["analysis_price"]].copy()
    drawdown_frame["running_peak"] = drawdown_frame["analysis_price"].cummax()
    drawdown_frame["drawdown"] = (
        drawdown_frame["analysis_price"] / drawdown_frame["running_peak"] - 1.0
    )
    return drawdown_frame


def _prepare_terminal_percentile_rows(terminal_summary: dict[str, float]) -> list[dict[str, str]]:
    """Prepare terminal outcome percentiles for PDF table rendering."""

    return [
        {"label": "5th Percentile", "value": _format_number(terminal_summary["p05_terminal_price"])},
        {"label": "25th Percentile", "value": _format_number(terminal_summary["p25_terminal_price"])},
        {"label": "Median", "value": _format_number(terminal_summary["median_terminal_price"])},
        {"label": "75th Percentile", "value": _format_number(terminal_summary["p75_terminal_price"])},
        {"label": "95th Percentile", "value": _format_number(terminal_summary["p95_terminal_price"])},
    ]


def _prepare_recent_headline_rows(sentiment_rows: list[dict[str, Any]], rows: int = 5) -> list[dict[str, str]]:
    """Prepare a concise recent-headlines block for PDF rendering."""

    recent_table = app_data.get_recent_sentiment_table(sentiment_rows, rows=rows)
    if recent_table.empty:
        return []

    prepared_rows: list[dict[str, str]] = []
    for _, row in recent_table.iterrows():
        prepared_rows.append(
            {
                "published_at": str(row["published_at"]),
                "publisher_name": str(row["publisher_name"] or "N/A"),
                "headline": str(row["headline"]),
                "sentiment_label": str(row["sentiment_label"]).title(),
                "sentiment_score": _format_number(float(row["sentiment_score"])),
            }
        )
    return prepared_rows


def build_asset_report_context(
    ticker: str,
    forecast_horizon: int = 63,
    simulation_count: int = 500,
) -> dict[str, Any]:
    """
    Build the full report context for a single selected asset.

    The returned structure is designed to support both HTML templating and PDF
    rendering.
    """

    metadata = app_data.load_asset_metadata(ticker)
    price_rows = app_data.load_price_history(ticker)
    price_frame = app_data.prepare_price_history_frame(price_rows)

    if metadata is None or price_frame.empty:
        raise ValueError(f"No reportable price history is available for {ticker}.")

    return_frame = build_return_frame(price_frame, price_column="analysis_price")
    drawdown_frame = _build_drawdown_frame(price_frame)
    rolling_volatility = compute_rolling_volatility(
        return_frame["daily_return"],
        window=ROLLING_VOLATILITY_WINDOW,
    ).dropna()
    risk_summary = build_risk_summary(
        price_frame,
        price_column="analysis_price",
        confidence_level=VAR_CONFIDENCE_LEVEL,
        volatility_window=ROLLING_VOLATILITY_WINDOW,
    )
    total_return = compute_total_return(price_frame, price_column="analysis_price")
    annualized_return = compute_annualized_return(price_frame, price_column="analysis_price")
    ml_summary = app_data.build_ml_forecast_summary(ticker)
    simulation_result = run_comparative_monte_carlo_simulation(
        price_frame,
        price_column="analysis_price",
        ml_forecast_snapshot=ml_summary["snapshot"],
        horizon_days=forecast_horizon,
        simulation_count=simulation_count,
    )
    try:
        sentiment_rows = app_data.load_recent_news_articles(ticker, limit=25)
    except OperationalError:
        sentiment_rows = []
    sentiment_summary = app_data.get_sentiment_summary(sentiment_rows)
    sentiment_trend = app_data.get_sentiment_trend_frame(sentiment_rows)
    recent_price_table = app_data.get_recent_price_table(price_frame, rows=8)

    coverage_start = price_frame.index.min()
    coverage_end = price_frame.index.max()

    return {
        "report_generated_at": datetime.now(),
        "ticker": metadata["ticker"],
        "metadata": metadata,
        "coverage": {
            "start_date": coverage_start,
            "end_date": coverage_end,
            "observation_count": len(price_frame),
            "latest_price": float(price_frame["analysis_price"].iloc[-1]),
        },
        "kpis": {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": risk_summary["annualized_volatility"],
            "max_drawdown": risk_summary["max_drawdown"],
            "historical_var": risk_summary["historical_var"],
            "expected_shortfall": risk_summary["expected_shortfall"],
        },
        "price_frame": price_frame,
        "return_frame": return_frame,
        "drawdown_frame": drawdown_frame,
        "rolling_volatility": rolling_volatility,
        "risk_summary": risk_summary,
        "simulation": simulation_result,
        "ml_forecast": ml_summary,
        "sentiment": {
            "summary": sentiment_summary,
            "trend": sentiment_trend,
            "recent_headlines": _prepare_recent_headline_rows(sentiment_rows, rows=5),
        },
        "terminal_percentiles": _prepare_terminal_percentile_rows(
            simulation_result["historical"]["terminal_summary"]
        ),
        "ml_terminal_percentiles": _prepare_terminal_percentile_rows(
            simulation_result["ml_informed"]["terminal_summary"]
        ),
        "recent_prices": _prepare_recent_price_rows(recent_price_table),
        "narrative": {
            "executive_summary": _build_executive_summary(
                metadata,
                total_return,
                annualized_return,
                risk_summary,
            ),
            "performance_commentary": _build_performance_commentary(
                total_return,
                annualized_return,
            ),
            "risk_commentary": _build_risk_commentary(risk_summary),
            "simulation_commentary": _build_simulation_commentary(
                simulation_result["historical"]["terminal_summary"],
                forecast_horizon,
            ),
            "ml_commentary": _build_ml_commentary(ml_summary),
            "comparative_simulation_commentary": _build_comparative_simulation_commentary(
                simulation_result["historical"]["terminal_summary"],
                simulation_result["ml_informed"]["terminal_summary"],
            ),
            "sentiment_commentary": _build_sentiment_commentary(sentiment_summary),
        },
        "methodology": {
            "data_source_note": (
                "Historical prices and asset metadata are sourced from the local database, "
                "which is populated through the project ingestion layer using yfinance as the initial provider. "
                "News sentiment uses stored article records sourced from the sentiment ingestion layer."
            ),
            "analytics_note": (
                "Return metrics use simple daily returns. Volatility is annualized using a 252-trading-day convention. "
                "Historical VaR and Expected Shortfall are computed from the empirical daily return distribution."
            ),
            "simulation_note": (
                "Monte Carlo price paths use a GBM-style process calibrated either from historical daily drift and "
                "volatility or from the latest model-implied return/risk overlay. Simulation outputs are scenario-oriented "
                "and should not be interpreted as forecasts with certainty."
            ),
            "caveat_note": (
                "The report is intended for analytical review and decision support. It does not constitute investment advice, "
                "valuation certainty, or a substitute for independent diligence."
            ),
        },
        "settings": {
            "forecast_horizon": forecast_horizon,
            "simulation_count": simulation_count,
            "rolling_volatility_window": ROLLING_VOLATILITY_WINDOW,
            "var_confidence_level": VAR_CONFIDENCE_LEVEL,
        },
    }
