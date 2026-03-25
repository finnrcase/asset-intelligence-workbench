"""
Reusable chart builders for the Streamlit asset dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import plotly.graph_objects as go

# Report-specific export settings are intentionally separate from the app charts.
REPORT_EXPORT_DPI = 260
REPORT_TITLE_SIZE = 14.5
REPORT_AXIS_LABEL_SIZE = 10.5
REPORT_TICK_LABEL_SIZE = 9.5
REPORT_LINE_WIDTH = 2.9
REPORT_CHART_SIZES = {
    "price_history": (10.0, 4.8),
    "cumulative_return": (10.0, 4.8),
    "rolling_volatility": (9.2, 4.0),
    "drawdown": (9.2, 4.0),
    "monte_carlo_paths": (11.4, 5.8),
    "terminal_distribution": (9.2, 4.8),
    "sentiment_trend": (10.4, 4.6),
    "forecast_history": (10.2, 4.6),
    "feature_drivers": (8.8, 4.8),
    "simulation_comparison": (10.8, 5.0),
}
APP_CHART_COLORS = {
    "ink": "#18242f",
    "muted": "#111111",
    "grid": "rgba(0, 0, 0, 0.12)",
    "line": "rgba(0, 0, 0, 0.58)",
    "surface": "rgba(255, 255, 255, 0)",
    "primary": "#111111",
    "primary_soft": "rgba(0, 0, 0, 0.22)",
    "secondary": "#111111",
    "positive": "#4f7c68",
    "positive_soft": "rgba(79, 124, 104, 0.14)",
    "negative": "#a05b52",
    "negative_soft": "rgba(160, 91, 82, 0.14)",
    "warm": "#b98249",
    "warm_soft": "rgba(185, 130, 73, 0.16)",
}


def _apply_standard_layout(figure: go.Figure, title: str, y_axis_title: str) -> go.Figure:
    """Apply the shared product-style chart layout."""

    figure.update_layout(
        title=title,
        template="plotly_white",
        height=340,
        margin=dict(l=18, r=18, t=64, b=18),
        hovermode="x unified",
        showlegend=False,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        paper_bgcolor=APP_CHART_COLORS["surface"],
        plot_bgcolor=APP_CHART_COLORS["surface"],
        title_x=0.02,
        title_font=dict(size=18, color=APP_CHART_COLORS["ink"]),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(24, 36, 47, 0.10)",
            font=dict(color=APP_CHART_COLORS["ink"], size=12),
        ),
        font=dict(size=12, color=APP_CHART_COLORS["muted"]),
    )
    figure.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=APP_CHART_COLORS["line"],
        tickfont=dict(color=APP_CHART_COLORS["muted"]),
        title_font=dict(color=APP_CHART_COLORS["muted"]),
        zeroline=False,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=APP_CHART_COLORS["grid"],
        showline=False,
        tickfont=dict(color=APP_CHART_COLORS["muted"]),
        title_font=dict(color=APP_CHART_COLORS["muted"]),
        zeroline=False,
    )
    return figure


def create_price_history_chart(frame: pd.DataFrame, price_column: str = "analysis_price") -> go.Figure:
    """Build a single-line price history chart."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame[price_column],
            mode="lines",
            line=dict(color=APP_CHART_COLORS["primary"], width=2.7),
            name="Price",
        )
    )
    return _apply_standard_layout(figure, "Price History", "Price")


def create_cumulative_return_chart(return_frame: pd.DataFrame) -> go.Figure:
    """Build a cumulative return chart from the return analytics frame."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=return_frame.index,
            y=return_frame["cumulative_return"],
            mode="lines",
            line=dict(color=APP_CHART_COLORS["positive"], width=2.7),
            name="Cumulative Return",
        )
    )
    figure.update_yaxes(tickformat=".0%")
    return _apply_standard_layout(figure, "Cumulative Return", "Return")


def create_rolling_volatility_chart(rolling_volatility: pd.Series) -> go.Figure:
    """Build a rolling volatility chart from an annualized volatility series."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=rolling_volatility.index,
            y=rolling_volatility,
            mode="lines",
            line=dict(color=APP_CHART_COLORS["warm"], width=2.6),
            name="Rolling Volatility",
        )
    )
    figure.update_yaxes(tickformat=".1%")
    return _apply_standard_layout(figure, "Rolling Volatility", "Volatility")


def create_monte_carlo_paths_chart(
    simulated_paths: pd.DataFrame,
    max_paths: int = 75,
) -> go.Figure:
    """Build a restrained Monte Carlo path chart using a subset of simulated paths."""

    figure = go.Figure()
    subset = simulated_paths.iloc[:, :max_paths]

    for column in subset.columns:
        figure.add_trace(
            go.Scatter(
                x=subset.index,
                y=subset[column],
                mode="lines",
                line=dict(color=APP_CHART_COLORS["primary_soft"], width=1.05),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    median_series = simulated_paths.median(axis=1)
    figure.add_trace(
        go.Scatter(
            x=median_series.index,
            y=median_series,
            mode="lines",
            line=dict(color=APP_CHART_COLORS["primary"], width=2.8),
            name="Median Path",
        )
    )
    return _apply_standard_layout(figure, "Monte Carlo Price Paths", "Price")


def create_terminal_distribution_chart(simulated_paths: pd.DataFrame) -> go.Figure:
    """Build a terminal-price distribution chart for simulated outcomes."""

    terminal_values = simulated_paths.iloc[-1]
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=terminal_values,
            nbinsx=40,
            marker=dict(color=APP_CHART_COLORS["secondary"]),
            opacity=0.9,
            name="Terminal Price",
        )
    )
    figure.update_layout(bargap=0.05)
    return _apply_standard_layout(figure, "Terminal Price Distribution", "Frequency")


def create_percentile_band_chart(percentile_bands: pd.DataFrame) -> go.Figure:
    """Build a percentile band chart for simulated price paths."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=percentile_bands.index,
            y=percentile_bands["p95"],
            mode="lines",
            line=dict(color="rgba(79, 124, 104, 0.0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=percentile_bands.index,
            y=percentile_bands["p05"],
            mode="lines",
            line=dict(color="rgba(79, 124, 104, 0.0)"),
            fill="tonexty",
            fillcolor=APP_CHART_COLORS["positive_soft"],
            name="5th-95th Percentile",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=percentile_bands.index,
            y=percentile_bands["p75"],
            mode="lines",
            line=dict(color="rgba(32, 74, 97, 0.0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=percentile_bands.index,
            y=percentile_bands["p25"],
            mode="lines",
            line=dict(color="rgba(32, 74, 97, 0.0)"),
            fill="tonexty",
            fillcolor="rgba(32, 74, 97, 0.16)",
            name="25th-75th Percentile",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=percentile_bands.index,
            y=percentile_bands["p50"],
            mode="lines",
            line=dict(color=APP_CHART_COLORS["primary"], width=2.8),
            name="Median",
        )
    )
    return _apply_standard_layout(figure, "Simulation Percentile Bands", "Price")


def create_sentiment_trend_chart(sentiment_trend: pd.DataFrame) -> go.Figure:
    """Build a restrained daily sentiment trend chart for stored news articles."""

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=sentiment_trend["published_date"],
            y=sentiment_trend["article_count"],
            marker_color="rgba(0, 0, 0, 0.42)",
            name="Article Count",
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sentiment_trend["published_date"],
            y=sentiment_trend["average_sentiment"],
            mode="lines+markers",
            line=dict(color=APP_CHART_COLORS["primary"], width=2.4),
            marker=dict(size=6, color="#f6f1ea", line=dict(color=APP_CHART_COLORS["primary"], width=1.5)),
            name="Average Sentiment",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=18, r=18, t=64, b=18),
        hovermode="x unified",
        paper_bgcolor=APP_CHART_COLORS["surface"],
        plot_bgcolor=APP_CHART_COLORS["surface"],
        xaxis_title="Date",
        yaxis=dict(
            title="Average Sentiment",
            range=[-1.0, 1.0],
            zeroline=True,
            zerolinecolor=APP_CHART_COLORS["line"],
            gridcolor=APP_CHART_COLORS["grid"],
        ),
        yaxis2=dict(
            title="Article Count",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=APP_CHART_COLORS["muted"])),
        font=dict(size=12),
        title="News Sentiment Trend",
        title_x=0.02,
        title_font=dict(size=18, color=APP_CHART_COLORS["ink"]),
    )
    figure.update_xaxes(showgrid=False, showline=True, linecolor=APP_CHART_COLORS["line"])
    return figure


def create_prediction_history_chart(prediction_history: pd.DataFrame) -> go.Figure:
    """Build a chart for recent model-implied expected return and downside probability."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prediction_history["as_of_date"],
            y=prediction_history["predicted_return_20d"],
            mode="lines+markers",
            line=dict(color=APP_CHART_COLORS["primary"], width=2.4),
            marker=dict(size=5, color="#f6f1ea", line=dict(color=APP_CHART_COLORS["primary"], width=1.4)),
            name="Expected Return",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=prediction_history["as_of_date"],
            y=prediction_history["downside_probability_20d"],
            mode="lines+markers",
            line=dict(color=APP_CHART_COLORS["negative"], width=2.0),
            marker=dict(size=5, color="#f6f1ea", line=dict(color=APP_CHART_COLORS["negative"], width=1.4)),
            name="Probability Negative",
            yaxis="y2",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=18, r=18, t=64, b=18),
        hovermode="x unified",
        paper_bgcolor=APP_CHART_COLORS["surface"],
        plot_bgcolor=APP_CHART_COLORS["surface"],
        xaxis_title="As Of Date",
        yaxis=dict(title="Expected Return", tickformat=".1%"),
        yaxis2=dict(
            title="Probability Negative",
            tickformat=".0%",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, 1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=APP_CHART_COLORS["muted"])),
        title="Model-Informed Forecast History",
        title_x=0.02,
        title_font=dict(size=18, color=APP_CHART_COLORS["ink"]),
        font=dict(size=12, color=APP_CHART_COLORS["muted"]),
    )
    figure.update_xaxes(showgrid=False, showline=True, linecolor=APP_CHART_COLORS["line"])
    figure.update_yaxes(gridcolor=APP_CHART_COLORS["grid"])
    return figure


def create_feature_driver_chart(feature_drivers: list[dict[str, object]]) -> go.Figure:
    """Build a bar chart for the strongest current forecast drivers."""

    sorted_drivers = sorted(feature_drivers, key=lambda row: abs(float(row["z_score"])), reverse=True)
    labels = [str(row["label"]).title() for row in sorted_drivers]
    z_scores = [float(row["z_score"]) for row in sorted_drivers]
    colors = [APP_CHART_COLORS["primary"] if score >= 0 else APP_CHART_COLORS["negative"] for score in z_scores]

    figure = go.Figure(
        go.Bar(
            x=z_scores,
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=18, r=18, t=64, b=18),
        xaxis_title="Standardized Deviation vs Recent History",
        yaxis_title="Current Drivers",
        title="Current Forecast Driver Context",
        paper_bgcolor=APP_CHART_COLORS["surface"],
        plot_bgcolor=APP_CHART_COLORS["surface"],
        title_x=0.02,
        title_font=dict(size=18, color=APP_CHART_COLORS["ink"]),
        font=dict(size=12, color=APP_CHART_COLORS["muted"]),
    )
    figure.update_xaxes(showgrid=True, gridcolor=APP_CHART_COLORS["grid"], showline=True, linecolor=APP_CHART_COLORS["line"])
    figure.update_yaxes(showgrid=False, autorange="reversed")
    return figure


def create_simulation_comparison_chart(
    historical_bands: pd.DataFrame,
    ml_bands: pd.DataFrame,
) -> go.Figure:
    """Build a comparison chart for historical-input versus ML-informed median paths."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=historical_bands.index,
            y=historical_bands["p50"],
            mode="lines",
            line=dict(color=APP_CHART_COLORS["secondary"], width=2.1),
            name="Historical Median",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ml_bands.index,
            y=ml_bands["p50"],
            mode="lines",
            line=dict(color=APP_CHART_COLORS["primary"], width=2.6),
            name="ML-Informed Median",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ml_bands.index,
            y=ml_bands["p95"],
            mode="lines",
            line=dict(color="rgba(32, 74, 97, 0.0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ml_bands.index,
            y=ml_bands["p05"],
            mode="lines",
            line=dict(color="rgba(32, 74, 97, 0.0)"),
            fill="tonexty",
            fillcolor=APP_CHART_COLORS["primary_soft"],
            name="ML-Informed 5th-95th",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=18, r=18, t=68, b=64),
        hovermode="x unified",
        xaxis_title="Forecast Step",
        yaxis_title="Price",
        title="Historical vs ML-Informed Scenario Comparison",
        paper_bgcolor=APP_CHART_COLORS["surface"],
        plot_bgcolor=APP_CHART_COLORS["surface"],
        title_x=0.02,
        title_font=dict(size=18, color=APP_CHART_COLORS["ink"]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color=APP_CHART_COLORS["muted"]),
        ),
        font=dict(size=12, color=APP_CHART_COLORS["muted"]),
    )
    figure.update_xaxes(showgrid=False, showline=True, linecolor=APP_CHART_COLORS["line"])
    figure.update_yaxes(showgrid=True, gridcolor=APP_CHART_COLORS["grid"])
    return figure


def _save_matplotlib_figure(output_path: str | Path) -> None:
    """Save and close the active matplotlib figure with consistent settings."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=REPORT_EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def _apply_report_axes_style(ax, title: str, y_axis_title: str) -> None:
    """Apply restrained report-friendly matplotlib styling."""

    ax.set_title(
        title,
        loc="left",
        fontsize=REPORT_TITLE_SIZE,
        fontweight="bold",
        color="#102131",
        pad=10,
    )
    ax.set_ylabel(y_axis_title, fontsize=REPORT_AXIS_LABEL_SIZE, color="#3d4e5f")
    ax.tick_params(axis="both", labelsize=REPORT_TICK_LABEL_SIZE, colors="#536272")
    ax.grid(axis="y", color="#dce3e9", linewidth=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color("#ccd6df")
    ax.spines["bottom"].set_color("#ccd6df")


def save_price_history_chart(
    frame: pd.DataFrame,
    output_path: str | Path,
    price_column: str = "analysis_price",
) -> str:
    """Render a static price history chart for PDF reporting."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["price_history"])
    ax.plot(frame.index, frame[price_column], color="#0f4c81", linewidth=REPORT_LINE_WIDTH)
    ax.set_xlabel("Date")
    _apply_report_axes_style(ax, "Price History", "Price")
    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_cumulative_return_chart(
    return_frame: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Render a static cumulative return chart for PDF reporting."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["cumulative_return"])
    ax.plot(
        return_frame.index,
        return_frame["cumulative_return"],
        color="#2e7d32",
        linewidth=REPORT_LINE_WIDTH,
    )
    ax.set_xlabel("Date")
    _apply_report_axes_style(ax, "Cumulative Return", "Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_rolling_volatility_chart(
    rolling_volatility: pd.Series,
    output_path: str | Path,
) -> str:
    """Render a static rolling volatility chart for PDF reporting."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["rolling_volatility"])
    ax.plot(
        rolling_volatility.index,
        rolling_volatility,
        color="#8a5a00",
        linewidth=REPORT_LINE_WIDTH - 0.1,
    )
    ax.set_xlabel("Date")
    _apply_report_axes_style(ax, "Rolling Volatility", "Volatility")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_drawdown_chart(
    drawdown_frame: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Render a static drawdown chart for PDF reporting."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["drawdown"])
    ax.fill_between(
        drawdown_frame.index,
        drawdown_frame["drawdown"],
        0,
        color="#b0453b",
        alpha=0.22,
    )
    ax.plot(
        drawdown_frame.index,
        drawdown_frame["drawdown"],
        color="#b0453b",
        linewidth=REPORT_LINE_WIDTH - 0.2,
    )
    ax.set_xlabel("Date")
    _apply_report_axes_style(ax, "Drawdown", "Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_monte_carlo_paths_chart(
    simulated_paths: pd.DataFrame,
    output_path: str | Path,
    max_paths: int = 75,
) -> str:
    """Render a static Monte Carlo path chart for PDF reporting."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["monte_carlo_paths"])
    subset = simulated_paths.iloc[:, :max_paths]
    for column in subset.columns:
        ax.plot(
            subset.index,
            subset[column],
            color="#0f4c81",
            linewidth=1.2,
            alpha=0.18,
        )

    median_series = simulated_paths.median(axis=1)
    ax.plot(median_series.index, median_series, color="#0f4c81", linewidth=3.3)
    ax.set_xlabel("Forecast Step")
    _apply_report_axes_style(ax, "Monte Carlo Price Paths", "Price")
    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_terminal_distribution_chart(
    simulated_paths: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Render a static terminal-price histogram for PDF reporting."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["terminal_distribution"])
    terminal_values = simulated_paths.iloc[-1]
    ax.hist(terminal_values, bins=36, color="#5b6770", edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Terminal Price")
    _apply_report_axes_style(ax, "Terminal Price Distribution", "Frequency")
    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_sentiment_trend_chart(
    sentiment_trend: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Render a static sentiment trend chart for PDF reporting."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["sentiment_trend"])
    ax.bar(
        sentiment_trend["published_date"],
        sentiment_trend["article_count"],
        color="#5b6770",
        alpha=0.28,
        width=0.8,
    )
    ax.set_xlabel("Date")
    _apply_report_axes_style(ax, "News Sentiment Trend", "Article Count")

    sentiment_ax = ax.twinx()
    sentiment_ax.plot(
        sentiment_trend["published_date"],
        sentiment_trend["average_sentiment"],
        color="#0f4c81",
        linewidth=REPORT_LINE_WIDTH - 0.3,
        marker="o",
        markersize=4,
    )
    sentiment_ax.set_ylabel("Average Sentiment", fontsize=REPORT_AXIS_LABEL_SIZE, color="#3d4e5f")
    sentiment_ax.tick_params(axis="y", labelsize=REPORT_TICK_LABEL_SIZE, colors="#536272")
    sentiment_ax.set_ylim(-1.0, 1.0)
    sentiment_ax.spines["top"].set_visible(False)
    sentiment_ax.spines["left"].set_visible(False)
    sentiment_ax.spines["right"].set_color("#ccd6df")
    sentiment_ax.spines["right"].set_linewidth(0.9)

    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_prediction_history_chart(
    prediction_history: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Render a static chart for recent model-informed forecast history."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["forecast_history"])
    ax.plot(
        prediction_history["as_of_date"],
        prediction_history["predicted_return_20d"],
        color="#0f4c81",
        linewidth=REPORT_LINE_WIDTH - 0.2,
        marker="o",
        markersize=4,
    )
    ax.set_xlabel("As Of Date")
    _apply_report_axes_style(ax, "Model-Informed Forecast History", "Expected Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    secondary_ax = ax.twinx()
    secondary_ax.plot(
        prediction_history["as_of_date"],
        prediction_history["downside_probability_20d"],
        color="#b0453b",
        linewidth=REPORT_LINE_WIDTH - 0.6,
    )
    secondary_ax.set_ylabel("Probability Negative", fontsize=REPORT_AXIS_LABEL_SIZE, color="#3d4e5f")
    secondary_ax.tick_params(axis="y", labelsize=REPORT_TICK_LABEL_SIZE, colors="#536272")
    secondary_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    secondary_ax.set_ylim(0.0, 1.0)
    secondary_ax.spines["top"].set_visible(False)
    secondary_ax.spines["left"].set_visible(False)
    secondary_ax.spines["right"].set_color("#ccd6df")
    secondary_ax.spines["right"].set_linewidth(0.9)

    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_feature_driver_chart(
    feature_drivers: list[dict[str, object]],
    output_path: str | Path,
) -> str:
    """Render a static horizontal bar chart for current forecast drivers."""

    ordered = sorted(feature_drivers, key=lambda row: abs(float(row["z_score"])), reverse=True)
    labels = [str(row["label"]).title() for row in ordered]
    z_scores = [float(row["z_score"]) for row in ordered]
    colors = ["#0f4c81" if score >= 0 else "#b0453b" for score in z_scores]

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["feature_drivers"])
    ax.barh(labels, z_scores, color=colors, alpha=0.88)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized Deviation vs Recent History")
    _apply_report_axes_style(ax, "Current Forecast Driver Context", "Drivers")
    _save_matplotlib_figure(output_path)
    return str(output_path)


def save_simulation_comparison_chart(
    historical_bands: pd.DataFrame,
    ml_bands: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Render a static historical-versus-ML simulation comparison chart."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["simulation_comparison"])
    ax.plot(
        historical_bands.index,
        historical_bands["p50"],
        color="#5b6770",
        linewidth=REPORT_LINE_WIDTH - 0.6,
        label="Historical Median",
    )
    ax.plot(
        ml_bands.index,
        ml_bands["p50"],
        color="#0f4c81",
        linewidth=REPORT_LINE_WIDTH,
        label="ML-Informed Median",
    )
    ax.fill_between(
        ml_bands.index,
        ml_bands["p05"],
        ml_bands["p95"],
        color="#0f4c81",
        alpha=0.10,
        label="ML-Informed 5th-95th",
    )
    ax.set_xlabel("Forecast Step")
    _apply_report_axes_style(ax, "Historical vs ML-Informed Scenarios", "Price")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save_matplotlib_figure(output_path)
    return str(output_path)


def create_ml_score_history_chart(prediction_history: pd.DataFrame) -> go.Figure:
    """Build a chart for recent composite score and directional probability history."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prediction_history["as_of_date"],
            y=prediction_history["composite_ml_score"],
            mode="lines+markers",
            line=dict(color=APP_CHART_COLORS["primary"], width=2.5),
            marker=dict(size=5, color="#f6f1ea", line=dict(color=APP_CHART_COLORS["primary"], width=1.4)),
            name="Composite Score",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=prediction_history["as_of_date"],
            y=prediction_history["probability_positive_20d"],
            mode="lines+markers",
            line=dict(color=APP_CHART_COLORS["positive"], width=2.0),
            marker=dict(size=5, color="#f6f1ea", line=dict(color=APP_CHART_COLORS["positive"], width=1.4)),
            name="Probability Positive",
            yaxis="y2",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=18, r=18, t=64, b=18),
        hovermode="x unified",
        paper_bgcolor=APP_CHART_COLORS["surface"],
        plot_bgcolor=APP_CHART_COLORS["surface"],
        xaxis_title="As Of Date",
        yaxis=dict(title="Composite Score", range=[-100, 100]),
        yaxis2=dict(
            title="Probability Positive",
            tickformat=".0%",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, 1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=APP_CHART_COLORS["muted"])),
        title="ML Signal History",
        title_x=0.02,
        title_font=dict(size=18, color=APP_CHART_COLORS["ink"]),
        font=dict(size=12, color=APP_CHART_COLORS["muted"]),
    )
    figure.update_xaxes(showgrid=False, showline=True, linecolor=APP_CHART_COLORS["line"])
    figure.update_yaxes(gridcolor=APP_CHART_COLORS["grid"])
    return figure



def create_pillar_contribution_chart(contribution_rows: list[dict[str, object]]) -> go.Figure:
    """Build a chart for latest pillar contributions."""

    labels = [str(row["pillar"]) for row in contribution_rows]
    values = [float(row["contribution"]) for row in contribution_rows]
    colors = [APP_CHART_COLORS["primary"] if value >= 0 else APP_CHART_COLORS["negative"] for value in values]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=18, r=18, t=64, b=18),
        xaxis_title="Signal Pillar",
        yaxis_title="Contribution",
        title="Latest Pillar Contribution Breakdown",
        paper_bgcolor=APP_CHART_COLORS["surface"],
        plot_bgcolor=APP_CHART_COLORS["surface"],
        title_x=0.02,
        title_font=dict(size=18, color=APP_CHART_COLORS["ink"]),
        font=dict(size=12, color=APP_CHART_COLORS["muted"]),
    )
    figure.update_xaxes(showgrid=False, showline=True, linecolor=APP_CHART_COLORS["line"])
    figure.update_yaxes(gridcolor=APP_CHART_COLORS["grid"])
    return figure



def create_feature_importance_chart(feature_rows: list[dict[str, object]]) -> go.Figure:
    """Build a chart for top feature importance rows."""

    ordered = sorted(feature_rows, key=lambda row: float(row.get("importance", 0.0)), reverse=True)
    labels = [str(row.get("feature", "")).replace("_", " ").title() for row in ordered]
    values = [float(row.get("importance", 0.0)) for row in ordered]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=APP_CHART_COLORS["primary"],
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=18, r=18, t=64, b=18),
        xaxis_title="Importance",
        yaxis_title="Feature",
        title="Top ML Feature Importance",
        paper_bgcolor=APP_CHART_COLORS["surface"],
        plot_bgcolor=APP_CHART_COLORS["surface"],
        title_x=0.02,
        title_font=dict(size=18, color=APP_CHART_COLORS["ink"]),
        font=dict(size=12, color=APP_CHART_COLORS["muted"]),
    )
    figure.update_xaxes(showgrid=True, gridcolor=APP_CHART_COLORS["grid"], showline=True, linecolor=APP_CHART_COLORS["line"])
    figure.update_yaxes(showgrid=False, autorange="reversed")
    return figure



def save_ml_score_history_chart(
    prediction_history: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Render a static composite-score history chart for PDF reporting."""

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["forecast_history"])
    ax.plot(
        prediction_history["as_of_date"],
        prediction_history["composite_ml_score"],
        color="#0f4c81",
        linewidth=REPORT_LINE_WIDTH - 0.2,
        marker="o",
        markersize=4,
    )
    ax.set_xlabel("As Of Date")
    _apply_report_axes_style(ax, "ML Signal History", "Composite Score")
    ax.set_ylim(-100, 100)

    secondary_ax = ax.twinx()
    secondary_ax.plot(
        prediction_history["as_of_date"],
        prediction_history["probability_positive_20d"],
        color="#2e7d32",
        linewidth=REPORT_LINE_WIDTH - 0.6,
    )
    secondary_ax.set_ylabel("Probability Positive", fontsize=REPORT_AXIS_LABEL_SIZE, color="#3d4e5f")
    secondary_ax.tick_params(axis="y", labelsize=REPORT_TICK_LABEL_SIZE, colors="#536272")
    secondary_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    secondary_ax.set_ylim(0.0, 1.0)
    secondary_ax.spines["top"].set_visible(False)
    secondary_ax.spines["left"].set_visible(False)
    secondary_ax.spines["right"].set_color("#ccd6df")
    secondary_ax.spines["right"].set_linewidth(0.9)

    _save_matplotlib_figure(output_path)
    return str(output_path)



def save_pillar_contribution_chart(
    contribution_rows: list[dict[str, object]],
    output_path: str | Path,
) -> str:
    """Render a static pillar-contribution chart for PDF reporting."""

    labels = [str(row["pillar"]) for row in contribution_rows]
    values = [float(row["contribution"]) for row in contribution_rows]
    colors = ["#0f4c81" if value >= 0 else "#b0453b" for value in values]

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["feature_drivers"])
    ax.bar(labels, values, color=colors, alpha=0.88)
    _apply_report_axes_style(ax, "Pillar Contribution Breakdown", "Contribution")
    _save_matplotlib_figure(output_path)
    return str(output_path)



def save_feature_importance_breakdown_chart(
    feature_rows: list[dict[str, object]],
    output_path: str | Path,
) -> str:
    """Render a static top-feature-importance chart for PDF reporting."""

    ordered = sorted(feature_rows, key=lambda row: float(row.get("importance", 0.0)), reverse=True)
    labels = [str(row.get("feature", "")).replace("_", " ").title() for row in ordered]
    values = [float(row.get("importance", 0.0)) for row in ordered]

    fig, ax = plt.subplots(figsize=REPORT_CHART_SIZES["feature_drivers"])
    ax.barh(labels, values, color="#0f4c81", alpha=0.88)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    _apply_report_axes_style(ax, "Top ML Feature Importance", "Feature")
    _save_matplotlib_figure(output_path)
    return str(output_path)
