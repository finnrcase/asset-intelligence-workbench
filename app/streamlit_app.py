"""
Streamlit application for the first-pass Asset Intelligence Workbench UI.
"""

from __future__ import annotations

from html import escape
import importlib
import logging
import math
import traceback
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import streamlit as st
import sqlalchemy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.database.connection as database_connection
from src.analytics.returns import build_return_frame
from src.analytics.returns import compute_annualized_return
from src.analytics.returns import compute_total_return
from src.analytics.risk import build_risk_summary
from src.analytics.risk import compute_rolling_volatility
APP_IMPORT_ERRORS: list[Exception] = []

try:
    from src.analytics import simulation as simulation_module

    simulation_module = importlib.reload(simulation_module)
except ModuleNotFoundError as exc:
    simulation_module = None
    APP_IMPORT_ERRORS.append(exc)

if simulation_module is not None:
    run_comparative_monte_carlo_simulation = getattr(
        simulation_module,
        "run_comparative_monte_carlo_simulation",
        None,
    )
    if run_comparative_monte_carlo_simulation is None:
        legacy_simulation = getattr(simulation_module, "run_monte_carlo_simulation")

        def run_comparative_monte_carlo_simulation(*args, **kwargs):
            """Fallback when the app is running against a stale simulation module."""

            historical_result = legacy_simulation(*args, **kwargs)
            return {
                "historical": historical_result,
                "ml_informed": historical_result,
            }

try:
    from src.utils import app_data

    app_data = importlib.reload(app_data)
except Exception as exc:
    app_data = None
    APP_IMPORT_ERRORS.append(exc)

try:
    from src.visuals import charts

    charts = importlib.reload(charts)
except ModuleNotFoundError as exc:
    charts = None
    APP_IMPORT_ERRORS.append(exc)


ROLLING_VOLATILITY_WINDOW = 21
VAR_CONFIDENCE_LEVEL = 0.95
DEFAULT_FORECAST_HORIZON = 63
DEFAULT_SIMULATION_COUNT = 500
DEFAULT_SENTIMENT_PAGE_SIZE = 12
ML_SUMMARY_TIMEOUT_SECONDS = 20
DEPLOY_MARKER = "asset-intelligence-workbench-build-2026-03-22-A"
LOGGER = logging.getLogger(__name__)
APP_THEME_CSS = """
<style>
:root {
    --app-bg: #f4f6f8;
    --page-bg: #eef2f6;
    --surface: #ffffff;
    --surface-muted: #f8fafc;
    --surface-soft: #f1f5f9;
    --ink: #0f1720;
    --muted: #536273;
    --muted-soft: #6d7d8f;
    --line: #d8e1ea;
    --line-strong: #c7d3e1;
    --accent: #5f98c6;
    --accent-strong: #3f79aa;
    --accent-soft: #e8f1f8;
    --success: #2d6a4f;
    --danger: #a33f3f;
    --shadow-soft: 0 10px 28px rgba(15, 23, 32, 0.06);
    --shadow-panel: 0 4px 14px rgba(15, 23, 32, 0.05);
    --radius-xl: 22px;
    --radius-lg: 18px;
    --radius-md: 14px;
    --radius-sm: 10px;
}

.stApp {
    background: linear-gradient(180deg, var(--page-bg) 0%, var(--app-bg) 100%);
    color: var(--ink);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

[data-testid="stHeader"] {
    background: rgba(244, 246, 248, 0.85);
    backdrop-filter: blur(8px);
}

[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 1520px;
    padding-top: 1.65rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: var(--ink);
}

h1 {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    letter-spacing: -0.02em;
}

body, p, li, label, [data-testid="stMarkdownContainer"] {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}

[data-testid="stMarkdownContainer"] p {
    color: var(--muted);
    line-height: 1.55;
}

[data-testid="stDataFrame"] *,
[data-testid="stTable"] *,
.stPlotlyChart *,
[data-testid="stMetric"] *,
[data-testid="stAlert"] *,
[data-testid="stStatusWidget"] *,
.hero-pill *,
.asset-pill *,
.asset-shell *,
.panel-shell *,
.summary-card *,
.inline-note *,
div[data-baseweb="select"] *,
.stTextInput *,
.stNumberInput *,
.stSelectbox *,
.stSlider * {
    color: var(--ink) !important;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-panel);
    padding: 0.95rem 1rem;
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    font-weight: 650;
    opacity: 0.9;
}

[data-testid="stMetricValue"] {
    color: var(--ink);
    font-size: 1.35rem;
    letter-spacing: -0.03em;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    min-height: 2.8rem;
    border: 1px solid var(--line-strong);
    background: var(--surface);
    color: var(--ink);
    font-weight: 600;
    box-shadow: none;
    transition: all 140ms ease;
}

.stButton > button[kind="primary"] {
    background: var(--accent-strong);
    color: white;
    border-color: var(--accent-strong);
}

.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent);
    background: var(--surface-muted);
}

div[data-baseweb="select"] > div,
.stTextInput > div > div > input,
.stNumberInput input {
    border-radius: 10px !important;
    border: 1px solid var(--line-strong) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    box-shadow: none !important;
}

div[data-baseweb="select"] input,
div[data-baseweb="select"] span,
div[data-baseweb="select"] svg,
.stTextInput input::placeholder,
.stTextInput textarea::placeholder {
    color: var(--ink) !important;
    fill: var(--ink) !important;
    opacity: 0.62 !important;
}

[data-testid="stTabs"] [role="tablist"] {
    gap: 0.5rem;
    margin-bottom: 1rem;
}

[data-testid="stTabs"] [role="tab"] {
    height: 2.75rem;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0 1rem;
    color: var(--muted);
    font-weight: 600;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: var(--surface);
    color: var(--ink);
    border-color: var(--accent);
    box-shadow: var(--shadow-panel);
}

[data-testid="stSlider"] [role="slider"] {
    box-shadow: 0 0 0 5px rgba(95, 152, 198, 0.18);
}

[data-testid="stSliderTrack"] {
    background: linear-gradient(90deg, var(--accent-soft), rgba(95, 152, 198, 0.4));
}

.stPlotlyChart,
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-panel);
    padding: 0.65rem 0.75rem;
}

[data-testid="stAlert"] {
    border-radius: var(--radius-md);
    border: 1px solid var(--line);
    box-shadow: none;
}

[data-testid="stStatusWidget"] {
    border-radius: var(--radius-lg);
    border: 1px solid var(--line);
    background: var(--surface);
}

[data-testid="stSpinner"] {
    color: var(--muted);
}

[data-testid="stSpinner"] * {
    color: var(--muted) !important;
    stroke: var(--muted) !important;
    border-top-color: var(--muted) !important;
    border-right-color: var(--muted) !important;
}

.app-hero {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.98));
    border: 1px solid var(--line);
    border-radius: 22px;
    box-shadow: var(--shadow-soft);
    padding: 1.5rem 1.6rem 1.2rem;
    margin-bottom: 1rem;
}

.hero-kicker, .section-kicker {
    color: var(--muted-soft);
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 700;
}

.hero-title {
    margin: 0.45rem 0 0.35rem;
    font-size: clamp(1.95rem, 4vw, 2.7rem);
    line-height: 1.05;
}

.hero-copy {
    max-width: 58rem;
    font-size: 0.98rem;
    line-height: 1.6;
    color: var(--muted);
    margin: 0;
}

.hero-pill-row, .asset-pill-grid, .summary-grid {
    display: grid;
    gap: 0.75rem;
    margin-top: 1.2rem;
}

.hero-pill-row {
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}

.hero-pill, .asset-pill {
    background: var(--surface-soft);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.9rem 1rem;
}

.pill-label {
    display: block;
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.28rem;
    font-weight: 650;
}

.pill-value {
    display: block;
    color: var(--ink);
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.35;
}

.asset-shell {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow-panel);
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}

.asset-topline {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 0.95rem;
}

.asset-ticker {
    font-size: 1.85rem;
    line-height: 1;
    letter-spacing: -0.04em;
    color: var(--ink);
    font-weight: 700;
}

.asset-subtitle {
    color: var(--ink);
    font-size: 0.98rem;
    margin-top: 0.24rem;
    opacity: 0.82;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.42rem 0.72rem;
    background: var(--accent-soft);
    color: var(--accent-strong);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.asset-pill-grid {
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}

.section-intro {
    margin: 1.4rem 0 0.9rem;
    padding: 0 0.1rem;
}

.section-title {
    margin: 0.18rem 0 0.25rem;
    font-size: 1.35rem;
    line-height: 1.15;
    letter-spacing: -0.03em;
}

.section-title-text,
.minor-label-text,
.inline-note-text {
    margin: 0;
}

.section-copy {
    max-width: 52rem;
    color: var(--ink);
    font-size: 0.94rem;
    line-height: 1.6;
    margin: 0;
    opacity: 0.82;
}

.minor-label {
    color: var(--ink);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 0.55rem;
    opacity: 0.78;
}

.inline-note {
    background: var(--surface-soft);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin: 0.45rem 0 1rem;
    color: var(--ink);
    opacity: 0.9;
}

.panel-shell {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow-panel);
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}

.panel-title {
    margin: 0;
    color: var(--ink);
    font-size: 1rem;
    font-weight: 650;
}

.panel-copy {
    margin: 0.3rem 0 0;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.5;
}

.summary-grid {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

.summary-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: var(--shadow-panel);
    padding: 1rem;
}

.summary-label {
    color: var(--muted);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.summary-value {
    margin-top: 0.4rem;
    color: var(--ink);
    font-size: 1.45rem;
    font-weight: 700;
    line-height: 1.15;
}

.summary-meta {
    margin-top: 0.4rem;
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.45;
}

.control-note {
    margin-top: 0.45rem;
    color: var(--muted);
    font-size: 0.88rem;
}
</style>
"""


def _log_startup_deploy_diagnostics() -> None:
    """Log a lightweight deployment consistency snapshot for Streamlit startup."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    LOGGER.info("DEPLOY_MARKER=%s", DEPLOY_MARKER)
    LOGGER.info("SQLALCHEMY_VERSION=%s", sqlalchemy.__version__)

    try:
        import src.database.connection as connection_module

        LOGGER.info("CONNECTION_MODULE_PATH=%s", Path(connection_module.__file__).resolve())
        connection_model_names = sorted(
            name
            for name in ("Base", "DataSource", "Asset", "HistoricalPrice", "MarketDataIngestionState", "IngestionRunLog")
            if hasattr(connection_module, name)
        )
        LOGGER.info("CONNECTION_ORM_MODELS=%s", connection_model_names)

        ingestion_state = getattr(connection_module, "MarketDataIngestionState", None)
        LOGGER.info("INGESTION_STATE_IMPORTED_FROM_EXPECTED_MODULE=%s", ingestion_state is not None)
        LOGGER.info("INGESTION_STATE_REPR=%r", ingestion_state)
        LOGGER.info("INGESTION_STATE_TYPE=%s", type(ingestion_state))
    except Exception as exc:
        LOGGER.exception("DEPLOY_DIAGNOSTIC_CONNECTION_ERROR=%s", exc)

    try:
        import src.data.storage.repository as repository_module

        LOGGER.info("REPOSITORY_MODULE_PATH=%s", Path(repository_module.__file__).resolve())
    except Exception as exc:
        LOGGER.exception("DEPLOY_DIAGNOSTIC_REPOSITORY_ERROR=%s", exc)


def _format_percent(value: float) -> str:
    """Format percentage-style values for KPI display."""

    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.2%}"


def _format_number(value: float) -> str:
    """Format plain numeric KPI values for display."""

    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:,.2f}"


def _apply_theme() -> None:
    """Inject the application-level visual system."""

    st.markdown(APP_THEME_CSS, unsafe_allow_html=True)


def _render_section_intro(kicker: str, title: str, description: str) -> None:
    """Render a consistent section heading with a softer product-like hierarchy."""

    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-kicker">{escape(kicker)}</div>
            <p class="section-title section-title-text">{escape(title)}</p>
            <p class="section-copy">{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_minor_label(text: str) -> None:
    """Render a small uppercase label without relying on raw div fragments."""

    st.markdown(
        f'<p class="minor-label minor-label-text">{escape(text)}</p>',
        unsafe_allow_html=True,
    )


def _render_inline_note(text: str) -> None:
    """Render a framed explanatory note with safer HTML markup."""

    st.markdown(
        f'<p class="inline-note inline-note-text">{escape(text)}</p>',
        unsafe_allow_html=True,
    )


def _render_asset_overview(metadata: dict[str, object], ticker: str, origin_label: str, price_frame) -> None:
    """Render a compact asset summary shell above the analysis sections."""

    latest_price = price_frame["analysis_price"].dropna().iloc[-1]
    latest_date = price_frame.index.max()
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown(
            f"""
            <div class="asset-pill">
                <div class="hero-kicker">Active Coverage</div>
                <div class="asset-ticker">{escape(ticker)}</div>
                <div class="asset-subtitle">{escape(str(metadata.get("asset_name") or "Stored asset analytics view"))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(
            f'<div style="display:flex; justify-content:flex-end;"><div class="status-badge">{escape(origin_label)}</div></div>',
            unsafe_allow_html=True,
        )

    overview_items = [
        ("Asset", metadata.get("asset_name") or ticker),
        ("Classification", metadata.get("asset_class") or "N/A"),
        ("Exchange / Currency", f"{metadata.get('exchange') or 'N/A'} / {metadata.get('currency') or 'N/A'}"),
        ("Primary Source", metadata.get("primary_source") or "N/A"),
        ("Latest Observation", str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date)),
        ("Latest Analysis Price", _format_number(float(latest_price))),
    ]
    pill_columns = st.columns(3)
    for index, (label, value) in enumerate(overview_items):
        with pill_columns[index % 3]:
            st.markdown(
                f"""
                <div class="asset-pill">
                    <span class="pill-label">{escape(label)}</span>
                    <span class="pill-value">{escape(str(value))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_simulation_metrics(terminal_summary: dict[str, float]) -> None:
    """Render top-line Monte Carlo terminal outcome metrics."""

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Median Terminal Price",
        _format_number(terminal_summary["median_terminal_price"]),
    )
    metric_columns[1].metric(
        "5th Percentile",
        _format_number(terminal_summary["p05_terminal_price"]),
    )
    metric_columns[2].metric(
        "95th Percentile",
        _format_number(terminal_summary["p95_terminal_price"]),
    )
    metric_columns[3].metric(
        "P(End Above Start)",
        _format_percent(terminal_summary["probability_above_start"]),
    )
    metric_columns[4].metric(
        "P(End Below Start)",
        _format_percent(terminal_summary["probability_below_start"]),
    )


def _render_ml_forecast_metrics(ml_summary: dict[str, object]) -> None:
    """Render top-line ML forecast KPIs for the selected asset."""

    snapshot = ml_summary["snapshot"]
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Model-Implied Expected 20-Day Return",
        _format_percent(snapshot["predicted_return_20d"]),
    )
    metric_columns[1].metric(
        "Probability of Negative Return",
        _format_percent(snapshot["downside_probability_20d"]),
    )
    metric_columns[2].metric(
        "Volatility Proxy",
        _format_percent(snapshot["predicted_volatility_20d"]),
    )
    metric_columns[3].metric(
        "Downside Volatility",
        _format_percent(snapshot["downside_volatility_20d"]),
    )
    metric_columns[4].metric(
        "Regime Context",
        snapshot["regime_label"],
    )


def _render_latest_ml_snapshot(ml_summary: dict[str, object]) -> None:
    """Render a compact latest-forecast snapshot when history is too short for a chart."""

    snapshot = ml_summary["snapshot"]
    snapshot_rows = [
        {
            "As Of Date": str(snapshot.get("as_of_date") or "N/A"),
            "Forecast Horizon": f"{int(snapshot.get('prediction_horizon_days') or 20)} trading days",
            "Expected Return": _format_percent(snapshot.get("predicted_return_20d")),
            "Probability Negative": _format_percent(snapshot.get("downside_probability_20d")),
            "Volatility Proxy": _format_percent(snapshot.get("predicted_volatility_20d")),
            "Regime": snapshot.get("regime_label") or "N/A",
        }
    ]
    st.markdown("**Latest Forecast Snapshot**")
    st.dataframe(snapshot_rows, use_container_width=True, hide_index=True)


def _render_header() -> None:
    """Render the page header."""

    st.markdown(
        """
        <section class="app-hero">
            <div class="hero-kicker">Asset Intelligence Platform</div>
            <h1 class="hero-title">Asset Intelligence Workbench</h1>
            <p class="hero-copy">
                A focused analytics workspace for stored market data, risk framing, model calibration,
                and analyst-ready reporting.
            </p>
            <div class="hero-pill-row">
                <div class="hero-pill">
                    <span class="pill-label">Built For</span>
                    <span class="pill-value">Finance, strategy, and infra research teams</span>
                </div>
                <div class="hero-pill">
                    <span class="pill-label">Core Lens</span>
                    <span class="pill-value">Market history, risk context, ML signals, and reporting</span>
                </div>
                <div class="hero-pill">
                    <span class="pill-label">Operating Model</span>
                    <span class="pill-value">Single-asset analysis on top of local SQL-backed retrieval</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state(message: str) -> None:
    """Render a consistent empty-state message."""

    st.info(message)


def _render_panel_header(title: str, description: str | None = None) -> None:
    """Render a consistent panel header above a control or output block."""

    description_markup = (
        f'<p class="panel-copy">{escape(description)}</p>'
        if description
        else ""
    )
    st.markdown(
        f"""
        <div class="panel-shell">
            <p class="panel-title">{escape(title)}</p>
            {description_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_card(label: str, value: str, meta: str | None = None) -> None:
    """Render a compact executive summary card."""

    meta_markup = f'<div class="summary-meta">{escape(meta)}</div>' if meta else ""
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{escape(label)}</div>
            <div class="summary-value">{escape(value)}</div>
            {meta_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chart_block(title: str, description: str | None = None) -> None:
    """Render a consistent heading for a chart or analytical block."""

    _render_panel_header(title, description)


def _render_status_message(status: dict[str, str] | None) -> None:
    """Render the most recent asset resolution message, if one exists."""

    if not status:
        return

    status_type = status.get("status")
    message = status.get("message", "")

    if not message:
        return

    if status_type in {"database"}:
        st.success(message)
    elif status_type in {"ingested"}:
        st.success(message)
    elif status_type in {"rate_limited"}:
        st.warning(message)
    else:
        st.error(message)


def _render_kpis(price_frame, risk_summary: dict[str, float]) -> None:
    """Render top-line KPI metrics for the selected asset."""

    total_return = compute_total_return(price_frame, price_column="analysis_price")
    annualized_return = compute_annualized_return(
        price_frame,
        price_column="analysis_price",
    )

    metric_columns = st.columns(6)
    metric_columns[0].metric("Total Return", _format_percent(total_return))
    metric_columns[1].metric("Annualized Return", _format_percent(annualized_return))
    metric_columns[2].metric(
        "Annualized Volatility",
        _format_percent(risk_summary["annualized_volatility"]),
    )
    metric_columns[3].metric("Max Drawdown", _format_percent(risk_summary["max_drawdown"]))
    metric_columns[4].metric("Historical VaR (95%)", _format_percent(risk_summary["historical_var"]))
    metric_columns[5].metric(
        "Expected Shortfall (95%)",
        _format_percent(risk_summary["expected_shortfall"]),
    )


def _render_sentiment_summary(summary: dict[str, object]) -> None:
    """Render top-line sentiment summary metrics."""

    metric_columns = st.columns(5)
    metric_columns[0].metric("Articles", f"{summary['article_count']:,}")
    metric_columns[1].metric(
        "Average Sentiment",
        _format_number(summary["average_sentiment"]) if summary["average_sentiment"] is not None else "N/A",
    )
    metric_columns[2].metric("Positive", f"{summary['positive_count']:,}")
    metric_columns[3].metric("Neutral", f"{summary['neutral_count']:,}")
    metric_columns[4].metric("Negative", f"{summary['negative_count']:,}")


def _load_pdf_report_module():
    """Load the PDF reporting module lazily so app startup does not require FPDF."""

    try:
        from src.reporting import pdf_report as report_module
    except ModuleNotFoundError as exc:
        if exc.name == "fpdf":
            return None, "PDF export is unavailable because the `fpdf` package is not installed."
        raise
    return importlib.reload(report_module), None


def _build_ml_summary_with_timeout(ticker: str) -> dict[str, object]:
    """Bound on-demand ML summary work so the Streamlit request cannot hang indefinitely."""

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(app_data.build_ml_forecast_summary, ticker)
    try:
        return future.result(timeout=ML_SUMMARY_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        future.cancel()
        LOGGER.warning(
            "Run analysis ML summary timed out for %s after %s seconds.",
            ticker,
            ML_SUMMARY_TIMEOUT_SECONDS,
        )
        return {
            "available": False,
            "snapshot": None,
            "prediction_history": __import__("pandas").DataFrame(),
            "feature_drivers": [],
            "feature_importance": [],
            "pillar_weights": [],
            "pillar_contributions": [],
            "build_status": {
                "success": False,
                "status": "ml_timeout",
                "message": (
                    f"Machine-learning summary build for {ticker.upper()} exceeded "
                    f"{ML_SUMMARY_TIMEOUT_SECONDS} seconds and was skipped for this run."
                ),
            },
            "target_definition": None,
            "interpretation": (
                f"Machine-learning summary build for {ticker.upper()} exceeded "
                f"{ML_SUMMARY_TIMEOUT_SECONDS} seconds and was skipped for this run. "
                "The rest of the analysis remains available."
            ),
        }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _build_analysis_signature(ticker: str, forecast_horizon: int, simulation_count: int) -> tuple[str, int, int]:
    """Build a stable cache key for the current analysis request."""

    return (ticker, forecast_horizon, simulation_count)


def _clear_analysis_state() -> None:
    """Reset cached analysis artifacts so outputs only show fresh runs."""

    st.session_state.analysis_outputs = None
    st.session_state.analysis_error = None


def main() -> None:
    """Run the first-pass Streamlit dashboard."""

    if "database_engine_ready" not in st.session_state:
        st.session_state.database_engine_ready = False
    if not st.session_state.database_engine_ready:
        database_connection.reset_database_engine()
        st.session_state.database_engine_ready = True
    else:
        getattr(database_connection, "ensure_database_engine", database_connection.get_engine)()
    _log_startup_deploy_diagnostics()

    st.set_page_config(
        page_title="Asset Intelligence Workbench",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _apply_theme()
    _render_header()

    if app_data is None or charts is None:
        missing_packages = sorted(
            {
                exc.name
                for exc in APP_IMPORT_ERRORS
                if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None)
            }
        )
        startup_errors = [
            exc for exc in APP_IMPORT_ERRORS if not isinstance(exc, ModuleNotFoundError)
        ]
        if startup_errors:
            startup_error = startup_errors[0]
            st.error(
                "The app cannot start because database startup validation failed.\n\n"
                f"{startup_error}"
            )
            st.code(
                traceback.format_exception_only(type(startup_error), startup_error)[-1].strip(),
                language="text",
            )
        else:
            missing_label = ", ".join(f"`{name}`" for name in missing_packages) or "`a required package`"
            st.error(
                "The app cannot start because required Python dependencies are missing.\n\n"
                f"Missing package(s): {missing_label}"
            )
            st.code("pip install -r requirements.txt", language="bash")
            st.info(
                "If you already installed dependencies, make sure Streamlit is running from the same "
                "Python environment or virtual environment as the project."
            )
        return

    available_assets = app_data.load_available_tickers()
    asset_labels = {asset["label"]: asset["ticker"] for asset in available_assets}
    label_by_ticker = {asset["ticker"]: asset["label"] for asset in available_assets}

    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = available_assets[0]["ticker"] if available_assets else ""
    if "asset_status" not in st.session_state:
        st.session_state.asset_status = None
    if "report_status" not in st.session_state:
        st.session_state.report_status = None
    if "latest_report_path" not in st.session_state:
        st.session_state.latest_report_path = None
    if "latest_report_name" not in st.session_state:
        st.session_state.latest_report_name = None
    if "open_report_status" not in st.session_state:
        st.session_state.open_report_status = None
    if "sentiment_status_by_ticker" not in st.session_state:
        st.session_state.sentiment_status_by_ticker = {}
    if "analysis_outputs" not in st.session_state:
        st.session_state.analysis_outputs = None
    if "analysis_error" not in st.session_state:
        st.session_state.analysis_error = None
    if "active_request_ticker" not in st.session_state:
        st.session_state.active_request_ticker = ""

    tab_input, tab_outputs = st.tabs(["Inputs", "Outputs"])

    selected_ticker = ""
    manual_ticker = ""
    load_clicked = False
    run_analysis_clicked = False
    generate_report_clicked = False
    forecast_horizon = st.session_state.get("forecast_horizon_input", DEFAULT_FORECAST_HORIZON)
    simulation_count = st.session_state.get("simulation_count_input", DEFAULT_SIMULATION_COUNT)

    with tab_input:
        _render_section_intro(
            "Control Center",
            "Configure the active analysis workspace",
            "Select stored coverage or ingest a new ticker, set simulation parameters, and prepare report generation from a single operating panel.",
        )

        if st.session_state.asset_status:
            _render_status_message(st.session_state.asset_status)

        control_left, control_right = st.columns([1.35, 1], gap="large")

        with control_left:
            _render_section_intro(
                "Asset Selection",
                "Choose the active asset",
                "Load from the local database or enter a new market symbol for on-demand ingestion.",
            )
            if available_assets:
                dropdown_options = list(asset_labels.keys())
                default_index = 0
                if st.session_state.active_ticker in label_by_ticker:
                    default_index = dropdown_options.index(label_by_ticker[st.session_state.active_ticker])

                selected_label = st.selectbox(
                    "Stored Asset",
                    options=dropdown_options,
                    index=default_index,
                    help="Load an asset that already exists in the local database.",
                )
                selected_ticker = asset_labels[selected_label]
            else:
                st.caption("No stored assets are currently available.")

            manual_ticker = st.text_input(
                "Manual Ticker",
                value="",
                placeholder="Examples: AAPL, SPY, BTC-USD",
                help="If provided, this input takes precedence over the dropdown and will fetch the asset if needed.",
            )
            st.caption(
                "Use the stored asset selector for local coverage, or enter a new ticker to fetch and ingest it on demand."
            )

            action_columns = st.columns(3)
            with action_columns[0]:
                load_clicked = st.button(
                    "Load Asset",
                    use_container_width=True,
                    type="primary",
                )
            with action_columns[1]:
                run_analysis_clicked = st.button(
                    "Run Analysis",
                    use_container_width=True,
                )
            with action_columns[2]:
                generate_report_clicked = st.button(
                    "Generate PDF Report",
                    use_container_width=True,
                )

        with control_right:
            _render_section_intro(
                "Simulation Settings",
                "Set the forward analysis range",
                "Preserve the current modeling behavior while tuning the horizon and path count used in scenario generation and PDF output.",
            )
            forecast_horizon = st.slider(
                "Forecast Horizon (Trading Days)",
                min_value=21,
                max_value=252,
                value=forecast_horizon,
                step=21,
                help="Forward simulation horizon expressed in trading days.",
                key="forecast_horizon_input",
            )
            simulation_count = st.slider(
                "Simulation Count",
                min_value=100,
                max_value=2000,
                value=simulation_count,
                step=100,
                help="Number of Monte Carlo paths to generate.",
                key="simulation_count_input",
            )
            st.markdown(
                '<p class="control-note">These controls feed the existing simulation and report-generation workflows without altering their underlying logic.</p>',
                unsafe_allow_html=True,
            )

        _render_section_intro(
            "Stored Coverage",
            "Current assets available in the local database",
            "This view reflects the same stored asset inventory the app already uses for the dropdown, presented as an operations-oriented coverage table.",
        )
        if available_assets:
            st.dataframe(
                available_assets,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No stored assets are currently available in the local database.")

    if load_clicked:
        requested_ticker = (manual_ticker or selected_ticker or st.session_state.active_ticker or "").strip().upper()
        load_status_indicator = st.status(
            f"Loading asset data for {requested_ticker or 'selected asset'}...",
            expanded=True,
        )
        load_status_indicator.write("Checking the local database and resolving the requested ticker.")
        resolution = app_data.resolve_asset_for_app(
            selected_ticker=selected_ticker,
            manual_ticker=manual_ticker,
        )
        load_status_indicator.write("Preparing the asset for the analytics workspace.")
        load_status_indicator.update(
            label=(
                f"Asset ready: {resolution.get('ticker') or requested_ticker or 'selected asset'}."
                if resolution.get("success")
                else f"Asset load finished with a status for {requested_ticker or 'the selected asset'}."
            ),
            state="complete" if resolution.get("success") else "error",
            expanded=not resolution.get("success"),
        )

        st.session_state.asset_status = resolution
        if resolution.get("success"):
            st.session_state.active_ticker = resolution["ticker"]
            st.session_state.active_request_ticker = resolution.get("requested_ticker") or resolution["ticker"]
            _clear_analysis_state()
            st.rerun()

    if not st.session_state.active_ticker:
        with tab_outputs:
            _render_empty_state(
                "Select an existing asset or enter a ticker to fetch one into the local database."
            )
        return

    available_assets = app_data.load_available_tickers()
    asset_dataset_result = app_data.load_asset_dataset_for_app(st.session_state.active_ticker)
    if not asset_dataset_result.get("success"):
        status = st.session_state.asset_status or {}
        if status.get("status") == "post_ingest_readback_failed":
            with tab_outputs:
                st.error(status.get("message", "The app could not read back the newly loaded asset data."))
                st.info(
                    "The ticker was resolved through the provider, but the local database readback did not complete cleanly. "
                    "Load the ticker again once the database write is available."
                )
            st.session_state.active_ticker = ""
            return
        with tab_outputs:
            if asset_dataset_result.get("status") in {"missing_metadata", "missing_price_history", "insufficient_price_history"}:
                _render_empty_state(asset_dataset_result.get("message", "Asset data is not available yet."))
            else:
                st.error(asset_dataset_result.get("message", "Asset data validation failed before downstream analysis."))
        return

    asset_dataset = asset_dataset_result["dataset"]
    metadata = asset_dataset.metadata
    price_rows = asset_dataset.price_rows
    price_frame = asset_dataset.price_frame

    if generate_report_clicked:
        report_status_indicator = st.status(
            f"Generating report for {st.session_state.active_ticker}...",
            expanded=True,
        )
        report_status_indicator.write("Loading the PDF report module.")

        try:
            pdf_report_module, pdf_report_error = _load_pdf_report_module()
            if pdf_report_module is None:
                raise RuntimeError(pdf_report_error)

            report_status_indicator.write("Running analytics and composing the PDF report.")
            report_path = pdf_report_module.generate_asset_pdf_report(
                ticker=st.session_state.active_ticker,
                forecast_horizon=forecast_horizon,
                simulation_count=simulation_count,
            )
            st.session_state.report_status = {
                "success": True,
                "path": report_path,
            }
            # Store the exact path returned by the generator and reuse only this file
            # for the success message, browser download, and local open actions.
            resolved_report_path = Path(report_path).resolve()
            st.session_state.latest_report_path = str(resolved_report_path)
            st.session_state.latest_report_name = resolved_report_path.name
            st.session_state.open_report_status = None
            report_status_indicator.update(
                label=f"Report ready for {st.session_state.active_ticker}.",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            st.session_state.report_status = {
                "success": False,
                "message": str(exc),
            }
            st.session_state.latest_report_path = None
            st.session_state.latest_report_name = None
            st.session_state.open_report_status = None
            report_status_indicator.update(
                label=f"Report generation failed for {st.session_state.active_ticker}.",
                state="error",
                expanded=True,
            )

    analysis_signature = _build_analysis_signature(
        st.session_state.active_ticker,
        forecast_horizon,
        simulation_count,
    )

    if run_analysis_clicked:
        LOGGER.info(
            "Run analysis started: requested_ticker=%s active_ticker=%s forecast_horizon=%s simulation_count=%s",
            st.session_state.active_request_ticker or st.session_state.active_ticker,
            st.session_state.active_ticker,
            forecast_horizon,
            simulation_count,
        )
        analysis_status_indicator = st.status(
            f"Running analysis for {st.session_state.active_ticker}...",
            expanded=True,
        )
        try:
            sentiment_status = st.session_state.sentiment_status_by_ticker.get(
                st.session_state.active_ticker
            )
            if sentiment_status is None:
                LOGGER.info("Run analysis sentiment fetch started: ticker=%s", st.session_state.active_ticker)
                analysis_status_indicator.write("Fetching recent sentiment context.")
                sentiment_status = app_data.ensure_sentiment_for_ticker(
                    st.session_state.active_ticker,
                    page_size=DEFAULT_SENTIMENT_PAGE_SIZE,
                )
                st.session_state.sentiment_status_by_ticker[st.session_state.active_ticker] = sentiment_status
                LOGGER.info("Run analysis sentiment fetch completed: ticker=%s", st.session_state.active_ticker)

            LOGGER.info("Run analysis analytics build started: ticker=%s", st.session_state.active_ticker)
            analysis_status_indicator.write("Building market analytics and risk outputs.")
            return_frame = build_return_frame(price_frame, price_column="analysis_price")
            ml_summary = _build_ml_summary_with_timeout(st.session_state.active_ticker)
            rolling_volatility = compute_rolling_volatility(
                return_frame["daily_return"],
                window=ROLLING_VOLATILITY_WINDOW,
            )
            risk_summary = build_risk_summary(
                price_frame,
                price_column="analysis_price",
                confidence_level=VAR_CONFIDENCE_LEVEL,
                volatility_window=ROLLING_VOLATILITY_WINDOW,
            )
            LOGGER.info("Run analysis analytics build completed: ticker=%s", st.session_state.active_ticker)

            LOGGER.info("Run analysis simulation started: ticker=%s", st.session_state.active_ticker)
            analysis_status_indicator.write("Preparing output tables and simulation results.")
            data_origin = (st.session_state.asset_status or {}).get("status", "database")
            origin_label = "Newly Ingested" if data_origin == "ingested" else "Local Database"
            sentiment_rows = app_data.load_recent_news_articles(
                st.session_state.active_ticker,
                limit=25,
            )
            sentiment_summary = app_data.get_sentiment_summary(sentiment_rows)
            recent_prices = app_data.get_recent_price_table(price_frame, rows=10)

            simulation_result = None
            simulation_error = None
            try:
                simulation_result = run_comparative_monte_carlo_simulation(
                    price_frame,
                    price_column="analysis_price",
                    ml_forecast_snapshot=ml_summary["snapshot"],
                    horizon_days=forecast_horizon,
                    simulation_count=simulation_count,
                )
            except ValueError as exc:
                simulation_error = str(exc)
            LOGGER.info(
                "Run analysis simulation completed: ticker=%s simulation_error=%s",
                st.session_state.active_ticker,
                simulation_error is not None,
            )

            st.session_state.analysis_outputs = {
                "signature": analysis_signature,
                "metadata": metadata,
                "price_frame": price_frame,
                "return_frame": return_frame,
                "ml_summary": ml_summary,
                "rolling_volatility": rolling_volatility,
                "risk_summary": risk_summary,
                "origin_label": origin_label,
                "sentiment_status": sentiment_status,
                "sentiment_rows": sentiment_rows,
                "sentiment_summary": sentiment_summary,
                "recent_prices": recent_prices,
                "simulation_result": simulation_result,
                "simulation_error": simulation_error,
            }
            st.session_state.analysis_error = None
            analysis_status_indicator.update(
                label=f"Analysis complete for {st.session_state.active_ticker}.",
                state="complete",
                expanded=False,
            )
            LOGGER.info("Run analysis finished: ticker=%s", st.session_state.active_ticker)
        except Exception as exc:
            _clear_analysis_state()
            st.session_state.analysis_error = str(exc)
            LOGGER.exception("Run analysis failed: ticker=%s", st.session_state.active_ticker)
            analysis_status_indicator.update(
                label=f"Analysis failed for {st.session_state.active_ticker}.",
                state="error",
                expanded=True,
            )
            analysis_status_indicator.write(str(exc))

    analysis_outputs = st.session_state.analysis_outputs
    has_current_outputs = (
        analysis_outputs is not None
        and analysis_outputs.get("signature") == analysis_signature
    )

    with tab_outputs:
        _render_section_intro(
            "Report Workspace",
            "Deliverables and analysis outputs",
            "Review generated deliverables first, then work through the complete analytical output stack using the active asset and current model settings.",
        )

        if st.session_state.report_status:
            if st.session_state.report_status.get("success"):
                latest_report_path = st.session_state.latest_report_path
                latest_report_name = st.session_state.latest_report_name
                st.success(
                    "PDF report generated successfully.\n\n"
                    f"`{latest_report_path}`"
                )
                if latest_report_name:
                    st.caption(f"Generated file: `{latest_report_name}`")

                if latest_report_path is None:
                    st.warning("The generated report path is not available in the current session.")
                else:
                    report_path = Path(latest_report_path).resolve()
                    if report_path.exists():
                        report_bytes = report_path.read_bytes()
                        action_columns = st.columns([1.2, 1.1, 3], gap="medium")
                        with action_columns[0]:
                            st.download_button(
                                label="Download Report",
                                data=report_bytes,
                                file_name=latest_report_name or report_path.name,
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        with action_columns[1]:
                            if st.button(
                                "Open Report Locally",
                                use_container_width=True,
                                key="outputs_open_report_button",
                            ):
                                pdf_report_module, pdf_report_error = _load_pdf_report_module()
                                if pdf_report_module is None:
                                    st.session_state.open_report_status = {
                                        "success": "false",
                                        "message": pdf_report_error,
                                    }
                                else:
                                    st.session_state.open_report_status = pdf_report_module.open_report_locally(
                                        report_path
                                    )
                    else:
                        st.warning(
                            "The latest generated report file could not be found. "
                            f"Expected path: `{report_path}`"
                        )
            else:
                st.error(
                    "Report generation failed.\n\n"
                    f"{st.session_state.report_status.get('message', 'Unknown error')}"
                )
        else:
            st.info(
                "Generate a PDF report from the Input tab to make the latest deliverable available for download here."
            )

        if st.session_state.open_report_status:
            if st.session_state.open_report_status.get("success") == "true":
                st.info(st.session_state.open_report_status["message"])
            else:
                st.warning(st.session_state.open_report_status["message"])

        if st.session_state.analysis_error and not has_current_outputs:
            st.error(
                "Analysis generation did not complete successfully.\n\n"
                f"{st.session_state.analysis_error}"
            )
            return

        if not has_current_outputs:
            if analysis_outputs is None:
                _render_empty_state(
                    "Run Analysis from the Inputs tab to generate charts, metrics, tables, and simulation outputs for the active asset."
                )
            else:
                st.info(
                    "The active inputs changed since the last completed analysis. Run Analysis again from the Inputs tab to refresh the output workspace."
                )
            return

        metadata = analysis_outputs["metadata"]
        price_frame = analysis_outputs["price_frame"]
        return_frame = analysis_outputs["return_frame"]
        ml_summary = analysis_outputs["ml_summary"]
        rolling_volatility = analysis_outputs["rolling_volatility"]
        risk_summary = analysis_outputs["risk_summary"]
        origin_label = analysis_outputs["origin_label"]
        sentiment_status = analysis_outputs["sentiment_status"]
        sentiment_rows = analysis_outputs["sentiment_rows"]
        sentiment_summary = analysis_outputs["sentiment_summary"]
        recent_prices = analysis_outputs["recent_prices"]
        simulation_result = analysis_outputs["simulation_result"]
        simulation_error = analysis_outputs["simulation_error"]

        _render_asset_overview(
            metadata=metadata,
            ticker=st.session_state.active_ticker,
            origin_label=origin_label,
            price_frame=price_frame,
        )

        _render_section_intro(
            "Market Frame",
            "Performance and risk at a glance",
            "Historical performance, realized risk, and recent price behavior remain grouped in one review workspace.",
        )
        _render_kpis(price_frame, risk_summary)

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.markdown("#### Price History")
            st.plotly_chart(
                charts.create_price_history_chart(price_frame),
                use_container_width=True,
                key="outputs_price_history_chart",
            )
        with chart_right:
            st.markdown("#### Cumulative Return")
            st.plotly_chart(
                charts.create_cumulative_return_chart(return_frame.dropna(subset=["cumulative_return"])),
                use_container_width=True,
                key="outputs_cumulative_return_chart",
            )

        rolling_volatility_clean = rolling_volatility.dropna()
        st.markdown("#### Rolling Volatility")
        if rolling_volatility_clean.empty:
            st.info(
                "Rolling volatility requires a longer return history before the full "
                f"{ROLLING_VOLATILITY_WINDOW}-day window is available."
            )
        else:
            st.plotly_chart(
                charts.create_rolling_volatility_chart(rolling_volatility_clean),
                use_container_width=True,
                key="outputs_rolling_volatility_chart",
            )

        _render_minor_label("Recent Price Observations")
        st.dataframe(recent_prices, use_container_width=True, hide_index=True)

        _render_section_intro(
            "Model Layer",
            "Machine learning signal calibration",
            "This layer combines historical market structure, downside and volatility context, and sentiment into a composite research weighting engine rather than a guaranteed forecast.",
        )

        if not ml_summary["available"]:
            st.info(ml_summary["interpretation"])
        else:
            snapshot = ml_summary["snapshot"]
            target_definition = ml_summary.get("target_definition") or {}
            model_summary = ml_summary.get("model_summary") or {}
            training_window = ml_summary.get("training_window") or {}

            _render_minor_label("Target Definition")
            st.info(target_definition.get("summary", "The current ML target definition is unavailable."))

            overview_columns = st.columns(4)
            overview_columns[0].metric("Composite ML Score", _format_number(snapshot.get("composite_ml_score")))
            overview_columns[1].metric(
                "Directional Signal",
                snapshot.get("directional_signal") or snapshot.get("regime_label") or "N/A",
            )
            overview_columns[2].metric("Confidence", _format_percent(snapshot.get("confidence_score")))
            overview_columns[3].metric(
                "Selected Model",
                model_summary.get("selected_model_name") or snapshot.get("selected_model_name") or "N/A",
            )

            secondary_columns = st.columns(5)
            secondary_columns[0].metric("Expected 20-Day Return", _format_percent(snapshot.get("predicted_return_20d")))
            secondary_columns[1].metric("Probability Positive", _format_percent(snapshot.get("probability_positive_20d")))
            secondary_columns[2].metric("History Score", _format_number(snapshot.get("history_score")))
            secondary_columns[3].metric("Risk Score", _format_number(snapshot.get("risk_score")))
            secondary_columns[4].metric("Sentiment Score", _format_number(snapshot.get("sentiment_score")))

            score_reason_rows = [
                {"Field": "Composite ML Score", "Reason": snapshot.get("composite_ml_score_reason")},
                {"Field": "Confidence", "Reason": snapshot.get("confidence_score_reason")},
                {"Field": "History Score", "Reason": snapshot.get("history_score_reason")},
                {"Field": "Risk Score", "Reason": snapshot.get("risk_score_reason")},
                {"Field": "Sentiment Score", "Reason": snapshot.get("sentiment_score_reason")},
            ]
            score_reason_rows = [row for row in score_reason_rows if row["Reason"]]
            if score_reason_rows:
                st.caption("ML coverage notes")
                st.dataframe(score_reason_rows, use_container_width=True, hide_index=True)

            model_detail_rows = [
                {
                    "Target": target_definition.get("name", "forward_return_20d"),
                    "Horizon": f"{int(target_definition.get('horizon_days') or 20)} trading days",
                    "Linear Weighting Engine": model_summary.get("selected_model_name") or snapshot.get("selected_model_name") or "ridge_regression",
                    "Nonlinear Challenger": snapshot.get("regression_model_name") or "random_forest_regressor",
                    "Direction Classifier": model_summary.get("classification_model_name") or snapshot.get("classification_model_name") or "random_forest_classifier",
                    "Feature Version": training_window.get("feature_version") or snapshot.get("feature_version") or "v1",
                    "As Of": str(snapshot.get("as_of_date") or "N/A"),
                }
            ]
            st.dataframe(model_detail_rows, use_container_width=True, hide_index=True)

            ml_chart_left, ml_chart_right = st.columns(2, gap="large")
            prediction_history = ml_summary["prediction_history"]
            with ml_chart_left:
                st.markdown("#### ML Score History")
                if not prediction_history.empty and prediction_history.shape[0] >= 2:
                    st.plotly_chart(
                        charts.create_ml_score_history_chart(prediction_history),
                        use_container_width=True,
                        key="outputs_ml_score_history_chart",
                    )
                else:
                    _render_latest_ml_snapshot(ml_summary)

            with ml_chart_right:
                st.markdown("#### Pillar Contribution")
                st.plotly_chart(
                    charts.create_pillar_contribution_chart(ml_summary.get("pillar_contributions") or []),
                    use_container_width=True,
                    key="outputs_pillar_contribution_chart",
                )

            _render_minor_label("Interpretation")
            st.info(ml_summary["interpretation"])

            feature_importance = ml_summary.get("feature_importance") or []
            if feature_importance:
                feature_chart_left, feature_chart_right = st.columns(2, gap="large")
                with feature_chart_left:
                    st.markdown("#### Feature Importance")
                    st.plotly_chart(
                        charts.create_feature_importance_chart(feature_importance[:8]),
                        use_container_width=True,
                        key="outputs_feature_importance_chart",
                    )
                with feature_chart_right:
                    st.markdown("#### Feature Detail")
                    st.dataframe(
                        [
                            {
                                "Feature": str(row.get("feature", "")).replace("_", " ").title(),
                                "Importance": _format_number(float(row.get("importance", 0.0))),
                            }
                            for row in feature_importance[:8]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

            pillar_weight_rows = ml_summary.get("pillar_weights") or []
            if pillar_weight_rows:
                _render_minor_label("Learned Pillar Weights")
                st.dataframe(
                    [
                        {
                            "Pillar": str(row.get("pillar", "")).title(),
                            "Weight": _format_percent(float(row.get("weight", 0.0))),
                        }
                        for row in pillar_weight_rows
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        _render_section_intro(
            "Context Layer",
            "News sentiment",
            "Recent headline and article context remains grouped with supporting sentiment views and raw recent records.",
        )

        sentiment_ui_message = sentiment_status.get("ui_message") if sentiment_status else None
        sentiment_ui_status = sentiment_status.get("status") if sentiment_status else None
        if sentiment_ui_message:
            if sentiment_ui_status in {"live_sentiment_loaded"}:
                st.success(sentiment_ui_message)
            elif sentiment_ui_status in {"cached_sentiment_loaded"}:
                st.info(sentiment_ui_message)
            else:
                st.warning(sentiment_ui_message)

        if sentiment_summary["article_count"] == 0:
            if not sentiment_ui_message:
                st.info("No recent sentiment records are available for this asset at the moment.")
        else:
            if (
                sentiment_status
                and sentiment_status.get("success")
                and sentiment_status.get("status") == "database"
                and not sentiment_ui_message
            ):
                st.caption("Recent sentiment was loaded from the local database cache.")

            _render_inline_note(
                "Recent news sentiment is based on stored headline and article records plus a lightweight lexical score intended for directional context rather than deep NLP inference."
            )
            _render_sentiment_summary(sentiment_summary)

            sentiment_trend = app_data.get_sentiment_trend_frame(sentiment_rows)
            st.markdown("#### Sentiment Trend")
            if sentiment_trend.shape[0] >= 2:
                st.plotly_chart(
                    charts.create_sentiment_trend_chart(sentiment_trend),
                    use_container_width=True,
                    key="outputs_sentiment_trend_chart",
                )
            else:
                st.info("At least two publication dates are needed before a sentiment trend chart becomes meaningful.")

            _render_minor_label("Recent Headlines")
            recent_sentiment = app_data.get_recent_sentiment_table(sentiment_rows, rows=8)
            st.dataframe(recent_sentiment, use_container_width=True, hide_index=True)

        _render_section_intro(
            "Scenario Layer",
            "Forward simulation",
            "Historical and ML-informed scenario analysis remains intact, reorganized into a cleaner review layout for terminal outcomes and path distributions.",
        )

        if simulation_error is not None:
            st.info(f"Simulation requires a sufficient clean return history. Detail: {simulation_error}")
        else:
            historical_simulation = simulation_result["historical"]
            ml_informed_simulation = simulation_result["ml_informed"]
            simulation_inputs = historical_simulation["inputs"]
            terminal_summary = historical_simulation["terminal_summary"]

            input_columns = st.columns(3)
            input_columns[0].metric("Historical Daily Drift", _format_percent(simulation_inputs["daily_drift"]))
            input_columns[1].metric("Historical Annualized Volatility", _format_percent(simulation_inputs["annualized_volatility"]))
            input_columns[2].metric("Return Observations", f"{int(simulation_inputs['observations']):,}")

            _render_inline_note(
                "The base scenario uses historical daily returns to estimate drift and volatility. The ML-informed scenario overlays the latest model-implied expected return and current risk proxy so analysts can compare a purely historical calibration with a current forecast context."
            )
            _render_simulation_metrics(terminal_summary)

            if ml_summary["available"]:
                _render_minor_label("ML-Informed Scenario Overlay")
                ml_input_columns = st.columns(4)
                ml_input_columns[0].metric("ML-Implied Daily Drift", _format_percent(ml_informed_simulation["inputs"]["daily_drift"]))
                ml_input_columns[1].metric("ML Volatility Input", _format_percent(ml_informed_simulation["inputs"]["annualized_volatility"]))
                ml_input_columns[2].metric("Downside Risk Context", _format_percent(ml_summary["snapshot"]["downside_probability_20d"]))
                ml_input_columns[3].metric("ML Regime", ml_summary["snapshot"]["regime_label"])

            simulation_chart_left, simulation_chart_right = st.columns(2, gap="large")
            with simulation_chart_left:
                st.markdown("#### Historical Path Distribution")
                st.plotly_chart(
                    charts.create_monte_carlo_paths_chart(historical_simulation["paths"]),
                    use_container_width=True,
                    key="outputs_historical_paths_chart",
                )
            with simulation_chart_right:
                st.markdown("#### Historical Terminal Distribution")
                st.plotly_chart(
                    charts.create_terminal_distribution_chart(historical_simulation["paths"]),
                    use_container_width=True,
                    key="outputs_historical_terminal_distribution_chart",
                )

            st.markdown("#### Historical Percentile Bands")
            st.plotly_chart(
                charts.create_percentile_band_chart(historical_simulation["bands"]),
                use_container_width=True,
                key="outputs_historical_percentile_band_chart",
            )

            if ml_summary["available"]:
                comparison_left, comparison_right = st.columns(2, gap="large")
                with comparison_left:
                    st.markdown("#### Historical vs ML-Informed Bands")
                    st.plotly_chart(
                        charts.create_simulation_comparison_chart(
                            historical_simulation["bands"],
                            ml_informed_simulation["bands"],
                        ),
                        use_container_width=True,
                        key="outputs_simulation_comparison_chart",
                    )
                with comparison_right:
                    st.markdown("#### ML-Informed Terminal Distribution")
                    st.plotly_chart(
                        charts.create_terminal_distribution_chart(ml_informed_simulation["paths"]),
                        use_container_width=True,
                        key="outputs_ml_informed_terminal_distribution_chart",
                    )

                comparison_summary = st.columns(3)
                comparison_summary[0].metric(
                    "Historical Median Terminal Price",
                    _format_number(historical_simulation["terminal_summary"]["median_terminal_price"]),
                )
                comparison_summary[1].metric(
                    "ML-Informed Median Terminal Price",
                    _format_number(ml_informed_simulation["terminal_summary"]["median_terminal_price"]),
                )
                comparison_summary[2].metric(
                    "Median Scenario Gap",
                    _format_number(
                        ml_informed_simulation["terminal_summary"]["median_terminal_price"]
                        - historical_simulation["terminal_summary"]["median_terminal_price"]
                    ),
                )

    if False:
        _render_section_intro(
            "Executive View",
            "Visual summary of the active asset",
            "This summary surfaces the main headline signals already produced by the platform so an analyst can scan the asset state before moving into the full output workspace.",
        )
        _render_asset_overview(
            metadata=metadata,
            ticker=st.session_state.active_ticker,
            origin_label=origin_label,
            price_frame=price_frame,
        )

        summary_columns = st.columns(5)
        with summary_columns[0]:
            _render_summary_card("Total Return", _format_percent(total_return))
        with summary_columns[1]:
            _render_summary_card("Annualized Return", _format_percent(annualized_return))
        with summary_columns[2]:
            _render_summary_card("Annualized Volatility", _format_percent(risk_summary["annualized_volatility"]))
        with summary_columns[3]:
            _render_summary_card("Sentiment Articles", f"{sentiment_summary['article_count']:,}")
        with summary_columns[4]:
            _render_summary_card(
                "Report Status",
                "Ready" if st.session_state.report_status and st.session_state.report_status.get("success") else "Not Generated",
            )

        if ml_summary["available"]:
            snapshot = ml_summary["snapshot"]
            ml_summary_columns = st.columns(4)
            with ml_summary_columns[0]:
                _render_summary_card(
                    "Composite ML Score",
                    _format_number(snapshot.get("composite_ml_score")),
                    snapshot.get("directional_signal") or snapshot.get("regime_label") or "N/A",
                )
            with ml_summary_columns[1]:
                _render_summary_card(
                    "Expected 20-Day Return",
                    _format_percent(snapshot.get("predicted_return_20d")),
                    f"Confidence {_format_percent(snapshot.get('confidence_score'))}",
                )
            with ml_summary_columns[2]:
                _render_summary_card(
                    "Probability Positive",
                    _format_percent(snapshot.get("probability_positive_20d")),
                    f"Downside {_format_percent(snapshot.get('downside_probability_20d'))}",
                )
            with ml_summary_columns[3]:
                _render_summary_card(
                    "Selected Model",
                    str(
                        (ml_summary.get("model_summary") or {}).get("selected_model_name")
                        or snapshot.get("selected_model_name")
                        or "N/A"
                    ),
                    f"As of {snapshot.get('as_of_date') or 'N/A'}",
                )
        else:
            st.info(ml_summary["interpretation"])

        summary_left, summary_right = st.columns([1.35, 1], gap="large")
        with summary_left:
            st.markdown("#### Performance Snapshot")
            st.dataframe(
                [
                    {"Metric": "Total Return", "Value": _format_percent(total_return)},
                    {"Metric": "Annualized Return", "Value": _format_percent(annualized_return)},
                    {"Metric": "Max Drawdown", "Value": _format_percent(risk_summary["max_drawdown"])},
                    {"Metric": "Historical VaR (95%)", "Value": _format_percent(risk_summary["historical_var"])},
                    {"Metric": "Expected Shortfall (95%)", "Value": _format_percent(risk_summary["expected_shortfall"])},
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("#### Recent Price Table")
            st.dataframe(recent_prices, use_container_width=True, hide_index=True)

        with summary_right:
            st.markdown("#### Coverage and Context")
            st.dataframe(
                [
                    {"Field": "Ticker", "Value": st.session_state.active_ticker},
                    {"Field": "Asset Name", "Value": str(metadata.get("asset_name") or st.session_state.active_ticker)},
                    {"Field": "Asset Class", "Value": str(metadata.get("asset_class") or "N/A")},
                    {"Field": "Exchange / Currency", "Value": f"{metadata.get('exchange') or 'N/A'} / {metadata.get('currency') or 'N/A'}"},
                    {"Field": "Primary Source", "Value": str(metadata.get("primary_source") or "N/A")},
                    {"Field": "Data Origin", "Value": origin_label},
                    {"Field": "Average Sentiment", "Value": _format_number(sentiment_summary["average_sentiment"]) if sentiment_summary["average_sentiment"] is not None else "N/A"},
                ],
                use_container_width=True,
                hide_index=True,
            )

            if simulation_error is None:
                terminal_summary = simulation_result["historical"]["terminal_summary"]
                st.markdown("#### Scenario Snapshot")
                st.dataframe(
                    [
                        {"Metric": "Median Terminal Price", "Value": _format_number(terminal_summary["median_terminal_price"])},
                        {"Metric": "5th Percentile", "Value": _format_number(terminal_summary["p05_terminal_price"])},
                        {"Metric": "95th Percentile", "Value": _format_number(terminal_summary["p95_terminal_price"])},
                        {"Metric": "P(End Above Start)", "Value": _format_percent(terminal_summary["probability_above_start"])},
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"Scenario summary is unavailable until simulation inputs are sufficient. Detail: {simulation_error}")

        summary_chart_left, summary_chart_right = st.columns(2, gap="large")
        with summary_chart_left:
            st.markdown("#### Price History")
            st.plotly_chart(
                charts.create_price_history_chart(price_frame),
                use_container_width=True,
                key="summary_price_history_chart",
            )
        with summary_chart_right:
            if simulation_error is None:
                st.markdown("#### Simulation Percentile Bands")
                st.plotly_chart(
                    charts.create_percentile_band_chart(simulation_result["historical"]["bands"]),
                    use_container_width=True,
                    key="summary_percentile_band_chart",
                )
            else:
                sentiment_trend = app_data.get_sentiment_trend_frame(sentiment_rows)
                st.markdown("#### Sentiment Overview")
                if sentiment_trend.shape[0] >= 2:
                    st.plotly_chart(
                        charts.create_sentiment_trend_chart(sentiment_trend),
                        use_container_width=True,
                        key="summary_sentiment_trend_chart",
                    )
                else:
                    st.info("Summary charts will expand as additional sentiment history becomes available.")

    return

    if st.session_state.report_status:
        if st.session_state.report_status.get("success"):
            latest_report_path = st.session_state.latest_report_path
            latest_report_name = st.session_state.latest_report_name
            st.success(
                "PDF report generated successfully.\n\n"
                f"`{latest_report_path}`"
            )
            if latest_report_name:
                st.caption(f"Generated file: `{latest_report_name}`")

            if latest_report_path is None:
                st.warning("The generated report path is not available in the current session.")
            else:
                # All follow-on actions intentionally reuse the exact latest generated file.
                report_path = Path(latest_report_path).resolve()
                if report_path.exists():
                    report_bytes = report_path.read_bytes()
                    action_columns = st.columns([1, 1, 3])
                    with action_columns[0]:
                        st.download_button(
                            label="Download Report",
                            data=report_bytes,
                            file_name=latest_report_name or report_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    with action_columns[1]:
                        if st.button("Open Report Locally", use_container_width=True):
                            pdf_report_module, pdf_report_error = _load_pdf_report_module()
                            if pdf_report_module is None:
                                st.session_state.open_report_status = {
                                    "success": "false",
                                    "message": pdf_report_error,
                                }
                            else:
                                st.session_state.open_report_status = pdf_report_module.open_report_locally(
                                report_path
                                )
                else:
                    st.warning(
                        "The latest generated report file could not be found. "
                        f"Expected path: `{report_path}`"
                    )
        else:
            st.error(
                "Report generation failed.\n\n"
                f"{st.session_state.report_status.get('message', 'Unknown error')}"
            )

    if st.session_state.open_report_status:
        if st.session_state.open_report_status.get("success") == "true":
            st.info(st.session_state.open_report_status["message"])
        else:
            st.warning(st.session_state.open_report_status["message"])

    with st.spinner(f"Building analytics output for {st.session_state.active_ticker}..."):
        return_frame = build_return_frame(price_frame, price_column="analysis_price")
        ml_summary = app_data.build_ml_forecast_summary(st.session_state.active_ticker)
        rolling_volatility = compute_rolling_volatility(
            return_frame["daily_return"],
            window=ROLLING_VOLATILITY_WINDOW,
        )
        risk_summary = build_risk_summary(
            price_frame,
            price_column="analysis_price",
            confidence_level=VAR_CONFIDENCE_LEVEL,
            volatility_window=ROLLING_VOLATILITY_WINDOW,
        )

    data_origin = (st.session_state.asset_status or {}).get("status", "database")
    origin_label = "Newly Ingested" if data_origin == "ingested" else "Local Database"
    _render_asset_overview(
        metadata=metadata,
        ticker=st.session_state.active_ticker,
        origin_label=origin_label,
        price_frame=price_frame,
    )

    _render_section_intro(
        "Market Frame",
        "Performance and risk at a glance",
        "",
    )
    _render_kpis(price_frame, risk_summary)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            charts.create_price_history_chart(price_frame),
            use_container_width=True,
        )
    with chart_right:
        st.plotly_chart(
            charts.create_cumulative_return_chart(return_frame.dropna(subset=["cumulative_return"])),
            use_container_width=True,
        )

    rolling_volatility_clean = rolling_volatility.dropna()
    if rolling_volatility_clean.empty:
        st.info(
            "Rolling volatility requires a longer return history before the full "
            f"{ROLLING_VOLATILITY_WINDOW}-day window is available."
        )
    else:
        st.plotly_chart(
            charts.create_rolling_volatility_chart(rolling_volatility_clean),
            use_container_width=True,
        )

    _render_minor_label("Recent Price Observations")
    recent_prices = app_data.get_recent_price_table(price_frame, rows=10)
    st.dataframe(recent_prices, use_container_width=True, hide_index=True)

    _render_section_intro(
        "Model Layer",
        "Machine learning signal calibration",
        "This layer combines historical market structure, downside and volatility context, and sentiment into a composite research weighting engine rather than a guaranteed forecast.",
    )

    if not ml_summary["available"]:
        st.info(ml_summary["interpretation"])
    else:
        snapshot = ml_summary["snapshot"]
        target_definition = ml_summary.get("target_definition") or {}
        model_summary = ml_summary.get("model_summary") or {}
        training_window = ml_summary.get("training_window") or {}

        _render_minor_label("Target Definition")
        st.info(target_definition.get("summary", "The current ML target definition is unavailable."))

        overview_columns = st.columns(4)
        overview_columns[0].metric(
            "Composite ML Score",
            _format_number(snapshot.get("composite_ml_score")),
        )
        overview_columns[1].metric(
            "Directional Signal",
            snapshot.get("directional_signal") or snapshot.get("regime_label") or "N/A",
        )
        overview_columns[2].metric(
            "Confidence",
            _format_percent(snapshot.get("confidence_score")),
        )
        overview_columns[3].metric(
            "Selected Model",
            model_summary.get("selected_model_name") or snapshot.get("selected_model_name") or "N/A",
        )

        secondary_columns = st.columns(5)
        secondary_columns[0].metric(
            "Expected 20-Day Return",
            _format_percent(snapshot.get("predicted_return_20d")),
        )
        secondary_columns[1].metric(
            "Probability Positive",
            _format_percent(snapshot.get("probability_positive_20d")),
        )
        secondary_columns[2].metric(
            "History Score",
            _format_number(snapshot.get("history_score")),
        )
        secondary_columns[3].metric(
            "Risk Score",
            _format_number(snapshot.get("risk_score")),
        )
        secondary_columns[4].metric(
            "Sentiment Score",
            _format_number(snapshot.get("sentiment_score")),
        )

        score_reason_rows = [
            {
                "Field": "Composite ML Score",
                "Reason": snapshot.get("composite_ml_score_reason"),
            },
            {
                "Field": "Confidence",
                "Reason": snapshot.get("confidence_score_reason"),
            },
            {
                "Field": "History Score",
                "Reason": snapshot.get("history_score_reason"),
            },
            {
                "Field": "Risk Score",
                "Reason": snapshot.get("risk_score_reason"),
            },
            {
                "Field": "Sentiment Score",
                "Reason": snapshot.get("sentiment_score_reason"),
            },
        ]
        score_reason_rows = [row for row in score_reason_rows if row["Reason"]]
        if score_reason_rows:
            st.caption("ML coverage notes")
            st.dataframe(score_reason_rows, use_container_width=True, hide_index=True)

        model_detail_rows = [
            {
                "Target": target_definition.get("name", "forward_return_20d"),
                "Horizon": f"{int(target_definition.get('horizon_days') or 20)} trading days",
                "Linear Weighting Engine": model_summary.get("selected_model_name") or snapshot.get("selected_model_name") or "ridge_regression",
                "Nonlinear Challenger": snapshot.get("regression_model_name") or "random_forest_regressor",
                "Direction Classifier": model_summary.get("classification_model_name") or snapshot.get("classification_model_name") or "random_forest_classifier",
                "Feature Version": training_window.get("feature_version") or snapshot.get("feature_version") or "v1",
                "As Of": str(snapshot.get("as_of_date") or "N/A"),
            }
        ]
        st.dataframe(model_detail_rows, use_container_width=True, hide_index=True)

        ml_chart_left, ml_chart_right = st.columns(2)
        prediction_history = ml_summary["prediction_history"]
        with ml_chart_left:
            if not prediction_history.empty and prediction_history.shape[0] >= 2:
                st.plotly_chart(
                    charts.create_ml_score_history_chart(prediction_history),
                    use_container_width=True,
                )
            else:
                _render_latest_ml_snapshot(ml_summary)

        with ml_chart_right:
            st.plotly_chart(
                charts.create_pillar_contribution_chart(ml_summary.get("pillar_contributions") or []),
                use_container_width=True,
            )

        _render_minor_label("Interpretation")
        st.info(ml_summary["interpretation"])

        feature_importance = ml_summary.get("feature_importance") or []
        if feature_importance:
            feature_chart_left, feature_chart_right = st.columns(2)
            with feature_chart_left:
                st.plotly_chart(
                    charts.create_feature_importance_chart(feature_importance[:8]),
                    use_container_width=True,
                )
            with feature_chart_right:
                st.dataframe(
                    [
                        {
                            "Feature": str(row.get("feature", "")).replace("_", " ").title(),
                            "Importance": _format_number(float(row.get("importance", 0.0))),
                        }
                        for row in feature_importance[:8]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        pillar_weight_rows = ml_summary.get("pillar_weights") or []
        if pillar_weight_rows:
            _render_minor_label("Learned Pillar Weights")
            st.dataframe(
                [
                    {
                        "Pillar": str(row.get("pillar", "")).title(),
                        "Weight": _format_percent(float(row.get("weight", 0.0))),
                    }
                    for row in pillar_weight_rows
                ],
                use_container_width=True,
                hide_index=True,
            )

    _render_section_intro(
        "Context Layer",
        "News sentiment",
        "",
    )

    sentiment_rows = app_data.load_recent_news_articles(
        st.session_state.active_ticker,
        limit=25,
    )
    sentiment_summary = app_data.get_sentiment_summary(sentiment_rows)

    sentiment_ui_message = sentiment_status.get("ui_message") if sentiment_status else None
    sentiment_ui_status = sentiment_status.get("status") if sentiment_status else None
    if sentiment_ui_message:
        if sentiment_ui_status in {"live_sentiment_loaded"}:
            st.success(sentiment_ui_message)
        elif sentiment_ui_status in {"cached_sentiment_loaded"}:
            st.info(sentiment_ui_message)
        else:
            st.warning(sentiment_ui_message)

    if sentiment_summary["article_count"] == 0:
        if not sentiment_ui_message:
            st.info(
                "No recent sentiment records are available for this asset at the moment."
            )
    else:
        if (
            sentiment_status
            and sentiment_status.get("success")
            and sentiment_status.get("status") == "database"
            and not sentiment_ui_message
        ):
            st.caption("Recent sentiment was loaded from the local database cache.")

        _render_inline_note(
            "Recent news sentiment is based on stored headline and article records plus a lightweight lexical score intended for directional context rather than deep NLP inference."
        )
        _render_sentiment_summary(sentiment_summary)

        sentiment_trend = app_data.get_sentiment_trend_frame(sentiment_rows)
        if sentiment_trend.shape[0] >= 2:
            st.plotly_chart(
                charts.create_sentiment_trend_chart(sentiment_trend),
                use_container_width=True,
            )
        else:
            st.info("At least two publication dates are needed before a sentiment trend chart becomes meaningful.")

        _render_minor_label("Recent Headlines")
        recent_sentiment = app_data.get_recent_sentiment_table(sentiment_rows, rows=8)
        st.dataframe(recent_sentiment, use_container_width=True, hide_index=True)

    _render_section_intro(
        "Scenario Layer",
        "Forward simulation",
        "",
    )

    try:
        simulation_result = run_comparative_monte_carlo_simulation(
            price_frame,
            price_column="analysis_price",
            ml_forecast_snapshot=ml_summary["snapshot"],
            horizon_days=forecast_horizon,
            simulation_count=simulation_count,
        )
    except ValueError as exc:
        st.info(f"Simulation requires a sufficient clean return history. Detail: {exc}")
        return

    historical_simulation = simulation_result["historical"]
    ml_informed_simulation = simulation_result["ml_informed"]
    simulation_inputs = historical_simulation["inputs"]
    terminal_summary = historical_simulation["terminal_summary"]

    input_columns = st.columns(3)
    input_columns[0].metric(
        "Historical Daily Drift",
        _format_percent(simulation_inputs["daily_drift"]),
    )
    input_columns[1].metric(
        "Historical Annualized Volatility",
        _format_percent(simulation_inputs["annualized_volatility"]),
    )
    input_columns[2].metric(
        "Return Observations",
        f"{int(simulation_inputs['observations']):,}",
    )

    _render_inline_note(
        "The base scenario uses historical daily returns to estimate drift and volatility. The ML-informed scenario overlays the latest model-implied expected return and current risk proxy so analysts can compare a purely historical calibration with a current forecast context."
    )
    _render_simulation_metrics(terminal_summary)

    if ml_summary["available"]:
        _render_minor_label("ML-Informed Scenario Overlay")
        ml_input_columns = st.columns(4)
        ml_input_columns[0].metric(
            "ML-Implied Daily Drift",
            _format_percent(ml_informed_simulation["inputs"]["daily_drift"]),
        )
        ml_input_columns[1].metric(
            "ML Volatility Input",
            _format_percent(ml_informed_simulation["inputs"]["annualized_volatility"]),
        )
        ml_input_columns[2].metric(
            "Downside Risk Context",
            _format_percent(ml_summary["snapshot"]["downside_probability_20d"]),
        )
        ml_input_columns[3].metric(
            "ML Regime",
            ml_summary["snapshot"]["regime_label"],
        )

    simulation_chart_left, simulation_chart_right = st.columns(2)
    with simulation_chart_left:
        st.plotly_chart(
            charts.create_monte_carlo_paths_chart(historical_simulation["paths"]),
            use_container_width=True,
        )
    with simulation_chart_right:
        st.plotly_chart(
            charts.create_terminal_distribution_chart(historical_simulation["paths"]),
            use_container_width=True,
        )

    st.plotly_chart(
        charts.create_percentile_band_chart(historical_simulation["bands"]),
        use_container_width=True,
    )

    if ml_summary["available"]:
        comparison_left, comparison_right = st.columns(2)
        with comparison_left:
            st.plotly_chart(
                charts.create_simulation_comparison_chart(
                    historical_simulation["bands"],
                    ml_informed_simulation["bands"],
                ),
                use_container_width=True,
            )
        with comparison_right:
            st.plotly_chart(
                charts.create_terminal_distribution_chart(ml_informed_simulation["paths"]),
                use_container_width=True,
            )

        comparison_summary = st.columns(3)
        comparison_summary[0].metric(
            "Historical Median Terminal Price",
            _format_number(historical_simulation["terminal_summary"]["median_terminal_price"]),
        )
        comparison_summary[1].metric(
            "ML-Informed Median Terminal Price",
            _format_number(ml_informed_simulation["terminal_summary"]["median_terminal_price"]),
        )
        comparison_summary[2].metric(
            "Median Scenario Gap",
            _format_number(
                ml_informed_simulation["terminal_summary"]["median_terminal_price"]
                - historical_simulation["terminal_summary"]["median_terminal_price"]
            ),
        )


if __name__ == "__main__":
    main()
