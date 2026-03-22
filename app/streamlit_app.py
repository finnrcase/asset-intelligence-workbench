"""
Streamlit application for the first-pass Asset Intelligence Workbench UI.
"""

from __future__ import annotations

import importlib
import logging
import math
import traceback
import sys
from pathlib import Path

import streamlit as st
import sqlalchemy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
DEPLOY_MARKER = "asset-intelligence-workbench-build-2026-03-22-A"
LOGGER = logging.getLogger(__name__)


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

    st.title("Asset Intelligence Workbench")
    st.caption("SQL-Driven Financial Analytics, Risk Simulation, and Reporting Platform")
    st.markdown(
        "Single-asset analytics view built on stored historical market data, "
        "SQL-backed retrieval, and reusable return/risk calculations."
    )


def _render_empty_state(message: str) -> None:
    """Render a consistent empty-state message."""

    st.info(message)


def _render_status_message(status: dict[str, str] | None) -> None:
    """Render the most recent asset resolution message, if one exists."""

    if not status:
        return

    status_type = status.get("status")
    message = status.get("message", "")

    if not message:
        return

    if status_type in {"database"}:
        st.info(message)
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


def main() -> None:
    """Run the first-pass Streamlit dashboard."""

    _log_startup_deploy_diagnostics()

    st.set_page_config(
        page_title="Asset Intelligence Workbench",
        layout="wide",
    )

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

    with st.sidebar:
        st.subheader("Asset Selection")
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
            selected_ticker = ""

        manual_ticker = st.text_input(
            "Manual Ticker",
            value="",
            placeholder="Examples: AAPL, SPY, BTC-USD",
            help="If provided, this input takes precedence over the dropdown and will fetch the asset if needed.",
        )

        load_clicked = st.button(
            "Load Asset",
            use_container_width=True,
            type="primary",
        )

        st.caption(
            "Use the dropdown for locally stored assets, or enter a new ticker to "
            "fetch and ingest it on demand."
        )

        st.divider()
        generate_report_clicked = st.button(
            "Generate PDF Report",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Simulation")
        forecast_horizon = st.slider(
            "Forecast Horizon (Trading Days)",
            min_value=21,
            max_value=252,
            value=DEFAULT_FORECAST_HORIZON,
            step=21,
            help="Forward simulation horizon expressed in trading days.",
        )
        simulation_count = st.slider(
            "Simulation Count",
            min_value=100,
            max_value=2000,
            value=DEFAULT_SIMULATION_COUNT,
            step=100,
            help="Number of Monte Carlo paths to generate.",
        )

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
            st.rerun()

    _render_status_message(st.session_state.asset_status)

    if not st.session_state.active_ticker:
        _render_empty_state(
            "Select an existing asset or enter a ticker to fetch one into the local database."
        )
        return

    available_assets = app_data.load_available_tickers()
    metadata = app_data.load_asset_metadata(st.session_state.active_ticker)
    price_rows = app_data.load_price_history(st.session_state.active_ticker)
    price_frame = app_data.prepare_price_history_frame(price_rows)

    if metadata is None or price_frame.empty:
        _render_empty_state(
            f"No historical price data is available for {st.session_state.active_ticker} in the local database."
        )
        return

    if price_frame["analysis_price"].dropna().shape[0] < 2:
        _render_empty_state(
            f"{st.session_state.active_ticker} does not have enough clean price observations for analytics yet."
        )
        return

    sentiment_status = st.session_state.sentiment_status_by_ticker.get(
        st.session_state.active_ticker
    )
    if sentiment_status is None:
        with st.spinner("Fetching recent sentiment..."):
            sentiment_status = app_data.ensure_sentiment_for_ticker(
                st.session_state.active_ticker,
                page_size=DEFAULT_SENTIMENT_PAGE_SIZE,
            )
        st.session_state.sentiment_status_by_ticker[st.session_state.active_ticker] = sentiment_status
        if sentiment_status.get("success") and sentiment_status.get("fetched"):
            st.rerun()

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

    metadata_columns = st.columns([2, 2, 2, 2])
    metadata_columns[0].markdown(f"**Asset**  \n{metadata['asset_name']}")
    metadata_columns[1].markdown(
        f"**Classification**  \n{metadata.get('asset_class') or 'N/A'}"
    )
    metadata_columns[2].markdown(
        f"**Exchange / Currency**  \n"
        f"{metadata.get('exchange') or 'N/A'} / {metadata.get('currency') or 'N/A'}"
    )
    metadata_columns[3].markdown(
        f"**Primary Source / Load Path**  \n"
        f"{metadata.get('primary_source') or 'N/A'} / {origin_label}"
    )

    st.divider()
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

    st.subheader("Recent Price Observations")
    recent_prices = app_data.get_recent_price_table(price_frame, rows=10)
    st.dataframe(recent_prices, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Machine Learning Signal Calibration")
    st.caption(
        "This layer combines historical market structure, downside/risk context, and sentiment into a "
        "composite decision-support score. It is framed as a weighting engine for research, not a guaranteed forecast."
    )

    if not ml_summary["available"]:
        st.info(ml_summary["interpretation"])
    else:
        snapshot = ml_summary["snapshot"]
        target_definition = ml_summary.get("target_definition") or {}
        model_summary = ml_summary.get("model_summary") or {}
        training_window = ml_summary.get("training_window") or {}

        st.markdown("**Target Definition**")
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

        st.markdown("**Interpretation**")
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
            st.markdown("**Learned Pillar Weights**")
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

    st.divider()
    st.subheader("News Sentiment")

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

        st.caption(
            "Recent news sentiment is based on stored headline/article records and a lightweight "
            "lexical sentiment score intended for directional context rather than deep NLP inference."
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

        st.markdown("**Recent Headlines**")
        recent_sentiment = app_data.get_recent_sentiment_table(sentiment_rows, rows=8)
        st.dataframe(recent_sentiment, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Forward Simulation")

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

    st.caption(
        "The base scenario uses historical daily returns to estimate drift and volatility. "
        "The ML-informed scenario overlays the latest model-implied expected return and current "
        "risk proxy so analysts can compare a purely historical calibration with a current forecast context."
    )
    _render_simulation_metrics(terminal_summary)

    if ml_summary["available"]:
        st.markdown("**ML-Informed Scenario Overlay**")
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
