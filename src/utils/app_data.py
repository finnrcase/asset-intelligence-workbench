"""
Application data helpers for the Streamlit layer.

These functions bridge the database query layer and the analytics/UI layer by
returning clean pandas structures suitable for charting and metric
calculation.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy.exc import OperationalError

from src.database.connection import initialize_database
from src.database.connection import session_scope
from src.database import queries as database_queries


DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_SENTIMENT_PAGE_SIZE = 12
DEFAULT_SENTIMENT_FRESHNESS_HOURS = 24
DEFAULT_ML_PREDICTION_HISTORY = 60


def load_available_tickers() -> list[dict[str, Any]]:
    """Return the available asset universe for app selection controls."""

    with session_scope() as session:
        assets = database_queries.get_asset_list(session, active_only=True)

    return [
        {
            "ticker": asset["ticker"],
            "label": f"{asset['ticker']} | {asset['asset_name']}",
            "asset_name": asset["asset_name"],
            "asset_class": asset["asset_class"],
            "exchange": asset["exchange"],
            "currency": asset["currency"],
        }
        for asset in assets
    ]


def load_asset_metadata(ticker: str) -> dict[str, Any] | None:
    """Return detailed metadata for a selected ticker."""

    with session_scope() as session:
        return database_queries.get_asset_metadata(session, ticker)


def load_price_history(ticker: str) -> list[dict[str, Any]]:
    """Return stored historical price rows for a selected ticker."""

    with session_scope() as session:
        return database_queries.get_price_history(session, ticker)


def normalize_app_ticker(ticker: str) -> str:
    """Normalize user-entered ticker text into the project storage format."""

    if not ticker or not ticker.strip():
        return ""
    return ticker.strip().upper()


def ticker_exists(ticker: str) -> bool:
    """Return True when a ticker already exists in the local asset universe."""

    normalized_ticker = normalize_app_ticker(ticker)
    if not normalized_ticker:
        return False
    return load_asset_metadata(normalized_ticker) is not None


def ingest_single_ticker(
    ticker: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Ingest one ticker into the local database using the existing market-data pipeline.

    Returns a structured result so the Streamlit layer can show clear user-facing
    status messages without catching provider/database exceptions directly.
    """

    normalized_ticker = normalize_app_ticker(ticker)
    if not normalized_ticker:
        return {
            "success": False,
            "ticker": "",
            "status": "invalid_input",
            "message": "Enter a ticker symbol before attempting to load an asset.",
        }

    initialize_database()

    existing_metadata = load_asset_metadata(normalized_ticker)
    existing_prices = load_price_history(normalized_ticker)
    if existing_metadata is not None and existing_prices:
        return {
            "success": True,
            "ticker": normalized_ticker,
            "status": "database",
            "message": f"{normalized_ticker} was loaded from the local database.",
        }

    try:
        from src.database.loaders import load_historical_prices
        from src.database.loaders import upsert_asset_metadata
        from src.ingestion.market_data import YFINANCE_SOURCE_NAME
        from src.ingestion.market_data import YFINANCE_SOURCE_TYPE
        from src.ingestion.market_data import YFINANCE_SOURCE_URL
        from src.ingestion.market_data import YFinanceMarketDataClient
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ingestion_import_error",
            "message": (
                "The on-demand ingestion components could not be loaded. "
                f"Import detail: {exc}"
            ),
        }

    client = YFinanceMarketDataClient()

    try:
        metadata = client.fetch_asset_metadata(normalized_ticker).as_dict()
        price_rows = client.fetch_normalized_price_rows(
            normalized_ticker,
            lookback_days=lookback_days,
        )
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "provider_error",
            "message": (
                f"Unable to resolve or download market data for {normalized_ticker}. "
                f"Provider detail: {exc}"
            ),
        }

    if not metadata.get("asset_name"):
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "missing_metadata",
            "message": f"{normalized_ticker} did not return usable asset metadata.",
        }

    if not price_rows:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "missing_prices",
            "message": f"{normalized_ticker} did not return daily historical price data.",
        }

    try:
        with session_scope() as session:
            upsert_asset_metadata(
                session=session,
                assets=[metadata],
                source_name=YFINANCE_SOURCE_NAME,
                source_type=YFINANCE_SOURCE_TYPE,
                source_url=YFINANCE_SOURCE_URL,
            )
            load_historical_prices(
                session=session,
                ticker=normalized_ticker,
                price_rows=price_rows,
                source_name=YFINANCE_SOURCE_NAME,
                source_type=YFINANCE_SOURCE_TYPE,
                source_url=YFINANCE_SOURCE_URL,
            )
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ingestion_error",
            "message": f"Failed to write {normalized_ticker} into the local database. Detail: {exc}",
        }

    return {
        "success": True,
        "ticker": normalized_ticker,
        "status": "ingested",
        "message": f"{normalized_ticker} was fetched from the provider and added to the local database.",
    }


def resolve_asset_for_app(
    selected_ticker: str | None = None,
    manual_ticker: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Resolve an asset for the app from either the dropdown or manual ticker input.

    Manual input takes precedence when supplied. Existing database assets are
    loaded directly; missing assets are ingested on demand.
    """

    manual_normalized = normalize_app_ticker(manual_ticker or "")
    if manual_normalized:
        return ingest_single_ticker(manual_normalized, lookback_days=lookback_days)

    selected_normalized = normalize_app_ticker(selected_ticker or "")
    if not selected_normalized:
        return {
            "success": False,
            "ticker": "",
            "status": "no_selection",
            "message": "Select an existing asset or enter a new ticker to continue.",
        }

    if ticker_exists(selected_normalized):
        return {
            "success": True,
            "ticker": selected_normalized,
            "status": "database",
            "message": f"{selected_normalized} was loaded from the local database.",
        }

    return ingest_single_ticker(selected_normalized, lookback_days=lookback_days)


def prepare_price_history_frame(price_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert queried price rows into a clean time-series DataFrame for the app.

    The app prefers `adjusted_close` when available and falls back to
    `close_price` otherwise.
    """

    if not price_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(price_rows).copy()
    frame["price_date"] = pd.to_datetime(frame["price_date"])

    numeric_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close",
        "volume",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["analysis_price"] = frame["adjusted_close"].where(
        frame["adjusted_close"].notna(),
        frame["close_price"],
    )
    frame = frame.sort_values("price_date").set_index("price_date")
    frame.index.name = "price_date"
    return frame


def get_recent_price_table(frame: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    """Return a clean recent-price table for terminal or app display."""

    if frame.empty:
        return pd.DataFrame()

    recent = frame.reset_index().sort_values("price_date", ascending=False).head(rows).copy()
    display_columns = [
        "price_date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close",
        "volume",
    ]
    available_columns = [column for column in display_columns if column in recent.columns]
    return recent[available_columns]


def load_recent_news_articles(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return recent stored news sentiment rows for a selected ticker."""

    with session_scope() as session:
        get_recent_news = getattr(database_queries, "get_recent_news_sentiment", None)
        if get_recent_news is None:
            return []
        try:
            return get_recent_news(session, ticker=ticker, limit=limit)
        except OperationalError:
            return []


def load_latest_ml_forecast(ticker: str) -> dict[str, Any] | None:
    """Return the latest ML forecast snapshot for an asset if available."""

    with session_scope() as session:
        get_latest_prediction = getattr(database_queries, "get_latest_ml_prediction", None)
        if get_latest_prediction is None:
            return None
        try:
            return get_latest_prediction(session, ticker=ticker)
        except OperationalError:
            return None


def load_ml_prediction_history(
    ticker: str,
    limit: int = DEFAULT_ML_PREDICTION_HISTORY,
) -> list[dict[str, Any]]:
    """Return recent stored ML prediction history for charting/reporting."""

    with session_scope() as session:
        get_prediction_history = getattr(database_queries, "get_ml_prediction_history", None)
        if get_prediction_history is None:
            return []
        try:
            return get_prediction_history(session, ticker=ticker, limit=limit)
        except OperationalError:
            return []


def load_ml_feature_driver_frame(ticker: str) -> pd.DataFrame:
    """Return recent feature history used to contextualize the latest forecast."""

    with session_scope() as session:
        get_driver_frame = getattr(database_queries, "get_feature_driver_frame", None)
        if get_driver_frame is None:
            return pd.DataFrame()
        try:
            return get_driver_frame(session, ticker=ticker)
        except OperationalError:
            return pd.DataFrame()


def ensure_ml_forecast_for_ticker(ticker: str) -> dict[str, Any]:
    """Build and store a model-informed forecast on demand when one is missing."""

    normalized_ticker = normalize_app_ticker(ticker)
    if not normalized_ticker:
        return {
            "success": False,
            "ticker": "",
            "status": "invalid_input",
            "message": "A valid ticker is required before an ML forecast can be built.",
        }

    existing_snapshot = load_latest_ml_forecast(normalized_ticker)
    if existing_snapshot is not None:
        return {
            "success": True,
            "ticker": normalized_ticker,
            "status": "database",
            "message": f"{normalized_ticker} model-informed forecast was loaded from the local database.",
        }

    try:
        import importlib

        from src.database import loaders as loaders_module
        from src.database import queries as queries_module
        from src.features import feature_store as feature_store_module
        from src.ml import train as train_module
        from src.ml import predict as predict_module

        loaders_module = importlib.reload(loaders_module)
        queries_module = importlib.reload(queries_module)
        feature_store_module = importlib.reload(feature_store_module)
        train_module = importlib.reload(train_module)
        predict_module = importlib.reload(predict_module)

        load_ml_model_run = getattr(loaders_module, "load_ml_model_run", None)
        refresh_feature_store = getattr(feature_store_module, "refresh_feature_store")
        train_models_from_feature_store = getattr(train_module, "train_models_from_feature_store")
        predict_from_feature_store = getattr(predict_module, "predict_from_feature_store")
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ml_import_error",
            "message": f"ML forecasting components could not be loaded. Detail: {exc}",
        }

    try:
        with session_scope() as session:
            refresh_feature_store(session=session, ticker=normalized_ticker)

            training_error: Exception | None = None
            training_result = None
            for training_scope in (None, normalized_ticker):
                try:
                    training_result = train_models_from_feature_store(
                        session=session,
                        ticker=training_scope,
                    )
                    break
                except Exception as exc:
                    training_error = exc

            if training_result is None:
                raise ValueError(
                    "Unable to train a model-informed forecast from the currently stored feature history."
                ) from training_error

            run_id = f"app_ml_run_{uuid4().hex}"
            if callable(load_ml_model_run):
                load_ml_model_run(
                    session=session,
                    run_record={
                        "run_id": run_id,
                        "run_timestamp": pd.Timestamp.utcnow().to_pydatetime(),
                        "regression_model_name": training_result["selected_models"]["regression"],
                        "classification_model_name": training_result["selected_models"]["classification"],
                        "training_start_date": str(training_result["training_frame_dates"]["start_date"]),
                        "training_end_date": str(training_result["training_frame_dates"]["end_date"]),
                        "evaluation_summary": training_result["evaluation"],
                        "feature_version": "v1",
                        "notes": f"On-demand app forecast build for {normalized_ticker}",
                    },
                )

            prediction_frame = predict_from_feature_store(
                session=session,
                training_result=training_result,
                ticker=normalized_ticker,
                model_run_id=run_id,
                write_to_sql=True,
            )
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ml_build_error",
            "message": (
                f"Unable to build a model-informed forecast for {normalized_ticker} from the current "
                f"local data. Detail: {exc}"
            ),
        }

    if prediction_frame.empty:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ml_no_prediction",
            "message": f"The forecast pipeline completed without producing a stored prediction for {normalized_ticker}.",
        }

    return {
        "success": True,
        "ticker": normalized_ticker,
        "status": "generated",
        "message": f"{normalized_ticker} model-informed forecast was generated from the local feature store.",
    }


def prepare_ml_prediction_history_frame(prediction_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare stored prediction history for charting and report usage."""

    if not prediction_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(prediction_rows).copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["prediction_generated_at"] = pd.to_datetime(
        frame["prediction_generated_at"],
        errors="coerce",
    )
    numeric_columns = [
        "predicted_return_20d",
        "downside_probability_20d",
        "prediction_horizon_days",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)


def _safe_float(value: Any) -> float | None:
    """Return a float when possible, otherwise None."""

    if value is None:
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _classify_return_outlook(expected_return: float | None) -> str:
    """Map expected return into a simple finance-oriented outlook label."""

    if expected_return is None:
        return "Unavailable"
    if expected_return >= 0.03:
        return "Constructive"
    if expected_return <= -0.03:
        return "Defensive"
    return "Balanced"


def _classify_downside_context(probability_negative: float | None) -> str:
    """Map downside probability into a planning-oriented regime label."""

    if probability_negative is None:
        return "Unavailable"
    if probability_negative >= 0.60:
        return "Elevated downside risk"
    if probability_negative <= 0.40:
        return "Contained downside risk"
    return "Mixed downside risk"


def derive_ml_feature_drivers(
    forecast_snapshot: dict[str, Any] | None,
    feature_history: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Derive lightweight driver context from the latest feature row versus history.

    This is intentionally a transparent context layer, not a claim of exact model
    attribution, because the current product stores predictions but not serialized
    feature-importance artifacts.
    """

    if not forecast_snapshot or feature_history.empty:
        return []

    latest_feature_date = pd.to_datetime(forecast_snapshot.get("as_of_date"), errors="coerce")
    if pd.isna(latest_feature_date):
        return []

    candidate_features = {
        "momentum_20d": "20-day momentum",
        "ma_distance_20d": "distance vs 20-day moving average",
        "drawdown_from_peak": "drawdown from recent peak",
        "realized_volatility_20d": "20-day realized volatility",
        "recent_realized_volatility_5d": "5-day realized volatility",
        "downside_volatility_20d": "20-day downside volatility",
        "rolling_volatility_20d": "20-day rolling volatility",
        "volume_ratio_20d": "volume vs 20-day average",
        "sentiment_mean_7d": "7-day average sentiment",
        "negative_article_share_7d": "7-day negative article share",
        "article_count_7d": "7-day article count",
    }

    history = feature_history.copy()
    history["feature_date"] = pd.to_datetime(history["feature_date"], errors="coerce")
    latest_row = history.loc[history["feature_date"] == latest_feature_date]
    if latest_row.empty:
        latest_row = history.tail(1)
    latest_row = latest_row.iloc[0]

    driver_rows: list[dict[str, Any]] = []
    for column, label in candidate_features.items():
        if column not in history.columns:
            continue
        series = pd.to_numeric(history[column], errors="coerce").dropna()
        current_value = pd.to_numeric(pd.Series([latest_row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(current_value) or len(series) < 20:
            continue
        mean_value = float(series.mean())
        std_value = float(series.std(ddof=1))
        z_score = 0.0 if std_value == 0.0 or pd.isna(std_value) else float((current_value - mean_value) / std_value)
        direction = "supportive" if current_value >= mean_value else "cautionary"
        driver_rows.append(
            {
                "feature": column,
                "label": label,
                "current_value": float(current_value),
                "historical_mean": mean_value,
                "z_score": z_score,
                "direction": direction,
            }
        )

    ranked = sorted(driver_rows, key=lambda row: abs(row["z_score"]), reverse=True)
    return ranked[:top_n]


def build_ml_forecast_summary(ticker: str) -> dict[str, Any]:
    """Build an app/report-friendly ML forecast summary for one asset."""

    snapshot = load_latest_ml_forecast(ticker)
    build_status = None
    if snapshot is None:
        build_status = ensure_ml_forecast_for_ticker(ticker)
        if build_status.get("success"):
            snapshot = load_latest_ml_forecast(ticker)

    prediction_history = prepare_ml_prediction_history_frame(load_ml_prediction_history(ticker))
    feature_history = load_ml_feature_driver_frame(ticker)
    drivers = derive_ml_feature_drivers(snapshot, feature_history)

    if snapshot is None:
        return {
            "available": False,
            "snapshot": None,
            "prediction_history": prediction_history,
            "feature_drivers": [],
            "build_status": build_status,
            "interpretation": (
                build_status["message"]
                if build_status and build_status.get("message")
                else "No stored model-informed forecast is currently available for this asset."
            ),
        }

    expected_return = _safe_float(snapshot.get("predicted_return_20d"))
    downside_probability = _safe_float(snapshot.get("downside_probability_20d"))
    predicted_volatility = _safe_float(snapshot.get("realized_volatility_20d"))
    short_horizon_volatility = _safe_float(snapshot.get("recent_realized_volatility_5d"))
    downside_volatility = _safe_float(snapshot.get("downside_volatility_20d"))
    forecast_horizon = int(snapshot.get("prediction_horizon_days") or 20)

    regime_label = (
        f"{_classify_return_outlook(expected_return)} / "
        f"{_classify_downside_context(downside_probability)}"
    )

    interpretation_parts = [
        f"The latest model-implied {forecast_horizon}-day expected return is "
        f"{expected_return:.2%}." if expected_return is not None else
        "The latest model-implied expected return is unavailable.",
        (
            f"The probability of a negative {forecast_horizon}-day return is "
            f"{downside_probability:.2%}, framing the current downside risk context as "
            f"{_classify_downside_context(downside_probability).lower()}."
        )
        if downside_probability is not None
        else "The latest downside-risk probability is unavailable."
    ]
    if predicted_volatility is not None:
        interpretation_parts.append(
            f"Recent realized volatility is {predicted_volatility:.2%}, which serves as a practical uncertainty proxy for scenario work."
        )
    if drivers:
        driver_labels = ", ".join(driver["label"] for driver in drivers[:3])
        interpretation_parts.append(
            f"The most unusual current drivers versus recent history are {driver_labels}."
        )

    return {
        "available": True,
        "build_status": build_status,
        "snapshot": {
            **snapshot,
            "predicted_return_20d": expected_return,
            "downside_probability_20d": downside_probability,
            "predicted_volatility_20d": predicted_volatility,
            "short_horizon_volatility_5d": short_horizon_volatility,
            "downside_volatility_20d": downside_volatility,
            "regime_label": regime_label,
        },
        "prediction_history": prediction_history,
        "feature_drivers": drivers,
        "interpretation": " ".join(interpretation_parts),
    }


def prepare_sentiment_frame(article_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare stored sentiment rows into app-friendly pandas structures."""

    if not article_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(article_rows).copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    frame["ingestion_timestamp"] = pd.to_datetime(
        frame["ingestion_timestamp"],
        errors="coerce",
    )
    frame["sentiment_score"] = pd.to_numeric(frame["sentiment_score"], errors="coerce")
    frame = frame.dropna(subset=["published_at", "sentiment_score"]).sort_values(
        "published_at",
        ascending=True,
    )
    frame["published_date"] = frame["published_at"].dt.normalize()
    return frame


def get_sentiment_summary(article_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute a lightweight sentiment summary for the selected asset."""

    frame = prepare_sentiment_frame(article_rows)
    if frame.empty:
        return {
            "article_count": 0,
            "average_sentiment": None,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "latest_published_at": None,
        }

    label_counts = frame["sentiment_label"].value_counts()
    return {
        "article_count": int(len(frame)),
        "average_sentiment": float(frame["sentiment_score"].mean()),
        "positive_count": int(label_counts.get("positive", 0)),
        "negative_count": int(label_counts.get("negative", 0)),
        "neutral_count": int(label_counts.get("neutral", 0)),
        "latest_published_at": frame["published_at"].max(),
    }


def get_sentiment_trend_frame(article_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate article sentiment into a daily trend frame for charting."""

    frame = prepare_sentiment_frame(article_rows)
    if frame.empty:
        return pd.DataFrame()

    trend = (
        frame.groupby("published_date", as_index=False)
        .agg(
            average_sentiment=("sentiment_score", "mean"),
            article_count=("headline", "count"),
        )
        .sort_values("published_date")
    )
    trend["average_sentiment"] = trend["average_sentiment"].round(4)
    return trend


def get_recent_sentiment_table(article_rows: list[dict[str, Any]], rows: int = 8) -> pd.DataFrame:
    """Return a clean recent-articles table for the app layer."""

    frame = prepare_sentiment_frame(article_rows)
    if frame.empty:
        return pd.DataFrame()

    recent = frame.sort_values("published_at", ascending=False).head(rows).copy()
    recent["published_at"] = recent["published_at"].dt.strftime("%Y-%m-%d %H:%M")
    display_columns = [
        "published_at",
        "publisher_name",
        "headline",
        "sentiment_label",
        "sentiment_score",
        "source_name",
        "url",
    ]
    return recent[display_columns]


def get_latest_sentiment_timestamp(article_rows: list[dict[str, Any]]) -> pd.Timestamp | None:
    """Return the most recent ingestion timestamp for stored sentiment rows."""

    frame = prepare_sentiment_frame(article_rows)
    if frame.empty or "ingestion_timestamp" not in frame:
        return None

    latest_timestamp = frame["ingestion_timestamp"].max()
    if pd.isna(latest_timestamp):
        return None
    return latest_timestamp


def sentiment_is_fresh(
    article_rows: list[dict[str, Any]],
    freshness_hours: int = DEFAULT_SENTIMENT_FRESHNESS_HOURS,
) -> bool:
    """Return True when stored sentiment is recent enough for app reuse."""

    latest_timestamp = get_latest_sentiment_timestamp(article_rows)
    if latest_timestamp is None:
        return False

    return latest_timestamp >= (pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=freshness_hours))


def sentiment_exists_for_ticker(ticker: str, minimum_articles: int = 1) -> bool:
    """Return True when stored sentiment records exist for the selected ticker."""

    return len(load_recent_news_articles(ticker, limit=minimum_articles)) >= minimum_articles


def ensure_sentiment_for_ticker(
    ticker: str,
    page_size: int = DEFAULT_SENTIMENT_PAGE_SIZE,
    freshness_hours: int = DEFAULT_SENTIMENT_FRESHNESS_HOURS,
) -> dict[str, Any]:
    """
    Ensure recent sentiment exists for a ticker, fetching on demand only when needed.

    The default policy is conservative for app reruns:
    - reuse stored sentiment when present and fresh enough
    - fetch only when no sentiment exists or the stored data is stale
    """

    normalized_ticker = normalize_app_ticker(ticker)
    if not normalized_ticker:
        return {
            "success": False,
            "ticker": "",
            "status": "invalid_input",
            "message": "A valid ticker is required before sentiment can be loaded.",
            "fetched": False,
        }

    stored_rows = load_recent_news_articles(normalized_ticker, limit=page_size)
    if stored_rows and sentiment_is_fresh(stored_rows, freshness_hours=freshness_hours):
        return {
            "success": True,
            "ticker": normalized_ticker,
            "status": "database",
            "message": f"{normalized_ticker} sentiment was loaded from the local database.",
            "fetched": False,
            "article_count": len(stored_rows),
        }

    try:
        from src.ingestion.bootstrap_sentiment import ingest_sentiment_for_ticker
    except Exception as exc:
        detail = str(exc)
        if "401" in detail or "invalid api key" in detail.lower() or "unauthorized" in detail.lower():
            return {
                "success": False,
                "ticker": normalized_ticker,
                "status": "credentials_error",
                "message": (
                    f"Stored sentiment is unavailable for {normalized_ticker}, and the configured news API "
                    "credentials were rejected by the provider. Add a valid API key to enable on-demand sentiment ingestion."
                ),
                "fetched": False,
            }
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ingestion_import_error",
            "message": f"Sentiment ingestion components could not be loaded. Detail: {exc}",
            "fetched": False,
        }

    initialize_database()

    try:
        summary = ingest_sentiment_for_ticker(
            ticker=normalized_ticker,
            page_size=page_size,
        )
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "provider_error",
            "message": (
                f"Unable to fetch recent sentiment for {normalized_ticker}. "
                f"Provider detail: {exc}"
            ),
            "fetched": False,
        }

    if summary.get("articles_loaded", 0) <= 0:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "no_data",
            "message": f"No recent sentiment articles were returned for {normalized_ticker}.",
            "fetched": True,
        }

    return {
        "success": True,
        "ticker": normalized_ticker,
        "status": "ingested",
        "message": (
            f"{normalized_ticker} sentiment was fetched and stored locally "
            f"({summary['articles_loaded']} articles)."
        ),
        "fetched": True,
        "article_count": summary["articles_loaded"],
    }
