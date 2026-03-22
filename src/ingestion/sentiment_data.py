"""
Provider-aware news sentiment ingestion helpers.

GNews is the default provider for recent asset coverage, while the module
retains Finnhub and NewsAPI as optional secondary implementations. Downstream
code depends only on the normalized article structure defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

from dotenv import load_dotenv


GNEWS_SOURCE_NAME = "gnews"
GNEWS_SOURCE_TYPE = "news_api"
GNEWS_SOURCE_URL = "https://gnews.io/"

FINNHUB_SOURCE_NAME = "finnhub"
FINNHUB_SOURCE_TYPE = "news_api"
FINNHUB_SOURCE_URL = "https://finnhub.io/"

NEWSAPI_SOURCE_NAME = "newsapi"
NEWSAPI_SOURCE_TYPE = "news_api"
NEWSAPI_SOURCE_URL = "https://newsapi.org/"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)

POSITIVE_TERMS = {
    "beat",
    "beats",
    "bullish",
    "gain",
    "gains",
    "growth",
    "improve",
    "improves",
    "improved",
    "optimistic",
    "outperform",
    "outperforms",
    "profit",
    "profits",
    "record",
    "resilient",
    "rise",
    "rises",
    "strong",
    "surge",
    "surges",
    "upgrade",
    "upgrades",
}

NEGATIVE_TERMS = {
    "bearish",
    "cut",
    "cuts",
    "decline",
    "declines",
    "declined",
    "downgrade",
    "downgrades",
    "drop",
    "drops",
    "fall",
    "falls",
    "fraud",
    "loss",
    "losses",
    "miss",
    "misses",
    "probe",
    "risk",
    "risks",
    "selloff",
    "slump",
    "weak",
}


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols for consistent storage and querying."""

    return ticker.strip().upper()


def build_news_query(ticker: str | None = None, company_name: str | None = None) -> str:
    """Build a human-readable asset query used for lineage and provider fallback."""

    parts: list[str] = []
    if ticker:
        parts.append(f'"{normalize_ticker(ticker)}"')
    if company_name:
        cleaned_name = company_name.strip()
        if cleaned_name:
            parts.append(f'"{cleaned_name}"')

    if not parts:
        raise ValueError("At least one of `ticker` or `company_name` is required.")

    return " OR ".join(parts)


def score_news_sentiment(text: str) -> float:
    """Compute a lightweight lexical sentiment score in the range [-1.0, 1.0]."""

    tokens = [
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in str(text).split()
        if token.strip(".,:;!?()[]{}\"'")
    ]
    if not tokens:
        return 0.0

    positive_count = sum(1 for token in tokens if token in POSITIVE_TERMS)
    negative_count = sum(1 for token in tokens if token in NEGATIVE_TERMS)
    recognized = positive_count + negative_count
    if recognized == 0:
        return 0.0

    raw_score = (positive_count - negative_count) / recognized
    return max(-1.0, min(1.0, round(raw_score, 4)))


def sentiment_label_from_score(score: float) -> str:
    """Convert a numeric sentiment score into a practical label."""

    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"


def _parse_published_at(value: str | int | float | None) -> datetime:
    """Parse provider publication timestamps into naive UTC datetimes."""

    if value in (None, ""):
        raise ValueError("Article publication timestamp is required.")

    if isinstance(value, (int, float)):
        numeric_value = float(value)
        # Some providers emit epoch milliseconds while others emit seconds.
        if numeric_value > 10_000_000_000:
            numeric_value /= 1000.0
        return datetime.fromtimestamp(numeric_value, tz=timezone.utc).replace(tzinfo=None)

    normalized = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _build_provider_article_id(url: str | None, headline: str, published_at: datetime) -> str:
    """Create a stable fallback article identifier when the provider omits one."""

    key = f"{url or ''}|{headline}|{published_at.isoformat()}"
    return sha256(key.encode("utf-8")).hexdigest()[:24]


def _load_sentiment_environment() -> None:
    """Load the project-level environment file for provider credentials."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_provider_availability() -> dict[str, bool]:
    """Return which supported sentiment providers are configured."""

    _load_sentiment_environment()
    return {
        "gnews": bool(str(os.getenv("GNEWS_API_KEY", "")).strip()),
        "finnhub": bool(str(os.getenv("FINNHUB_API_KEY", "")).strip()),
        "newsapi": bool(str(os.getenv("NEWSAPI_API_KEY") or os.getenv("NEWS_API_KEY") or "").strip()),
    }


def build_default_news_provider() -> NewsSentimentProvider:
    """
    Select the best configured provider using a stable fallback order.

    Preference order:
    1. GNews
    2. Finnhub
    3. NewsAPI
    """

    availability = get_provider_availability()
    if availability["gnews"]:
        LOGGER.info("Selected sentiment provider: gnews (GNEWS_API_KEY detected=true)")
        return GNewsClient()
    if availability["finnhub"]:
        LOGGER.info(
            "Selected sentiment provider: finnhub (GNEWS_API_KEY detected=false, fallback_provider_used=true)"
        )
        return FinnhubNewsClient()
    if availability["newsapi"]:
        LOGGER.info(
            "Selected sentiment provider: newsapi (GNEWS_API_KEY detected=false, fallback_provider_used=true)"
        )
        return NewsAPIClient()

    LOGGER.info(
        "Selected sentiment provider: none (GNEWS_API_KEY detected=false, fallback_provider_used=false, live_fetch_skipped=true)"
    )
    raise ValueError(
        "No configured news sentiment provider is available. "
        "Set GNEWS_API_KEY, FINNHUB_API_KEY, or NEWSAPI_API_KEY to enable live sentiment fetching."
    )


def _request_json(url: str, headers: dict[str, str] | None = None) -> Any:
    """Execute a provider request and return parsed JSON with clearer error messages."""

    request = Request(
        url,
        headers={
            "User-Agent": "AssetIntelligenceWorkbench/1.0",
            **(headers or {}),
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            detail = payload.get("error") or payload.get("message") or str(payload)
        except Exception:
            detail = str(exc)
        raise ValueError(f"{exc.code} {exc.reason}: {detail}") from exc


def _clean_optional_text(value: Any) -> str | None:
    """Normalize provider text-like values into clean optional strings."""

    if value in (None, "", "None", "null"):
        return None
    cleaned = str(value).strip()
    return cleaned or None


@dataclass(frozen=True)
class ProviderNewsArticle:
    """Normalized provider article record ready for the loader layer."""

    headline: str
    summary: str | None
    url: str
    published_at: datetime
    publisher_name: str | None
    provider_article_id: str
    query_text: str
    sentiment_score: float
    sentiment_label: str

    def as_dict(self) -> dict[str, Any]:
        """Return a loader-friendly dictionary representation."""

        return {
            "provider_article_id": self.provider_article_id,
            "publisher_name": self.publisher_name,
            "headline": self.headline,
            "summary": self.summary,
            "url": self.url,
            "published_at": self.published_at,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "query_text": self.query_text,
            "ingestion_timestamp": datetime.utcnow(),
        }


class NewsSentimentProvider(Protocol):
    """Provider interface for future sentiment/news integrations."""

    source_name: str
    source_type: str
    source_url: str

    def fetch_recent_articles(
        self,
        ticker: str | None = None,
        company_name: str | None = None,
        page_size: int = 20,
    ) -> list[ProviderNewsArticle]:
        """Fetch and normalize recent articles for an asset query."""


class GNewsClient:
    """GNews-backed news client using the search endpoint for recent coverage."""

    source_name = GNEWS_SOURCE_NAME
    source_type = GNEWS_SOURCE_TYPE
    source_url = GNEWS_SOURCE_URL

    def __init__(self, api_key: str | None = None) -> None:
        _load_sentiment_environment()
        resolved_key = api_key or os.getenv("GNEWS_API_KEY")
        self.api_key = resolved_key.strip() if isinstance(resolved_key, str) else None

    def fetch_recent_articles(
        self,
        ticker: str | None = None,
        company_name: str | None = None,
        page_size: int = 20,
    ) -> list[ProviderNewsArticle]:
        """Fetch recent asset coverage from GNews and normalize it."""

        if not self.api_key:
            raise ValueError("GNEWS_API_KEY is required to fetch news sentiment data.")

        query_text = build_news_query(ticker=ticker, company_name=company_name)
        normalized_page_size = max(1, min(int(page_size), 10))
        end_timestamp = datetime.utcnow()
        start_timestamp = end_timestamp - timedelta(days=14)
        params = urlencode(
            {
                "q": query_text,
                "lang": "en",
                "in": "title,description",
                "sortby": "publishedAt",
                "max": normalized_page_size,
                "from": start_timestamp.replace(microsecond=0).isoformat() + "Z",
                "to": end_timestamp.replace(microsecond=0).isoformat() + "Z",
                "apikey": self.api_key,
            }
        )
        url = f"https://gnews.io/api/v4/search?{params}"
        payload = _request_json(url)

        if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
            raise ValueError("GNews returned an unexpected response shape for article search.")

        return self.normalize_articles(payload["articles"], query_text=query_text)

    def normalize_articles(
        self,
        articles: list[dict[str, Any]],
        query_text: str,
    ) -> list[ProviderNewsArticle]:
        """Normalize GNews payloads into internal article records."""

        normalized_articles: list[ProviderNewsArticle] = []
        for article in articles:
            try:
                normalized_articles.append(
                    normalize_gnews_article(article=article, query_text=query_text)
                )
            except ValueError:
                continue
        return normalized_articles


class FinnhubNewsClient:
    """Finnhub-backed finance-news client using the company-news endpoint."""

    source_name = FINNHUB_SOURCE_NAME
    source_type = FINNHUB_SOURCE_TYPE
    source_url = FINNHUB_SOURCE_URL

    def __init__(self, api_key: str | None = None) -> None:
        _load_sentiment_environment()
        resolved_key = api_key or os.getenv("FINNHUB_API_KEY")
        self.api_key = resolved_key.strip() if isinstance(resolved_key, str) else None

    def fetch_recent_articles(
        self,
        ticker: str | None = None,
        company_name: str | None = None,
        page_size: int = 20,
    ) -> list[ProviderNewsArticle]:
        """Fetch recent company news from Finnhub and normalize it."""

        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY is required to fetch news sentiment data.")
        if not ticker:
            raise ValueError("Finnhub company news requires a ticker symbol.")

        normalized_ticker = normalize_ticker(ticker)
        end_date = date.today()
        start_date = end_date - timedelta(days=14)
        query_text = build_news_query(ticker=normalized_ticker, company_name=company_name)

        params = urlencode(
            {
                "symbol": normalized_ticker,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
                "token": self.api_key,
            }
        )
        url = f"https://finnhub.io/api/v1/company-news?{params}"
        payload = _request_json(url)
        if not isinstance(payload, list):
            raise ValueError("Finnhub returned an unexpected response shape for company news.")

        normalized_articles = self.normalize_articles(payload, query_text=query_text)
        return normalized_articles[:page_size]

    def normalize_articles(
        self,
        articles: list[dict[str, Any]],
        query_text: str,
    ) -> list[ProviderNewsArticle]:
        """Normalize Finnhub company-news payloads into internal article records."""

        return [
            normalize_finnhub_article(article=article, query_text=query_text)
            for article in articles
            if article.get("headline") and article.get("url") and article.get("datetime")
        ]


class NewsAPIClient:
    """Optional NewsAPI client retained as a secondary provider implementation."""

    source_name = NEWSAPI_SOURCE_NAME
    source_type = NEWSAPI_SOURCE_TYPE
    source_url = NEWSAPI_SOURCE_URL

    def __init__(self, api_key: str | None = None) -> None:
        _load_sentiment_environment()
        resolved_key = api_key or os.getenv("NEWSAPI_API_KEY") or os.getenv("NEWS_API_KEY")
        self.api_key = resolved_key.strip() if isinstance(resolved_key, str) else None

    def fetch_recent_articles(
        self,
        ticker: str | None = None,
        company_name: str | None = None,
        page_size: int = 20,
    ) -> list[ProviderNewsArticle]:
        """Fetch recent articles from NewsAPI and normalize them."""

        if not self.api_key:
            raise ValueError("NEWSAPI_API_KEY is required to fetch news sentiment data.")

        query_text = build_news_query(ticker=ticker, company_name=company_name)
        params = urlencode(
            {
                "q": query_text,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
            }
        )
        url = f"https://newsapi.org/v2/everything?{params}"
        payload = _request_json(url, headers={"X-Api-Key": self.api_key})

        status = payload.get("status")
        if status != "ok":
            raise ValueError(payload.get("message", "NewsAPI returned an invalid response."))

        articles = payload.get("articles", [])
        return self.normalize_articles(articles=articles, query_text=query_text)

    def normalize_articles(
        self,
        articles: list[dict[str, Any]],
        query_text: str,
    ) -> list[ProviderNewsArticle]:
        """Normalize NewsAPI payloads into internal article records."""

        return [
            normalize_newsapi_article(article=article, query_text=query_text)
            for article in articles
            if article.get("title") and article.get("url") and article.get("publishedAt")
        ]


def normalize_finnhub_article(
    article: dict[str, Any],
    query_text: str,
) -> ProviderNewsArticle:
    """Normalize a single Finnhub article payload into the internal structure."""

    headline = _clean_optional_text(article.get("headline")) or _clean_optional_text(article.get("title"))
    if not headline:
        raise ValueError("Finnhub article is missing a usable headline.")

    url = _clean_optional_text(article.get("url"))
    if not url:
        raise ValueError("Finnhub article is missing a usable URL.")

    summary = (
        _clean_optional_text(article.get("summary"))
        or _clean_optional_text(article.get("description"))
        or _clean_optional_text(article.get("snippet"))
    )
    published_at = _parse_published_at(article.get("datetime") or article.get("publishedAt"))
    publisher_name = _clean_optional_text(article.get("source"))
    raw_provider_id = _clean_optional_text(article.get("id"))
    provider_article_id = raw_provider_id or _build_provider_article_id(url, headline, published_at)
    sentiment_input = " ".join(part for part in [headline, summary or ""] if part).strip()
    score = score_news_sentiment(sentiment_input)

    return ProviderNewsArticle(
        headline=headline,
        summary=summary,
        url=url,
        published_at=published_at,
        publisher_name=publisher_name,
        provider_article_id=provider_article_id,
        query_text=query_text,
        sentiment_score=score,
        sentiment_label=sentiment_label_from_score(score),
    )


def normalize_gnews_article(
    article: dict[str, Any],
    query_text: str,
) -> ProviderNewsArticle:
    """Normalize a single GNews article payload into the internal structure."""

    headline = _clean_optional_text(article.get("title"))
    if not headline:
        raise ValueError("GNews article is missing a usable headline.")

    url = _clean_optional_text(article.get("url"))
    if not url:
        raise ValueError("GNews article is missing a usable URL.")

    summary = (
        _clean_optional_text(article.get("description"))
        or _clean_optional_text(article.get("content"))
    )
    published_at = _parse_published_at(article.get("publishedAt"))

    publisher_name = None
    source = article.get("source")
    if isinstance(source, dict):
        publisher_name = _clean_optional_text(source.get("name"))

    raw_provider_id = _clean_optional_text(article.get("id"))
    provider_article_id = raw_provider_id or _build_provider_article_id(url, headline, published_at)
    sentiment_input = " ".join(part for part in [headline, summary or ""] if part).strip()
    score = score_news_sentiment(sentiment_input)

    return ProviderNewsArticle(
        headline=headline,
        summary=summary,
        url=url,
        published_at=published_at,
        publisher_name=publisher_name,
        provider_article_id=provider_article_id,
        query_text=query_text,
        sentiment_score=score,
        sentiment_label=sentiment_label_from_score(score),
    )


def normalize_newsapi_article(
    article: dict[str, Any],
    query_text: str,
) -> ProviderNewsArticle:
    """Normalize a single NewsAPI article payload into the internal structure."""

    headline = _clean_optional_text(article.get("title"))
    if not headline:
        raise ValueError("NewsAPI article is missing a usable headline.")

    summary = _clean_optional_text(article.get("description"))
    url = _clean_optional_text(article.get("url"))
    if not url:
        raise ValueError("NewsAPI article is missing a usable URL.")
    published_at = _parse_published_at(article.get("publishedAt"))
    publisher_name = None
    if isinstance(article.get("source"), dict):
        publisher_name = _clean_optional_text(article["source"].get("name"))

    sentiment_input = " ".join(part for part in [headline, summary or ""] if part).strip()
    score = score_news_sentiment(sentiment_input)
    provider_article_id = _build_provider_article_id(url, headline, published_at)

    return ProviderNewsArticle(
        headline=headline,
        summary=summary,
        url=url,
        published_at=published_at,
        publisher_name=publisher_name,
        provider_article_id=provider_article_id,
        query_text=query_text,
        sentiment_score=score,
        sentiment_label=sentiment_label_from_score(score),
    )


def fetch_recent_news_sentiment(
    ticker: str | None = None,
    company_name: str | None = None,
    page_size: int = 20,
    provider: NewsSentimentProvider | None = None,
) -> list[dict[str, Any]]:
    """Fetch normalized recent articles using the default or supplied provider."""

    resolved_provider = provider or build_default_news_provider()
    return [
        article.as_dict()
        for article in resolved_provider.fetch_recent_articles(
            ticker=ticker,
            company_name=company_name,
            page_size=page_size,
        )
    ]
