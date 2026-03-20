PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS data_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    source_url TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    asset_name TEXT NOT NULL,
    asset_class TEXT,
    exchange TEXT,
    currency TEXT NOT NULL DEFAULT 'USD',
    sector TEXT,
    industry TEXT,
    country TEXT,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    primary_source_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (primary_source_id) REFERENCES data_sources (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS historical_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    price_date DATE NOT NULL,
    open_price NUMERIC(18, 6),
    high_price NUMERIC(18, 6),
    low_price NUMERIC(18, 6),
    close_price NUMERIC(18, 6) NOT NULL,
    adjusted_close NUMERIC(18, 6),
    volume INTEGER,
    ingestion_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets (id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES data_sources (id) ON DELETE RESTRICT,
    CONSTRAINT uq_historical_prices_asset_source_date UNIQUE (asset_id, source_id, price_date)
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    provider_article_id TEXT,
    publisher_name TEXT,
    headline TEXT NOT NULL,
    summary TEXT,
    url TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    sentiment_score NUMERIC(8, 4) NOT NULL,
    sentiment_label TEXT NOT NULL,
    query_text TEXT,
    ingestion_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets (id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES data_sources (id) ON DELETE RESTRICT,
    CONSTRAINT uq_news_articles_asset_source_url UNIQUE (asset_id, source_id, url),
    CONSTRAINT uq_news_articles_asset_source_provider_id UNIQUE (asset_id, source_id, provider_article_id)
);

CREATE INDEX IF NOT EXISTS ix_assets_ticker ON assets (ticker);
CREATE INDEX IF NOT EXISTS ix_assets_asset_class ON assets (asset_class);
CREATE INDEX IF NOT EXISTS ix_historical_prices_asset_date
    ON historical_prices (asset_id, price_date);
CREATE INDEX IF NOT EXISTS ix_historical_prices_date
    ON historical_prices (price_date);
CREATE INDEX IF NOT EXISTS ix_historical_prices_source_date
    ON historical_prices (source_id, price_date);
CREATE INDEX IF NOT EXISTS ix_news_articles_asset_published_at
    ON news_articles (asset_id, published_at);
CREATE INDEX IF NOT EXISTS ix_news_articles_source_published_at
    ON news_articles (source_id, published_at);
CREATE INDEX IF NOT EXISTS ix_news_articles_published_at
    ON news_articles (published_at);
