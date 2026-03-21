CREATE TABLE IF NOT EXISTS technical_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    feature_date DATE NOT NULL,
    analysis_price NUMERIC(18, 6),
    close_price NUMERIC(18, 6),
    adjusted_close NUMERIC(18, 6),
    volume INTEGER,
    daily_return NUMERIC(18, 8),
    return_lag_1d NUMERIC(18, 8),
    return_lag_5d NUMERIC(18, 8),
    return_lag_10d NUMERIC(18, 8),
    rolling_mean_return_5d NUMERIC(18, 8),
    rolling_mean_return_20d NUMERIC(18, 8),
    rolling_volatility_10d NUMERIC(18, 8),
    rolling_volatility_20d NUMERIC(18, 8),
    realized_volatility_20d NUMERIC(18, 8),
    recent_realized_volatility_5d NUMERIC(18, 8),
    momentum_5d NUMERIC(18, 8),
    momentum_10d NUMERIC(18, 8),
    momentum_20d NUMERIC(18, 8),
    ma_distance_10d NUMERIC(18, 8),
    ma_distance_20d NUMERIC(18, 8),
    drawdown_from_peak NUMERIC(18, 8),
    rolling_drawdown_20d NUMERIC(18, 8),
    downside_volatility_20d NUMERIC(18, 8),
    intraday_range_pct NUMERIC(18, 8),
    volume_change_1d NUMERIC(18, 8),
    volume_ratio_20d NUMERIC(18, 8),
    volume_zscore_20d NUMERIC(18, 8),
    target_forward_return_20d NUMERIC(18, 8),
    target_negative_return_20d INTEGER CHECK (target_negative_return_20d IN (0, 1)),
    feature_version TEXT NOT NULL DEFAULT 'v1',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets (id) ON DELETE CASCADE,
    CONSTRAINT uq_technical_features_asset_date UNIQUE (asset_id, feature_date)
);

CREATE TABLE IF NOT EXISTS sentiment_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    feature_date DATE NOT NULL,
    article_count_1d INTEGER,
    sentiment_mean_1d NUMERIC(18, 8),
    sentiment_mean_7d NUMERIC(18, 8),
    sentiment_std_7d NUMERIC(18, 8),
    negative_article_share_7d NUMERIC(18, 8),
    positive_article_share_7d NUMERIC(18, 8),
    article_count_7d INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets (id) ON DELETE CASCADE,
    CONSTRAINT uq_sentiment_features_asset_date UNIQUE (asset_id, feature_date)
);

CREATE INDEX IF NOT EXISTS ix_technical_features_asset_date
    ON technical_features (asset_id, feature_date);
CREATE INDEX IF NOT EXISTS ix_technical_features_feature_date
    ON technical_features (feature_date);
CREATE INDEX IF NOT EXISTS ix_sentiment_features_asset_date
    ON sentiment_features (asset_id, feature_date);
