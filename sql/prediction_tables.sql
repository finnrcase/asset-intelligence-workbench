CREATE TABLE IF NOT EXISTS ml_model_runs (
    run_id TEXT PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    regression_model_name TEXT NOT NULL,
    classification_model_name TEXT NOT NULL,
    training_start_date DATE,
    training_end_date DATE,
    evaluation_summary TEXT,
    feature_version TEXT NOT NULL DEFAULT 'v1',
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    prediction_horizon_days INTEGER NOT NULL,
    model_run_id TEXT,
    regression_model_name TEXT NOT NULL,
    classification_model_name TEXT NOT NULL,
    selected_model_name TEXT,
    model_family TEXT,
    target_name TEXT,
    predicted_return_20d NUMERIC(18, 8),
    downside_probability_20d NUMERIC(18, 8),
    probability_positive_20d NUMERIC(18, 8),
    predicted_negative_return_flag INTEGER NOT NULL CHECK (predicted_negative_return_flag IN (0, 1)),
    composite_ml_score NUMERIC(18, 8),
    confidence_score NUMERIC(18, 8),
    directional_signal TEXT,
    history_score NUMERIC(18, 8),
    risk_score NUMERIC(18, 8),
    sentiment_score NUMERIC(18, 8),
    history_contribution NUMERIC(18, 8),
    risk_contribution NUMERIC(18, 8),
    sentiment_contribution NUMERIC(18, 8),
    pillar_weights_json TEXT,
    feature_importance_json TEXT,
    top_features_json TEXT,
    prediction_generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets (id) ON DELETE CASCADE,
    FOREIGN KEY (model_run_id) REFERENCES ml_model_runs (run_id) ON DELETE SET NULL,
    CONSTRAINT uq_ml_predictions_asset_asof_horizon_models UNIQUE (
        asset_id,
        as_of_date,
        prediction_horizon_days,
        regression_model_name,
        classification_model_name
    )
);

CREATE INDEX IF NOT EXISTS ix_ml_predictions_asset_asof
    ON ml_predictions (asset_id, as_of_date);
CREATE INDEX IF NOT EXISTS ix_ml_predictions_model_run
    ON ml_predictions (model_run_id);
