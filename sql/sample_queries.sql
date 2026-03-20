-- Asset Intelligence Workbench
-- Sample SQL Queries
--
-- These examples are written for SQLite and are intended to make the project
-- schema easier to understand before building deeper analytics. Each query is
-- realistic for a finance-oriented workflow and mirrors the kinds of lookups
-- the Python query layer performs.


-- 1. List all tracked assets
-- Purpose:
-- Show the security master table that defines the asset universe available for
-- downstream analytics, reporting, and ingestion checks.
SELECT
    id,
    ticker,
    asset_name,
    asset_class,
    exchange,
    currency,
    is_active,
    created_at
FROM assets
ORDER BY ticker;


-- 2. Get metadata for one ticker
-- Purpose:
-- Retrieve the descriptive fields for a single instrument, including the
-- primary source associated with that asset record.
SELECT
    a.ticker,
    a.asset_name,
    a.asset_class,
    a.exchange,
    a.currency,
    a.sector,
    a.industry,
    a.country,
    a.is_active,
    ds.source_name AS primary_source,
    a.created_at,
    a.updated_at
FROM assets AS a
LEFT JOIN data_sources AS ds
    ON a.primary_source_id = ds.id
WHERE a.ticker = 'AAPL';


-- 3. Show the 10 most recent price rows for a ticker
-- Purpose:
-- Inspect recent stored market data for one asset and confirm the daily price
-- history is loading as expected.
SELECT
    a.ticker,
    hp.price_date,
    hp.open_price,
    hp.high_price,
    hp.low_price,
    hp.close_price,
    hp.adjusted_close,
    hp.volume,
    hp.ingestion_timestamp
FROM historical_prices AS hp
INNER JOIN assets AS a
    ON hp.asset_id = a.id
WHERE a.ticker = 'AAPL'
ORDER BY hp.price_date DESC
LIMIT 10;


-- 4. Count historical price rows by ticker
-- Purpose:
-- Measure coverage by asset and quickly identify where one symbol may have more
-- or less data history than another.
SELECT
    a.ticker,
    COUNT(*) AS price_row_count
FROM historical_prices AS hp
INNER JOIN assets AS a
    ON hp.asset_id = a.id
GROUP BY a.ticker
ORDER BY price_row_count DESC, a.ticker ASC;


-- 5. Join assets with historical prices
-- Purpose:
-- Demonstrate the core relational join in the project: descriptive asset
-- metadata linked to daily observations in the time-series table.
SELECT
    a.ticker,
    a.asset_name,
    a.asset_class,
    hp.price_date,
    hp.close_price,
    hp.volume
FROM assets AS a
INNER JOIN historical_prices AS hp
    ON a.id = hp.asset_id
ORDER BY a.ticker, hp.price_date DESC
LIMIT 25;


-- 6. Show min and max dates available for each asset
-- Purpose:
-- Check the available historical window for every tracked symbol. This is
-- useful before computing returns, volatility, or rolling measures.
SELECT
    a.ticker,
    MIN(hp.price_date) AS first_price_date,
    MAX(hp.price_date) AS last_price_date,
    COUNT(*) AS total_rows
FROM historical_prices AS hp
INNER JOIN assets AS a
    ON hp.asset_id = a.id
GROUP BY a.ticker
ORDER BY a.ticker;


-- 7. Show basic summary statistics from stored price data
-- Purpose:
-- Provide a simple profile of the stored market data without doing full
-- analytics yet. This helps validate the price ranges and overall data quality.
SELECT
    a.ticker,
    MIN(hp.close_price) AS min_close_price,
    MAX(hp.close_price) AS max_close_price,
    AVG(hp.close_price) AS avg_close_price,
    MIN(hp.volume) AS min_volume,
    MAX(hp.volume) AS max_volume,
    AVG(hp.volume) AS avg_volume
FROM historical_prices AS hp
INNER JOIN assets AS a
    ON hp.asset_id = a.id
GROUP BY a.ticker
ORDER BY a.ticker;


-- 8. Show price row counts by data source and ticker
-- Purpose:
-- Make the source-tracking layer visible. This becomes more important if you
-- later ingest from multiple vendors or reconcile competing data feeds.
SELECT
    ds.source_name,
    a.ticker,
    COUNT(*) AS price_row_count
FROM historical_prices AS hp
INNER JOIN assets AS a
    ON hp.asset_id = a.id
INNER JOIN data_sources AS ds
    ON hp.source_id = ds.id
GROUP BY ds.source_name, a.ticker
ORDER BY ds.source_name, a.ticker;


-- 9. Find the most recently ingested records
-- Purpose:
-- Surface the latest ingestion activity for auditability and operational
-- troubleshooting.
SELECT
    a.ticker,
    hp.price_date,
    hp.close_price,
    hp.ingestion_timestamp,
    ds.source_name
FROM historical_prices AS hp
INNER JOIN assets AS a
    ON hp.asset_id = a.id
INNER JOIN data_sources AS ds
    ON hp.source_id = ds.id
ORDER BY hp.ingestion_timestamp DESC, a.ticker ASC
LIMIT 15;
