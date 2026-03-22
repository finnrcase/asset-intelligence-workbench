# Asset Intelligence Workbench

SQL-Driven Financial Analytics, Risk Simulation, and Reporting Platform

The Asset Intelligence Workbench is a portfolio project designed to feel like a real internal analyst tool. It combines structured SQL-backed data storage, market data ingestion, return and risk analytics, Monte Carlo simulation, live news sentiment context, and a detailed PDF report in a single workflow.

The goal of the project was to build something closer to a finance, treasury, strategic finance, or risk decision-support tool that could plausibly be used inside a modern company.

https://asset-intelligence-workbench-eq2zbxelprxwjnnxendyqp.streamlit.app/

## Database path

The local development SQLite database lives at `data/app.db` by default.

- Override it with `SQLITE_DB_PATH` when needed.
- Configure market-data freshness with `MARKET_DATA_METADATA_FRESHNESS_HOURS` and `MARKET_DATA_PRICES_FRESHNESS_HOURS`.
- The app resolves relative SQLite paths against the repository root, not the current working directory.
- The app validates the resolved SQLite path on startup and logs the fully resolved absolute path.
- Startup fails early if the parent directory is not writable, the database file cannot be created, or SQLite cannot open the file for writes.

### Market Data Architecture

The market-data path is now separated into a small storage-backed pipeline:

- provider layer: `src/data/providers/market_data_provider.py`
- ingestion service: `src/data/ingestion/service.py`
- SQL repository: `src/data/storage/repository.py`
- SQL query interface for app/report reads: `src/data/queries/market_data_queries.py`

The intended flow is:

`API -> normalize -> persist to SQL -> query from SQL -> analytics/reporting/app`

The app and reporting layers should read market data from SQL through the query/repository layer rather than calling the provider directly.

### Resetting the local database safely

1. Stop the Streamlit app or any Python process using the database.
2. Delete `data/app.db`.
3. Restart the app so the schema is recreated automatically.

Older local database files from the previous layout can also be deleted if you are no longer using them:

- `data/processed/asset_intelligence.db`
- `data/app.db-wal`
- `data/app.db-shm`
- `data/app.db-journal`
- `data/processed/asset_intelligence.db-wal`
- `data/processed/asset_intelligence.db-shm`
- `data/processed/asset_intelligence.db-journal`

If you are deploying to a hosted environment with a read-only repo mount, choose a writable location explicitly with `SQLITE_DB_PATH` before startup.

### SQLite write debug

Run this before starting the app if you want to verify the exact local DB target and writeability outside Streamlit:

```bash
python -m src.database.debug_sqlite_write
```

## What the project does

The user can enter or select a financial instrument such as a stock, ETF, or crypto asset by entering its ticker/CUSIP at the top right. The system then:

- pulls and stores historical market data on the asset
- retrieves and stores recent news based sentiment data using an API
- structures both datasets in a SQL database
- calculates return and downside risk metrics
- runs a simulation analysis
- displays the output in a Streamlit app
- generates a multi-page analyst-style PDF briefing

## Core features

### 1. Market data ingestion
Historical asset data is pulled from external APIs and normalized into a local SQL-backed store.

### 2. SQL database design
The project uses a structured relational layer for assets, historical prices, data sources, and news articles. This makes the workflow reusable and closer to a real analytics system than a notebook only project.

### 3. Return analytics
The platform calculates:
- daily returns
- cumulative returns
- total return
- annualized return

### 4. Risk analytics
The platform calculates:
- annualized volatility
- rolling volatility
- maximum drawdown
- historical Value at Risk (VaR)
- Expected Shortfall

### 5. Monte Carlo simulation
The project includes forward simulation logic so the workbench can produce scenario oriented outputs rather than just historical description.

### 6. News sentiment layer
Recent news coverage is ingested, stored, and scored. This adds directional context without overcomplicating the architecture.

### 7. Machine Learning Signal Calibration Layer
The ML subsystem is framed as a dynamic weighting engine rather than a toy price predictor. It combines three explicit research pillars:

- history / technical / market-structure signals
- risk / downside / scenario signals
- sentiment / external-information signals

The model target is forward 20-trading-day return rather than raw price level. That choice is more finance-oriented because returns are comparable across assets, less dominated by nominal price scale, and more aligned with decision-support workflows.

The workflow is intentionally structured:

- raw market and sentiment data are ingested and stored in SQL
- engineered features are built into history, risk, and sentiment pillars
- standardized `history_score`, `risk_score`, and `sentiment_score` are created before model training
- a ridge-based weighting engine learns an interpretable baseline allocation across signals
- a random-forest challenger model is trained as a nonlinear comparison benchmark
- outputs are evaluated with time-aware train/test splits and expanding-window validation
- scored results are written back to SQL for reuse by the app and the PDF report

The app and report expose:

- composite ML score
- directional signal
- confidence indicator
- pillar contribution breakdown
- learned pillar weights
- top feature importance

This keeps the ML layer grounded as an investment-research support tool rather than a black-box trading bot.

### 8. Multi-page PDF briefing
The reporting layer generates a formal PDF-style asset briefing with:
- summary
- performance overview
- risk analysis
- simulation section
- sentiment section
- methodology notes

## Repository structure

```text
asset-intelligence-workbench/
├── app/                # Streamlit UI
├── data/               # Local database / processed data
├── sql/                # Schema and SQL artifacts
├── src/
│   ├── analytics/      # Returns, risk, simulation logic
│   ├── database/       # ORM models, loaders, queries, connections
│   ├── ingestion/      # Market and sentiment ingestion workflows
│   ├── reporting/      # Report context + PDF generation
│   ├── utils/          # App-facing data preparation and config
│   └── visuals/        # Charts and visual helpers
├── tests/              # Focused unit tests
├── requirements.txt
└── README.md


## Running the ML layer locally

1. Activate the project environment.
2. Start the app with `streamlit run app\streamlit_app.py`.
3. Load or ingest a ticker with enough historical data.
4. Open the `Machine Learning Signal Calibration` section in the app.
5. Generate a PDF report to include the ML signal page in the briefing.

Focused verification commands:

```bash
python -m unittest tests.test_features tests.test_ml_models tests.test_ml_integration tests.test_app_data
```
