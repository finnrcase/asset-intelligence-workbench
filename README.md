# Asset Intelligence Workbench

SQL-Driven Financial Analytics, Risk Simulation, and Reporting Platform

The Asset Intelligence Workbench is a portfolio project designed to feel like a real internal analyst tool. It combines structured SQL-backed data storage, market data ingestion, return and risk analytics, Monte Carlo simulation, live news sentiment context, and a detailed PDF report in a single workflow.

The goal of the project was to build something closer to a finance, treasury, strategic finance, or risk decision-support tool that could plausibly be used inside a modern company.

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

### 7. Multi-page PDF briefing
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