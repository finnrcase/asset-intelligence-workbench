# Asset Intelligence Workbench

SQL-Driven Financial Analytics, Risk Simulation, and Reporting Platform

Asset Intelligence Workbench is a portfolio project designed to feel like a real internal analyst tool rather than a toy dashboard. It combines structured SQL-backed data storage, market data ingestion, return and risk analytics, Monte Carlo simulation, news sentiment context, and multi-page PDF reporting in a single workflow.

The goal of the project was to build something closer to a finance, treasury, strategic finance, or risk decision-support tool that could plausibly be used inside a modern company.

## What the project does

A user can enter or select a financial instrument such as a stock, ETF, or crypto asset. The system then:

- pulls and stores historical market data
- retrieves and stores recent news-based sentiment data
- structures both datasets in a SQL database
- calculates return and downside risk metrics
- runs forward simulation analysis
- displays the output in a Streamlit app
- generates a multi-page analyst-style PDF briefing

## Why I built it

A lot of portfolio finance projects stop at a chart or a simple dashboard. I wanted to build something more complete and more realistic: a full workflow that starts with ingestion and storage, moves through analytics, and ends with decision-ready outputs.

This project was built to demonstrate:

- finance and economics reasoning
- SQL-backed data workflows
- Python analytics and simulation
- modular data pipeline design
- sentiment/alternative-data integration
- reporting and presentation polish

## Core features

### 1. Market data ingestion
Historical asset data is pulled from external APIs and normalized into a local SQL-backed store.

### 2. SQL database design
The project uses a structured relational layer for assets, historical prices, data sources, and news articles. This makes the workflow reusable and closer to a real analytics system than a notebook-only project.

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
The project includes forward simulation logic so the workbench can produce scenario-oriented outputs rather than just historical description.

### 6. News sentiment layer
Recent news coverage is ingested, stored, and scored using a lightweight lexical sentiment approach. This adds directional context without overcomplicating the architecture.

### 7. Streamlit app
The user-facing application presents:
- asset metadata
- historical performance charts
- risk metrics
- sentiment summaries
- simulation outputs
- PDF report generation

### 8. Multi-page PDF briefing
The reporting layer generates a formal PDF-style asset briefing with:
- executive summary
- performance overview
- risk analysis
- simulation section
- sentiment section
- methodology notes

## Project architecture

The workflow is:

**API / provider ingestion -> SQL storage -> query layer -> analytics and simulation -> app and PDF outputs**

This separation was intentional. I wanted the project to reflect a more realistic internal-tool architecture rather than mixing everything into one script.

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