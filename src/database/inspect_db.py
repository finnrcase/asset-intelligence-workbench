"""
Database inspection utility for Asset Intelligence Workbench.

This script makes the SQL layer visible from the terminal by listing tables,
printing row counts, and running a few representative raw SQL queries against
the configured database.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import text

from src.database.connection import session_scope
from src.utils.config import get_config


def print_section(title: str) -> None:
    """Print a simple section divider for terminal readability."""

    print(f"\n{title}")
    print("-" * len(title))


def print_rows(rows: Sequence[dict], max_rows: int | None = None) -> None:
    """Render a sequence of dictionaries in a compact table-like format."""

    if not rows:
        print("No rows returned.")
        return

    display_rows = list(rows[:max_rows] if max_rows is not None else rows)
    columns = list(display_rows[0].keys())
    widths = {
        column: max(len(str(column)), max(len(str(row.get(column, ""))) for row in display_rows))
        for column in columns
    }

    header = " | ".join(f"{column:<{widths[column]}}" for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)

    print(header)
    print(separator)
    for row in display_rows:
        print(" | ".join(f"{str(row.get(column, '')):<{widths[column]}}" for column in columns))

    if max_rows is not None and len(rows) > max_rows:
        print(f"... showing {max_rows} of {len(rows)} rows")


def fetch_all_as_dicts(session, sql: str, params: dict | None = None) -> list[dict]:
    """Execute a raw SQL statement and return rows as dictionaries."""

    result = session.execute(text(sql), params or {})
    return [dict(row._mapping) for row in result]


def main() -> None:
    """Inspect the configured database using a few representative raw SQL queries."""

    config = get_config()
    print("Asset Intelligence Workbench Database Inspection")
    print(f"Configured database: {config.database_url}")

    with session_scope() as session:
        print_section("Tables")
        tables = fetch_all_as_dicts(
            session,
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """,
        )
        print_rows(tables)

        print_section("Row Counts")
        row_counts = []
        for table_name in ("data_sources", "assets", "historical_prices"):
            count_rows = fetch_all_as_dicts(
                session,
                f"SELECT '{table_name}' AS table_name, COUNT(*) AS row_count FROM {table_name}",
            )
            row_counts.extend(count_rows)
        print_rows(row_counts)

        print_section("Assets")
        assets = fetch_all_as_dicts(
            session,
            """
            SELECT
                ticker,
                asset_name,
                asset_class,
                exchange,
                currency
            FROM assets
            ORDER BY ticker
            """,
        )
        print_rows(assets)

        sample_ticker = assets[0]["ticker"] if assets else "AAPL"

        print_section(f"Recent Prices: {sample_ticker}")
        recent_prices = fetch_all_as_dicts(
            session,
            """
            SELECT
                a.ticker,
                hp.price_date,
                hp.open_price,
                hp.high_price,
                hp.low_price,
                hp.close_price,
                hp.volume
            FROM historical_prices AS hp
            INNER JOIN assets AS a
                ON hp.asset_id = a.id
            WHERE a.ticker = :ticker
            ORDER BY hp.price_date DESC
            LIMIT 10
            """,
            {"ticker": sample_ticker},
        )
        print_rows(recent_prices)

        print_section("Joined Asset + Price Example")
        joined_rows = fetch_all_as_dicts(
            session,
            """
            SELECT
                a.ticker,
                a.asset_name,
                ds.source_name,
                hp.price_date,
                hp.close_price,
                hp.adjusted_close,
                hp.volume
            FROM historical_prices AS hp
            INNER JOIN assets AS a
                ON hp.asset_id = a.id
            INNER JOIN data_sources AS ds
                ON hp.source_id = ds.id
            ORDER BY hp.price_date DESC, a.ticker ASC
            LIMIT 15
            """,
        )
        print_rows(joined_rows)


if __name__ == "__main__":
    main()
