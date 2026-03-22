"""
Database connection and ORM model definitions.

The project uses SQLite for local development today, but the engine/session
setup and SQLAlchemy models are structured to support a future move to
PostgreSQL with minimal application code changes.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from sqlalchemy import Boolean
from sqlalchemy import Date
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import Numeric
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import Session
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker

from src.utils.config import get_config


config = get_config()


class Base(DeclarativeBase):
    """Base class for declarative ORM models."""


class DataSource(Base):
    """
    Reference table for upstream data providers.

    Tracking the source separately supports lineage, ingestion auditability, and
    future reconciliation between multiple vendors.
    """

    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    assets: Mapped[list["Asset"]] = relationship(back_populates="primary_source")
    price_history: Mapped[list["HistoricalPrice"]] = relationship(
        back_populates="data_source"
    )


class Asset(Base):
    """Security master record for supported financial instruments."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    asset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    asset_class: Mapped[str | None] = mapped_column(String(50), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(50), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(100), nullable=True)
    country: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    primary_source_id: Mapped[int | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    primary_source: Mapped[DataSource | None] = relationship(back_populates="assets")
    price_history: Mapped[list["HistoricalPrice"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class HistoricalPrice(Base):
    """Daily market data for an asset at a specific observation date."""

    __tablename__ = "historical_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[int] = mapped_column(
        ForeignKey("data_sources.id", ondelete="RESTRICT"),
        nullable=False,
    )
    price_date: Mapped[date] = mapped_column(Date, nullable=False)
    open_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    high_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    low_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    close_price: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    adjusted_close: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), nullable=True
    )
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingestion_timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    asset: Mapped[Asset] = relationship(back_populates="price_history")
    data_source: Mapped[DataSource] = relationship(back_populates="price_history")

    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "source_id",
            "price_date",
            name="uq_historical_prices_asset_source_date",
        ),
        Index("ix_historical_prices_asset_date", "asset_id", "price_date"),
        Index("ix_historical_prices_date", "price_date"),
    )


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """Enable SQLite foreign key enforcement on every new connection."""

    if not config.is_sqlite:
        return

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine for the configured backend."""

    connect_args = {"check_same_thread": False} if config.is_sqlite else {}
    engine = create_engine(
        config.database_url,
        echo=config.sqlalchemy_echo,
        future=True,
        connect_args=connect_args,
    )
    _enable_sqlite_foreign_keys(engine)
    return engine


ENGINE = get_engine()
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, future=True)


def reset_database_engine() -> Engine:
    """Dispose and rebuild the shared engine/session factory for the current process."""

    global ENGINE
    global SessionLocal

    ENGINE.dispose()
    ENGINE = get_engine()
    SessionLocal.configure(bind=ENGINE)
    return ENGINE


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Provide a transactional session boundary.

    This pattern keeps session handling explicit in loaders, queries, or future
    service modules without scattering commit/rollback boilerplate everywhere.
    """

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def initialize_database(schema_path: Path | None = None) -> None:
    """
    Initialize the database schema.

    If a schema file is provided, it is executed directly. Otherwise SQLAlchemy
    metadata is used to create the tables represented by the ORM models.
    """

    if config.is_sqlite:
        config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    if schema_path is not None:
        sql_text = schema_path.read_text(encoding="utf-8")
        with ENGINE.begin() as connection:
            for statement in [chunk.strip() for chunk in sql_text.split(";") if chunk.strip()]:
                connection.exec_driver_sql(statement)
        return

    Base.metadata.create_all(bind=ENGINE)
