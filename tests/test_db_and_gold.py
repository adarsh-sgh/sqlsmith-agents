import sqlite3

import pytest

from sqlsmith.db import run_sql, schema_text, table_names
from sqlsmith.gold import GOLD


def test_every_gold_query_runs_and_returns_rows(conn):
    assert 40 <= len(GOLD) <= 60
    assert len({g.question for g in GOLD}) == len(GOLD)
    for g in GOLD:
        df = run_sql(conn, g.sql)
        assert len(df) > 0, g.question
        # Aggregates must not be NULL: the seed data has to cover every question.
        assert not df.isna().any().any(), g.question


def test_schema_helpers_and_readonly_guard(conn):
    assert table_names(conn) == ["categories", "customers", "order_items", "orders", "products", "reviews"]
    subset = schema_text(conn, ["orders", "customers"])
    assert "CREATE TABLE orders" in subset and "CREATE TABLE reviews" not in subset
    with pytest.raises(sqlite3.Error):
        run_sql(conn, "DELETE FROM customers")
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        run_sql(conn, "SELECT nope FROM customers")
