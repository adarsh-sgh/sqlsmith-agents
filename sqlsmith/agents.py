"""DSPy signatures and modules: planner -> writer -> critic/repair, plus a single-agent fast path."""

import copy
import sqlite3
from dataclasses import dataclass, field

import dspy

from sqlsmith.db import run_sql, schema_text, table_names


class PlanTables(dspy.Signature):
    """Pick the minimal set of tables needed to answer the question. Output a comma-separated list of table names."""

    question: str = dspy.InputField()
    ddl: str = dspy.InputField(desc="SQLite DDL for all tables")
    tables: str = dspy.OutputField(desc="comma-separated table names, e.g. 'orders, customers'")


class WriteSQL(dspy.Signature):
    """Write one SQLite SELECT statement answering the question. Return only SQL, no markdown fences."""

    question: str = dspy.InputField()
    ddl: str = dspy.InputField(desc="SQLite DDL for the relevant tables")
    sql: str = dspy.OutputField(desc="a single SQLite SELECT statement")


class RepairSQL(dspy.Signature):
    """The SQL failed when executed. Fix it so it runs on SQLite and answers the question."""

    question: str = dspy.InputField()
    ddl: str = dspy.InputField()
    sql: str = dspy.InputField(desc="the failing SQL")
    error: str = dspy.InputField(desc="SQLite error message")
    fixed_sql: str = dspy.OutputField(desc="corrected SQLite SELECT statement")


@dataclass
class Trace:
    sql: str
    tables: list = field(default_factory=list)
    attempts: int = 1
    errors: list = field(default_factory=list)
    path: str = "multi"
    ok: bool = True


def clean_sql(sql):
    sql = sql.strip()
    if sql.startswith("```"):
        sql = sql.strip("`").lstrip("sql").strip()
    return sql.rstrip(";").strip()


class _SharesConnection:
    """Optimizers deep-copy programs; sqlite connections can't be copied, so copy predictors only."""

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        for k, v in self.__dict__.items():
            if isinstance(v, dspy.Module):
                setattr(new, k, copy.deepcopy(v, memo))
        return new


class Critic(_SharesConnection, dspy.Module):
    """Executes SQL against the DB; on error asks the repair agent for a fix, up to max_repairs times."""

    def __init__(self, conn, max_repairs=2):
        super().__init__()
        self.conn = conn
        self.repair = dspy.Predict(RepairSQL)
        self.max_repairs = max_repairs

    def forward(self, question, schema, sql, trace):
        while True:
            try:
                run_sql(self.conn, sql)
                trace.sql, trace.ok = sql, True
                return trace
            except sqlite3.Error as e:
                trace.errors.append(str(e))
                if trace.attempts > self.max_repairs:
                    trace.sql, trace.ok = sql, False
                    return trace
                trace.attempts += 1
                sql = clean_sql(self.repair(question=question, ddl=schema, sql=sql, error=str(e)).fixed_sql)


class Text2SQL(_SharesConnection, dspy.Module):
    """Multi-agent pipeline. `single=True` skips the planner and sends the full schema to the writer."""

    def __init__(self, conn, single=False, max_repairs=2):
        super().__init__()
        self.conn = conn
        self.single = single
        self.plan = dspy.ChainOfThought(PlanTables)
        self.write = dspy.Predict(WriteSQL)
        self.critic = Critic(conn, max_repairs)
        self.all_tables = table_names(conn)
        self.full_schema = schema_text(conn)

    def forward(self, question):
        tables, schema = [], self.full_schema
        if not self.single:
            raw = self.plan(question=question, ddl=self.full_schema).tables
            tables = [t.strip() for t in raw.split(",") if t.strip() in self.all_tables]
            if tables:
                schema = schema_text(self.conn, tables)
        sql = clean_sql(self.write(question=question, ddl=schema).sql)
        trace = Trace(sql=sql, tables=tables, path="single" if self.single else "multi")
        trace = self.critic(question=question, schema=schema, sql=sql, trace=trace)
        return dspy.Prediction(sql=trace.sql, trace=trace)


class RoutedText2SQL(_SharesConnection, dspy.Module):
    """Routes easy questions (per the difficulty classifier) to the single-agent path."""

    def __init__(self, conn, router, max_repairs=2):
        super().__init__()
        self.router = router
        self.multi = Text2SQL(conn, single=False, max_repairs=max_repairs)
        self.fast = Text2SQL(conn, single=True, max_repairs=max_repairs)

    def forward(self, question):
        difficulty = self.router.predict(question)
        pred = (self.fast if difficulty == "easy" else self.multi)(question=question)
        pred.trace.path = f"{pred.trace.path}:{difficulty}"
        return pred
