"""Execution-accuracy eval harness: pandas result comparison + MLflow run tracking."""

import sqlite3
import time

import dspy
import mlflow
import pandas as pd

from sqlsmith.db import run_sql
from sqlsmith.gold import GOLD


def _canonical(df):
    """Order-insensitive view of a result: a sorted list of per-column value multisets, names ignored."""
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            s = s.round(4)
        cols.append(tuple(sorted(map(str, s.tolist()))))
    return sorted(cols)


def results_match(pred_df, gold_df):
    return pred_df.shape == gold_df.shape and _canonical(pred_df) == _canonical(gold_df)


def exec_match(conn, pred_sql, gold_sql):
    try:
        return results_match(run_sql(conn, pred_sql), run_sql(conn, gold_sql))
    except sqlite3.Error:
        return False


def make_metric(conn):
    """DSPy-style metric(example, prediction, trace) usable by evaluators and optimizers."""

    def metric(example, pred, trace=None):
        return exec_match(conn, pred.sql, example.gold_sql)

    return metric


def to_examples(gold):
    return [dspy.Example(question=g.question, gold_sql=g.sql, difficulty=g.difficulty).with_inputs("question") for g in gold]


def run_eval(program, conn, gold=GOLD):
    """Run every gold question through `program`; return (metrics, per-question DataFrame)."""
    rows = []
    for g in gold:
        t0 = time.perf_counter()
        pred = program(question=g.question)
        t = pred.trace
        rows.append(
            {
                "question": g.question, "difficulty": g.difficulty, "pred_sql": pred.sql, "gold_sql": g.sql,
                "correct": exec_match(conn, pred.sql, g.sql), "executed": t.ok, "attempts": t.attempts,
                "repaired": t.attempts > 1, "path": t.path, "latency_s": round(time.perf_counter() - t0, 4),
            }
        )
    df = pd.DataFrame(rows)
    metrics = {
        "execution_accuracy": df.correct.mean(),
        "execution_rate": df.executed.mean(),
        "repair_rate": df.repaired.mean(),
        "avg_attempts": df.attempts.mean(),
        "n_questions": len(df),
    }
    for d, sub in df.groupby("difficulty"):
        metrics[f"accuracy_{d}"] = sub.correct.mean()
    return metrics, df


def tracked_eval(program, conn, params, gold=GOLD, tracking_uri="file:./mlruns", experiment="sqlsmith", run_name=None):
    """run_eval wrapped in an MLflow run: params, metrics, and the per-question table are logged."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        metrics, df = run_eval(program, conn, gold)
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_table(df, "results.json")
        metrics["run_id"] = run.info.run_id
    return metrics, df
