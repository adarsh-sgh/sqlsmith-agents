"""End-to-end flows through planner -> writer -> critic with the deterministic fake LM."""

from sqlsmith.agents import RoutedText2SQL, Text2SQL
from sqlsmith.evaluate import exec_match, run_eval
from sqlsmith.gold import GOLD
from sqlsmith.lm import FakeLM
from sqlsmith.router import train_router


def _by_bucket(buckets):
    return next(g for g in GOLD if FakeLM.bucket(g.question) in buckets)


def test_multi_agent_repairs_broken_sql(conn, fake_lm):
    g = _by_bucket(fake_lm.break_first)
    pred = Text2SQL(conn)(question=g.question)
    t = pred.trace
    assert t.attempts == 2 and t.ok and "no_such_col" in t.errors[0]
    assert t.tables and all(tbl in g.sql for tbl in t.tables)
    assert exec_match(conn, pred.sql, g.sql)


def test_repair_gives_up_after_max_attempts(conn, fake_lm):
    # Every bucket breaks and the "fix" is broken too, so the critic must stop at max_repairs.
    lm = FakeLM(break_first=range(10))
    lm._value = lambda name, inputs: "SELECT still_bad FROM customers"
    import dspy

    dspy.configure(lm=lm)
    t = Text2SQL(conn, max_repairs=2)(question="How many customers are there?").trace
    assert not t.ok and t.attempts == 3 and len(t.errors) == 3


def test_single_path_skips_planner_and_router_splits_traffic(conn, fake_lm):
    easy = "How many customers are there?"
    hard = "Who are the top 3 customers by total spend? Show name and total spend."
    assert Text2SQL(conn, single=True)(question=easy).trace.tables == []

    routed = RoutedText2SQL(conn, train_router("sklearn"))
    assert routed(question=easy).trace.path == "single:easy"
    assert routed(question=hard).trace.path == "multi:complex"


def test_full_eval_reflects_injected_failures(conn, fake_lm):
    metrics, df = run_eval(Text2SQL(conn), conn)
    assert metrics["n_questions"] == len(GOLD)
    assert metrics["execution_rate"] == 1.0  # every broken query was repaired
    assert 0 < metrics["repair_rate"] < 0.5
    assert 0.7 < metrics["execution_accuracy"] < 1.0  # the "wrong result" bucket is uncatchable
    wrong = df[~df.correct]
    assert wrong.executed.all() and set(map(FakeLM.bucket, wrong.question)) == fake_lm.wrong
