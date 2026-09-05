import mlflow
import pandas as pd

from sqlsmith.agents import Text2SQL
from sqlsmith.evaluate import results_match, tracked_eval
from sqlsmith.gold import GOLD


def test_results_match_ignores_order_and_names_but_not_values():
    a = pd.DataFrame({"name": ["x", "y"], "total": [1.00001, 2.0]})
    assert results_match(a, pd.DataFrame({"n": ["y", "x"], "t": [2.0, 1.0]}))
    assert not results_match(a, pd.DataFrame({"name": ["x", "y"], "total": [1.5, 2.0]}))
    assert not results_match(a, a.head(1))
    assert not results_match(a, a[["name"]])


def test_tracked_eval_logs_run_to_mlflow(conn, fake_lm, tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    metrics, df = tracked_eval(
        Text2SQL(conn), conn, params={"lm": "fake"}, gold=GOLD[:6], tracking_uri=uri, experiment="t", run_name="r"
    )
    assert len(df) == 6 and metrics["n_questions"] == 6
    run = mlflow.get_run(metrics["run_id"])
    assert run.data.params == {"lm": "fake"}
    assert run.data.metrics["execution_accuracy"] == metrics["execution_accuracy"]
    assert any(a.path == "results.json" for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id))
