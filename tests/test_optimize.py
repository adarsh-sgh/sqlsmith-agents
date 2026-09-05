import json

from sqlsmith.optimize import optimize, split_gold


def test_bootstrap_fewshot_compiles_and_saves(conn, fake_lm, tmp_path):
    train, val = split_gold()
    assert len(train) + len(val) == 54 and {g.difficulty for g in train} == {"easy", "medium", "hard"}
    out = tmp_path / "compiled.json"
    report = optimize(conn, max_demos=3, save_path=str(out))
    assert report["demos"]["write"] == 3 and report["demos"]["plan.predict"] == 3
    assert 0 < report["demos"]["critic.repair"] <= 3  # only bootstrapped from questions that needed repair
    assert report["after"]["execution_accuracy"] >= report["before"]["execution_accuracy"]
    saved = json.loads(out.read_text())
    assert {d["question"] for d in saved["write"]["demos"]} <= {g.question for g in train}
