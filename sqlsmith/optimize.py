"""Prompt optimization: dspy.BootstrapFewShot over the pipeline with execution match as the metric."""

import random
from pathlib import Path

from dspy.teleprompt import BootstrapFewShot

from sqlsmith.agents import Text2SQL
from sqlsmith.evaluate import make_metric, run_eval, to_examples
from sqlsmith.gold import GOLD


def split_gold(gold=GOLD, train_frac=0.5, seed=0):
    """Deterministic, difficulty-stratified train/val split."""
    rng = random.Random(seed)
    train, val = [], []
    for d in ("easy", "medium", "hard"):
        items = [g for g in gold if g.difficulty == d]
        rng.shuffle(items)
        k = int(len(items) * train_frac)
        train += items[:k]
        val += items[k:]
    return train, val


def optimize(conn, max_demos=4, save_path="compiled/text2sql.json", seed=0):
    """Compile Text2SQL with bootstrapped demos; return before/after validation metrics and demo counts."""
    train, val = split_gold(seed=seed)
    program = Text2SQL(conn)
    before, _ = run_eval(program, conn, val)

    optimizer = BootstrapFewShot(
        metric=make_metric(conn), max_bootstrapped_demos=max_demos, max_labeled_demos=0, max_rounds=1
    )
    compiled = optimizer.compile(program, trainset=to_examples(train))
    after, _ = run_eval(compiled, conn, val)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    compiled.save(save_path)
    demos = {name: len(p.demos) for name, p in compiled.named_predictors()}
    return {"before": before, "after": after, "demos": demos, "n_train": len(train), "n_val": len(val)}
