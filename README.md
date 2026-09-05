# sqlsmith-agents

Multi-agent Text2SQL over a SQLite e-commerce DB, built on DSPy, with an execution-accuracy eval
harness (pandas + MLflow), a `BootstrapFewShot` prompt-optimization run, and a scikit-learn / PyTorch
difficulty router that sends easy questions down a cheaper single-module path.

```
question ──> Router (TF-IDF + LogReg | torch.nn.Linear) ──easy──> Writer ─────┐
                 │                                                          │
               complex                                                      v
                 └──> Planner (CoT: pick tables) ──> Writer (SQL) ──> Critic ──exec──> SQLite
                                                                       ^  │ error
                                                                       └──┘ Repair (<=2x)
eval: run 54 gold questions ──> execute pred & gold ──> pandas compare ──> MLflow run (./mlruns)
```

Every LM call goes through `dspy.BaseLM`, so the pipeline is pluggable: `ClaudeLM` wraps the Anthropic
SDK (`claude-sonnet-5`), `FakeLM` is a deterministic oracle over the gold set that speaks DSPy's
ChatAdapter format and injects failures per question (bogus column -> repair loop; silently wrong
result -> caught only by the eval). Tests and the default CLI need no API key.

## Run

```
make setup                                   # python3 -m venv .venv && pip install
make test                                    # pytest, 13 tests, ~4s, no network
make eval                                    # 54 questions with the fake LM, logs to ./mlruns
make optimize                                # BootstrapFewShot, saves compiled/text2sql.json
make train-router                            # 5-fold CV for both router backends
make ask Q="Which category generates the most revenue?"
ANTHROPIC_API_KEY=... make eval LM=claude    # same harness against Claude
.venv/bin/mlflow ui                          # browse runs
```

## Results (fake LM, so numbers describe the harness, not a model)

| run | exec acc | easy | medium | hard | repair rate |
|---|---|---|---|---|---|
| `eval` routed, 54 q | 0.870 | 0.889 | 0.947 | 0.765 | 0.204 |
| `optimize` val (28 q) before -> after | 0.821 -> 0.821 | | | | demos: plan 4, write 4, repair 2 |

Router 5-fold accuracy (106 labelled questions, majority baseline 0.66): sklearn 0.906, torch 0.887.

The optimizer's before/after are equal by construction: the fake LM answers from a lookup table and
ignores demos. The run still exercises the full compile path (metric-gated trace bootstrapping,
demo selection per predictor, save/load), so swapping in `--lm claude` gives a real number.

## Design decisions

- Critic executes rather than reviews: SQLite is the judge of syntax, the LM only sees the error string.
  Wrong-but-valid SQL is deliberately left to the eval metric so the two failure modes stay separate.
- Execution accuracy compares result multisets column-wise with float rounding, ignoring row order and
  column names; the gold SQL is the spec, not the string.
- Router is binary (easy vs complex) because that is the only decision it makes; 3-way labels were 0.6 CV
  on 106 examples. Seven structural features (entity nouns, "each/per/top/than", opener) added ~8 points
  over TF-IDF alone. The torch model is the same linear classifier so the two backends are comparable.
- The fake LM breaks a fixed 30% of questions (crc32 buckets) so repair and accuracy < 1 are observable
  and reproducible in CI without a key.
- `dspy` programs are deep-copied by optimizers; modules share one sqlite connection via a
  `__deepcopy__` that copies predictors only.

## Next

- MIPROv2 with a real LM and a held-out split larger than 28 questions.
- Schema-aware retrieval (column descriptions, sample values) in the planner for a bigger DB (Spider).
- Route on router confidence, and log per-path cost and latency to MLflow.
