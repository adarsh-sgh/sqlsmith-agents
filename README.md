# sqlsmith-agents

Multi-agent Text2SQL over a SQLite e-commerce DB, built on DSPy, with an execution-accuracy eval
harness (pandas + MLflow), a `BootstrapFewShot` prompt-optimization run, and a scikit-learn / PyTorch
difficulty router that sends easy questions down a cheaper single-module path. The router and the
pipeline are served behind a FastAPI inference service with micro-batched routing, router versioning
through the MLflow model registry, a concurrency benchmark, and an offline-vs-online parity check.

```
question ──> Router (TF-IDF + LogReg | torch.nn.Linear) ──easy──> Writer ─────┐
                 │                                                          │
               complex                                                      v
                 └──> Planner (CoT: pick tables) ──> Writer (SQL) ──> Critic ──exec──> SQLite
                                                                       ^  │ error
                                                                       └──┘ Repair (<=2x)
eval: run 54 gold questions ──> execute pred & gold ──> pandas compare ──> MLflow run (./mlruns)

serve: POST /v1/text2sql ──> MicroBatcher (<=32 q or 5ms) ──> router.predict_many ──> thread pool
                                                                  (models:/sqlsmith-router@champion)   └─ per-thread SQLite + pipelines
```

Every LM call goes through `dspy.BaseLM`, so the pipeline is pluggable: `ClaudeLM` wraps the Anthropic
SDK (`claude-sonnet-5`), `FakeLM` is a deterministic oracle over the gold set that speaks DSPy's
ChatAdapter format and injects failures per question (bogus column -> repair loop; silently wrong
result -> caught only by the eval). Tests and the default CLI need no API key.

## Run

```
make setup                                   # python3 -m venv .venv && pip install
make test                                    # pytest, 17 tests, ~12s, no network
make eval                                    # 54 questions with the fake LM, logs to ./mlruns
make optimize                                # BootstrapFewShot, saves compiled/text2sql.json
make train-router                            # 5-fold CV for both router backends
make ask Q="Which category generates the most revenue?"
ANTHROPIC_API_KEY=... make eval LM=claude    # same harness against Claude
make publish-router                          # train, CV-gate, register a new version, move alias `champion`
make serve ROUTER=registry                   # FastAPI on :8000, router loaded from the registry alias
make bench C=16 N=200                        # p50/p95/p99 + rps against the app in-process (or --url http://host:8000)
make parity                                  # served endpoint vs run_eval on all 54 gold questions
.venv/bin/mlflow ui                          # browse runs and model versions
```

Endpoints: `GET /healthz`, `GET /v1/model` (router source/version, batcher counters), `POST /v1/route`
(`{"questions": [...]}`), `POST /v1/text2sql` (`{"question": "...", "with_rows": false}` -> sql, path,
attempts, errors, rows, latency_ms).

## Results (fake LM, so numbers describe the harness, not a model)

| run | exec acc | easy | medium | hard | repair rate |
|---|---|---|---|---|---|
| `eval` routed, 54 q | 0.870 | 0.889 | 0.947 | 0.765 | 0.204 |
| `optimize` val (28 q) before -> after | 0.821 -> 0.821 | | | | demos: plan 4, write 4, repair 2 |

Router 5-fold accuracy (106 labelled questions, majority baseline 0.66): sklearn 0.906, torch 0.887.

Serving, 400 requests to `/v1/text2sql`, fake LM, 4 workers, M-series laptop. In-process numbers measure the
batcher + thread pool + DSPy pipeline; the HTTP rows add uvicorn and the loopback stack.

| transport | concurrency | p50 ms | p95 ms | p99 ms | req/s |
|---|---|---|---|---|---|
| in-process (ASGI) | 16 | 14.1 | 20.8 | 32.2 | 1047 |
| in-process (ASGI) | 64 | 53.2 | 61.7 | 73.6 | 1146 |
| HTTP (uvicorn) | 16 | 54.9 | 199.8 | 287.8 | 205 |
| HTTP (uvicorn) | 64 | 131.7 | 330.6 | 337.6 | 373 |

Throughput plateaus around 1.1k req/s in-process because the fake LM makes the pipeline CPU-bound under the
GIL; with a real LM the pool is I/O-bound and `--workers` becomes the knob. Parity: 54/54 questions score
identically offline and online (0.870 both sides), so the served path introduces no drift.

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
- Serving batches the router, not the LM: routing is one vectorised sklearn/torch call so coalescing
  concurrent requests (up to 32 or 5 ms) is nearly free, whereas LM calls are per question and only gain
  from concurrency, so they fan out to a thread pool. Each worker thread owns its SQLite connection and
  pipeline objects because sqlite connections are not shareable and DSPy modules carry state.
- The registry stores both router backends behind one `pyfunc` entry; `load_router` unwraps the native
  object so serving calls `predict_many` directly. Promotion is gated on 5-fold CV accuracy (`--min-cv-acc`),
  so a worse version is recorded but never becomes `champion`.
- Parity compares per-question correctness *and* routing path between `run_eval` and the endpoint, not just
  the aggregate, so a router version mismatch or a path-specific bug shows up as a named question.

## Next

- MIPROv2 with a real LM and a held-out split larger than 28 questions.
- Schema-aware retrieval (column descriptions, sample values) in the planner for a bigger DB (Spider).
- Route on router confidence, and log per-path cost and latency to MLflow.
- Bench with `--lm claude` to size `--workers` under real LM latency; add a request-level cache keyed on question.
