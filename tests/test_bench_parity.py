"""Benchmark and offline-vs-online parity run in-process against the ASGI app."""

from sqlsmith.bench import bench
from sqlsmith.gold import GOLD
from sqlsmith.parity import parity
from sqlsmith.serve import Service, create_app


def test_bench_reports_percentiles_and_parity_holds():
    service = Service(lm="fake", router="sklearn", workers=4, max_batch=16, max_wait_ms=2)
    app = create_app(service)

    b = bench(app=app, concurrency=8, requests=40, warmup=4)
    assert b["errors"] == 0 and b["requests"] == 40
    assert 0 < b["p50_ms"] <= b["p95_ms"] <= b["p99_ms"] and b["rps"] > 0
    assert service.batcher.batches < 44  # concurrent requests were coalesced into fewer router calls

    p = parity(service=service, gold=GOLD)
    assert p["n"] == len(GOLD) and p["parity"] == 1.0 and p["mismatches"] == []
    assert 0.7 < p["offline_accuracy"] == p["online_accuracy"] < 1.0  # same injected failures on both sides
