"""Latency/throughput benchmark: N concurrent clients against /v1/text2sql, in-process (ASGI) or over HTTP."""

import asyncio
import statistics
import time

import httpx

from sqlsmith.gold import GOLD


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def run_bench(app=None, base_url=None, endpoint="/v1/text2sql", concurrency=16, requests=200, questions=None, warmup=4):
    """Fire `requests` POSTs with at most `concurrency` in flight; return latency percentiles (ms) and rps."""
    questions = questions or [g.question for g in GOLD]
    transport = httpx.ASGITransport(app=app) if app is not None else None
    lat, errors = [], 0
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(transport=transport, base_url=base_url or "http://bench", timeout=60) as client:
        async def one(i, record=True):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(endpoint, json={"question": questions[i % len(questions)]})
                if record:
                    lat.append((time.perf_counter() - t0) * 1000)
                    errors += r.status_code != 200

        await asyncio.gather(*(one(i, record=False) for i in range(warmup)))  # thread-local pipelines warm up
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(requests)))
        wall = time.perf_counter() - t0

    return {
        "endpoint": endpoint, "concurrency": concurrency, "requests": requests, "errors": errors,
        "p50_ms": round(_pct(lat, 50), 2), "p95_ms": round(_pct(lat, 95), 2), "p99_ms": round(_pct(lat, 99), 2),
        "mean_ms": round(statistics.fmean(lat), 2), "rps": round(requests / wall, 1), "wall_s": round(wall, 3),
    }


def bench(**kwargs):
    return asyncio.run(run_bench(**kwargs))
