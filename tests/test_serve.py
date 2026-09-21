"""Inference service: batched routing, threaded Text2SQL with repair, and the micro-batcher's coalescing."""

import asyncio
import math

from fastapi.testclient import TestClient

from sqlsmith.evaluate import exec_match
from sqlsmith.gold import GOLD
from sqlsmith.lm import FakeLM
from sqlsmith.serve import MicroBatcher, Service, create_app


def test_service_routes_batches_and_serves_text2sql(conn):
    service = Service(lm="fake", router="sklearn", workers=2, max_batch=8, max_wait_ms=2)
    with TestClient(create_app(service)) as c:
        assert c.get("/healthz").json()["ok"]
        assert c.get("/v1/model").json()["router"] == {"source": "trained", "backend": "sklearn"}

        easy, hard = "How many customers are there?", "Who are the top 3 customers by total spend? Show name and total spend."
        assert c.post("/v1/route", json={"questions": [easy, hard]}).json()["difficulty"] == ["easy", "complex"]
        assert c.post("/v1/route", json={"questions": []}).status_code == 422

        r = c.post("/v1/text2sql", json={"question": easy, "with_rows": True}).json()
        assert r["path"] == "single:easy" and r["executed"] and r["rows"] == [{"COUNT(*)": 60}]

        broken = next(g for g in GOLD if FakeLM.bucket(g.question) in FakeLM().break_first and g.difficulty == "hard")
        r = c.post("/v1/text2sql", json={"question": broken.question}).json()
        assert r["path"] == "multi:complex" and r["attempts"] == 2 and "no_such_col" in r["errors"][0]
        assert r["executed"] and r["rows"] is None and exec_match(conn, r["sql"], broken.sql)

        # concurrent traffic lands on one shared batcher; every request is counted exactly once
        before = c.get("/v1/model").json()["batcher"]["items"]
        for g in GOLD[:10]:
            assert c.post("/v1/text2sql", json={"question": g.question}).status_code == 200
        assert c.get("/v1/model").json()["batcher"]["items"] == before + 10


def test_micro_batcher_coalesces_concurrent_submits_and_propagates_errors():
    calls = []

    def fn(xs):
        calls.append(len(xs))
        if "boom" in xs:
            raise ValueError("boom")
        return [x * 2 for x in xs]

    async def scenario():
        b = MicroBatcher(fn, max_batch=8, max_wait_ms=20)
        outs = await asyncio.gather(*(b.submit(i) for i in range(20)))
        assert outs == [i * 2 for i in range(20)]
        assert len(calls) <= math.ceil(20 / 8) + 1 and max(calls) == 8  # coalesced, not 20 single calls
        results = await asyncio.gather(b.submit("boom"), b.submit(1), return_exceptions=True)
        assert all(isinstance(r, ValueError) for r in results)  # whole batch fails together
        assert await b.submit(3) == 6  # loop survives a failed batch
        await b.stop()

    asyncio.run(scenario())
