"""FastAPI inference service: micro-batched router inference, Text2SQL fan-out on a bounded thread pool.

Every /v1/text2sql request is routed as part of a batch (one vectorised router call per flush), then the
chosen DSPy pipeline runs on a worker thread that owns its own SQLite connection and pipeline objects.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

import dspy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sqlsmith.agents import Text2SQL
from sqlsmith.db import build_db, run_sql
from sqlsmith.lm import DEFAULT_MODEL, make_lm
from sqlsmith.registry import make_router


class MicroBatcher:
    """Coalesces concurrent submit() calls into one fn(list): flushes at max_batch items or after max_wait_ms."""

    def __init__(self, fn, max_batch=32, max_wait_ms=5):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000
        self.queue = self.task = None
        self.batches = self.items = 0

    def _ensure_started(self):
        if self.task is None or self.task.done():
            self.queue = asyncio.Queue()
            self.task = asyncio.get_running_loop().create_task(self._loop())

    async def submit(self, item):
        self._ensure_started()
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    async def stop(self):
        if self.task:
            self.task.cancel()

    async def _loop(self):
        loop = asyncio.get_running_loop()
        while True:
            pending = [await self.queue.get()]
            deadline = time.monotonic() + self.max_wait
            while len(pending) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            self.batches, self.items = self.batches + 1, self.items + len(pending)
            try:
                outs = await loop.run_in_executor(None, self.fn, [i for i, _ in pending])
                for (_, fut), out in zip(pending, outs):
                    if not fut.done():
                        fut.set_result(out)
            except Exception as e:  # noqa: BLE001 - propagate to every waiter in the batch
                for _, fut in pending:
                    if not fut.done():
                        fut.set_exception(e)


class Service:
    def __init__(self, lm="fake", model=DEFAULT_MODEL, router="sklearn", workers=4, max_batch=32, max_wait_ms=5, tracking_uri="file:./mlruns"):
        self.lm = make_lm(lm, model)
        self.router, self.router_info = make_router(router, tracking_uri)
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="text2sql")
        self.local = threading.local()
        self.batcher = MicroBatcher(self.router.predict_many, max_batch, max_wait_ms)
        self.started = time.time()

    def _worker(self):
        """Per-thread SQLite connection and pipelines (sqlite connections are not shareable across threads)."""
        if not hasattr(self.local, "conn"):
            self.local.conn = build_db()
            self.local.multi = Text2SQL(self.local.conn)
            self.local.fast = Text2SQL(self.local.conn, single=True)
        return self.local

    def run_pipeline(self, question, difficulty, with_rows):
        w = self._worker()
        t0 = time.perf_counter()
        with dspy.context(lm=self.lm):
            pred = (w.fast if difficulty == "easy" else w.multi)(question=question)
        t = pred.trace
        rows = run_sql(w.conn, pred.sql).to_dict("records") if with_rows and t.ok else None
        return {
            "sql": pred.sql, "path": f"{t.path}:{difficulty}", "tables": t.tables, "attempts": t.attempts,
            "executed": t.ok, "errors": t.errors, "rows": rows, "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    async def route(self, questions):
        return await asyncio.gather(*(self.batcher.submit(q) for q in questions))

    async def text2sql(self, question, with_rows=False):
        difficulty = await self.batcher.submit(question)
        return await asyncio.get_running_loop().run_in_executor(self.pool, self.run_pipeline, question, difficulty, with_rows)


class RouteRequest(BaseModel):
    questions: List[str] = Field(min_length=1, max_length=256)


class Text2SQLRequest(BaseModel):
    question: str = Field(min_length=1)
    with_rows: bool = False


class Text2SQLResponse(BaseModel):
    sql: str
    path: str
    tables: List[str]
    attempts: int
    executed: bool
    errors: List[str]
    rows: Optional[List[dict]] = None
    latency_ms: float


def create_app(service=None, **kwargs):
    service = service or Service(**kwargs)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await service.batcher.stop()
        service.pool.shutdown(wait=False)

    app = FastAPI(title="sqlsmith", lifespan=lifespan)
    app.state.service = service

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "uptime_s": round(time.time() - service.started, 1)}

    @app.get("/v1/model")
    async def model():
        return {"router": service.router_info, "lm": service.lm.model,
                "batcher": {"max_batch": service.batcher.max_batch, "max_wait_ms": service.batcher.max_wait * 1000,
                            "batches": service.batcher.batches, "items": service.batcher.items}}

    @app.post("/v1/route")
    async def route(req: RouteRequest):
        return {"difficulty": await service.route(req.questions)}

    @app.post("/v1/text2sql", response_model=Text2SQLResponse)
    async def text2sql(req: Text2SQLRequest):
        try:
            return await service.text2sql(req.question, req.with_rows)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"{type(e).__name__}: {e}")

    return app
