"""Offline-vs-online eval parity: the served endpoint must score every gold question exactly like run_eval does.

Catches serving-only drift: a different router version, a pipeline built with other settings, a request path
that mangles SQL, or a thread-local pipeline that diverges from the one the eval harness constructs.
"""

import asyncio

import dspy
import httpx
import pandas as pd

from sqlsmith.agents import RoutedText2SQL
from sqlsmith.db import build_db
from sqlsmith.evaluate import exec_match, run_eval
from sqlsmith.gold import GOLD
from sqlsmith.lm import make_lm
from sqlsmith.registry import make_router
from sqlsmith.serve import create_app


async def run_parity(service=None, base_url=None, router="sklearn", gold=GOLD, concurrency=8):
    """In-process: offline side reuses the service's router and LM. Remote: offline side builds `router` itself."""
    conn = build_db()
    if service is not None:
        lm, offline_router, app = service.lm, service.router, create_app(service)
    else:
        lm, (offline_router, _), app = make_lm("fake"), make_router(router), None
    with dspy.context(lm=lm):
        _, offline = run_eval(RoutedText2SQL(conn, offline_router), conn, gold)

    sem = asyncio.Semaphore(concurrency)
    transport = httpx.ASGITransport(app=app) if app is not None else None
    async with httpx.AsyncClient(transport=transport, base_url=base_url or "http://parity", timeout=60) as client:
        async def one(g):
            async with sem:
                r = await client.post("/v1/text2sql", json={"question": g.question})
                r.raise_for_status()
                body = r.json()
                return {"question": g.question, "online_sql": body["sql"], "online_path": body["path"],
                        "online_correct": exec_match(conn, body["sql"], g.sql)}

        online = pd.DataFrame(await asyncio.gather(*(one(g) for g in gold)))

    df = offline[["question", "difficulty", "correct", "path", "pred_sql"]].merge(online, on="question")
    df["match"] = (df.correct == df.online_correct) & (df.path == df.online_path)
    mism = df[~df.match]
    return {
        "n": len(df), "parity": float(df.match.mean()),
        "offline_accuracy": float(df.correct.mean()), "online_accuracy": float(df.online_correct.mean()),
        "mismatches": mism[["question", "correct", "online_correct", "path", "online_path"]].to_dict("records"),
    }


def parity(**kwargs):
    return asyncio.run(run_parity(**kwargs))
