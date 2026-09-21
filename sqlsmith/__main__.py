"""CLI: python -m sqlsmith {ask,eval,optimize,train-router,publish-router,serve,bench,parity}"""

import argparse
import json

import dspy

from sqlsmith.agents import RoutedText2SQL, Text2SQL
from sqlsmith.db import build_db, run_sql
from sqlsmith.evaluate import tracked_eval
from sqlsmith.lm import make_lm
from sqlsmith.optimize import optimize
from sqlsmith.registry import make_router, publish_router
from sqlsmith.router import cross_val_accuracy


def main(argv=None):
    p = argparse.ArgumentParser(prog="sqlsmith")
    p.add_argument("--lm", default="fake", help="fake | claude")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--router", default="sklearn", help="sklearn | torch | registry[:alias] | none")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("eval")
    sub.add_parser("optimize")
    sub.add_parser("train-router")
    pub = sub.add_parser("publish-router")
    pub.add_argument("--backend", default="sklearn")
    pub.add_argument("--alias", default="champion")
    pub.add_argument("--min-cv-acc", type=float, default=0.8)
    srv = sub.add_parser("serve")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--workers", type=int, default=4)
    srv.add_argument("--max-batch", type=int, default=32)
    srv.add_argument("--max-wait-ms", type=float, default=5)
    ben = sub.add_parser("bench")
    ben.add_argument("--url", help="benchmark a running server; default runs the app in-process")
    ben.add_argument("--concurrency", type=int, default=16)
    ben.add_argument("--requests", type=int, default=200)
    par = sub.add_parser("parity")
    par.add_argument("--url", help="check a running server; default runs the app in-process")
    args = p.parse_args(argv)

    if args.cmd == "train-router":
        for b in ("sklearn", "torch"):
            print(f"{b:8s} 5-fold accuracy: {cross_val_accuracy(b):.3f}")
        return
    if args.cmd == "publish-router":
        print(json.dumps(publish_router(args.backend, args.alias, args.min_cv_acc), indent=2))
        return
    if args.cmd in ("serve", "bench", "parity"):
        return _serving(args)

    conn = build_db()
    dspy.configure(lm=make_lm(args.lm, args.model))
    program = Text2SQL(conn) if args.router == "none" else RoutedText2SQL(conn, make_router(args.router)[0])

    if args.cmd == "ask":
        pred = program(question=args.question)
        t = pred.trace
        print(f"-- path={t.path} tables={t.tables} attempts={t.attempts} executed={t.ok}")
        print(pred.sql)
        if t.ok:
            print(run_sql(conn, pred.sql).to_string(index=False))
    elif args.cmd == "eval":
        metrics, df = tracked_eval(program, conn, params={"lm": args.lm, "model": args.model, "router": args.router})
        print(df[["difficulty", "correct", "attempts", "path"]].to_string())
        print(json.dumps(metrics, indent=2))
    elif args.cmd == "optimize":
        print(json.dumps(optimize(conn), indent=2))


def _serving(args):
    from sqlsmith.serve import Service, create_app

    if args.cmd == "serve":
        import uvicorn

        app = create_app(lm=args.lm, model=args.model, router=args.router, workers=args.workers,
                         max_batch=args.max_batch, max_wait_ms=args.max_wait_ms)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.cmd == "bench":
        from sqlsmith.bench import bench

        app = None if args.url else create_app(lm=args.lm, model=args.model, router=args.router)
        print(json.dumps(bench(app=app, base_url=args.url, concurrency=args.concurrency, requests=args.requests), indent=2))
    elif args.cmd == "parity":
        from sqlsmith.parity import parity

        service = None if args.url else Service(lm=args.lm, model=args.model, router=args.router)
        print(json.dumps(parity(service=service, base_url=args.url, router=args.router), indent=2))


if __name__ == "__main__":
    main()
