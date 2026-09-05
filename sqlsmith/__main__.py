"""CLI: python -m sqlsmith {ask,eval,optimize,train-router}"""

import argparse
import json

import dspy

from sqlsmith.agents import RoutedText2SQL, Text2SQL
from sqlsmith.db import build_db, run_sql
from sqlsmith.evaluate import tracked_eval
from sqlsmith.lm import make_lm
from sqlsmith.optimize import optimize
from sqlsmith.router import cross_val_accuracy, train_router


def main(argv=None):
    p = argparse.ArgumentParser(prog="sqlsmith")
    p.add_argument("--lm", default="fake", help="fake | claude")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--router", default="sklearn", help="sklearn | torch | none")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("eval")
    sub.add_parser("optimize")
    sub.add_parser("train-router")
    args = p.parse_args(argv)

    if args.cmd == "train-router":
        for b in ("sklearn", "torch"):
            print(f"{b:8s} 5-fold accuracy: {cross_val_accuracy(b):.3f}")
        return

    conn = build_db()
    dspy.configure(lm=make_lm(args.lm, args.model))
    program = Text2SQL(conn) if args.router == "none" else RoutedText2SQL(conn, train_router(args.router))

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


if __name__ == "__main__":
    main()
