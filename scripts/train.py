#!/usr/bin/env python
"""Training entrypoint.

  python scripts/train.py --config configs/tiny.yaml --corpus data/corpus_train.jsonl --mode fresh
  python scripts/train.py --resume runs/tiny_.../           # explicit resume
  python scripts/train.py --inference runs/tiny_.../
  python scripts/train.py --finetune runs/tiny_.../
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from nova.training import engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT, "configs", "tiny.yaml"))
    ap.add_argument("--mode", choices=["fresh", "resume", "inference", "finetune"], default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--inference", default=None)
    ap.add_argument("--finetune", default=None)
    ap.add_argument("--corpus-train", default=os.path.join(PROJECT, "data", "corpus_train.jsonl"))
    ap.add_argument("--corpus-val", default=os.path.join(PROJECT, "data", "corpus_val.jsonl"))
    ap.add_argument("--runs-root", default=os.path.join(PROJECT, "runs"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-name", default="tiny")
    ap.add_argument("--prompt", default=None, help="prompt for --inference")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    mode = args.mode or ("resume" if args.resume else
                         "inference" if args.inference else
                         "finetune" if args.finetune else "fresh")
    args.corpus_train = os.path.abspath(args.corpus_train)
    args.corpus_val = os.path.abspath(args.corpus_val)
    args.runs_root = os.path.abspath(args.runs_root)
    if not os.path.exists(args.corpus_train):
        raise SystemExit(f"corpus not found: {args.corpus_train} (run scripts/build_corpus.py first)")

    if mode == "fresh":
        rd, tr = engine.fresh_flow(args)
        print(f"\nrun complete: {rd}\n  best val: {tr.best_val} @ epoch {tr.best_epoch}")
    elif mode == "resume":
        args.resume = args.resume or ""
        rd, tr = engine.resume_flow(args)
        print(f"\nresumed run: {rd}")
    elif mode == "inference":
        args.inference = args.inference or ""
        rd = engine.inference_flow(args)
    elif mode == "finetune":
        args.finetune = args.finetune or ""
        rd, tr = engine.finetune_flow(args)
        print(f"\nfine-tune run: {rd}")


if __name__ == "__main__":
    main()
