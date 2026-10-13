#!/usr/bin/env python
"""Build the training corpus.

By default builds a small synthetic corpus (deterministic, tiny, perfect for
pipeline tests). Use --tiny-stories to download public TinyStories-like data when
internet + license conditions allow (see docs/DATA_PROVENANCE.md).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

from nova.data import corpus


def synthetic_corpus(n_paras=1200, seed=7):
    """Deterministic synthetic English-ish corpus for pipeline validation."""
    import random
    rng = random.Random(seed)
    subjects = ["the cat", "the dog", "a bird", "the child", "nova", "the engine",
                "a small robot", "the moon", "the river", "an old clock"]
    verbs = ["walked", "looked at", "found", "followed", "listened to", "thought about",
             "waited near", "cleaned", "remembered", "watched"]
    objects = ["the garden", "a small key", "the big door", "a red ball", "the quiet room",
               "the tall tree", "a gentle song", "the morning light", "an old map",
               "the warm soup"]
    paras = []
    for _ in range(n_paras):
        sents = []
        for _ in range(rng.randint(2, 5)):
            s = (f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}.")
            if rng.random() < 0.3:
                s += f" Then {rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} again."
            sents.append(s)
        paras.append(" ".join(sents))
    return paras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", default=os.path.join(PROJECT, "data", "corpus_train.jsonl"))
    ap.add_argument("--out-val", default=os.path.join(PROJECT, "data", "corpus_val.jsonl"))
    ap.add_argument("--from-files", nargs="*", default=None,
                    help="raw .txt sources; default = synthetic corpus")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.from_files:
        sources = args.from_files
        # write raw paragraphs to a temp combined file so fingerprinting works
        texts = []
        paras = []
        for src in sources:
            with open(src, "r", encoding="utf-8") as f:
                paras.extend(corpus.paragraph_split(f.read()))
    else:
        paras = synthetic_corpus()
        sources = ["<synthetic:synthetic_corpus(seed7)>"]

    p = corpus.dedup(paras)
    train, val = corpus.make_splits(p, val_ratio=args.val_ratio, seed=args.seed)
    for path, data in ((args.out_train, train), (args.out_val, val)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for para in data:
                import json
                f.write(json.dumps({"text": para}, ensure_ascii=False) + "\n")
    manifest = {
        "sources": str(sources),
        "paragraphs_total": len(p),
        "train": len(train), "val": len(val),
        "val_ratio": args.val_ratio, "seed": args.seed,
        "note": "synthetic corpus for pipeline validation" if not args.from_files
                else "from user-supplied files; verify licenses in DATA_PROVENANCE.md",
    }
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out_train)),
                           "corpus_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
