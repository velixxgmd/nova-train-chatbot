#!/usr/bin/env python
"""Prepare a real-text corpus for Nova Assistant (Stage B).

Downloads the official TinyStories validation text from Hugging Face (the only
file small enough for this laptop at ~19 MB) plus a bounded slice of the
training file streamed to a fixed output budget, all with provenance and
fingerprints recorded in a manifest.

License: CDLA-Sharing-1.0 (verified on the HF dataset card). Redistribution of
the data itself requires keeping those terms; training a model on it does not.
Source: https://huggingface.co/datasets/roneneldan/TinyStories

Usage:
  python scripts/prepare_data.py --train-chars 20000000 --dir data/tinystories_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINY_STORIES = {
    "name": "TinyStories",
    "version": "main (V1)",
    "authors": "Ronen Eldan, Yuanzhi Li (Microsoft)",
    "paper": "arXiv:2305.07759 — \"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?\"",
    "license": "CDLA-Sharing-1.0",
    "source_urls": {
        "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        "card": "https://huggingface.co/datasets/roneneldan/TinyStories",
    },
}

_STORY_SEP = "<|endoftext|>"
_ws = re.compile(r"[ \t]+")
_blanks = re.compile(r"\n{3,}")


def conservative_normalize(text: str) -> str:
    """Normalize spacing/blank lines without touching capitalization or punctuation."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _ws.sub(" ", t)
    t = "\n\n".join(p.strip() for p in t.split("\n\n") if p.strip())
    return t + ("\n" if not t.endswith("\n") else "")


def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_download(url: str, dest: str, max_bytes: int | None = None,
                    resume: bool = True) -> tuple[int, str]:
    """Streaming download with Range-resume support. Returns (bytes, final_hash_of_written)."""
    Partial = type("P", (), {})()  # marker only
    tmp = dest + ".part"
    offset = 0
    if resume and os.path.exists(tmp):
        offset = os.path.getsize(tmp)
    headers = {"User-Agent": "nova-assistant-corpus-prep/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    written = offset
    with urllib.request.urlopen(req, timeout=60) as resp:
        total_hdr = resp.headers.get("Content-Length")
        mode = "ab" if offset else "wb"
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                if max_bytes and written + len(chunk) > max_bytes:
                    chunk = chunk[: max_bytes - written]
                    f.write(chunk)
                    written += len(chunk)
                    break
                f.write(chunk)
                written += len(chunk)
    if max_bytes and written < max_bytes:
        pass  # file simply ended before the budget
    os.replace(tmp, dest)
    return written, sha256_path(dest)


def iter_stories(path: str):
    """Yield (story_text, is_hard_eot_sep) stories from a TinyStories .txt file."""
    buf = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == _STORY_SEP:
                if buf:
                    yield "".join(buf), True
                    buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(PROJECT, "data", "tinystories_v1"))
    ap.add_argument("--train-chars", type=int, default=20_000_000,
                    help="char budget for the training text (20 MB ≈ 1/96 of full train file)")
    ap.add_argument("--val-text", type=int, default=None,
                    help="char budget for validation (default: the whole valid file)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw = os.path.join(args.out_dir, "raw")
    os.makedirs(raw, exist_ok=True)

    t0 = time.time()
    files = {
        "train": os.path.join(raw, "TinyStories-train.head.txt"),
        "valid": os.path.join(raw, "TinyStories-valid.txt"),
    }
    stats = {}
    for split, url in [("train", TINY_STORIES["source_urls"]["train"]),
                       ("valid", TINY_STORIES["source_urls"]["valid"])]:
        dest = files[split]
        budget = args.train_chars if split == "train" else args.val_text
        if os.path.exists(dest) and not args.force:
            stats[split] = {"cached": True, "bytes": os.path.getsize(dest)}
            print(f"using cached {split}: {dest} ({stats[split]['bytes']:,} bytes)")
        else:
            n, _ = stream_download(url, dest, max_bytes=budget)
            stats[split] = {"cached": False, "bytes": n}
            print(f"downloaded {split}: {n:,} bytes in {time.time()-t0:.1f}s")

    # ---- process: validate, normalize, dedup ------------------------------
    seen = set()
    records = {"train": [], "valid": []}
    rejected = {"train": 0, "valid": 0}
    for split in ("train", "valid"):
        path = files[split]
        local_seen = set()
        for story, _eot in iter_stories(path):
            text = conservative_normalize(story)
            if len(text) < 40 or not text.strip():
                rejected[split] += 1
                continue
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key in seen or key in local_seen:
                rejected[split] += 1    # counts as duplicate
                continue
            local_seen.add(key)
            seen.add(key)
            records[split].append(text)
        # leakage prevention: a story may not appear in both splits after all processing
        if split == "valid":
            dupes_removed = 0
            cleaned = []
            seen_train = set(hashlib.sha256(x.encode('utf-8')).hexdigest() for x in records["train"])
            for text in records[split]:
                k = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if k in seen_train:
                    dupes_removed += 1
                    continue
                cleaned.append(text)
            records[split] = cleaned
            rejected[split] += dupes_removed

    out_train = os.path.join(args.out_dir, "corpus_train.jsonl")
    out_val = os.path.join(args.out_dir, "corpus_val.jsonl")
    for path, data in ((out_train, records["train"]), (out_val, records["valid"])):
        with open(path, "w", encoding="utf-8") as f:
            for text in data:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": TINY_STORIES,
        "created": datetime.now().isoformat(timespec="seconds"),
        "raw_files": {k: {"path": os.path.relpath(files[k], PROJECT), "sha256": sha256_path(files[k]),
                          "bytes": os.path.getsize(files[k])} for k in files},
        "download_stats": stats,
        "preprocessing": {"normalize": "collapse runs of spaces + blank lines; preserve case/punctuation",
                          "dedup": "exact-duplicate by sha256 of normalized text",
                          "leakage_control": "story-level: val stories whose hash appears in train are removed; paragraph overlaps impossible because stories are atomic units"},
        "counts": {s: {"records": len(records[s]), "rejected_or_dupes": rejected[s],
                       "chars": sum(len(x) for x in records[s])} for s in records},
        "outputs": {"train": os.path.relpath(out_train, PROJECT),
                     "val": os.path.relpath(out_val, PROJECT)},
    }
    with open(os.path.join(args.out_dir, "corpus_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    # a corpus fingerprint = hash of both outputs' sha256 concatenated
    fp = hashlib.sha256((sha256_path(out_train) + sha256_path(out_val)).encode()).hexdigest()
    manifest["corpus_fingerprint"] = fp
    with open(os.path.join(args.out_dir, "corpus_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in manifest.items() if k != "dataset"}, indent=2)[:1800])
    print(f"corpus_fingerprint: {fp}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
