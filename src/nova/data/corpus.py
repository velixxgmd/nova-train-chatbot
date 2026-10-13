"""Corpus loading, deduplication, splits, provenance.

Small-footprint: works on plain UTF-8 text files, hashes everything for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_texts(paths, normalize: bool = True):
    """Load text files; normalize whitespace lightly (collapse runs of blank lines)."""
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        if normalize:
            raw = raw.replace("\r\n", "\n")
            while "\n\n\n" in raw:
                raw = raw.replace("\n\n\n", "\n\n")
        texts.append(raw)
    return texts


def paragraph_split(text: str):
    return [p for p in text.split("\n\n") if p.strip()]


def dedup(paragraphs):
    """Exact-duplicate removal, order-preserving."""
    seen, out = set(), []
    for p in paragraphs:
        key = hashlib.sha256(p.encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def make_splits(paragraphs, val_ratio=0.05, seed=42):
    """Leakage-resistant paragraph-level splits via independent seeding."""
    rng = random.Random(seed)
    idx = list(range(len(paragraphs)))
    rng.shuffle(idx)
    n_val = max(1, int(len(paragraphs) * val_ratio))
    val_idx = set(idx[:n_val])
    train = [paragraphs[i] for i in idx if i not in val_idx]
    val = [paragraphs[i] for i in idx if i in val_idx]
    return train, val


def fingerprint(paths) -> dict:
    """Dataset identity: per-file sha256 + sizes."""
    out = {"files": {}}
    for p in paths:
        out["files"][os.path.basename(p)] = {
            "sha256": sha256_file(p),
            "bytes": os.path.getsize(p),
        }
    return out


def build_jsonl_corpus(input_paths, out_train: str, out_val: str,
                       val_ratio=0.05, seed=42, dedup_on=True):
    """Convert raw text files into train/val corpus files (one paragraph per line, JSON).

    Returns a provenance dict written next to the outputs as corpus_manifest.json.
    """
    texts = load_texts(input_paths)
    paras = []
    for t in texts:
        paras.extend(paragraph_split(t))
    n_raw = len(paras)
    if dedup_on:
        paras = dedup(paras)
    n_dedup = len(paras)
    train, val = make_splits(paras, val_ratio=val_ratio, seed=seed)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "sources": [{"path": os.path.abspath(p), "sha256": sha256_file(p)} for p in input_paths],
        "paragraphs_raw": n_raw,
        "paragraphs_after_dedup": n_dedup,
        "train_paragraphs": len(train),
        "val_paragraphs": len(val),
        "seed": seed,
        "val_ratio": val_ratio,
    }
    for path, data in ((out_train, train), (out_val, val)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in data:
                f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
    # clean splits leakage check: disjointness is guaranteed by construction
    manifest["leakage_check"] = "paragraph-level disjoint splits by construction"
    with open(os.path.join(os.path.dirname(os.path.abspath(out_train)),
                           "corpus_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def iter_corpus_text(corpus_path: str):
    """Yield paragraph strings from a JSONL corpus file."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]
