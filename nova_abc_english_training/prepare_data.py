#!/usr/bin/env python
"""Download, audit, filter, deduplicate and sample the ABC English datasets.

Sources (official Hugging Face dataset repositories, revisions pinned in the
experiment config):

  A. OpenAssistant/oasst1           — conversation trees    (Apache-2.0)
  B. databricks/databricks-dolly-15k — instruction/response (CC BY-SA 3.0)
  C. HuggingFaceTB/cosmopedia-100k   — educational prose    (Apache-2.0)

Pipeline (deterministic, resumable, provenance-preserving):

  1. download pinned revisions through huggingface_hub (cached by HF)
  2. build per-source candidate pools with quality + English filters,
     recording how many examples each filter removed and why
  3. global exact/normalized dedup across ALL sources
  4. seeded train/val split per source + cross-split leakage removal
  5. deterministic source-aware sampling to the configured *token* targets
     (50% OASST1 / 30% Dolly / 20% Cosmopedia, measured with the BASELINE
     tokenizer — never row counts)
  6. write JSONL partitions + DATA_MANIFEST.json with hashes and stats

Usage:
  python prepare_data.py --config configs/abc_ft_v1.json
  python prepare_data.py --quick            # small fast pool for tests/smoke
  python prepare_data.py --force            # rebuild even if cached
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc_data import (  # noqa: E402
    PKG_ROOT, chat_conversation_to_chunks, conservative_normalize, is_english,
    load_config, load_json, norm_hash, now_iso, save_json, sha256_bytes,
    sha256_file, text_to_chunks, baseline_tokenizer,
)

DOWNLOADS: dict = {}   # filled at runtime for the manifest


class StageTimer:
    """Prints stage name + elapsed seconds as each pipeline stage finishes."""

    def __init__(self):
        self.t0 = time.time()
        self.marks = []

    def mark(self, label: str):
        now = time.time()
        elapsed = now - self.t0
        self.marks.append((label, elapsed))
        print(f"[timer] {label}: {elapsed:.1f}s (stage took "
              f"{elapsed - (self.marks[-2][1] if len(self.marks) > 1 else 0):.1f}s)",
              flush=True)


TIMER = StageTimer()


class ReasonCounters:
    def __init__(self):
        self.c = collections.Counter()

    def add(self, reason: str, n: int = 1):
        self.c[reason] += n

    def as_dict(self):
        return dict(self.c)


# ---------------------------------------------------------------------------
# 1. downloads (pinned, cached by the Hugging Face hub)
# ---------------------------------------------------------------------------
def ensure_downloaded(data_cfg: dict) -> dict:
    from huggingface_hub import hf_hub_download
    out = {}
    for src in ("oasst1", "dolly15k"):
        s = data_cfg[src]
        t0 = time.time()
        path = hf_hub_download(s["repo"], s["file"], revision=s["revision"],
                               repo_type="dataset")
        out[src] = {"path": path, "repo": s["repo"], "revision": s["revision"],
                    "file": s["file"], "license": s["license"],
                    "sha256": sha256_file(path), "bytes": os.path.getsize(path),
                    "download_s": round(time.time() - t0, 1)}
        print(f"[download] {src}: {s['file']} ({out[src]['bytes']:,} bytes, "
              f"{out[src]['download_s']}s)")
    s = data_cfg["cosmopedia100k"]
    files = []
    for f in s["files"]:
        t0 = time.time()
        path = hf_hub_download(s["repo"], f, revision=s["revision"],
                               repo_type="dataset")
        files.append({"path": path, "file": f, "sha256": sha256_file(path),
                      "bytes": os.path.getsize(path),
                      "download_s": round(time.time() - t0, 1)})
        print(f"[download] cosmopedia100k: {f} ({files[-1]['bytes']:,} bytes, "
              f"{files[-1]['download_s']}s)")
    out["cosmopedia100k"] = {"repo": s["repo"], "revision": s["revision"],
                             "license": s["license"], "files": files}
    return out


# ---------------------------------------------------------------------------
# 2a. OASST1 — message trees -> valid English conversation branches
# ---------------------------------------------------------------------------
def _oasst_message_ok(msg: dict, cfg: dict, rc: ReasonCounters):
    if msg.get("deleted"):
        rc.add("oasst1:deleted")
        return False
    if msg.get("lang") != "en":
        rc.add("oasst1:not_english_metadata")
        return False
    if msg.get("review_result") is False:
        rc.add("oasst1:review_rejected")
        return False
    if (msg.get("review_count") or 0) < cfg["min_review_count"]:
        rc.add("oasst1:insufficient_review")
        return False
    labels = msg.get("labels") or []
    if isinstance(labels, dict):     # jsonl export: {name: {value, count}, ...}
        labels = [{"name": k, **(v if isinstance(v, dict) else {"value": v})}
                  for k, v in labels.items()]
    for lab in labels:
        if lab.get("name") in ("spam", "lang_mismatch") and (lab.get("value") or 0) >= 0.5:
            rc.add(f"oasst1:label_{lab['name']}")
            return False
    det = msg.get("detoxify") or {}
    if isinstance(det, dict) and (det.get("toxicity") or 0) > cfg["max_toxicity"]:
        rc.add("oasst1:toxic")
        return False
    text = (msg.get("text") or "").strip()
    if len(text) < cfg["min_text_chars"]:
        rc.add("oasst1:too_short")
        return False
    if len(text) > cfg["max_text_chars"]:
        rc.add("oasst1:too_long")
        return False
    if is_english(text) is False:
        rc.add("oasst1:non_english")
        return False
    return True


def build_oasst_pool(path: str, cfg: dict, rc: ReasonCounters, pool_cap: int):
    """Reconstruct valid conversation branches from the message tree."""
    messages = {}
    children: dict = collections.defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            messages[m["message_id"]] = m
            children[m.get("parent_id")].append(m["message_id"])
    for kids in children.values():
        kids.sort(key=lambda mid: (messages[mid].get("created_date") or "",
                                   mid))       # deterministic order
    print(f"[oasst1] {len(messages):,} messages, "
          f"{len([m for m in messages.values() if not m.get('parent_id')]):,} trees")
    TIMER.mark("oasst1: load messages")

    # messages appear on many root->leaf paths; each passes the filter only once
    ok_cache: dict = {}

    def msg_ok(mid):
        if mid not in ok_cache:
            ok_cache[mid] = _oasst_message_ok(messages[mid], cfg, rc)
        return ok_cache[mid]

    convos = []
    trees = sorted((mid for mid, m in messages.items() if not m.get("parent_id")),
                   key=lambda mid: (messages[mid].get("created_date") or "", mid))
    rng = random.Random(1234)     # pool-level cap sampling (deterministic)

    def dfs(mid, path_ids):
        ok = msg_ok(mid)
        path_ids = path_ids + [mid]
        kids = children.get(mid, [])
        if ok and kids:
            for k in kids:
                yield from dfs(k, path_ids)
        elif ok:
            yield path_ids
        elif not kids:
            rc.add("oasst1:branch_dropped")

    for root in trees:
        kept_this_tree = 0
        for path_ids in dfs(root, []):
            msgs = [messages[mid] for mid in path_ids]
            # strict role alternation, prompter first, assistant last
            roles = [m["role"] for m in msgs]
            expect = ["prompter" if i % 2 == 0 else "assistant"
                      for i in range(len(roles))]
            if roles != expect or roles[-1] != "assistant":
                rc.add("oasst1:bad_role_order")
                continue
            n_turns = len(msgs) // 2
            if n_turns < 1:
                rc.add("oasst1:too_few_turns")
                continue
            msgs = msgs[: cfg["max_turns"] * 2]
            turns = [(msgs[i]["text"].strip(), msgs[i + 1]["text"].strip())
                     for i in range(0, len(msgs) - 1, 2)]
            convos.append({
                "kind": "chat",
                "source": "oasst1",
                "turns": turns,
                "meta": {"tree_id": msgs[0]["message_tree_id"],
                         "path": ">/".join(path_ids)[:200],
                         "n_turns": len(turns),
                         "lang": "en"},
            })
            rc.add("oasst1:kept")
            kept_this_tree += 1
            if kept_this_tree >= cfg["max_conversations_per_tree"]:
                break
        # trees whose every branch died before a leaf are counted upstream
    if pool_cap and len(convos) > pool_cap:
        idx = list(range(len(convos)))
        rng.shuffle(idx)
        keep = set(idx[:pool_cap])
        convos = [convos[i] for i in sorted(keep)]
        rc.add("oasst1:pool_cap_trim", len(idx) - pool_cap)
    print(f"[oasst1] candidate conversations: {len(convos):,}")
    return convos


# ---------------------------------------------------------------------------
# 2b. Dolly 15k — instruction/context/response
# ---------------------------------------------------------------------------
def build_dolly_pool(path: str, cfg: dict, rc: ReasonCounters, pool_cap: int):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            instr = (d.get("instruction") or "").strip()
            resp = (d.get("response") or "").strip()
            ctx = (d.get("context") or "").strip()
            if len(instr) < cfg["min_instruction_chars"]:
                rc.add("dolly15k:instruction_too_short")
                continue
            if len(resp) < cfg["min_response_chars"]:
                rc.add("dolly15k:response_too_short")
                continue
            if len(resp) > cfg["max_response_chars"]:
                rc.add("dolly15k:response_too_long")
                continue
            if is_english(instr) is False or is_english(resp) is False:
                rc.add("dolly15k:non_english")
                continue
            user = (f"Use the following context to answer the instruction.\n\n"
                    f"Context: {ctx}\n\nInstruction: {instr}") if ctx else instr
            rows.append({
                "kind": "chat",
                "source": "dolly15k",
                "turns": [(user, resp)],
                "meta": {"category": d.get("category"),
                         "had_context": bool(ctx),
                         "instruction": instr[:200]},
            })
            rc.add("dolly15k:kept")
    if pool_cap and len(rows) > pool_cap:
        rng = random.Random(1234)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        rows = [rows[i] for i in sorted(idx[:pool_cap])]
        rc.add("dolly15k:pool_cap_trim")
    print(f"[dolly15k] candidate rows: {len(rows):,}")
    return rows


# ---------------------------------------------------------------------------
# 2c. Cosmopedia 100k — educational prose (pure LM objective)
# ---------------------------------------------------------------------------
def _repetitive(text: str, max_ratio: float) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 4:
        return (1 - len(set(lines)) / len(lines)) > max_ratio
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) >= 4:
        return (1 - len(set(paras)) / len(paras)) > max_ratio
    return False


def build_cosmo_pool(files: list, cfg: dict, rc: ReasonCounters, pool_cap: int,
                     scan_limit: int = 0):
    """Reservoir-sample parquet rows. `scan_limit` bounds rows examined
    (0 = scan all 100k); the limit never weakens any quality filter —
    every scanned row passes through the identical checks."""
    import pyarrow.parquet as pq
    rng = random.Random(1234)
    reservoir: list = []
    seen = 0
    t_read = t_filter = 0.0
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=512,
                                     columns=["text", "prompt", "seed_data",
                                              "format", "audience"]):
            t_read += 0
            cols = {name: batch.column(name).to_pylist()
                    for name in ("text", "prompt", "seed_data", "format", "audience")}
            for i in range(batch.num_rows):
                seen += 1
                t_f0 = time.time()
                text = (cols["text"][i] or "").strip()
                if len(text) < cfg["min_text_chars"]:
                    rc.add("cosmopedia100k:too_short"); t_filter += time.time() - t_f0; continue
                if len(text) > cfg["max_text_chars"]:
                    rc.add("cosmopedia100k:too_long"); t_filter += time.time() - t_f0; continue
                if _repetitive(text, cfg["max_repeated_line_ratio"]):
                    rc.add("cosmopedia100k:repetitive"); t_filter += time.time() - t_f0; continue
                if is_english(text) is False:
                    rc.add("cosmopedia100k:non_english"); t_filter += time.time() - t_f0; continue
                rc.add("cosmopedia100k:passed_filters")
                rec = {"kind": "text", "source": "cosmopedia100k",
                       "text": text,
                       "meta": {"format": cols["format"][i],
                                "audience": cols["audience"][i],
                                "seed_data": cols["seed_data"][i],
                                "prompt_head": (cols["prompt"][i] or "")[:160]}}
                t_filter += time.time() - t_f0
                if pool_cap and len(reservoir) < pool_cap:
                    reservoir.append(rec)
                elif pool_cap:
                    j = rng.randint(0, seen - 1)
                    if j < pool_cap:
                        reservoir[j] = rec
                else:
                    reservoir.append(rec)
            if scan_limit and seen >= scan_limit:
                break
        if scan_limit and seen >= scan_limit:
            break
        print(f"  [cosmopedia100k] {os.path.basename(fp)} done: {seen:,} rows scanned, "
              f"{len(reservoir):,} in pool", flush=True)
    print(f"[cosmopedia100k] scanned {seen:,} rows -> pool {len(reservoir):,} "
          f"(filter time {t_filter:.1f}s, ~{t_filter*1000/max(1,seen):.1f} ms/row)", flush=True)
    return reservoir


# ---------------------------------------------------------------------------
# 3-5. dedup, split, token sampling
# ---------------------------------------------------------------------------
def global_dedup(pools: dict, rc: ReasonCounters):
    seen = set()
    out = {}
    for src in ("oasst1", "dolly15k", "cosmopedia100k"):   # deterministic order
        keep = []
        for rec in pools[src]:
            key = (norm_hash(" ".join(u + " " + a for u, a in rec["turns"]))
                   if rec["kind"] == "chat" else norm_hash(rec["text"]))
            if key in seen:
                rc.add(f"{src}:duplicate")
                continue
            seen.add(key)
            rec["norm_hash"] = key
            keep.append(rec)
        out[src] = keep
    return out


def split_and_sample(pools: dict, tok, data_cfg: dict, seed: int,
                     rc: ReasonCounters, total_train_tokens: int):
    """Per-source seeded train/val split, leakage removal, token-target sample."""
    targets = {src: int(total_train_tokens * ratio)
               for src, ratio in data_cfg["mixture_targets_by_tokens"].items()}
    rng = random.Random(seed)
    val_frac = data_cfg["val_fraction_per_source"]
    max_ex = data_cfg["max_example_tokens"]

    def rec_tokens(rec):
        if rec["kind"] == "chat":
            # measure real masked-template tokens (worst-case single chunk)
            chunks, dropped = chat_conversation_to_chunks(rec["turns"], tok, max_ex)
            if dropped:
                rc.add(f"{rec['source']}:oversize_turn_dropped", dropped)
            rec["_chunks"] = chunks
        else:
            rec["_chunks"] = text_to_chunks(rec["text"], tok, max_ex)
        return sum(len(c[0]) for c in rec["_chunks"])

    result = {"train": {"chat": [], "text": []}, "val": {"chat": [], "text": []}}
    stats = {}
    val_hashes = set()
    for src in ("oasst1", "dolly15k", "cosmopedia100k"):
        pool = pools[src]
        order = list(range(len(pool)))
        rng.shuffle(order)
        n_val = max(1, int(len(pool) * val_frac))
        val_idx = set(order[:n_val])
        src_train, src_val = [], []
        for i, rec in enumerate(pool):
            (src_val if i in val_idx else src_train).append(rec)
        # leakage: no train record may share a normalized hash with any val record
        for rec in src_val:
            val_hashes.add(rec["norm_hash"])
        cleaned = []
        for rec in src_train:
            if rec["norm_hash"] in val_hashes:
                rc.add(f"{src}:train_val_leakage_removed")
                continue
            cleaned.append(rec)
        src_train = cleaned
        # val tokens
        val_toks = 0
        for rec in src_val:
            val_toks += rec_tokens(rec)
            result["val"][rec["kind"]].append(rec)
        # train: greedy fill to the source token target in shuffled order
        used, kept = 0, []
        for rec in src_train:
            t = rec_tokens(rec)
            if used + t > targets[src] * 1.10:      # 10% overshoot allowance
                continue
            kept.append(rec)
            used += t
        for rec in kept:
            result["train"][rec["kind"]].append(rec)
        stats[src] = {"pool": len(pool), "val_records": len(src_val),
                      "val_tokens": val_toks, "train_records": len(kept),
                      "train_tokens": used, "target_tokens": targets[src],
                      "achieved_ratio": round(used / max(1, total_train_tokens), 4)}
        print(f"[sample] {src}: train {len(kept):,} recs / {used:,} tokens "
              f"(target {targets[src]:,}) | val {len(src_val):,} recs / {val_toks:,} tokens")
    return result, stats


def _strip_private(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def write_partition(path: str, records: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_strip_private(rec), ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "configs", "abc_ft_v1.json"))
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("--quick", action="store_true",
                    help="tiny pools + token targets (tests/smoke only)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    data_cfg = dict(cfg["data"])
    out_dir = os.path.join(PKG_ROOT, data_cfg["out_dir"])
    proc_dir = os.path.join(out_dir, "processed")
    manifest_path = os.path.join(PKG_ROOT, "DATA_MANIFEST.json")

    total_tokens = data_cfg["total_train_tokens"]
    pool_caps = {"oasst1": 0, "dolly15k": 0, "cosmopedia100k": 0}   # 0 = unlimited
    scan_limit = 0          # 0 = scan every row of cosmopedia
    if args.quick:
        total_tokens = 60_000
        pool_caps = {"oasst1": 400, "dolly15k": 400, "cosmopedia100k": 300}
        # --quick also bounds how much of cosmopedia we EXAMINE (30x the pool
        # target), so the smoke pipeline runs in minutes, not hours. Every row
        # that is examined passes through the same quality filters.
        scan_limit = pool_caps["cosmopedia100k"] * 40
        out_dir = os.path.join(PKG_ROOT, data_cfg["out_dir"] + "_quick")
        proc_dir = os.path.join(out_dir, "processed")
        manifest_path = os.path.join(out_dir, "DATA_MANIFEST.json")

    if os.path.exists(manifest_path) and not args.force:
        man = load_json(manifest_path)
        if man.get("reused_ok"):
            print(f"[cache] processed data already exists ({manifest_path}); "
                  f"reuse or pass --force to rebuild.")
            return

    rc = ReasonCounters()
    downloads = ensure_downloaded(data_cfg)
    TIMER.mark("downloads (cached by HF hub)")

    tok, tok_info = baseline_tokenizer(cfg["baseline_run"], cfg["baseline_checkpoint"])
    print(f"[baseline] tokenizer vocab {tok.vocab_size}, fingerprint "
          f"{tok_info['tokenizer_fingerprint'][:16]}…")

    # pools (resumable per source via intermediate files keyed on inputs+filters)
    inter = os.path.join(out_dir, "intermediate")
    os.makedirs(inter, exist_ok=True)
    pools = {}
    for src in ("oasst1", "dolly15k", "cosmopedia100k"):
        sig = sha256_bytes(json.dumps(
            {"src": src, "revision": data_cfg[src]["revision"],
             "filters": {k: v for k, v in data_cfg[src].items()
                         if k not in ("repo", "revision", "license", "file", "files")},
             "pool_cap": pool_caps[src]}, sort_keys=True).encode())
        cache = os.path.join(inter, f"pool_{src}.jsonl")
        meta = os.path.join(inter, f"pool_{src}.sig.json")
        if os.path.exists(cache) and os.path.exists(meta) and \
                load_json(meta).get("sig") == sig and not args.force:
            pools[src] = [json.loads(l) for l in open(cache, encoding="utf-8") if l.strip()]
            print(f"[cache] {src}: reused pool ({len(pools[src]):,} records)")
        else:
            print(f"[build] {src}: scanning…", flush=True)
            t_src = time.time()
            if src == "oasst1":
                pools[src] = build_oasst_pool(downloads[src]["path"], data_cfg[src], rc, pool_caps[src])
            elif src == "dolly15k":
                pools[src] = build_dolly_pool(downloads[src]["path"], data_cfg[src], rc, pool_caps[src])
            else:
                pools[src] = build_cosmo_pool([f["path"] for f in downloads[src]["files"]],
                                              data_cfg[src], rc, pool_caps[src],
                                              scan_limit=scan_limit)
            # atomic cache write: temp file + rename, so an interrupt never
            # leaves a half-written pool that would be trusted next run
            tmp = cache + f".tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                for rec in pools[src]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, cache)
            save_json(meta, {"sig": sig, "created": now_iso(), "count": len(pools[src])})
            print(f"[build] {src}: {len(pools[src]):,} records in "
                  f"{time.time() - t_src:.1f}s", flush=True)
    TIMER.mark("pool building (all sources)")

    pools = global_dedup(pools, rc)
    TIMER.mark("global dedup")
    result, sample_stats = split_and_sample(pools, tok, data_cfg, cfg["seed"],
                                            rc, total_tokens)
    TIMER.mark("split + token-target sampling")

    # fixed held-out eval prompts must never overlap training data
    eval_cfg = load_json(os.path.join(PKG_ROOT, "configs", "eval_prompts.json"))
    train_hashes = {r["norm_hash"] for k in result["train"] for r in result["train"][k]}
    overlap = 0
    for p in eval_cfg["prompts"]:
        for turn in p["turns"]:
            if norm_hash(turn) in train_hashes:
                overlap += 1
    if overlap:
        raise RuntimeError(f"eval prompt overlaps training data ({overlap}) — aborting")

    # write partitions
    outs = {
        "train_chat": os.path.join(proc_dir, "train_chat.jsonl"),
        "train_text": os.path.join(proc_dir, "train_text.jsonl"),
        "val_chat": os.path.join(proc_dir, "val_chat.jsonl"),
        "val_text": os.path.join(proc_dir, "val_text.jsonl"),
    }
    write_partition(outs["train_chat"], result["train"]["chat"])
    write_partition(outs["train_text"], result["train"]["text"])
    write_partition(outs["val_chat"], result["val"]["chat"])
    write_partition(outs["val_text"], result["val"]["text"])
    TIMER.mark("write partitions")

    def part_stats(records, kind, key):
        toks = chars = 0
        for rec in records:
            if kind == "chat":
                chars += sum(len(u) + len(a) for u, a in rec["turns"])
            else:
                chars += len(rec["text"])
            toks += sum(len(c[0]) for c in rec["_chunks"])
        return {"records": len(records), "tokens": toks, "chars": chars,
                "avg_tokens_per_record": round(toks / len(records), 1) if records else 0,
                "disk_bytes": os.path.getsize(outs[key])}

    manifest = {
        "experiment": cfg["experiment"],
        "created": now_iso(),
        "seed": cfg["seed"],
        "command": " ".join(sys.argv),
        "datasets": {
            src: {k: v for k, v in downloads[src].items() if k != "path"}
            for src in downloads},
        "baseline_tokenizer": tok_info,
        "filter_config": {s: data_cfg[s] for s in
                          ("oasst1", "dolly15k", "cosmopedia100k")},
        "mixture_targets_by_tokens": data_cfg["mixture_targets_by_tokens"],
        "total_train_tokens_target": total_tokens,
        "removed_by_reason": rc.as_dict(),
        "sample_stats": sample_stats,
        "outputs": {
            "train_chat": part_stats(result["train"]["chat"], "chat", "train_chat"),
            "train_text": part_stats(result["train"]["text"], "text", "train_text"),
            "val_chat": part_stats(result["val"]["chat"], "chat", "val_chat"),
            "val_text": part_stats(result["val"]["text"], "text", "val_text"),
        },
        "realized_mixture_by_train_tokens": {},
        "eval_prompt_overlap_with_train": overlap,
        "files": {},
        "environment": _env_info(),
        "elapsed_s": round(time.time() - t0, 1),
        "reused_ok": True,
    }
    tr = manifest["outputs"]
    tot = tr["train_chat"]["tokens"] + tr["train_text"]["tokens"]
    manifest["realized_mixture_by_train_tokens"] = {
        "oasst1+dolly15k (chat)": round(tr["train_chat"]["tokens"] / max(1, tot), 4),
        "cosmopedia100k (text)": round(tr["train_text"]["tokens"] / max(1, tot), 4),
        "chat_split_note": "chat tokens are split per source in sample_stats",
    }
    for name, path in outs.items():
        manifest["files"][name] = {"path": os.path.relpath(path, PKG_ROOT),
                                   "sha256": sha256_file(path),
                                   "bytes": os.path.getsize(path)}
    save_json(manifest_path, manifest)
    print(f"\n[done] manifest -> {os.path.relpath(manifest_path, PKG_ROOT)} "
          f"in {manifest['elapsed_s']}s")
    print(json.dumps({"outputs": manifest["outputs"],
                      "removed_by_reason": manifest["removed_by_reason"]},
                     indent=2)[:2000])


def _env_info() -> dict:
    import platform
    import importlib.metadata as md
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ("torch", "tokenizers", "huggingface_hub", "pyarrow", "langid", "numpy"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


if __name__ == "__main__":
    main()
