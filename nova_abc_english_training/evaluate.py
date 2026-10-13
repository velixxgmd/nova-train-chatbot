#!/usr/bin/env python
"""Baseline vs fine-tuned evaluation for the ABC English experiment.

Quantitative: masked validation loss per source on FIXED seeded windows —
identical windows for both checkpoints, drawn from the held-out val splits
that never trained.

Qualitative: identical fixed prompts (configs/eval_prompts.json) through both
checkpoints with identical sampling settings, plus automated sanity flags
(English, repetition, distinct-2, end-of-turn stopping). Open-ended quality is
left to the documented manual rubric; no benchmark scores are invented.

Usage:
  python evaluate.py --finetuned runs/abc_ft_v1_2026...
  python evaluate.py --finetuned runs/... --limit 3 --data-dir <quick>/processed
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc_data import (  # noqa: E402
    PKG_ROOT, is_english, load_config, load_json, now_iso, sample_windows,
    save_json, sha256_file,
)
from nova import chat_template as ct  # noqa: E402
from nova.inference.generator import Generator  # noqa: E402
from nova.tokenization.tokenizer import NovaTokenizer  # noqa: E402
from train import build_streams  # noqa: E402

SOURCES = ("oasst1", "dolly15k", "cosmopedia100k")
MAX_NEW = 96


def _open_generator(run_dir: str, device: str):
    for name in ("best", "latest"):
        if os.path.exists(os.path.join(run_dir, "checkpoints", f"{name}.pt")):
            return Generator.from_checkpoint(run_dir, checkpoint_name=name,
                                             device=device), name
    raise SystemExit(f"no usable checkpoint in {run_dir}")


def distinct2(text: str) -> float:
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    bi = collections.Counter(zip(toks, toks[1:]))
    return len(bi) / (len(toks) - 1)


def max_ngram_repeat(text: str, n: int = 4) -> int:
    toks = text.split()
    if len(toks) < n:
        return 0
    grams = collections.Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return max(grams.values()) if grams else 0


def generate_tracked(gen: Generator, prompt: str, seed: int):
    """Generate and report whether an end-of-turn was hit before the limit."""
    pieces = []
    for piece in gen.stream(prompt, max_new_tokens=MAX_NEW, temperature=0.8,
                            top_k=50, seed=seed):
        pieces.append(piece)
    text = "".join(pieces)
    stopped_early = len(pieces) < MAX_NEW
    return text, stopped_early


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "configs", "abc_ft_v1.json"))
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (tests)")
    ap.add_argument("--out", default=os.path.join(PKG_ROOT, "reports", "EVAL_REPORT.md"))
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    tr_cfg = cfg["training"]
    device = ("cuda" if __import__("torch").cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    baseline_run = args.baseline or os.path.join(PKG_ROOT, cfg["baseline_run"])

    tok_b = NovaTokenizer.from_file(os.path.join(baseline_run, "tokenizer.json"))
    gen_b, ck_b = _open_generator(baseline_run, device)
    gen_f, ck_f = _open_generator(os.path.abspath(args.finetuned), device)
    fp_b = NovaTokenizer.file_fingerprint(os.path.join(baseline_run, "tokenizer.json"))
    fp_f = NovaTokenizer.file_fingerprint(os.path.join(args.finetuned, "tokenizer.json"))
    if fp_b != fp_f:
        raise SystemExit("tokenizer mismatch between baseline and fine-tuned run — "
                         "refusing to compare")

    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Inherit the fine-tuned run's data dir so we score the same val split it
        # trained on (a --smoke run uses the quick set, a real run the full set).
        rc_path = os.path.join(os.path.abspath(args.finetuned), "run_config.json")
        recorded = load_json(rc_path).get("data_dir") if os.path.exists(rc_path) else None
        data_dir = os.path.join(PKG_ROOT, recorded) if recorded else \
            os.path.join(PKG_ROOT, data_cfg["out_dir"], "processed")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"data dir not found: {data_dir} — run prepare_data.py "
                         f"(use --quick for the smoke set) or pass --data-dir")
    _v, val_streams, _s, _d = build_streams(data_dir, tok_b, tr_cfg["seq_len"], 0)

    # ---- quantitative: same fixed windows for both models ------------------
    per_source = {}
    weighted = {"baseline": 0.0, "finetuned": 0.0}
    wsum = 0
    for i, src in enumerate(SOURCES):
        vs = val_streams.get(src)
        if vs is None or len(vs[0]) <= tr_cfg["seq_len"] + 1:
            continue
        wins = sample_windows(vs[0], vs[1], tr_cfg["val_windows_per_source"],
                              tr_cfg["seq_len"], seed=cfg["seed"] + 100 + i)
        import torch
        res = {}
        for label, gen in (("baseline", gen_b), ("finetuned", gen_f)):
            gen.model.eval()
            tot = 0.0
            with torch.no_grad():
                for x, y in wins:
                    logits, loss = gen.model(x[None].to(device), targets=y[None].to(device))
                    tot += float(loss)
            res[label] = tot / len(wins)
        res["delta"] = res["finetuned"] - res["baseline"]
        res["windows"] = len(wins)
        res["tokens_scored"] = len(wins) * tr_cfg["seq_len"]
        per_source[src] = res
        weighted["baseline"] += res["baseline"] * len(wins)
        weighted["finetuned"] += res["finetuned"] * len(wins)
        wsum += len(wins)
    weighted = {k: v / wsum for k, v in weighted.items()} if wsum else weighted
    weighted["delta"] = weighted["finetuned"] - weighted["baseline"]

    # ---- qualitative: fixed prompts through both checkpoints ----------------
    eval_cfg = load_json(os.path.join(PKG_ROOT, "configs", "eval_prompts.json"))
    prompts = eval_cfg["prompts"]
    if args.limit:
        prompts = prompts[:args.limit]

    def render_prompt(turns):
        convo = ct.Conversation()
        it = iter(range(0, len(turns), 2))
        for i in it:
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            convo.add(turns[i], nxt)
        return ct.render(convo, tok_b, gen_b.model.cfg.max_seq_len - MAX_NEW)

    rows = []
    for p in prompts:
        prompt_text = tok_b.decode(render_prompt(p["turns"]))
        greedy_b = gen_b.generate(prompt_text, max_new_tokens=MAX_NEW,
                                  temperature=0.0, seed=0)
        greedy_f = gen_f.generate(prompt_text, max_new_tokens=MAX_NEW,
                                  temperature=0.0, seed=0)
        samp_b, stop_b = generate_tracked(gen_b, prompt_text, seed=42)
        samp_f, stop_f = generate_tracked(gen_f, prompt_text, seed=42)
        rows.append({
            "id": p["id"], "category": p["category"], "prompt": p["turns"],
            "baseline_greedy": greedy_b, "finetuned_greedy": greedy_f,
            "baseline_sampled": samp_b, "finetuned_sampled": samp_f,
            "flags": {
                "baseline": {"english": is_english(samp_b),
                             "distinct2": round(distinct2(samp_b), 3),
                             "max_4gram_repeat": max_ngram_repeat(samp_b),
                             "stopped_at_eot": stop_b},
                "finetuned": {"english": is_english(samp_f),
                              "distinct2": round(distinct2(samp_f), 3),
                              "max_4gram_repeat": max_ngram_repeat(samp_f),
                              "stopped_at_eot": stop_f},
            },
        })
        print(f"[eval] {p['id']}: base {len(samp_b)}ch / ft {len(samp_f)}ch")

    # ---- write report -------------------------------------------------------
    results = {
        "created": now_iso(), "device": device,
        "baseline": {"run": os.path.abspath(baseline_run), "checkpoint": ck_b},
        "finetuned": {"run": os.path.abspath(args.finetuned), "checkpoint": ck_f},
        "quant": {"per_source": per_source, "weighted": weighted,
                  "val_windows_per_source": tr_cfg["val_windows_per_source"],
                  "seq_len": tr_cfg["seq_len"]},
        "prompts": rows,
        "eval_set_version": eval_cfg["version"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    json_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                             "eval_results.json")
    save_json(json_path, results)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    md = []
    md.append("# ABC English — baseline vs fine-tuned evaluation\n")
    md.append(f"- generated: {results['created']} | device: {device} | "
              f"elapsed {results['elapsed_s']}s")
    md.append(f"- baseline: `{results['baseline']['run']}` ({ck_b})")
    md.append(f"- finetuned: `{results['finetuned']['run']}` ({ck_f})")
    md.append(f"- eval set: {eval_cfg['version']} "
              f"({len(prompts)} prompts, fixed, never trained on)\n")
    md.append("## Quantitative — masked validation loss (identical fixed windows)\n")
    md.append("| source | baseline | finetuned | delta | windows |")
    md.append("|---|---|---|---|---|")
    for src, r in per_source.items():
        md.append(f"| {src} | {fmt(r['baseline'])} | {fmt(r['finetuned'])} | "
                  f"{fmt(r['delta'])} | {r['windows']} |")
    if per_source:
        md.append(f"| **weighted** | {fmt(weighted['baseline'])} | "
                  f"{fmt(weighted['finetuned'])} | {fmt(weighted['delta'])} | {wsum} |")
    md.append("\n## Qualitative — fixed prompts (manual rubric: 0=bad, 1=partial, 2=good)\n")
    for r in rows:
        md.append(f"### {r['id']} ({r['category']})")
        for t in r["prompt"]:
            md.append(f"- **user**: {t}")
        md.append(f"- **baseline**: {r['baseline_sampled']!r}")
        md.append(f"- **finetuned**: {r['finetuned_sampled']!r}")
        md.append(f"- flags base: `{r['flags']['baseline']}` / "
                  f"ft: `{r['flags']['finetuned']}`")
        md.append("- manual rubric (base/ft): ___ / ___")
        md.append("")
    md.append("## Honesty notes\n")
    md.append("- Loss deltas are measured, not estimated; generation quality on a "
              "1.7M-parameter model is limited — scores of 2 mean 'usable for its "
              "scale', not 'human-like'.")
    md.append("- Greedy + one seeded sample per prompt; stochastic variation exists. "
              "Re-running with the same seeds reproduces this report exactly.")
    md.append("- No benchmark leaderboard numbers are claimed or implied.\n")
    md.append(f"Raw results: `{os.path.relpath(json_path, PKG_ROOT)}`")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[done] report -> {os.path.relpath(args.out, PKG_ROOT)}")
    print(json.dumps({"weighted": weighted, "per_source": per_source}, indent=2))


if __name__ == "__main__":
    main()
