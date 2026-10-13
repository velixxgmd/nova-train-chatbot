#!/usr/bin/env python
"""Masked conversational fine-tuning of the Stage B baseline on ABC English data.

A NEW trainer lives here because the existing engine
(nova_assistant/src/nova/training/engine.py) trains unmasked random raw-text
windows and cannot apply assistant-only loss masking. Everything compatible is
reused unchanged from the `nova` package: NovaLM + baseline weights, the frozen
baseline tokenizer, chat_template.build_training_example (loss masks), and the
checkpoint format expected by nova.inference.generator.Generator.

Modes:
  python train.py --config configs/abc_ft_v1.json      # completely NEW run
  python train.py --resume runs/abc_ft_v1_2026...      # CONTINUE an existing run
                                                        # exactly where it stopped
                                                        # (mid-epoch if it stopped
                                                        # mid-epoch)
  python train.py --resume runs/... --restart-epoch    # redo the interrupted
                                                        # epoch from its start
  python train.py --smoke                              # small smoke test

The baseline run directory is opened READ-ONLY and never written to.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from abc_data import (  # noqa: E402
    PKG_ROOT, DEFAULT_SYSTEM, batch_windows, build_stream, load_baseline,
    load_config, load_json, now_iso, sample_windows, save_json, sha256_bytes,
    sha256_file, chat_conversation_to_chunks, text_to_chunks,
)
from nova.models.model import count_parameters  # noqa: E402
from nova.tokenization.tokenizer import NovaTokenizer  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
def peak_rss_mb():
    """Peak resident-set size in MB, or None if it cannot be measured.

    Returns None rather than 0 on failure so a broken probe is never
    mistaken for "used no memory".
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                                   ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            hproc = kernel32.GetCurrentProcess()
            c = PMC(); c.cb = ctypes.sizeof(PMC)
            ok = psapi.GetProcessMemoryInfo(hproc, ctypes.byref(c), c.cb)
            if not ok or c.PeakWorkingSetSize == 0:
                return None            # probe failed -> report unknown, not 0
            return round(c.PeakWorkingSetSize / (1 << 20), 1)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def fmt_dur(s):
    s = max(0, int(s)); h, rem = divmod(s, 3600); m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_streams(data_dir: str, tok, seq_len: int, smoke: int = 0):
    """Per-source (ids, labels) streams + counts. Sources stay separated so
    their contributions remain auditable."""
    parts = {"train_chat": os.path.join(data_dir, "train_chat.jsonl"),
             "train_text": os.path.join(data_dir, "train_text.jsonl"),
             "val_chat": os.path.join(data_dir, "val_chat.jsonl"),
             "val_text": os.path.join(data_dir, "val_text.jsonl")}
    for p in parts.values():
        if not os.path.exists(p):
            raise SystemExit(f"missing data partition: {p} (run prepare_data.py first)")

    stats = {s: {"chunks": 0, "tokens": 0} for s in
             ("oasst1", "dolly15k", "cosmopedia100k")}
    per_source = {s: [] for s in stats}          # train
    val_per_source = {s: [] for s in stats}
    dropped_overlong = 0

    for rec in read_jsonl(parts["train_chat"]):
        if smoke and len(per_source[rec["source"]]) >= smoke:
            continue
        chunks, dropped = chat_conversation_to_chunks(rec["turns"], tok, seq_len,
                                                      rec.get("system", DEFAULT_SYSTEM))
        dropped_overlong += dropped
        per_source[rec["source"]] += chunks
    for rec in read_jsonl(parts["train_text"]):
        if smoke and len(per_source["cosmopedia100k"]) >= smoke:
            continue
        per_source["cosmopedia100k"] += text_to_chunks(rec["text"], tok, seq_len)
    for rec in read_jsonl(parts["val_chat"]):
        chunks, dropped = chat_conversation_to_chunks(rec["turns"], tok, seq_len,
                                                      rec.get("system", DEFAULT_SYSTEM))
        val_per_source[rec["source"]] += chunks
    for rec in read_jsonl(parts["val_text"]):
        val_per_source["cosmopedia100k"] += text_to_chunks(rec["text"], tok, seq_len)

    streams, val_streams = {}, {}
    for s in stats:
        streams[s] = build_stream(per_source[s]) if per_source[s] else None
        val_streams[s] = build_stream(val_per_source[s]) if val_per_source[s] else None
        stats[s] = {"train_chunks": len(per_source[s]),
                    "train_tokens": sum(len(c[0]) for c in per_source[s]),
                    "val_chunks": len(val_per_source[s]),
                    "val_tokens": sum(len(c[0]) for c in val_per_source[s])}
    return streams, val_streams, stats, dropped_overlong


# ---------------------------------------------------------------------------
class AbcTrainer:
    """Masked, source-aware fine-tuning loop with atomic checkpoints."""

    def __init__(self, run_dir, tok, model, tr_cfg, streams, val_streams,
                 manifest_fp, baseline_info, device, resume=None):
        self.run_dir, self.tok, self.cfg = run_dir, tok, tr_cfg
        self.model, self.streams, self.val_streams = model, streams, val_streams
        self.manifest_fp, self.baseline_info = manifest_fp, baseline_info
        self.device = device
        self.ckpts = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.ckpts, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        self.opt = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"],
                                     weight_decay=tr_cfg["weight_decay"],
                                     betas=(0.9, 0.95))
        self.sources = [s for s in ("oasst1", "dolly15k", "cosmopedia100k")
                        if streams.get(s) is not None]
        self.mix = [len(streams[s][0]) for s in self.sources]
        total = sum(self.mix)
        self.mix_p = [m / total for m in self.mix]
        self.tok_per_step = tr_cfg["batch_size"] * tr_cfg["seq_len"] * tr_cfg["accum_steps"]
        self.steps_per_epoch = int(tr_cfg.get("steps_per_epoch_override") or
                                   max(1, total // self.tok_per_step))
        self.global_step = 0
        self.epoch = 0
        self.best_val = None
        self.best_epoch = None
        self.history = []
        self.rng = random.Random(tr_cfg["seed"])
        self.amp = (tr_cfg.get("mixed_precision", "auto") == "auto" or
                    tr_cfg.get("mixed_precision") is True) and device == "cuda"
        self.amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._restart_epoch = bool(resume and resume.get("restart_epoch"))
        # fixed validation windows (same windows for every model compared later)
        self.val_windows = {}
        for i, s in enumerate(self.sources):
            vs = val_streams.get(s)
            if vs is not None and len(vs[0]) > tr_cfg["seq_len"] + 1:
                self.val_windows[s] = sample_windows(
                    vs[0], vs[1], tr_cfg["val_windows_per_source"],
                    tr_cfg["seq_len"], seed=tr_cfg["seed"] + 100 + i)
        if resume:
            self._restore(resume)

    # ------------------------------------------------------------ checkpoints
    def _state(self):
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "config": self.cfg, "progress": {
                    "epoch": self.epoch, "global_step": self.global_step,
                    "best_val": self.best_val, "best_epoch": self.best_epoch,
                    "history": self.history[-20:]},
                "manifest_fp": self.manifest_fp,
                "baseline": {"run_dir": self.baseline_info["run_dir"],
                             "checkpoint": self.baseline_info["checkpoint"],
                             "ckpt_sha256": self.baseline_info["checkpoint_pt_sha256"]},
                "rng": {"py": random.getstate(), "np": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "trainer": self.rng.getstate()}}

    def save(self, name, mark_best=False) -> bool:
        state = self._state()
        pt_tmp = os.path.join(self.ckpts, f"{name}.pt.tmp{os.getpid()}")
        js_tmp = os.path.join(self.ckpts, f"{name}.json.tmp{os.getpid()}")
        try:
            with open(pt_tmp, "wb") as f:
                torch.save(state, f); f.flush(); os.fsync(f.fileno())
            meta = {"name": name, "is_best": mark_best,
                    "progress": state["progress"],
                    "model_config": self.model.cfg.to_dict(),
                    "vocab_size": self.tok.vocab_size,
                    "tokenizer_fingerprint": NovaTokenizer.file_fingerprint(
                        os.path.join(self.run_dir, "tokenizer.json")),
                    "manifest_fp": self.manifest_fp,
                    "baseline": state["baseline"],
                    "device": self.device,
                    "created": now_iso()}
            with open(js_tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f); f.flush(); os.fsync(f.fileno())
            probe = torch.load(pt_tmp, map_location="cpu", weights_only=False)
            assert "model" in probe and "opt" in probe
            os.replace(pt_tmp, os.path.join(self.ckpts, f"{name}.pt"))
            os.replace(js_tmp, os.path.join(self.ckpts, f"{name}.json"))
            return True
        except Exception as e:
            for t in (pt_tmp, js_tmp):
                if os.path.exists(t):
                    try: os.remove(t)
                    except OSError: pass
            self.log(f"CHECKPOINT SAVE FAILED ({name}): {e}")
            return False

    def _restore(self, info):
        pt, mj = info["pt_path"], info["meta_path"]
        st = torch.load(pt, map_location=self.device, weights_only=False)
        meta = load_json(mj)
        if meta["model_config"] != self.model.cfg.to_dict():
            raise RuntimeError("resume refused: model config mismatch")
        if meta["vocab_size"] != self.tok.vocab_size:
            raise RuntimeError("resume refused: vocab mismatch")
        if meta.get("tokenizer_fingerprint") and meta["tokenizer_fingerprint"] != \
                NovaTokenizer.file_fingerprint(os.path.join(self.run_dir, "tokenizer.json")):
            raise RuntimeError("resume refused: tokenizer fingerprint mismatch")
        if meta.get("manifest_fp") and meta["manifest_fp"] != self.manifest_fp:
            raise RuntimeError("resume refused: dataset manifest fingerprint mismatch")
        # drift guard: the .pt weights are authoritative for progress. If the
        # sidecar .json disagrees we keep going (weights win) but say so loudly.
        jp = meta.get("progress") or {}
        if jp and jp.get("global_step") != st["progress"].get("global_step"):
            self.log(f"WARNING: checkpoint drift — {os.path.basename(mj)} says step "
                     f"{jp.get('global_step')} but {os.path.basename(pt)} holds step "
                     f"{st['progress'].get('global_step')}; trusting the .pt")
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        p = st["progress"]
        self.global_step = p["global_step"]
        self.epoch = self.global_step // self.steps_per_epoch + 1
        if getattr(self, "_restart_epoch", False) and self.global_step % self.steps_per_epoch:
            # operator asked to redo the interrupted epoch from its first step
            # index; weights/opt state are the mid-epoch ones we saved
            redo = self.global_step % self.steps_per_epoch
            self.global_step -= redo
            self.epoch = self.global_step // self.steps_per_epoch + 1
            self.log(f"--restart-epoch: rewinding {redo} steps to redo "
                     f"epoch {self.epoch} from its start (weights/opt kept)")
        self.best_val, self.best_epoch, self.history = p["best_val"], p["best_epoch"], p.get("history", [])
        r = st["rng"]
        random.setstate(r["py"]); np.random.set_state(r["np"])
        torch.set_rng_state(r["torch"].cpu())
        self.rng.setstate(r["trainer"])
        ep = self.global_step // self.steps_per_epoch + 1
        self.log(f"resumed into epoch {ep}, global step {self.global_step} "
                 f"(epoch step {self.global_step % self.steps_per_epoch}), "
                 f"best val {self.best_val}")

    # ---------------------------------------------------------------- helpers
    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with open(os.path.join(self.run_dir, "logs", "train_log.txt"), "a",
                  encoding="utf-8") as f:
            f.write(line + "\n")
        print(msg, flush=True)

    def _lr(self):
        s = self.global_step
        if s < self.cfg["warmup_steps"]:
            return self.cfg["lr"] * (s + 1) / self.cfg["warmup_steps"]
        span = self.steps_per_epoch * self.cfg["epochs"] - self.cfg["warmup_steps"]
        if span <= 0:
            return self.cfg["min_lr"]
        p = min(1.0, (s - self.cfg["warmup_steps"]) / span)
        return self.cfg["min_lr"] + (self.cfg["lr"] - self.cfg["min_lr"]) * 0.5 * \
            (1 + math.cos(math.pi * p))

    def _pick_source(self):
        r = self.rng.random()
        acc = 0.0
        for s, p in zip(self.sources, self.mix_p):
            acc += p
            if r <= acc:
                return s
        return self.sources[-1]

    def eval_val(self) -> float:
        """Weighted mean masked val loss over the fixed per-source windows."""
        self.model.eval()
        losses, weights = [], []
        with torch.no_grad():
            for s, wins in self.val_windows.items():
                tot = 0.0
                for x, y in batch_windows(wins, self.cfg["batch_size"]):
                    x, y = x.to(self.device), y.to(self.device)
                    _, loss = self.model(x, targets=y)
                    tot += float(loss) * len(x)
                losses.append((s, tot / len(wins)))
                weights.append(len(wins))
        self.model.train()
        wsum = sum(weights)
        return sum(l * w for (_, l), w in zip(losses, weights)) / wsum, losses

    # ------------------------------------------------------------------- loop
    def train(self):
        """Step-driven loop: state is fully defined by global_step, so a
        resumed run continues exactly where the interrupted epoch was left
        off (mid-epoch), while an epoch-boundary checkpoint starts the next
        epoch cleanly."""
        cfg = self.cfg
        self.model.train()
        t0 = time.time()
        target_steps = self.steps_per_epoch * cfg["epochs"]
        try:
            while self.global_step < target_steps:
                epoch_num = self.global_step // self.steps_per_epoch + 1
                step_in = self.global_step % self.steps_per_epoch
                self.epoch = epoch_num
                ep_loss, ep_n = 0.0, 0
                self.opt.zero_grad(set_to_none=True)
                if step_in > 0:
                    self.log(f"continuing epoch {epoch_num} from step "
                             f"{step_in}/{self.steps_per_epoch} "
                             f"(glob {self.global_step})")
                    # a resumed epoch only runs its remaining steps; scale the
                    # denominator so the logged mean covers the whole epoch
                    ep_n = step_in
                for step in range(step_in, self.steps_per_epoch):
                    lr = self._lr()
                    for g in self.opt.param_groups:
                        g["lr"] = lr
                    for a in range(cfg["accum_steps"]):
                        s = self._pick_source()
                        ids, labels = self.streams[s]
                        # cfg["batch_size"] windows per forward, so the
                        # configured batch actually reaches the model (grad
                        # accumulation multiplies it by accum_steps).
                        wins = sample_windows(ids, labels, cfg["batch_size"],
                                              cfg["seq_len"],
                                              seed=self.rng.randint(0, 2**31 - 1))
                        x = torch.stack([w[0] for w in wins]).to(self.device)
                        y = torch.stack([w[1] for w in wins]).to(self.device)
                        with torch.autocast(self.device, dtype=self.amp_dtype,
                                            enabled=self.amp):
                            _, loss = self.model(x, targets=y)
                        raw = float(loss.detach())
                        (loss / cfg["accum_steps"]).backward()
                    if not math.isfinite(raw):
                        raise RuntimeError(f"non-finite loss at step {self.global_step}")
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               cfg["grad_clip"])
                    if not math.isfinite(float(grad_norm)):
                        raise RuntimeError(f"non-finite gradients at step {self.global_step}")
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.global_step += 1
                    ep_loss += raw; ep_n += 1
                    if self.global_step % cfg["eval_every_steps"] == 0:
                        val, per_src = self.eval_val()
                        improved = self.best_val is None or val < self.best_val
                        if improved:
                            self.best_val, self.best_epoch = val, self.epoch
                        rec = {"ts": time.strftime("%H:%M:%S"), "epoch": self.epoch,
                               "global_step": self.global_step, "train_loss": raw,
                               "val_loss": val, "val_per_source": dict(per_src),
                               "lr": lr, "is_best": improved,
                               "elapsed_s": round(time.time() - t0, 1),
                               "peak_gpu_mb": (torch.cuda.max_memory_allocated() / (1 << 20)
                                               if self.device == "cuda" else None),
                               "peak_rss_mb": peak_rss_mb()}
                        self.history.append(rec)
                        with open(os.path.join(self.run_dir, "logs",
                                                "metrics.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                        self.log(f"step {self.global_step}: train {raw:.4f} | "
                                 f"val {val:.4f}" + (" ★" if improved else ""))
                        if improved:
                            self.save("best", mark_best=True)
                    if self.global_step % cfg["checkpoint_every_steps"] == 0:
                        self.save("latest")
                    if step % 10 == 0 or step == self.steps_per_epoch - 1:
                        done = time.time() - t0
                        rate = self.global_step / done if done else 0
                        rem = (cfg["epochs"] - self.epoch) * self.steps_per_epoch + \
                              (self.steps_per_epoch - step - 1)
                        print(f"\repoch {self.epoch}/{cfg['epochs']} step "
                              f"{step+1}/{self.steps_per_epoch} (glob {self.global_step}) "
                              f"loss {raw:.4f} lr {lr:.2e} {rate:.2f} steps/s "
                              f"ETA {fmt_dur(rem / rate if rate else 0)}", end="", flush=True)
                val, per_src = self.eval_val()
                improved = self.best_val is None or val < self.best_val
                if improved:
                    self.best_val, self.best_epoch = val, self.epoch
                rec = {"ts": time.strftime("%H:%M:%S"), "epoch_end": self.epoch,
                       "global_step": self.global_step,
                       "train_loss": ep_loss / max(1, ep_n), "val_loss": val,
                       "val_per_source": dict(per_src), "is_best": improved,
                       "elapsed_s": round(time.time() - t0, 1),
                       "peak_gpu_mb": (torch.cuda.max_memory_allocated() / (1 << 20)
                                       if self.device == "cuda" else None),
                       "peak_rss_mb": peak_rss_mb()}
                self.history.append(rec)
                with open(os.path.join(self.run_dir, "logs", "metrics.jsonl"), "a",
                          encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print()
                self.log(f"epoch {self.epoch} done: train {rec['train_loss']:.4f} "
                         f"val {val:.4f}" + (" ★ new best" if improved else ""))
                self.save("latest")
                if improved:
                    self.save("best", mark_best=True)
        except KeyboardInterrupt:
            self.log("interrupted — saving resumable checkpoint")
            self.save("latest")
        except torch.cuda.OutOfMemoryError:
            self.log("CUDA OUT OF MEMORY — saving checkpoint and stopping. "
                     "Do NOT rerun unchanged: lower training.batch_size or "
                     "training.seq_len in the config, then --resume.")
            self.save("latest")
            raise
        return self.best_val


# ---------------------------------------------------------------------------
def make_run_dir(runs_root, prefix):
    d = os.path.join(runs_root, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=False)
    return d


def write_run_config(run_dir, cfg, tok_info, data_stats, manifest_fp, device,
                     tr_cfg, smoke, data_dir=None):
    rc = {"created": now_iso(), "experiment": cfg["experiment"],
          "mode": "smoke" if smoke else "fresh",
          "config_file": os.path.relpath(CFG_PATH, PKG_ROOT),
          "resolved_training": tr_cfg,
          "baseline": tok_info, "data_stats": data_stats,
          "data_dir": os.path.relpath(data_dir, PKG_ROOT) if data_dir else None,
          "manifest_fp": manifest_fp, "device": device,
          "torch": torch.__version__,
          "cuda_available": torch.cuda.is_available(),
          "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    save_json(os.path.join(run_dir, "run_config.json"), rc)
    save_json(os.path.join(PKG_ROOT, "TRAINING_CONFIG.json"), {
        "experiment": cfg["experiment"], "written": now_iso(),
        "run_dir": os.path.relpath(run_dir, PKG_ROOT),
        "resolved_training": tr_cfg, "baseline": tok_info,
        "dataset_manifest_fingerprint": manifest_fp, "device": device,
        "note": "snapshot of the resolved settings for reproducibility"})
    return rc


def main():
    global CFG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PKG_ROOT, "configs", "abc_ft_v1.json"))
    ap.add_argument("--data-dir", default=None,
                    help="default: <out_dir>/processed from the config")
    ap.add_argument("--resume", default=None,
                    help="existing run dir: continue exactly where it left off "
                         "(mid-epoch if the checkpoint is mid-epoch)")
    ap.add_argument("--restart-epoch", action="store_true",
                    help="with --resume: redo the interrupted epoch from its "
                         "start instead of continuing mid-epoch")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny deterministic smoke run (never a real result)")
    ap.add_argument("--device", default=None, choices=[None, "auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    CFG_PATH = args.config
    cfg = load_config(args.config)
    tr_cfg = dict(cfg["training"])
    data_cfg = cfg["data"]
    smoke_cfg = cfg.get("smoke", {})
    smoke = 0
    if args.smoke:
        smoke = smoke_cfg.get("examples_per_source", 120)
        tr_cfg["epochs"] = 1
        tr_cfg["val_windows_per_source"] = smoke_cfg.get("val_windows_per_source", 8)
        tr_cfg["eval_every_steps"] = 2
        tr_cfg["checkpoint_every_steps"] = 100000      # end-of-run saves only
        tr_cfg["batch_size"] = 8
        tr_cfg["accum_steps"] = 1
        tr_cfg["steps_per_epoch_override"] = smoke_cfg.get("steps", 15)
    if args.epochs:
        tr_cfg["epochs"] = args.epochs
    if args.device:
        tr_cfg["device"] = args.device
    tr_cfg.setdefault("seed", cfg.get("seed", 42))   # seed lives at config top level
    device = resolve_device(tr_cfg.get("device", "auto"))
    set_seed(tr_cfg["seed"])

    if args.data_dir:
        data_dir = args.data_dir
    elif args.resume:
        # a resumed run MUST use the exact data dir it started on, otherwise the
        # dataset fingerprint check below would (correctly) refuse the resume
        rc_probe = load_json(os.path.join(os.path.abspath(args.resume), "run_config.json"))
        rd = rc_probe.get("data_dir")
        data_dir = os.path.join(PKG_ROOT, rd) if rd else os.path.join(
            PKG_ROOT, data_cfg["out_dir"], "processed")
    elif args.smoke:
        # smoke runs always use the small quick-prepared dataset
        data_dir = os.path.join(PKG_ROOT, data_cfg["out_dir"] + "_quick", "processed")
    else:
        data_dir = os.path.join(PKG_ROOT, data_cfg["out_dir"], "processed")
    manifest = None
    for cand in (os.path.join(os.path.dirname(data_dir), "DATA_MANIFEST.json"),
                 os.path.join(PKG_ROOT, "DATA_MANIFEST.json")):
        if os.path.exists(cand):
            manifest = load_json(cand)
            break
    manifest_fp = (manifest or {}).get("files", {})
    manifest_fp = sha256_bytes(json.dumps(manifest_fp, sort_keys=True).encode()) if manifest_fp else "no-manifest"

    if args.resume:
        run_dir = os.path.abspath(args.resume)
        if not os.path.isdir(run_dir):
            raise SystemExit(f"resume dir not found: {run_dir}")
        rc = load_json(os.path.join(run_dir, "run_config.json"))
        if rc.get("manifest_fp") and rc["manifest_fp"] != manifest_fp:
            raise SystemExit("resume refused: dataset changed since the run started")
        tr_cfg = rc["resolved_training"]
        if args.epochs:
            tr_cfg["epochs"] = args.epochs
        tok = NovaTokenizer.from_file(os.path.join(run_dir, "tokenizer.json"))
        tok_info = {"run_dir": rc["baseline"]["run_dir"],
                    "checkpoint": rc["baseline"]["checkpoint"],
                    "tokenizer_fingerprint": NovaTokenizer.file_fingerprint(
                        os.path.join(run_dir, "tokenizer.json")),
                    "vocab_size": tok.vocab_size,
                    "model_config": rc["baseline"].get("model_config")}
        pt = os.path.join(run_dir, "checkpoints", "latest.pt")
        mj = os.path.join(run_dir, "checkpoints", "latest.json")
        if not (os.path.exists(pt) and os.path.exists(mj)):
            raise SystemExit("resume refused: latest checkpoint missing/incomplete")
        streams, val_streams, dstats, dropped = build_streams(data_dir, tok, tr_cfg["seq_len"], smoke)
        model, _, binfo = load_baseline(rc["baseline"]["run_dir"],
                                        rc["baseline"].get("checkpoint", "best"), "cpu")
        trainer = AbcTrainer(run_dir, tok, model.to(device), tr_cfg, streams,
                             val_streams, manifest_fp, binfo, device,
                             resume={"pt_path": pt, "meta_path": mj,
                                     "restart_epoch": args.restart_epoch})
        trainer.log(f"resuming run {run_dir} on {device}")
        best = trainer.train()
        print(f"\nresumed run complete: {run_dir}\n  best val: {best}")
        return

    # -------- new run (default) -------------------------------------------
    # A completely new run always creates a fresh timestamped run directory;
    # the baseline and every previous run are left untouched.
    baseline_run = os.path.join(PKG_ROOT, cfg["baseline_run"])
    model, tok, binfo = load_baseline(baseline_run, cfg["baseline_checkpoint"], device)
    if next(model.parameters()).device.type != device:
        raise RuntimeError("model device does not match requested device — refusing")
    tr_cfg.setdefault("device", device)
    streams, val_streams, dstats, dropped = build_streams(data_dir, tok,
                                                          tr_cfg["seq_len"], smoke)
    if dropped:
        print(f"[warn] {dropped} single turns exceeded seq_len and were dropped")
    real_tokens = sum(v["train_tokens"] for v in dstats.values())
    if not args.smoke and real_tokens < 10_000:
        raise SystemExit(f"prepared data too small ({real_tokens} tokens) — "
                         f"run prepare_data.py fully first")

    prefix = "smoke_abc" if args.smoke else tr_cfg.get("run_name", "abc_ft")
    run_dir = make_run_dir(os.path.join(PKG_ROOT, "runs"), prefix)
    shutil.copy2(os.path.join(baseline_run, "tokenizer.json"),
                 os.path.join(run_dir, "tokenizer.json"))
    write_run_config(run_dir, cfg, binfo, dstats, manifest_fp, device, tr_cfg,
                     args.smoke, data_dir=data_dir)

    trainer = AbcTrainer(run_dir, tok, model, tr_cfg, streams, val_streams,
                         manifest_fp, binfo, device)
    trainer.log(f"baseline: {binfo['run_dir']} ({binfo['checkpoint']}, "
                f"sha {binfo['checkpoint_pt_sha256'][:12]}…, "
                f"{count_parameters(model):,} params)")
    trainer.log(f"device: {device} | amp: {trainer.amp} | steps/epoch: "
                f"{trainer.steps_per_epoch} | mix: "
                f"{dict(zip(trainer.sources, [round(p, 3) for p in trainer.mix_p]))}")
    trainer.log(f"data: {json.dumps(dstats)}")
    best = trainer.train()
    trainer.save("latest")
    print(f"\nrun complete: {run_dir}\n  best val: {best} @ epoch {trainer.best_epoch}")

    # ---- integrity probe: reload the saved best/late checkpoint fresh ------
    ck = "best" if os.path.exists(os.path.join(run_dir, "checkpoints", "best.pt")) else "latest"
    meta = load_json(os.path.join(run_dir, "checkpoints", f"{ck}.json"))
    fresh = torch.nn.Module()  # placeholder to avoid state aliasing
    from nova.models.model import NovaLM, NovaModelConfig
    fresh = NovaLM(NovaModelConfig.from_dict(meta["model_config"]))
    st = torch.load(os.path.join(run_dir, "checkpoints", f"{ck}.pt"),
                    map_location="cpu", weights_only=False)
    fresh.load_state_dict(st["model"])
    trainer.log(f"checkpoint reload verified ({ck}): {count_parameters(fresh):,} params, "
                f"val {trainer.best_val:.4f}")
    gen_txt = None
    try:
        from nova.inference.generator import Generator
        g = Generator.from_checkpoint(run_dir, checkpoint_name=ck, device=device)
        gen_txt = g.generate("<system> You are Nova, a helpful AI assistant. "
                             "<user> what is a mammal? <assistant>",
                             max_new_tokens=40, temperature=0.8, seed=42)
        trainer.log(f"generation probe: {gen_txt!r}")
    except Exception as e:
        trainer.log(f"generation probe FAILED: {e}")
    save_json(os.path.join(run_dir, "smoke_summary.json"), {
        "best_val": trainer.best_val, "steps": trainer.global_step,
        "elapsed_s": trainer.history[-1].get("elapsed_s") if trainer.history else None,
        "peak_rss_mb": peak_rss_mb(), "device": device, "amp": trainer.amp,
        "generation": gen_txt})


if __name__ == "__main__":
    main()
