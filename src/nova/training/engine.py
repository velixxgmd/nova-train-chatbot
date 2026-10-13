"""Training engine: fresh/resume/inference/finetune modes, fused checkpoints,
structured logs, live progress, graceful interruption.

Inherited discipline from the NumPy upgrade (src/nova_v4_upgraded.py):
  - fresh is the default; resume is ALWAYS explicit
  - fused checkpoint must reload before a save is reported as success
  - best-validation checkpoints are never overwritten by worse results
  - run directories isolate each experiment
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
from dataclasses import dataclass

import numpy as np
import torch

from nova.models.model import NovaLM, NovaModelConfig, count_parameters
from nova.tokenization.tokenizer import NovaTokenizer, SPECIAL_TOKENS
from nova.data.corpus import fingerprint, iter_corpus_text
from nova import chat_template as ct

# Windows console: force UTF-8 so logs with non-ASCII never crash training
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_interactive() -> bool:
    return sys.stdout.isatty() and "idlelib" not in sys.modules


def live_line(msg: str, final: bool = False):
    if is_interactive():
        try:
            import shutil as sh
            cols = sh.get_terminal_size(fallback=(100, 24)).columns
            width = max(min(cols - 1, 120), 40)
            sys.stdout.write(f"\r{msg[:width]:<{width}}")
            if final:
                sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            print(msg, flush=True)
    else:
        print(msg, flush=True)


def fmt_dur(s: float) -> str:
    s = max(0, int(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


@dataclass
class TrainConfig:
    run_name: str = "tiny"
    epochs: int = 1
    seq_len: int = 128
    batch_size: int = 8
    accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    steps_per_epoch: int = 100
    eval_every_steps: int = 100
    seed: int = 42
    mixed_precision: bool = False      # only honored when device is cuda
    display_every: int = 10
    step_log_every: int = 50
    max_lr_decay_epochs: int = 0       # 0 = cosine horizon = total steps

    model: dict = None                 # NovaModelConfig kwargs


class Trainer:
    def __init__(self, run_dir: str, tok: NovaTokenizer, corpus_train: str,
                 corpus_val: str, cfg: TrainConfig, resume_info: dict | None = None):
        self.run_dir = run_dir
        self.tok = tok
        self.corpus_train = corpus_train
        self.corpus_val = corpus_val
        self.cfg = cfg
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        self.ckpts = os.path.join(run_dir, "checkpoints")
        self.device = detect_device()
        self.log_lines = []

        model_kwargs = dict(cfg.model or {})
        model_kwargs["vocab_size"] = tok.vocab_size   # tokenizer vocab wins
        model_cfg = NovaModelConfig(**model_kwargs)
        self.model = NovaLM(model_cfg).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay,
                                     betas=(0.9, 0.95))
        self.total_steps_planned = cfg.epochs * cfg.steps_per_epoch
        self.global_step = 0
        self.epoch = 0
        self.best_val = None
        self.best_epoch = None
        self.history = []
        self.finished = False
        self.amp_enabled = cfg.mixed_precision and self.device == "cuda"
        self.amp_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._load_corpus_cache()
        if resume_info:
            self._restore(resume_info)

    # ------------------------------------------------------------ data cache
    def _load_corpus_cache(self):
        """Tokenize the whole (small) corpus into one long id tensor for speed."""
        self.train_ids = self._corpus_to_ids(self.corpus_train)
        self.val_ids = self._corpus_to_ids(self.corpus_val)
        assert len(self.train_ids) > self.cfg.seq_len + 1, "corpus too small for seq_len"

    def _corpus_to_ids(self, path):
        ids = []
        for para in iter_corpus_text(path):
            ids += self.tok.encode(para) + [self.tok.eos_id]
        return torch.tensor(ids, dtype=torch.long)

    # ---------------------------------------------------------- checkpointing
    def _ckpt_state(self):
        return {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "config": self.cfg.__dict__,
            "tokenizer_file": "tokenizer.json",
            "progress": {"epoch": self.epoch, "global_step": self.global_step,
                         "best_val": self.best_val, "best_epoch": self.best_epoch,
                         "history": self.history[-20:], "finished": self.finished},
            "torch_rng": torch.get_rng_state(),
            "py_rng": random.getstate(),
            "np_rng": np.random.get_state(),
        }

    def save(self, name: str, mark_best: bool = False) -> bool:
        state = self._ckpt_state()
        state["is_best"] = mark_best
        pt_tmp = os.path.join(self.ckpts, f"{name}.pt.tmp{os.getpid()}")
        json_tmp = os.path.join(self.ckpts, f"{name}.json.tmp{os.getpid()}")
        try:
            with open(pt_tmp, "wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            meta = {
                "name": name, "is_best": mark_best,
                "progress": state["progress"],
                "model_config": self.model.cfg.to_dict(),
                "vocab_size": self.tok.vocab_size,
                "tokenizer_fingerprint": NovaTokenizer.file_fingerprint(
                    os.path.join(self.run_dir, "tokenizer.json")),
                "corpus": {"train": os.path.basename(self.corpus_train),
                            "val": os.path.basename(self.corpus_val)},
            }
            with open(json_tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f)
                f.flush()
                os.fsync(f.fileno())
            # verify reload before swap
            probe = torch.load(pt_tmp, map_location="cpu", weights_only=False)
            if "model" not in probe or "opt" not in probe:
                raise RuntimeError("reloaded checkpoint lacks model/opt state")
            os.replace(pt_tmp, os.path.join(self.ckpts, f"{name}.pt"))
            os.replace(json_tmp, os.path.join(self.ckpts, f"{name}.json"))
            return True
        except Exception as e:
            for t in (pt_tmp, json_tmp):
                try:
                    if os.path.exists(t):
                        os.remove(t)
                except OSError:
                    pass
            self.log(f"CHECKPOINT SAVE FAILED ({name}): {e}")
            return False

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with open(os.path.join(self.run_dir, "logs", "train_log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(msg, flush=True)

    def epoch_metrics(self, rec):
        with open(os.path.join(self.run_dir, "logs", "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def step_metrics(self, rec):
        with open(os.path.join(self.run_dir, "logs", "step_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _restore(self, info: dict):
        st = torch.load(info["pt_path"], map_location=self.device, weights_only=False)
        compat = st["config"]
        with open(info["meta_path"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        # compatibility checks
        model_cfg = comp_model_cfg = NovaModelConfig.from_dict(meta["model_config"])
        if model_cfg.to_dict() != self.model.cfg.to_dict():
            raise RuntimeError(
                f"checkpoint model config mismatch: {model_cfg.to_dict()} vs run {self.model.cfg.to_dict()}")
        if meta["vocab_size"] != self.tok.vocab_size:
            raise RuntimeError(
                f"checkpoint vocab {meta['vocab_size']} != run vocab {self.tok.vocab_size}")
        tf = meta.get("tokenizer_fingerprint")
        cur_tf = NovaTokenizer.file_fingerprint(os.path.join(self.run_dir, "tokenizer.json"))
        if tf and tf != cur_tf:
            raise RuntimeError("tokenizer fingerprint mismatch — refusing to resume")
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        p = st["progress"]
        self.epoch = p["epoch"]
        self.global_step = p["global_step"]
        self.best_val = p["best_val"]
        self.best_epoch = p["best_epoch"]
        self.history = p.get("history", [])
        self.finished = p.get("finished", False)
        torch.set_rng_state(st["torch_rng"].cpu() if st["torch_rng"].device.type != "cpu" else st["torch_rng"])
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        self.log(f"restored step {self.global_step}, epoch {self.epoch}, "
                 f"best val {self.best_val} @ {self.best_epoch}")

    # ------------------------------------------------------------- LR schedule
    def _lr(self):
        """Cosine with warmup, based on global step."""
        s = self.global_step
        if s < self.cfg.warmup_steps:
            return self.cfg.lr * (s + 1) / self.cfg.warmup_steps
        span = self.total_steps_planned - self.cfg.warmup_steps
        if span <= 0:
            return self.cfg.min_lr
        p = min(1.0, (s - self.cfg.warmup_steps) / span)
        return self.cfg.min_lr + (self.cfg.lr - self.cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    # ---------------------------------------------------------------- batches
    def random_batch(self, ids, batch_size, seq_len):
        n = len(ids) - seq_len - 1
        idx = torch.randint(0, n, (batch_size,))
        off = torch.arange(seq_len)
        x = ids[idx[:, None] + off]
        y = ids[idx[:, None] + off + 1]
        return x.to(self.device), y.to(self.device)

    def eval_val(self, max_batches=16):
        self.model.eval()
        n = len(self.val_ids) - self.cfg.seq_len - 1
        gen = torch.Generator().manual_seed(self.cfg.seed + 7)
        losses = []
        with torch.no_grad():
            for _ in range(max_batches):
                idx = torch.randint(0, n, (self.cfg.batch_size,), generator=gen)
                off = torch.arange(self.cfg.seq_len)
                x = self.val_ids[idx[:, None] + off].to(self.device)
                y = self.val_ids[idx[:, None] + off + 1].to(self.device)
                _, loss = self.model(x, targets=y)
                losses.append(float(loss))
        self.model.train()
        return sum(losses) / len(losses)

    # ------------------------------------------------------------------- loop
    def train(self, stop_checker=None):
        cfg = self.cfg
        self.model.train()
        start = time.time()
        t_render = 0.0
        sess_steps = 0
        try:
            while self.epoch < cfg.epochs:
                self.epoch += 1
                epoch_t0 = time.time()
                ep_loss, ep_n = 0.0, 0
                self.opt.zero_grad(set_to_none=True)
                for step in range(cfg.steps_per_epoch):
                    lr = self._lr()
                    for g in self.opt.param_groups:
                        g["lr"] = lr
                    x, y = self.random_batch(self.train_ids, cfg.batch_size, cfg.seq_len)
                    ctx = torch.autocast(self.device, dtype=self.amp_dtype, enabled=self.amp_enabled)
                    with ctx:
                        _, loss = self.model(x, targets=y)
                        raw_loss = float(loss.detach())
                        (loss / cfg.accum_steps).backward()
                    if (step + 1) % cfg.accum_steps == 0 or step == cfg.steps_per_epoch - 1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                        self.opt.step()
                        self.opt.zero_grad(set_to_none=True)
                        self.global_step += 1
                        sess_steps += 1
                        ep_loss += raw_loss
                        ep_n += 1
                        if (step + 1) % cfg.display_every == 0 or step == cfg.steps_per_epoch - 1:
                            dt = time.time() - start
                            sps = sess_steps / dt if dt else 0.0
                            remaining = (cfg.epochs - self.epoch) * cfg.steps_per_epoch + \
                                        (cfg.steps_per_epoch - step - 1)
                            eta = remaining / sps if sps > 0 else None
                            live_line(f"epoch {self.epoch}/{cfg.epochs} | step {step+1}/{cfg.steps_per_epoch} "
                                      f"(glob {self.global_step:,}) | batch loss {raw_loss:.4f} | lr {lr:.2e} | "
                                      f"{sps:.1f} steps/s | ETA {fmt_dur(eta) if eta else '--'}")
                            t_render += 0.0
                        if self.global_step % cfg.step_log_every == 0:
                            r = self._rate()
                            self.step_metrics({"ts": time.strftime("%H:%M:%S"),
                                               "global_step": self.global_step,
                                               "batch_loss": raw_loss, "lr": lr,
                                               "steps_per_s": r})
                        if stop_checker and stop_checker():
                            raise KeyboardInterrupt
                val = self.eval_val()
                is_best = self.best_val is None or val < self.best_val
                if is_best:
                    self.best_val, self.best_epoch = val, self.epoch
                dur = time.time() - epoch_t0
                rec = {"ts": time.strftime("%H:%M:%S"), "epoch": self.epoch,
                       "global_step": self.global_step,
                       "train_loss": ep_loss / max(1, ep_n), "val_loss": val,
                       "val_ppl": math.exp(min(val, 20)), "lr": lr,
                       "duration_s": dur, "is_best": is_best,
                       "steps_per_s": cfg.steps_per_epoch / dur}
                self.history.append(rec)
                self.epoch_metrics(rec)
                if self.save("checkpoint"):
                    if is_best:
                        self.save("best", mark_best=True)
                live_line(f"epoch {self.epoch}/{cfg.epochs} done | train {rec['train_loss']:.4f} "
                          f"| val {val:.4f} | ppl {rec['val_ppl']:.2f} | {dur:.1f}s", final=True)
                if is_best:
                    self.log(f"  ★ new best val {self.best_val:.4f} @ epoch {self.best_epoch}")
            self.finished = True
            self.save("checkpoint")
        except KeyboardInterrupt:
            self.log("Interrupted — saving resumable checkpoint…")
            self.save("checkpoint")
            self.log(f"resumable state saved at step {self.global_step}")

    def _rate(self):
        return getattr(self, "_last_steps_per_s", None)


# ---------------------------------------------------------------------------
# Mode flows
# ---------------------------------------------------------------------------
def fresh_flow(args):
    from nova.data.corpus import load_texts, paragraph_split, dedup
    run_dir = new_run_dir(args.runs_root, prefix=args.run_name)
    tok = train_or_load_tokenizer(run_dir, args)
    cfg = train_config_from_args(args)
    trainer = Trainer(run_dir, tok, args.corpus_train, args.corpus_val, cfg)
    write_run_config(run_dir, args, tok, cfg, mode="fresh")
    trainer.log(f"model params: {count_parameters(trainer.model):,} | device: {trainer.device}")
    trainer.train()
    return run_dir, trainer


def resume_flow(args):
    run_dir, cname = resolve_run(args.resume)
    if not run_dir:
        raise SystemExit(f"cannot resolve run dir from {args.resume}")
    meta = json.load(open(os.path.join(run_dir, "run_config.json"), encoding="utf-8"))
    load_tokenizer_into_run(run_dir)
    tok = NovaTokenizer.from_file(os.path.join(run_dir, "tokenizer.json"))
    cfg = train_config_from_args(args)
    if args.epochs > cfg.epochs - 1:
        cfg.epochs = args.epochs
    pt = os.path.join(run_dir, "checkpoints", f"{cname}.pt")
    mj = os.path.join(run_dir, "checkpoints", f"{cname}.json")
    if not (os.path.exists(pt) and os.path.exists(mj)):
        raise SystemExit(f"refusing: checkpoint {cname} missing or incomplete in {run_dir}")
    trainer = Trainer(run_dir, tok, args.corpus_train, args.corpus_val, cfg,
                      resume_info={"pt_path": pt, "meta_path": mj})
    # extend epochs if requested
    if args.epochs and args.epochs > trainer.cfg.epochs:
        trainer.cfg.epochs = args.epochs
        trainer.total_steps_planned = args.epochs * trainer.cfg.steps_per_epoch
    trainer.finished = False
    trainer.train()
    return run_dir, trainer


def inference_flow(args):
    run_dir, cname = resolve_run(args.inference)
    if not run_dir:
        raise SystemExit(f"cannot resolve run dir from {args.inference}")
    from nova.inference.generator import Generator
    gen = Generator.from_checkpoint(run_dir, checkpoint_name=cname)
    if args.prompt:
        print(gen.generate(args.prompt, max_new_tokens=args.max_tokens,
                           temperature=args.temperature, stream=False))
    else:
        for p, r in gen.eval_prompts(max_new_tokens=48).items():
            print(f"{p!r} -> {r!r}")
    return run_dir


def finetune_flow(args):
    run_dir, cname = resolve_run(args.finetune)
    pt = os.path.join(run_dir, "checkpoints", f"{cname}.pt")
    st = torch.load(pt, map_location="cpu", weights_only=False)
    new_run = new_run_dir(args.runs_root, prefix=f"{args.run_name}-ft")
    tok = NovaTokenizer.from_file(os.path.join(run_dir, "tokenizer.json"))
    tok.save(os.path.join(new_run, "tokenizer.json"))
    cfg = train_config_from_args(args)
    trainer = Trainer(new_run, tok, args.corpus_train, args.corpus_val, cfg)
    trainer.model.load_state_dict(st["model"])   # weights only: fresh optimizer
    write_run_config(new_run, args, tok, cfg, mode="finetune",
                     source={"run_dir": run_dir, "checkpoint": cname})
    trainer.log(f"fine-tune from {run_dir}/{cname}: weights loaded, optimizer fresh")
    trainer.train()
    return new_run, trainer


# ---------------------------------------------------------------------------
def new_run_dir(runs_root, prefix="run"):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(runs_root, f"{prefix}_{stamp}")
    os.makedirs(d, exist_ok=True)
    return d


def train_config_from_args(args) -> TrainConfig:
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    tc = y.get("training", {})
    mc = y.get("model", {})
    return TrainConfig(
        run_name=args.run_name if hasattr(args, "run_name") else tc.get("run_name", "tiny"),
        epochs=args.epochs if args.epochs is not None else tc.get("epochs", 1),
        seq_len=tc.get("seq_len", 128), batch_size=tc.get("batch_size", 8),
        accum_steps=tc.get("accum_steps", 1), lr=tc.get("lr", 3e-4),
        min_lr=tc.get("min_lr", 3e-5), warmup_steps=tc.get("warmup_steps", 50),
        weight_decay=tc.get("weight_decay", 0.1), grad_clip=tc.get("grad_clip", 1.0),
        steps_per_epoch=tc.get("steps_per_epoch", 100),
        eval_every_steps=tc.get("eval_every_steps", 100),
        seed=tc.get("seed", 42), mixed_precision=tc.get("mixed_precision", False),
        display_every=tc.get("display_every", 10),
        step_log_every=tc.get("step_log_every", 50),
        model=mc)


def train_or_load_tokenizer(run_dir, args):
    tr = os.path.join(run_dir, "tokenizer.json")
    if os.path.exists(tr):
        return NovaTokenizer.from_file(tr)
    tok = NovaTokenizer.train_from_iter(
        (p for p in iter_corpus_text(args.corpus_train)),
        vocab_size=args.tokenizer_vocab if hasattr(args, "tokenizer_vocab") else 8000)
    tok.save(tr)
    return tok


def load_tokenizer_into_run(run_dir):
    """Ensure a run dir has its frozen tokenizer.json (copied from corpus-derived
    snapshot if missing)."""
    tr = os.path.join(run_dir, "tokenizer.json")
    if not os.path.exists(tr):
        raise SystemExit(f"{run_dir} has no frozen tokenizer.json — refusing guesswork")
    return tr


def write_run_config(run_dir, args, tok, cfg: TrainConfig, mode: str, source=None):
    rc = {"mode": mode, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
          "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in vars(args).items()},
          "vocab_size": tok.vocab_size,
          "tokenizer_fingerprint": NovaTokenizer.file_fingerprint(
              os.path.join(run_dir, "tokenizer.json")),
          "corpus": {"train": os.path.abspath(args.corpus_train),
                      "val": os.path.abspath(args.corpus_val),
                      "train_fingerprint": fingerprint([args.corpus_train]),
                      "val_fingerprint": fingerprint([args.corpus_val])},
          "training": cfg.__dict__ | {"model": cfg.model},
          "source": source}
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(rc, f, indent=2)


def resolve_run(path):
    """Accept run dir or run-dir/checkpoints/<name>. Returns (run_dir, name)."""
    p = os.path.abspath(path)
    if os.path.basename(os.path.dirname(p)) == "checkpoints":
        return os.path.dirname(os.path.dirname(p)), os.path.basename(p)
    if os.path.isdir(os.path.join(p, "checkpoints")):
        name = "best" if os.path.exists(os.path.join(p, "checkpoints", "best.pt")) else "checkpoint"
        return p, name
    return None, None
