"""Shared helpers for the Nova ABC English fine-tuning package.

Everything here is deliberately conservative:
  - language filtering only removes text when it is *confident* it is not English,
  - normalization never touches case, punctuation or wording,
  - all randomness is seeded so any run can be reproduced exactly.

This module also bootstraps the import path for the existing `nova` package
(located at `nova_assistant/src`), which provides the model, tokenizer, chat
template and generator. We reuse those components; we never modify them.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
NOVA_ASSISTANT_ROOT = os.path.dirname(PKG_ROOT)
NOVA_SRC = os.path.join(NOVA_ASSISTANT_ROOT, "src")
if NOVA_SRC not in sys.path:
    sys.path.insert(0, NOVA_SRC)

import torch  # noqa: E402

from nova import chat_template as ct  # noqa: E402
from nova.models.model import NovaLM, NovaModelConfig, count_parameters  # noqa: E402
from nova.tokenization.tokenizer import NovaTokenizer  # noqa: E402

DEFAULT_SYSTEM = "You are Nova, a helpful AI assistant."
MAX_SENTINEL = 1 << 30          # "infinite" budget used to measure real lengths
IGNORE_INDEX = -100


def now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_config(path: str) -> dict:
    return load_json(path)


# ---------------------------------------------------------------------------
# Conservative text normalization + dedup keys
# ---------------------------------------------------------------------------
_ws = re.compile(r"[ \t]+")
_blank = re.compile(r"\n{3,}")


def conservative_normalize(text: str) -> str:
    """Collapse whitespace runs only. Case, punctuation and wording untouched."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _ws.sub(" ", t)
    t = _blank.sub("\n\n", t)
    return t.strip()


def norm_hash(text: str) -> str:
    """Dedup key: normalization-insensitive exact hash."""
    return sha256_bytes(conservative_normalize(text).lower().encode("utf-8"))


# ---------------------------------------------------------------------------
# Language identification (conservative)
# ---------------------------------------------------------------------------
_EN_STOPWORDS = {
    "the", "and", "is", "are", "was", "were", "of", "to", "in", "that", "it",
    "for", "on", "with", "as", "at", "by", "this", "from", "or", "an", "be",
    "can", "you", "your", "not", "but", "they", "have", "has", "had", "i",
    "we", "he", "she", "his", "her", "their", "there", "which", "what",
    "when", "who", "how", "why", "if", "then", "than", "so", "no", "yes",
    "do", "does", "did", "will", "would", "about", "into", "over", "under",
    "after", "before", "also", "more", "most", "some", "such", "only", "own",
    "same", "too", "very", "my", "me", "him", "them", "these", "those", "a",
}

_langid = None


def _get_langid():
    global _langid
    if _langid is None:
        import langid
        _langid = langid
    return _langid


def english_signal(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    words = re.findall(r"[a-zA-Z']+", text.lower())
    non_ascii_letters = sum(1 for c in letters if ord(c) > 127)
    stop = sum(1 for w in words if w in _EN_STOPWORDS)
    return {
        "n_letters": len(letters),
        "n_words": len(words),
        "non_latin_ratio": (non_ascii_letters / len(letters)) if letters else 0.0,
        "stopword_ratio": (stop / len(words)) if words else 0.0,
    }


# Language identity is a document-level property, so langid only ever sees a
# bounded prefix. Without this, classify() costs ~370 ms per 4 KB document
# (measured) and a full Cosmopedia scan would take ~10 hours; with a 400-char
# prefix it costs ~5 ms and the decision is unchanged in practice.
_LANGID_PREFIX = 400


def is_english(text: str):
    """Conservative English check.

    Returns True (looks English), False (confidently NOT English), or
    None (uncertain — callers decide; we never delete valid English on
    uncertainty). Designed to keep technical vocabulary, code, names,
    numbers and quotations in English text.
    """
    sig = english_signal(text)
    if sig["n_letters"] == 0:
        return None                     # numbers/symbols only: no signal
    # clearly non-Latin script and no English function-word support
    if sig["non_latin_ratio"] > 0.20 and sig["stopword_ratio"] < 0.10:
        return False
    # fast path: ASCII-only letters with strong English function-word support
    if sig["non_latin_ratio"] == 0.0 and sig["stopword_ratio"] >= 0.15:
        return True
    lang, _conf = _get_langid().classify(text[:_LANGID_PREFIX])
    if lang == "en":
        return True
    # langid disagrees with English: keep it if English function words support it
    if sig["stopword_ratio"] >= 0.10:
        return True
    if len(text) < 40:
        return None                     # too short to be confident
    return False


# ---------------------------------------------------------------------------
# Chunk building (uses the existing, regression-tested chat template)
# ---------------------------------------------------------------------------
def render_masked(turns, tok, system: str = DEFAULT_SYSTEM):
    """(user, assistant) turns -> (ids, labels) with assistant-only loss."""
    convo = ct.Conversation(system=system,
                            turns=[{"user": u, "assistant": a} for u, a in turns])
    return ct.build_training_example(convo, tok, MAX_SENTINEL)


def chat_conversation_to_chunks(turns, tok, max_seq_len: int,
                                system: str = DEFAULT_SYSTEM):
    """Split a multi-turn conversation into chunks that fit max_seq_len.

    Chunks break at turn boundaries (each chunk is a valid conversation
    prefix), so nothing is fabricated and every chunk ends with a labeled
    assistant end-of-turn. Returns (chunks, dropped_single_turns).
    """
    chunks, dropped = [], 0
    cur = []
    for user, assistant in turns:
        candidate = cur + [(user, assistant)]
        ids, _labels = render_masked(candidate, tok, system)
        if len(ids) <= max_seq_len:
            cur = candidate
            continue
        if cur:
            chunks.append(render_masked(cur, tok, system))
        single, _ = render_masked([(user, assistant)], tok, system)
        if len(single) <= max_seq_len:
            cur = [(user, assistant)]
        else:
            dropped += 1
            cur = []
    if cur:
        chunks.append(render_masked(cur, tok, system))
    return chunks, dropped


def text_to_chunks(text: str, tok, max_seq_len: int):
    """Language-model objective chunks for raw prose (no fabricated roles).

    The whole text gets real labels; long documents are split into
    consecutive token windows, the document ends with one labeled eos.
    """
    ids = tok.encode(text) + [tok.eos_id]
    if len(ids) <= max_seq_len:
        return [(ids, list(ids))]
    step = max_seq_len
    return [(ids[i:i + step], ids[i:i + step]) for i in range(0, len(ids), step)]


# ---------------------------------------------------------------------------
# Token streams + deterministic window sampling
# ---------------------------------------------------------------------------
def build_stream(chunks) -> tuple[torch.Tensor, torch.Tensor]:
    ids: list = []
    for c_ids, _ in chunks:
        ids += c_ids
    labels: list = []
    for _c_ids, c_labels in chunks:
        labels += c_labels
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def sample_windows(ids: torch.Tensor, labels: torch.Tensor, n_windows: int,
                   seq_len: int, seed: int, max_tries_per_window: int = 50):
    """Deterministic masked-window sample. Windows with zero labeled tokens
    are rejected and resampled (bounded), so loss is never NaN."""
    rng = random.Random(seed)
    n = len(ids) - seq_len - 1
    if n <= 0:
        raise ValueError(f"stream too short ({len(ids)}) for seq_len {seq_len}")
    out = []
    tries = 0
    while len(out) < n_windows and tries < n_windows * max_tries_per_window:
        tries += 1
        i = rng.randint(0, n)
        y = labels[i + 1: i + 1 + seq_len]
        if (y != IGNORE_INDEX).any():
            out.append((ids[i: i + seq_len].clone(), y.clone()))
    if not out:
        raise ValueError("could not sample any window with labeled tokens")
    return out


def batch_windows(windows, batch_size: int):
    for i in range(0, len(windows), batch_size):
        xs = torch.stack([w[0] for w in windows[i:i + batch_size]])
        ys = torch.stack([w[1] for w in windows[i:i + batch_size]])
        yield xs, ys


# ---------------------------------------------------------------------------
# Baseline checkpoint handling (read-only)
# ---------------------------------------------------------------------------
def baseline_tokenizer(baseline_run: str, checkpoint_name: str = "best"):
    """Tokenizer-only baseline load with the same integrity checks
    (used by data preparation, where building the model is unnecessary)."""
    run = os.path.abspath(baseline_run)
    mj = os.path.join(run, "checkpoints", f"{checkpoint_name}.json")
    tok_path = os.path.join(run, "tokenizer.json")
    meta = load_json(mj)
    tok = NovaTokenizer.from_file(tok_path)
    if meta["vocab_size"] != tok.vocab_size:
        raise RuntimeError("baseline vocab mismatch — refusing to prepare data")
    tfp = NovaTokenizer.file_fingerprint(tok_path)
    if meta.get("tokenizer_fingerprint") and meta["tokenizer_fingerprint"] != tfp:
        raise RuntimeError("baseline tokenizer fingerprint mismatch — refusing")
    info = {"run_dir": run, "tokenizer_fingerprint": tfp,
            "vocab_size": tok.vocab_size, "model_config": meta["model_config"]}
    return tok, info


def load_baseline(baseline_run: str, checkpoint_name: str = "best",
                  device: str = "cpu"):
    """Load the baseline model + tokenizer with strict integrity checks.

    Raises loudly on any mismatch (vocab, tokenizer fingerprint, weights).
    Never writes anything into the baseline run directory.
    """
    run = os.path.abspath(baseline_run)
    pt = os.path.join(run, "checkpoints", f"{checkpoint_name}.pt")
    mj = os.path.join(run, "checkpoints", f"{checkpoint_name}.json")
    tok_path = os.path.join(run, "tokenizer.json")
    for p in (pt, mj, tok_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"baseline file missing: {p}")
    meta = load_json(mj)
    tok = NovaTokenizer.from_file(tok_path)
    if meta["vocab_size"] != tok.vocab_size:
        raise RuntimeError(f"baseline vocab mismatch: ckpt {meta['vocab_size']} "
                           f"vs tokenizer {tok.vocab_size}")
    tfp = NovaTokenizer.file_fingerprint(tok_path)
    if meta.get("tokenizer_fingerprint") and meta["tokenizer_fingerprint"] != tfp:
        raise RuntimeError("baseline tokenizer fingerprint mismatch — refusing to load")
    cfg = NovaModelConfig.from_dict(meta["model_config"])
    st = torch.load(pt, map_location="cpu", weights_only=False)
    model = NovaLM(cfg)
    model.load_state_dict(st["model"])          # strict
    model.to(device).eval()
    info = {
        "run_dir": run,
        "checkpoint": checkpoint_name,
        "checkpoint_pt_sha256": sha256_file(pt),
        "tokenizer_fingerprint": tfp,
        "vocab_size": tok.vocab_size,
        "model_config": cfg.to_dict(),
        "param_count": count_parameters(model),
        "progress": st.get("progress"),
    }
    return model, tok, info
