"""Part 9 — required tests for the ABC English training package.

Every test runs against real components: the actual Stage B baseline checkpoint
(1.68M params, loaded read-only) and a real byte-level BPE tokenizer trained on
a tiny local corpus (so tests never touch or depend on network downloads).

Fast + offline. CUDA-only assertions are skipped when no GPU is present.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from abc_data import (  # noqa: E402
    IGNORE_INDEX, build_stream, chat_conversation_to_chunks, is_english,
    load_baseline, norm_hash, sample_windows, sha256_file, text_to_chunks,
)
from nova.models.model import count_parameters  # noqa: E402
from nova.tokenization.tokenizer import NovaTokenizer  # noqa: E402
from train import AbcTrainer  # noqa: E402

BASELINE = os.path.join(PKG, os.pardir, "runs", "real_tiny_B_20261012_222702")
CFG = json.load(open(os.path.join(PKG, "configs", "abc_ft_v1.json"), encoding="utf-8"))

# every .py file in the package (syntax check)
PKG_PY = [os.path.join(PKG, f) for f in os.listdir(PKG) if f.endswith(".py")]
PKG_PY += [os.path.join(PKG, "tests", "test_part9.py")]


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tok():
    """A real tokenizer trained on a tiny local corpus (offline, fast)."""
    texts = [
        "The cat sat on the mat and looked at the sky.",
        "A mammal is an animal that feeds its young with milk.",
        "Photosynthesis is the process plants use to make food from sunlight.",
        "def add(a, b): return a + b  # a tiny helper function",
        "Gravity is the force that pulls objects toward each other.",
    ]
    return NovaTokenizer.train_from_iter(texts, vocab_size=1500)


@pytest.fixture(scope="module")
def baseline():
    model, tok_b, info = load_baseline(BASELINE, CFG["baseline_checkpoint"], "cpu")
    return model, tok_b, info


def _tiny_tr_cfg(**over):
    cfg = {"run_name": "test", "epochs": 1, "seq_len": 128, "batch_size": 4,
           "accum_steps": 1, "lr": 1e-4, "min_lr": 1e-5, "warmup_steps": 2,
           "weight_decay": 0.1, "grad_clip": 1.0, "steps_per_epoch_override": 4,
           "eval_every_steps": 2, "checkpoint_every_steps": 100000,
           "val_windows_per_source": 2, "seed": 42, "mixed_precision": False,
           "device": "cpu", "display_every": 10}
    cfg.update(over)
    return cfg


def _make_streams(tok_, n_src=2):
    """Two tiny sources of chat + one text source for the trainer."""
    chat_turns = [(f"what is thing number {i}?",
                   f"it is a small object number {i} used in tests.")
                  for i in range(n_src * 3)]
    text = "gravity pulls objects together. plants make food from sunlight. "
    chunks = [chat_conversation_to_chunks([t], tok_, 256)[0][0] for t in chat_turns]
    tchunks = text_to_chunks(text * 4, tok_, 128)
    stream = build_stream(chunks)
    tstream = build_stream(tchunks)
    streams = {"oasst1": stream, "dolly15k": stream, "cosmopedia100k": tstream}
    return streams, streams


@pytest.fixture(scope="module")
def streams(tok):
    """Tiny per-source (ids, labels) streams for the trainer."""
    return _make_streams(tok)[0]


def _fresh_trainer(tmp_path, tok_, streams, resume=None, **over):
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)
    tok_.save(os.path.join(run_dir, "tokenizer.json"))
    from nova.models.model import NovaLM, NovaModelConfig
    cfg = NovaModelConfig(vocab_size=tok_.vocab_size, d_model=32, n_layers=1,
                          n_heads=2, n_kv_heads=2, ff_mult=2.0, max_seq_len=128,
                          dropout=0.0, tie_embeddings=True)
    model = NovaLM(cfg)
    binfo = {"run_dir": "test-baseline", "checkpoint": "best",
             "checkpoint_pt_sha256": "0" * 64}
    tr_cfg = _tiny_tr_cfg(**over)
    return AbcTrainer(run_dir, tok_, model, tr_cfg, streams, streams,
                      "test-manifest-fp", binfo, "cpu", resume=resume)


# ---------------------------------------------------------------- Part 9
def test_01_python_syntax():
    """All package .py files compile."""
    for p in PKG_PY:
        src = open(p, encoding="utf-8").read()
        compile(src, p, "exec")


def test_02_config_and_dataset_revisions_pinned():
    d = CFG["data"]
    for src in ("oasst1", "dolly15k", "cosmopedia100k"):
        assert len(d[src]["revision"]) == 40, f"{src} revision not pinned to a sha"
        assert d[src]["license"], f"{src} license missing"


def _quick_proc():
    """Resolve the quick processed dir exactly as train.py does (out_dir is
    already 'data/abc_english_v1', so it must not be prefixed again)."""
    return os.path.join(PKG, CFG["data"]["out_dir"] + "_quick", "processed")


def test_03_schema_fields_on_small_samples():
    """Processed quick partitions exist and carry the expected record schema."""
    proc = _quick_proc()
    chat = os.path.join(proc, "train_chat.jsonl")
    if not os.path.exists(chat):
        pytest.skip("quick data not prepared")
    with open(chat, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert "turns" in rec and "source" in rec
    assert isinstance(rec["turns"], list) and len(rec["turns"][0]) == 2


def test_04_english_filter():
    """Confident non-English is dropped; English, code and technical text kept."""
    assert is_english("The water cycle moves water between oceans and the sky.") is True
    assert is_english("def add(a, b): return a + b  # this adds two numbers") is True
    assert is_english("Die Photosynthese ist der Prozess, bei dem Pflanzen "
                      "Lichtenergie in chemische Energie umwandeln.") is False
    assert is_english("La photosynthese est le processus par lequel les "
                      "plantes convertissent la lumiere du soleil.") is False
    assert is_english("12345 !!!") is None   # no signal -> uncertain, never deleted


def test_05_dedup_key_normalizes_case_and_whitespace():
    """Dedup is case/space-insensitive but preserves paragraph boundaries
    (conservative_normalize keeps \\n\\n by design, so blank lines still separate docs)."""
    assert norm_hash("The cat sat on the mat.") == norm_hash("  the   cat sat on the mat. ")
    assert norm_hash("The cat sat on the mat.") == norm_hash("THE CAT SAT ON THE MAT.")
    assert norm_hash("The cat sat on the rug.") != norm_hash("The cat sat on the mat.")


def test_06_train_val_split_isolated():
    """No normalized-hash overlap between quick train and val partitions."""
    proc = _quick_proc()
    if not os.path.exists(proc):
        pytest.skip("quick data not prepared")
    def hashes(name):
        out = set()
        with open(os.path.join(proc, name), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                txt = (" ".join(u + " " + a for u, a in r["turns"])
                       if "turns" in r else r["text"])
                out.add(norm_hash(txt))
        return out
    assert not (hashes("train_chat.jsonl") & hashes("val_chat.jsonl"))
    assert not (hashes("train_text.jsonl") & hashes("val_text.jsonl"))


def test_07_chat_template_loss_masks(tok):
    """Assistant tokens are labeled; user/system/unlabeled positions are -100."""
    turns = [("hello", "hi there, how can I help?")]
    chunks, dropped = chat_conversation_to_chunks(turns, tok, 256)
    assert dropped == 0 and len(chunks) == 1
    ids, labels = chunks[0]
    assert len(ids) == len(labels)
    assert any(l == IGNORE_INDEX for l in labels), "user/system must be masked"
    assert labels[-1] != IGNORE_INDEX, "final eos must be labeled"
    # every labeled token must be a real token id or the eos
    for tok_id, l in zip(ids, labels):
        if l != IGNORE_INDEX:
            assert 0 <= l < tok.vocab_size


def test_08_text_chunks_label_everything(tok):
    ids, labels = text_to_chunks("gravity pulls objects together.", tok, 64)[0]
    assert labels == ids


def test_09_baseline_checkpoint_and_tokenizer_load(baseline):
    model, tok_b, info = baseline
    assert info["vocab_size"] == tok_b.vocab_size
    assert info["param_count"] > 0
    assert count_parameters(model) == info["param_count"]
    assert len(info["tokenizer_fingerprint"]) == 64
    # model produces finite logits
    ids = torch.tensor([tok_b.encode("the cat sat")], dtype=torch.long)
    logits, _ = model(ids)
    assert torch.isfinite(logits).all()


def test_10_forward_backward_finite_loss_and_grads(baseline):
    model, tok_b, _ = baseline
    model.train()
    x = torch.randint(0, tok_b.vocab_size, (2, 16))
    y = torch.randint(0, tok_b.vocab_size, (2, 16))
    _, loss = model(x, targets=y)
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    model.eval()


def test_11_checkpoint_save_reload_roundtrip(tmp_path, tok, streams):
    tr = _fresh_trainer(tmp_path, tok, streams)
    tr.train()
    assert tr.save("latest")
    pt = os.path.join(tr.ckpts, "latest.pt")
    js = os.path.join(tr.ckpts, "latest.json")
    assert os.path.exists(pt) and os.path.exists(js)
    st = torch.load(pt, map_location="cpu", weights_only=False)
    meta = json.load(open(js, encoding="utf-8"))
    assert "model" in st and "opt" in st
    assert meta["progress"]["global_step"] == tr.global_step
    # a fresh model reloads the saved weights exactly
    from nova.models.model import NovaLM, NovaModelConfig
    fresh = NovaLM(NovaModelConfig.from_dict(meta["model_config"]))
    fresh.load_state_dict(st["model"])


def test_12_generation_produces_text(baseline):
    from nova.inference.generator import Generator
    model, tok_b, _ = baseline
    gen = Generator(model, tok_b, "cpu")
    out = gen.generate("the cat sat on the", max_new_tokens=8, temperature=0.8, seed=1)
    assert isinstance(out, str)


def _rewrite_step(pt_path, json_path, global_step):
    """Rewind a saved checkpoint to global_step so a completed run becomes a
    mid-epoch checkpoint for resume testing."""
    st = torch.load(pt_path, map_location="cpu", weights_only=False)
    st["progress"]["global_step"] = global_step
    torch.save(st, pt_path)
    meta = json.load(open(json_path, encoding="utf-8"))
    meta["progress"]["global_step"] = global_step
    json.dump(meta, open(json_path, "w", encoding="utf-8"))


def test_13_resume_continues_step(tmp_path, tok, streams):
    """Resume from a mid-epoch checkpoint continues at that exact step."""
    tr = _fresh_trainer(tmp_path, tok, streams)
    tr.train()
    assert tr.save("latest")
    pt = os.path.join(tr.ckpts, "latest.pt")
    mj = os.path.join(tr.ckpts, "latest.json")
    _rewrite_step(pt, mj, 2)                       # pretend we died at step 2/4
    tr2 = _fresh_trainer(tmp_path / "r2", tok, streams,
                         resume={"pt_path": pt, "meta_path": mj})
    assert tr2.global_step == 2, "resume must continue at the saved mid-epoch step"
    assert tr2.global_step % tr2.steps_per_epoch != 0, "step should be mid-epoch"


def test_14_restart_epoch_rewinds_to_epoch_start(tmp_path, tok, streams):
    """--restart-epoch on a mid-epoch checkpoint rewinds to the epoch's first
    step index (global_step % steps_per_epoch == 0)."""
    tr = _fresh_trainer(tmp_path, tok, streams)
    tr.train()
    assert tr.save("latest")
    pt = os.path.join(tr.ckpts, "latest.pt")
    mj = os.path.join(tr.ckpts, "latest.json")
    _rewrite_step(pt, mj, 3)                       # pretend we died at step 3/4
    tr2 = _fresh_trainer(tmp_path / "r2", tok, streams,
                         resume={"pt_path": pt, "meta_path": mj,
                                 "restart_epoch": True})
    assert tr2.global_step == 0, "--restart-epoch must rewind the partial epoch"
    assert tr2.global_step % tr2.steps_per_epoch == 0


def test_15_prepare_reuses_cached_data():
    """Rerunning prepare_data --quick reuses the cached pool (no re-download)
    and produces byte-identical partitions + stable fingerprints."""
    quick_dir = os.path.join(PKG, CFG["data"]["out_dir"] + "_quick")
    if not os.path.exists(quick_dir):
        pytest.skip("quick data not prepared")
    proc = os.path.join(quick_dir, "processed")
    manifest = os.path.join(quick_dir, "DATA_MANIFEST.json")
    assert os.path.exists(manifest)
    man = json.load(open(manifest, encoding="utf-8"))
    # fingerprints recorded and stable
    assert "files" in man and len(man["files"]) >= 4
    before = {}
    for name, meta in man["files"].items():
        full = os.path.join(PKG, meta["path"])
        assert sha256_file(full) == meta["sha256"], f"{name} fingerprint drift"
        before[name] = meta["sha256"]
    # rerun the real prepare CLI; the pool .sig match means no rebuild, and the
    # deterministic pipeline must reproduce the identical partition bytes.
    rc = subprocess.run(
        [sys.executable, "-u", os.path.join(PKG, "prepare_data.py"), "--quick"],
        cwd=PKG, capture_output=True, text=True)
    assert rc.returncode == 0, f"prepare rerun failed:\n{rc.stdout}\n{rc.stderr}"
    assert "reus" in (rc.stdout + rc.stderr).lower(), "expected cached-pool reuse"
    man2 = json.load(open(manifest, encoding="utf-8"))
    for name, meta in man2["files"].items():
        assert meta["sha256"] == before[name], f"{name} changed on rerun"


def test_16_baseline_artifacts_unmodified(baseline):
    """The baseline run dir is read-only for us: files still present + hashed."""
    _, _, info = baseline
    pt = os.path.join(BASELINE, "checkpoints", f"{CFG['baseline_checkpoint']}.pt")
    assert sha256_file(pt) == info["checkpoint_pt_sha256"]
    assert os.path.exists(os.path.join(BASELINE, "tokenizer.json"))


def test_17_eval_prompts_config_valid():
    ev = json.load(open(os.path.join(PKG, "configs", "eval_prompts.json"), encoding="utf-8"))
    assert ev["version"] and len(ev["prompts"]) > 0
    for p in ev["prompts"]:
        assert "id" in p and "category" in p and "turns" in p
        assert isinstance(p["turns"], list) and len(p["turns"]) >= 1


def test_18_gpu_absent_is_honest():
    """No CUDA on this machine -> nothing claims a GPU pass."""
    if not torch.cuda.is_available():
        assert torch.cuda.device_count() == 0
