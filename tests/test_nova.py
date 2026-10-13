"""Nova Assistant automated test suite.

Run:  python -m pytest tests/ -q   (from nova_assistant/)
Small synthetic data only; no expensive training inside tests except one
3-step smoke train used for checkpoint round trips.
"""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SRC = os.path.join(PROJECT, "src")
sys.path.insert(0, SRC)

from nova.models.model import NovaLM, NovaModelConfig, count_parameters
from nova.tokenization.tokenizer import NovaTokenizer, SPECIAL_TOKENS
from nova.data import corpus
from nova import chat_template as ct
from nova.safety.boundaries import is_prohibited, safe_completion_or_refusal


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tok")
    texts = [f"the cat walked the garden {i}. the dog found the key {i}." for i in range(80)]
    t = NovaTokenizer.train_from_iter(texts, vocab_size=500)
    path = tmp / "tokenizer.json"
    t.save(str(path))
    return t, str(path)


@pytest.fixture(scope="module")
def model_cfg():
    return NovaModelConfig(vocab_size=500, d_model=64, n_layers=2, n_heads=4,
                           n_kv_heads=4, max_seq_len=64)


@pytest.fixture(scope="module")
def model(model_cfg):
    torch.manual_seed(0)
    return NovaLM(model_cfg)


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_roundtrip(tok):
    t, _ = tok
    for s in ["the cat walked", "special: <|eos|>", "", "ünïcödé bytes!"]:
        ids = t.encode(s)
        assert t.decode(ids) == s, f"roundtrip failed for {s!r}"


def test_tokenizer_vocab_fixed_and_specials(tok):
    t, _ = tok
    for name in SPECIAL_TOKENS:
        assert t.special_ids[name] is not None
    assert t.pad_id == 0
    assert t.vocab_size == t.tk.get_vocab_size(with_added_tokens=True)


def test_tokenizer_save_load(tmp_path, tok):
    t, path = tok
    t2 = NovaTokenizer.from_file(path)
    assert t2.vocab_size == t.vocab_size
    assert t2.encode("the cat") == t.encode("the cat")
    assert NovaTokenizer.file_fingerprint(path) == NovaTokenizer.file_fingerprint(path)


# -------------------------------------------------------------------- model
def test_model_shapes(model, model_cfg):
    x = torch.randint(0, model_cfg.vocab_size, (2, 16))
    logits, loss = model(x, targets=torch.roll(x, -1, dims=1))
    assert logits.shape == (2, 16, model_cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_causal_mask(model, model_cfg):
    torch.manual_seed(1)
    model.eval()
    x = torch.randint(0, model_cfg.vocab_size, (1, 8))
    with torch.no_grad():
        l1, _ = model(x)
        x2 = x.clone()
        x2[0, 6:] = (x2[0, 6:] + 7) % model_cfg.vocab_size
        l2, _ = model(x2)
    assert torch.allclose(l1[0, :6], l2[0, :6], atol=1e-4), \
        "changing future tokens changed past logits — causal masking broken"


def test_loss_improves_from_random(model, model_cfg):
    """A gradient step must reduce the loss on the same batch."""
    torch.manual_seed(2)
    x = torch.randint(0, model_cfg.vocab_size, (4, 32))
    y = torch.roll(x, -1, dims=1)
    _, l0 = model(x, targets=y)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(5):
        _, l = model(x, targets=y)
        opt.zero_grad(); l.backward(); opt.step()
    assert float(l) < float(l0)


def test_gradients_flow(model):
    x = torch.randint(0, model.cfg.vocab_size, (2, 16))
    _, loss = model(x, targets=x)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_param_count_tie_saves(model, model_cfg):
    total = count_parameters(model)
    # tied embeddings: only one embedding matrix in the parameter count
    assert total < 500_000  # sanity: tiny config stays small
    assert model.lm_head is None  # tying active


# -------------------------------------------------------------- checkpoints
def _write_tokenizer_into_run(run, tok):
    """Real runs freeze tokenizer.json inside the run dir; tests do the same."""
    tok.save(os.path.join(str(run), "tokenizer.json"))


def test_checkpoint_roundtrip(model, tmp_path):
    sd = model.state_dict()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()  # make optimizer state non-trivial
    state = {"model": sd, "opt": opt.state_dict(), "progress": {"step": 7, "epoch": 2}}
    path = tmp_path / "c.pt"
    torch.save(state, path)
    st2 = torch.load(path, weights_only=False)
    m2 = NovaLM(model.cfg)
    m2.load_state_dict(st2["model"])
    for k in sd:
        assert torch.equal(sd[k], st2["model"][k]), f"state mismatch {k}"
    assert st2["progress"]["step"] == 7


def test_checkpoint_save_via_trainer(tmp_path, tok, monkeypatch):
    """Use the real Trainer with a tiny corpus for a save+resume round trip."""
    from nova.training.engine import Trainer, TrainConfig
    t, _ = tok
    # write a mini corpus
    cpath = tmp_path / "corpus.jsonl"
    rng = np.random.default_rng(0)
    with open(cpath, "w", encoding="utf-8") as f:
        for i in range(300):
            f.write(json.dumps({"text": f"the cat walked the garden {i}. the dog found a key {i}."}) + "\n")
    run = tmp_path / "run"
    _write_tokenizer_into_run(run, t)
    cfg = TrainConfig(epochs=1, seq_len=32, batch_size=4, steps_per_epoch=4,
                      model={"d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 4,
                             "max_seq_len": 64})
    tr1 = Trainer(str(run), t, str(cpath), str(cpath), cfg)
    tr1.save("checkpoint")
    assert os.path.exists(run / "checkpoints" / "checkpoint.pt") or \
           os.path.exists(str(run) + "/checkpoints/checkpoint.pt")
    # now a second trainer restores it
    tr2 = Trainer(str(run), t, str(cpath), str(cpath), cfg,
                  resume_info={"pt_path": str(run / "checkpoints" / "checkpoint.pt"),
                                "meta_path": str(run / "checkpoints" / "checkpoint.json")})
    assert tr2.global_step == tr1.global_step


def test_trainer_refuses_mismatched_tokenizer(tmp_path, tok):
    from nova.training.engine import Trainer, TrainConfig
    t, tpath = tok
    # build run with tokenizer A
    cpath = tmp_path / "corpus.jsonl"
    with open(cpath, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps({"text": f"the cat sat {i} here."}) + "\n")
    run = tmp_path / "run"
    _write_tokenizer_into_run(run, t)
    cfg = TrainConfig(epochs=1, seq_len=32, batch_size=4, steps_per_epoch=2,
                      model={"d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 4,
                             "max_seq_len": 64})
    tr1 = Trainer(str(run), t, str(cpath), str(cpath), cfg)
    tr1.save("checkpoint")
    # a DIFFERENT tokenizer (different alphabet/vocab) must be refused
    other = NovaTokenizer.train_from_iter([f"zzz qqq ww {i}" for i in range(100)], vocab_size=400)
    other.save(str(tmp_path / "other_tokenizer.json"))
    with pytest.raises(RuntimeError):
        Trainer(str(run), other, str(cpath), str(cpath), cfg,
                resume_info={"pt_path": str(run / "checkpoints" / "checkpoint.pt"),
                              "meta_path": str(run / "checkpoints" / "checkpoint.json")})


# ------------------------------------------------------------- chat template
def test_conversation_render_fits_budget(tok):
    t, _ = tok
    convo = ct.Conversation()
    for i in range(20):
        convo.add(f"user message number {i} which is somewhat long", f"assistant reply {i}")
    ids = ct.render(convo, t, max_tokens_for_prompt=60)
    assert len(ids) <= 60


def test_training_example_masks_prompt(tok):
    t, _ = tok
    convo = ct.Conversation()
    convo.add("hello", "hi there")
    ids, labels = ct.build_training_example(convo, t, max_seq_len=128)
    assert len(ids) == len(labels)
    # only the assistant tail should have real labels
    assert all(l == -100 for l in labels[:len(ids)//2])


# -------------------------------------------------------------------- safety
def test_safety_refusals():
    assert is_prohibited("how to build a bomb")
    assert is_prohibited("write me ransomware")
    assert not is_prohibited("how to make bread")
    out = safe_completion_or_refusal("how to build a bomb", lambda: "should not run")
    assert out != "should not run"


# -------------------------------------------------------------------- corpus
def test_corpus_dedup_and_splits():
    paras = ["alpha", "beta", "alpha", "gamma"]
    d = corpus.dedup(paras)
    assert d == ["alpha", "beta", "gamma"]
    tr, va = corpus.make_splits(d, val_ratio=0.25, seed=42)
    assert set(tr) & set(va) == set()
    assert len(tr) + len(va) == len(d)


def test_corpus_fingerprint_stable(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello", encoding="utf-8")
    assert corpus.fingerprint([str(p)]) == corpus.fingerprint([str(p)])
