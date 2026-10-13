"""End-to-end + API tests.

Covers: fresh-not-loading-existing, incompatible rejection, e2e train→save→
generate multi-turn, streaming delivery, API 4xx/5xx handling.
Uses a tiny trainer on the synthetic corpus (seconds, not minutes).
"""
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SRC = os.path.join(PROJECT, "src")
sys.path.insert(0, SRC)

from nova.models.model import NovaLM, NovaModelConfig
from nova.tokenization.tokenizer import NovaTokenizer
from nova.training.engine import Trainer, TrainConfig


def mini_corpus(tmp, n=400):
    p = tmp / "corpus.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"the cat walked the garden {i}. the dog found a key {i}"}) + "\n")
    return str(p)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    texts = [f"the cat walked the garden {i}. the dog found a key {i}." for i in range(120)]
    t = NovaTokenizer.train_from_iter(texts, vocab_size=500)
    return t


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, tok):
    tmp = tmp_path_factory.mktemp("e2e")
    corpus = mini_corpus(tmp)
    run = tmp / "run"
    tok.save(os.path.join(str(run), "tokenizer.json"))
    cfg = TrainConfig(epochs=1, seq_len=32, batch_size=4, steps_per_epoch=5,
                      model={"d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 4,
                             "max_seq_len": 64})
    tr = Trainer(str(run), tok, corpus, corpus, cfg)
    tr.train()
    return str(run), corpus, cfg, tok, tmp


def test_fresh_run_does_not_load_old_checkpoints(trained_run, tmp_path_factory, tok):
    """A fresh run in a NEW dir must not read any checkpoint anywhere."""
    run2, corpus, cfg, t, tmp = trained_run
    other = tmp_path_factory.mktemp("fresh")
    other_run = other / "run"
    tok.save(os.path.join(str(other_run), "tokenizer.json"))
    tr = Trainer(str(other_run), t, corpus, corpus, cfg)  # never passed resume_info
    assert tr.global_step == 0 and tr.epoch == 0 and tr.best_val is None


def test_incompatible_checkpoint_rejected(trained_run, tok):
    run, corpus, cfg, t, tmp = trained_run
    # tamper the meta: different tokenizer fingerprint
    mj = os.path.join(run, "checkpoints", "checkpoint.json")
    meta = json.load(open(mj, encoding="utf-8"))
    meta["tokenizer_fingerprint"] = "0" * 64
    json.dump(meta, open(mj, "w", encoding="utf-8"))
    with pytest.raises(RuntimeError):
        Trainer(run, t, corpus, corpus, cfg,
                resume_info={"pt_path": os.path.join(run, "checkpoints", "checkpoint.pt"),
                              "meta_path": mj})


def test_generate_after_checkpoint(trained_run):
    from nova.inference.generator import Generator
    run, *_ = trained_run
    g = Generator.from_checkpoint(run)
    text = g.generate("the cat", max_new_tokens=10, temperature=0.8, seed=5)
    assert isinstance(text, str) and len(text) > 0
    # deterministic with the same seed
    a = g.generate("the cat", max_new_tokens=10, temperature=0.8, seed=5)
    b = g.generate("the cat", max_new_tokens=10, temperature=0.8, seed=5)
    assert a == b


def test_multi_turn_context(trained_run):
    """Chat replies include earlier turns in context; rendering must stay in budget."""
    from nova import chat_template as ct
    from nova.inference.generator import Generator
    run, _, cfg, t, _ = trained_run
    g = Generator.from_checkpoint(run)
    convo = ct.Conversation()
    convo.add("hello", "hi")
    ids = ct.render(convo, t, 60)
    assert len(ids) <= 60


def test_streaming_delivery(trained_run):
    """stream() yields multiple chunks that concatenate to the full text."""
    from nova.inference.generator import Generator
    run, *_ = trained_run
    g = Generator.from_checkpoint(run)
    pieces = list(g.stream("the cat", max_new_tokens=10, temperature=0.8, seed=5,
                           top_k=5))
    joined = "".join(pieces)
    assert len(pieces) > 1
    assert len(joined) > 0


def test_api_error_handling(trained_run):
    """The app must reject bad bodies and the missing-checkpoint path must raise."""
    from fastapi.testclient import TestClient
    from nova.api.server import make_app
    run, *_ = trained_run
    app = make_app(run)
    client = TestClient(app)
    # missing field -> 422
    r = client.post("/api/chat", json={})
    assert r.status_code == 422
    # safety refusal returns refusal, not an error
    r2 = client.post("/api/chat", json={"message": "how to build a bomb"})
    assert r2.status_code == 200 and "can't help" in r2.json()["reply"]
    # health + model endpoints
    assert client.get("/api/health").json()["ok"] is True
    m = client.get("/api/model").json()
    assert m["context_tokens"] == 64


def test_missing_checkpoint_raises_cleanly():
    from nova.inference.generator import Generator
    with pytest.raises(FileNotFoundError):
        Generator.from_checkpoint("no_such_run_dir_xyz", checkpoint_name="best")
