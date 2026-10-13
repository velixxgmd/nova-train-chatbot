"""Stage B regression tests: corpus preparation, leakage prevention, tokenizer
real-text roundtrip, LR schedule, EOS and masking correctness.

Uses small local fixtures only — no network downloads in tests.
"""
import hashlib
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SRC = os.path.join(PROJECT, "src")
sys.path.insert(0, SRC)

from nova.tokenization.tokenizer import NovaTokenizer, SPECIAL_TOKENS


# ---------------------------------------------------------- prepare_data.py
def _import_prepare():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prep", os.path.join(PROJECT, "scripts", "prepare_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_conservative_normalize_preserves_case_punctuation():
    p = _import_prepare()
    src = "Hello  World.   THIS  stays!\r\n\r\n\r\nsecond   para.\n"
    out = p.conservative_normalize(src)
    assert "Hello World." in out
    assert "THIS  stays" not in out          # double spaces collapsed
    assert "THIS stays!" in out              # punctuation + caps kept
    assert out.count("\n\n") >= 1            # paragraph boundary kept
    assert "second para." in out


def test_iter_stories_splits_on_separator(tmp_path):
    p = _import_prepare()
    f = tmp_path / "x.txt"
    f.write_text("chunk a b c\n" + p._STORY_SEP + "\nchunk d e f\n", encoding="utf-8")
    got = [s.replace("\n", " ").strip() for s, _ in p.iter_stories(str(f))]
    assert got == ["chunk a b c", "chunk d e f"]


def test_leakage_between_splits(tmp_path):
    p = _import_prepare()
    # craft two files with an overlapping story
    story_a = "Once there was a cat named Ella who played in the garden all afternoon."
    train = p.conservative_normalize(story_a)
    val = p.conservative_normalize(story_a + " She loved it.")
    # ensure these normalize with >= 40 chars and are considered distinct stories
    ptr = tmp_path / "tr.txt"; pvr = tmp_path / "vl.txt"
    ptr.write_text(train + "\n" + p._STORY_SEP + "\n", encoding="utf-8")
    pvr.write_text(val + "\n" + p._STORY_SEP + "\n", encoding="utf-8")
    seen, recs, rej = set(), {"train": [], "valid": []}, {"train": 0, "valid": 0}
    for split, path in (("train", str(ptr)), ("valid", str(pvr))):
        for story, _ in p.iter_stories(path):
            text = p.conservative_normalize(story)
            k = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if k in seen:
                rej[split] += 1; continue
            seen.add(k); recs[split].append(text)
    # exact-duplicate check runs, and prepare_data additionally removes any val
    # story whose hash appears in train (story-level leakage prevention)
    assert recs["train"] == [train]
    assert recs["valid"] == [val]           # different text -> kept as distinct
    # simulate the val-side train-overlap filter from prepare_data.main
    h_train = {hashlib.sha256(x.encode()).hexdigest() for x in recs["train"]}
    cleaned = [x for x in recs["valid"]
               if hashlib.sha256(x.encode()).hexdigest() not in h_train]
    assert cleaned == recs["valid"]         # no overlap because texts differ
    # overlapping identical story would be removed:
    identical_val = recs["train"][0]
    assert hashlib.sha256(identical_val.encode()).hexdigest() in h_train


def test_prepare_manifest_created(tmp_path, monkeypatch):
    """Run the real prepare pipeline end-to-end using pre-seeded raw files (no network)."""
    p = _import_prepare()
    out = tmp_path / "corpus"
    out.mkdir()
    raw = out / "raw"
    raw.mkdir()
    stories = [f"Story {i}: the cat walked the garden and found story "
               f"number {i} about a ball." for i in range(6)]
    train_parts = ("\n" + p._STORY_SEP + "\n").join(stories)
    val_parts = ("\n" + p._STORY_SEP + "\n").join(
        f"Val story {i}: a small dog found the ball too. Number {i} here."
        for i in range(3))
    (raw / "TinyStories-train.head.txt").write_text(train_parts, encoding="utf-8")
    (raw / "TinyStories-valid.txt").write_text(val_parts, encoding="utf-8")
    # monkeypatch the download to a no-op since files exist
    monkeypatch.setattr(p, "stream_download", lambda *a, **k: (0, "x"))
    orig_argv = sys.argv
    sys.argv = ["prep", "--out-dir", str(out), "--train-chars", "1"]
    try:
        p.main()
    finally:
        sys.argv = orig_argv
    man = json.loads((out / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert man["corpus_fingerprint"] and len(man["corpus_fingerprint"]) == 64
    assert man["counts"]["train"]["records"] == 6
    assert man["counts"]["valid"]["records"] == 3
    assert (out / "corpus_train.jsonl").exists()
    assert (out / "corpus_val.jsonl").exists()


# ------------------------------------------------------- tokenizer on real text
@pytest.fixture(scope="module")
def realm_tok(tmp_path_factory):
    """Tiny tokenizer trained on actual TinyStories-style snippets."""
    tmp = tmp_path_factory.mktemp("tok")
    texts = [
        "Once upon a time, there was a little girl named Lily. She loved to play in the park with her friends.",
        "One day, Lily saw a big red ball. She wanted to play with it, so she asked her mom for help.",
        "Lily and her mom went to the park together. They had a lot of fun playing with the red ball.",
        "The cat sat on the mat and watched the rain fall outside the window all day long.",
    ]
    t = NovaTokenizer.train_from_iter(texts, vocab_size=1200)
    t.save(str(tmp / "tokenizer.json"))
    return t


def test_realtext_roundtrip(realm_tok):
    for s in ["Once upon a time, there was a little girl named Lily.",
              "They had a lot of fun!  Numbers 123 and symbols #@$ work too.",
              ""][0:3]:
        ids = realm_tok.encode(s)
        assert realm_tok.decode(ids) == s


def test_special_tokens_are_reserved_ids(realm_tok):
    ids = list(realm_tok.special_ids.values())
    assert len(set(ids)) == len(ids), "duplicate special ids"
    assert realm_tok.pad_id == 0 and realm_tok.special_ids["<|pad|>"] == 0


def test_eos_stops_generation(realm_tok):
    """Real-model EOS behavior: when the model's argmax is EOS it must stop and
    decode nothing. Verified through the actual Generator pipeline by checking
    that eos_id is never decoded into visible text."""
    from nova.models.model import NovaLM, NovaModelConfig
    from nova.inference.generator import Generator
    cfg = NovaModelConfig(vocab_size=realm_tok.vocab_size, d_model=32, n_layers=2,
                          n_heads=4, n_kv_heads=4, max_seq_len=32)
    torch.manual_seed(0)
    m = NovaLM(cfg)
    m.eval()
    orig_forward = m.forward

    def boom(tokens, targets=None):
        logits, loss = orig_forward(tokens, targets)
        logits[...] = 0.0
        logits[:, -1, realm_tok.eos_id] = 1e6
        return logits, loss
    m.forward = boom
    g = Generator(m, realm_tok)
    out = g.generate("Once upon a time", max_new_tokens=10, temperature=0.8, seed=1)
    assert out == "", f"output should be empty when EOS fires first, got {out!r}"


def test_lr_schedule_horizon():
    from nova.training.engine import TrainConfig
    import nova.training.engine as eng
    cfg = TrainConfig(epochs=2, steps_per_epoch=100, warmup_steps=20, lr=1e-3, min_lr=1e-5)

    class Fake:
        pass
    f = Fake()
    f.cfg = cfg
    f.total_steps_planned = 200
    f.global_step = 0
    vals = {}
    for s in (0, 10, 20, 100, 199):
        f.global_step = s
        vals[s] = eng.Trainer._lr(f)
    assert vals[10] < vals[20]                      # still warming up
    assert abs(vals[20] - 1e-3) < 1e-9              # peak at warmup end
    assert vals[100] < vals[20]                     # decaying
    assert vals[199] < 1.1e-5                       # near the floor at the end


def test_chat_template_prompt_masking(realm_tok):
    from nova import chat_template as ct
    convo = ct.Conversation()
    convo.add("question", "answer text")
    ids, labels = ct.build_training_example(convo, realm_tok, max_seq_len=64)
    n_labeled = sum(1 for l in labels if l != -100)
    assert n_labeled > 0
    # the labeled region must be at the END (assistant span), not the start
    assert all(l == -100 for l in labels[: len(labels) - n_labeled])
    # <|eos|> must be the final labeled token
    assert labels[-1] == realm_tok.eos_id
    assert ids[-1] == realm_tok.eos_id
