# Nova Assistant

A separate, extensible foundation for a real conversational AI — distinct from the old
Nova V4 research model (`../Final model/`, untouched) and the CLI proof-of-concept
(`../nova_chat.py`).

**Honest status:** this repository ships a *working end-to-end vertical slice*: a modern
decoder-only LM (PyTorch), byte-level BPE tokenizer, fresh/resume/inference/fine-tune
training modes with verified checkpoints, a FastAPI streaming chat backend, and a web UI.
The default config trains a ~1M-param model in minutes on a CPU laptop and generates
coherent text on the corpus it saw. It is **not** yet an assistant that follows complex
instructions — see "Roadmap" for the staged path (instruction data is Stage C).

## Setup (Windows, CPU)

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers fastapi "uvicorn[standard]" pyyaml pydantic numpy pytest
```

No GPU is required (or assumed); CUDA is used only if explicitly detected.

## 1. Build a corpus

```bat
:: synthetic pipeline-validation corpus (default)
python scripts/build_corpus.py
:: or from your own text files (verify licenses, see docs/DATA_PROVENANCE.md)
python scripts/build_corpus.py --from-files path\to\a.txt path\to\b.txt
```

## 2. Train

**Stage B (real language, TinyStories — already run, see the results below):**

```bat
:: 1) prepare the real corpus (downloads TinyStories train slice + valid, builds manifest)
python scripts\prepare_data.py --train-chars 2000000

:: 2) smoke test (~4 min), then the full run (~10 min on the 8 GB laptop)
python scripts\train.py --config configs\real_tiny.yaml --mode smoke --corpus-train data\tinystories_v1\corpus_train.jsonl --corpus-val data\tinystories_v1\corpus_val.jsonl
python scripts\train.py --config configs\real_tiny.yaml --mode fresh --corpus-train data\tinystories_v1\corpus_train.jsonl --corpus-val data\tinystories_v1\corpus_val.jsonl --run-name real_tiny_B

:: 3) resume the latest run later if needed
python scripts\train.py --config configs\real_tiny.yaml --mode resume --run-name real_tiny_B
```

**Stage B result (measured 2026-10-12):** validation loss 3.880 → **3.460 (best)**,
val PPL 48.4 → **31.8** over 2 epochs × 600 steps (1,676,032-param model, ~10 min,
CPU, 697 MB peak RSS). In-domain generations are TinyStories-style and grammatically
plausible; coherence still breaks after a few sentences — expected at this scale.
Full details and sample generations: `reports/STAGE_B_FINAL_REPORT.md`.

**Stage A (synthetic pipeline-validation corpus) — kept unchanged as the baseline:**
```bat
:: fresh (default) — creates runs/tiny_<timestamp>/ with frozen tokenizer, checkpoints, logs
python scripts\train.py --config configs\tiny.yaml --mode fresh --epochs 2

:: resume explicitly (never implicit)
python scripts\train.py --resume runs\tiny_20261012_210913 --epochs 4

:: inference-only
python scripts\train.py --inference runs\tiny_20261012_210913 --prompt "the cat looked at" --max-tokens 40

:: fine-tune weights into a NEW run with a fresh optimizer
python scripts\train.py --finetune runs\tiny_20261012_210913
```

What you get per run:

| Path | Contents |
|---|---|
| `runs/<id>/tokenizer.json` | frozen tokenizer (fingerprint-checked on every load) |
| `runs/<id>/checkpoints/checkpoint.pt/.json` | latest resumable state (model+optimizer+RNG+progress) |
| `runs/<id>/checkpoints/best.pt/.json` | best-validation model (never overwritten by worse results) |
| `runs/<id>/logs/metrics.jsonl` | per-epoch train/val loss, PPL, LR, duration, is_best |
| `runs/<id>/logs/step_metrics.jsonl` | per-N-step batch losses and step rates |
| `runs/<id>/run_config.json` | full reproducibility metadata (args, corpus fingerprints, tokenizer fingerprint) |

Checkpoint discipline (inherited from the previous project): saves are atomic
(temp file + fsync + replace), each save is reload-verified before being declared
successful, resume hashes are checked (model config, vocab size, tokenizer
fingerprint), and mismatched/incomplete checkpoints are refused with clear errors —
never silently ignored.

## 3. Chat

```bat
:: point server.yaml at your run dir, then:
python scripts\chat_server.py
:: open http://127.0.0.1:8001/
```

Features: streaming (SSE) token-by-token replies, conversation list with
rename/delete, conversations persisted in localStorage (survive refresh), safe
markdown rendering (model output is never inserted as raw HTML), dark/light theme
(choice persists), settings dialog (temperature, max new tokens — both honored by
the API), server health + model info pills, stop button, copy button, example
prompt cards on an honest welcome screen, mobile drawer layout with keyboard and
reduced-motion accessibility support, and a basic safety refusal layer.

Legacy single-file UI preserved at `src/nova/app/ui/index.legacy-v1.html.bak`.

API:

| Route | Purpose |
|---|---|
| `POST /api/chat` | `{message, temperature?, max_new_tokens?}` → `{reply}` |
| `POST /api/chat/stream` | same body → SSE `data: {"piece": …}` events |
| `POST /api/new` | resets server conversation |
| `GET /api/model` | model info + explicit capability note |
| `GET /api/health` | liveness |

## Hardware expectations

Measured on the dev laptop (i5-8365U, 8 GB RAM, CPU):

| Config | Params | RAM | Training speed | Use |
|---|---|---|---|---|
| test cfg (this repo's tests) | ~100 K | <1 GB | seconds | CI |
| tiny (default) | ~1.0 M | ~1.5 GB | ~3.4 steps/s @ bs 8×2 | pipeline validation |
| nano (planned) | ~10 M | ~2.5 GB | est. 0.2–0.5 steps/s | experiment |
| base (planned) | ~50 M | >6 GB | not viable on this machine | needs bigger machine/GPU |

Configs are data; to train larger models on other hardware, add a YAML and run the
same commands. `--resume` lets you continue anywhere the checkpoints can be copied.

## Tests

```bat
python -m pytest tests\ -q
```

32 tests: tokenizer roundtrip/specials/fingerprint, model shapes, causal-masking
invariance, loss-reduction-under-grad, gradient finiteness, weight-tying, checkpoint
roundtrips (torch-level and Trainer-level), tokenizer-mismatch refusal,
fresh-run isolation, chat-template truncation, corpus leakage control, split
integrity, and API/streaming/refusal behavior.
fresh-run isolation, chat-template masking and context budgeting, dedup/split
integrity, safety refusals, e2e generate-after-checkpoint determinism, streaming
chunking, API validation error handling, missing-checkpoint errors.

## Honest limitations

- The default model is trained on a **synthetic corpus** for pipeline validation only.
  Real capability requires real corpora (Stage B) and instruction/dialogue data (Stage C).
- The safety layer is a keyword filter, not a classifier.
- The server holds a single conversation (single-user demo). Multi-session support is
  on the roadmap.
- No tools/function-calling execution; the `tools/` package is deliberately empty.
- Generation is one-forward-per-token (no KV cache) — fine at this scale, and the
  roadmap includes a cached path for larger models.

## Roadmap

1. **Stage B** — real public corpus (TinyStories-style, license-documented),
   larger `nano` config, learning-per-compare metrics across seeds.
2. **Stage C** — instruction/dialogue dataset with role tokens and loss masking
   (template already supports `<user>`/`<assistant>` masking via
   `build_training_example`), instruction-following eval suite.
3. **Stage D** — conversation memory polish, stop-sequence tuning, repetition
   controls, quantized CPU inference for the UI, multi-user sessions with auth.
4. **Stage E** — evaluation-driven iteration; keep comparing against preserved
   baselines (`../runs/EVAL_BASELINE.json`).
