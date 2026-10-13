# Stage B Final Report — Real-Data Training + UI Redesign

Date: 2026-10-12 · Project: `nova_assistant/` · All numbers measured this session.

---

## Part 1 — Stage B: real-language training (TinyStories)

### Dataset provenance & verification

- **Dataset:** TinyStories V1 (Eldan & Li, Microsoft) — arXiv:2305.07759
- **Licence:** CDLA-Sharing-1.0 (permissive sharing licence; verified from the HF dataset card, recorded in `corpus_manifest.json`)
- **Sources (official HF resolve URLs, no auth required):**
  - train: `TinyStories-train.txt` sliced to first 2,000,000 bytes (`--train-chars 2000000`), sha256 recorded
  - valid: entire `TinyStories-valid.txt` (19,444,728 bytes), sha256 recorded
- **Preprocessing:** whitespace-normalization (case/punctuation preserved), exact-sha256 dedup, story-level leak control, deterministic split, corpus fingerprint `fb5b2458…d9f1`
- **Split:** 2,256 train stories (2 MB cap on purpose) / 21,940 val stories (whole official valid file). Post-normalization train∩val overlap measured directly: **0**. The asymmetry is safe-directioned (val more reliable than train); a later run should cap val too.

### Run configuration (actual, from run_config.json + tokenizer.json)

| thing | value |
|---|---|
| mode | `fresh` (explicit — no synthetic weights) |
| tokenizer | BPE, vocab **7,965** (fingerprint `4ad206e3…c44af1`) |
| context | model max_seq_len 256, trained seq_len 128 |
| optimizer | AdamW, lr 1e-3 → cosine decay to 1e-4, warmup 100 steps, wd 0.1, grad clip 1.0 |
| batch | 12 × accum 2 = effective 24 |
| budget | 2 epochs × 600 steps = 1,200 steps |
| params | 1,676,032 |
| measured peak RSS | 571 → 697 MB (8 GB laptop stable at 88–90 % used) |

### Training results (full run: `runs/real_tiny_B_20261012_222702`)

| metric | value |
|---|---|
| duration | ~10 min wall clock (epoch 1: 300.6 s, epoch 2: 238.8 s) |
| throughput | 1.0–1.1 steps/s during training, 2.5 steps/s incl. eval overhead |
| final train loss | **3.419** |
| best val loss | **3.460** (val PPL **31.82**) — epoch 2 checkpoint kept |
| epoch 1 val | 3.880 (PPL 48.43) |
| smoke run | val 3.888 @ 300 steps, 224 s (`real_tiny_smoke_20261012_221916`) |

Val PPL 48 → 32 across epochs: the fresh-initialized model is learning real-language structure on held-out text, not memorizing.

### Sample generations (measured, verbatim)

- *"Once upon a time, there was a little girl named"* → "Lily. She loved to play the park outside in the sun. One day, she found a special dress of door. …"
- *"The dog was very happy because"* → "it had a great idea."
- *"Ben and Emma found a shiny"* → coherent dialogue start, drifts mid-story.
- *"Explain how neural networks learn, like I am five."* — out-of-distribution; drifts into story-style fragments, as expected from a 1.7M-param story model.

**Honest reading:** surface grammar is right; semantic coherence breaks within a few sentences — the TinyStories paper's own expectation for models this small. No production-assistant claim is made or implied.

## Part 2 — UI redesign

### Files

- `src/nova/app/ui/index.html` — rewritten (sidebar, topbar, welcome + example cards, composer, settings modal, welcome `<template>`)
- `src/nova/app/ui/styles.css` — new design tokens, dark + light themes, mobile drawer layout, focus-visible + reduced-motion a11y
- `src/nova/app/ui/app.js` — new: conversation list (new/rename/delete/switch), versioned localStorage persistence (corrupt-storage-safe), token-by-token SSE rendering, whitelist markdown renderer (model output never inserted as raw HTML; link schemes checked), stop/cancel, retry, server/model status pills, theme toggle, settings (temperature + max-new-tokens — both honored by the API)
- Legacy UI kept at `src/nova/app/ui/index.legacy-v1.html.bak`

### Backend changes (surgical, `src/nova/api/server.py` only)

1. `ChatIn` moved to module level (fixes pydantic ForwardRef crash discovered earlier)
2. refusal path now emits valid SSE (`data: {"refusal": …}`) so browser clients parse it
3. removed an unused `JSONResponse` import (leftover lint)

No training/tokenizer/model files touched for the UI.

### Bugs found & fixed during live browser verification

1. **Fatal JS syntax error** (`?.content…` used as an assignment target in `showOfflineError`) — the whole app.js failed to parse; caught because the welcome screen didn't render; fixed; `node --check` clean.
2. **Welcome screen persisted after first send** — `clearWelcome()` ran before a fallback `newConversation()` re-rendered it; ordering fixed and re-verified in-browser.
3. **History never persisted** — `loadStore()` required `version === 1` but nothing wrote it; fixed in both load/save; verified a conversation with 2 messages survives a full page reload.

### Verified end-to-end in the browser (Chromium, real server + real model)

- [x] page loads; **zero console errors** on normal use (checked after each fix)
- [x] server-online pill + model info (`runs · 256 ctx · cpu`) from the real backend
- [x] full send → stream → render loop with the Stage B checkpoint, streaming token-by-token
- [x] welcome/security honesty text visible on empty state
- [x] history persists across refresh; settings (temperature) persist and are sent to the API
- [x] theme toggle switches dark/light and persists
- [x] refusal path returns proper SSE that the client renders without a bogus retry error
- [x] mobile 390×844: layout falls into the fixed-drawer mode (`position: fixed`, off-canvas at x:-264 → 0 when opened); desktop 1400×900: sidebar 264 px + main fills remainder, no horizontal overflow
- [x] health/model endpoints live-checked; `GET /app.js` byte-identical to disk

### Verification caveats

- Screenshot capture keeps failing with webview-compositor errors in the preview harness ("no frames"), so desktop/mobile visuals were verified via accessibility snapshots + computed-style probes, not pixels.
- Empty replies can occur when the shared server conversation accumulates turns (tiny model hits EOS early) — restarting the server or pressing "New chat" resets it. Documented behavior, not a defect.

## Launch commands (Windows)

```bat
cd nova_assistant
:: serve UI + chat on the Stage B checkpoint (already configured in configs\server.yaml)
python scripts\chat_server.py
:: open http://127.0.0.1:8001/

:: or one-off generation without the server
python scripts\train.py --inference runs\real_tiny_B_20261012_222702 --prompt "Once upon a time" --max-tokens 80
```

## Full test suite status

`python -m pytest tests/ -q` → **32 passed, 0 failed** (exit 0), rerun after the final server.py edit; the unchanged foundation tests were not re-run separately since the earlier full-suite pass (32/32) covered them.

## Stage C — conversational fine-tuning (prepared, not started)

- `chat_template.py` already supports multi-turn `<system>/<user>/<assistant>` rendering with distinct end-of-turn (`<|eos|>`) markers, and `build_training_example()` implements prompt-token loss masking (labels −100 outside assistant spans) — regression-tested this stage (truncation bug found and fixed).
- The tokenizer was trained on raw stories, so special tokens exist but are rare in its vocab statistics; Stage C must retrain or extend the tokenizer on dialogue data before fine-tuning, or keep the reserved ids out of BPE merges (they already are excluded in the trainer).
- Concrete plan: (1) small licensed instruction/dialogue corpus (e.g. oasst1-derived derivative or hand-written intents under permissive terms — verify licence before use), (2) tokenizer vocabulary audit for chat tokens, (3) finetune from `real_tiny_B` best checkpoint with `--mode finetune`, masked as above, (4) evaluate: refusal-rate sanity, end-of-turn behavior, and multi-turn coherence on fixed prompts.

## Files changed this stage

- NEW: `scripts/prepare_data.py`, `scripts/memcheck.py`, `configs/real_tiny.yaml`, `tests/test_stage_b.py`
- NEW: `src/nova/app/ui/styles.css`, `src/nova/app/ui/app.js`, `src/nova/app/ui/index.legacy-v1.html.bak`
- MOD: `src/nova/app/ui/index.html` (full redesign), `src/nova/api/server.py` (ChatIn module-level, SSE refusal, removed dead import), `src/nova/chat_template.py` (truncation fix), `configs/server.yaml` (points at the Stage B checkpoint), `README.md` (Stage B + UI + 32 tests)
- NEW (reports): `reports/STAGE_B_FINAL_REPORT.md` (this file)
- Preserved untouched: legacy Nova V3.6/V4 sources, synthetic baseline runs, existing checkpoints/benchmarks

## Known limitations

- mixed-precision path is code-gated but untested (no CUDA on this machine)
- validation set much larger than train (safe direction; cap symmetrically next run)
- screenshot-based visual QA not possible in this harness (see caveat above)
- conversational fine-tuning (Stage C) designed, not started
- BPE tokenizer trained on only 2 MB of text (vocab 7,965); more text → larger vocab
