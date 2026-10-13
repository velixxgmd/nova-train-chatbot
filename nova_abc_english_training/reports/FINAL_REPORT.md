# ABC English Training Kit — Final Deliverable Report

Everything below is measured on this machine. Nothing is estimated or invented.
No CUDA GPU was available here, so no GPU number is claimed anywhere.

## 1. Folder

`nova_assistant/nova_abc_english_training/` (inside the existing `nova_assistant`
project, as the task preferred).

## 2. Files created

- `prepare_data.py`, `train.py`, `evaluate.py`, `abc_data.py`
- `configs/abc_ft_v1.json`, `configs/eval_prompts.json`
- `tests/test_part9.py`
- `requirements.txt`, `.gitignore`, `README.md`
- `reports/COMPATIBILITY_REPORT.md`, `reports/FINAL_REPORT.md`
- `reports/EVAL_REPORT.md`, `reports/eval_results.json` (written by `evaluate.py`)
- `TRAINING_CONFIG.json` (written by `train.py` at run time)
- `data/abc_english_v1_quick/DATA_MANIFEST.json` (+ processed partitions) —
  generated, gitignored

## 3. Dataset versions, licenses, source counts

| Source | Revision | License | Pool (quick) |
|--------|----------|---------|--------------|
| OpenAssistant/oasst1 | `fdf72ae0827c` | Apache-2.0 | 400 conversations |
| databricks/databricks-dolly-15k | `bdd27f4d94b9` | CC BY-SA 3.0 | 400 instruction/response |
| HuggingFaceTB/cosmopedia-100k | `8fdb6dada238` | Apache-2.0 | 300 prose docs |

Licenses and revisions were read from the official Hugging Face dataset cards
(verified live). Dolly's CC BY-SA 3.0 carries attribution + share-alike duties on
**redistribution of the data**; it does not, on its face, impose terms on trained
weights. That nuance is documented, not glossed over.

## 4. Examples and tokens retained (quick set)

- `train_chat`: 627 records / 52,792 tokens / 908,428 chars / ~1.13 MB
- `train_text`: 10 records / 13,189 tokens / 42,846 chars
- `val_chat`: 40 records / 4,842 tokens
- `val_text`: 15 records / 15,613 tokens

Removed by reason: `oasst1:oversize_turn_dropped` 393, `dolly15k:oversize_turn_dropped` 167.

## 5. Train / val / eval split

- Per-source held-out split: 5% (`val_fraction_per_source`), taken before
  token-target sampling, so val is never trained on.
- Eval prompts (`configs/eval_prompts.json`) are a fixed hand-written set;
  `eval_prompt_overlap_with_train = 0` (asserted at prepare time and in tests).

## 6. Mixture (by measured training tokens, not rows)

Targets 50/30/20. Achieved on the quick set:

| Source | train_tokens | achieved share |
|--------|--------------|----------------|
| oasst1 | 32,998 | 0.550 |
| dolly15k | 19,794 | 0.330 |
| cosmopedia100k | 13,189 | 0.220 |

Combined chat (oasst+dolly) = 0.800, cosmopedia prose = 0.200 — matching the
80/20 chat-vs-prose intent. The quick set is deliberately small; the full
`data/abc_english_v1/` run targets 1.5M train tokens and will track the ratios more
tightly.

## 7. Model & tokenizer compatibility

**Compatible — no blockers.** Baseline: 1,676,032 params, vocab 7,965, tokenizer
fingerprint `4ad206e3…`. Loads via strict `load_state_dict`; vocab size and
tokenizer fingerprint are verified, never resized or substituted. Full detail in
`reports/COMPATIBILITY_REPORT.md`. The new `AbcTrainer` exists only because the
existing engine cannot do assistant-only loss masking; the model, tokenizer, and
chat template are reused unchanged.

## 8. Training defaults and why

| Setting | Value | Reason |
|---------|-------|--------|
| epochs | 1 | first experiment; no automatic multi-epoch runs |
| seq_len | 256 | matches baseline `max_seq_len`; max_example_tokens=250 fits one chunk |
| batch_size | 16 | conservative for 6 GB VRAM / 24 GB RAM target |
| accum_steps | 2 | effective batch 32 without extra memory |
| lr | 1e-4 | typical safe fine-tune LR (lower than from-scratch) |
| warmup_steps | 30 | short warmup for a short run |
| grad_clip | 1.0 | standard stability guard |
| mixed_precision | auto | bf16 only when device==cuda, verified at runtime |
| device | auto | cuda if genuinely available, else cpu |

## 9. Tests executed and actual results

`python -m pytest tests/ -q` → **18 passed, 0 skipped, 0 failed** (~16 s, CPU).

Covers all Part-9 requirements: python syntax, dataset revision pinning, schema
on small samples, English filter (incl. non-Latin → False, symbols → None),
dedup keying, train/val isolation, chat-template loss masks, text-chunk labels,
baseline checkpoint+tokenizer load, finite forward/backward + grads,
checkpoint save/reload, generation, mid-epoch resume, `--restart-epoch` rewind,
prepare reuse + fingerprint stability, baseline artifact immutability,
eval-prompt config validity, and honest GPU-absence.

GPU-dependent assertions are skipped when no CUDA device is present — they were
not run, and are not claimed as passing.

## 10. Disk / memory / runtime

- Quick dataset on disk: ~1.13 MB (train_chat) + ~57 KB (val_text) + small others.
- Smoke train (15 steps, CPU): **8.9 s**, peak RSS **475.3 MB** (measured; see
  `runs/*/smoke_summary.json`). The memory probe was fixed during verification to
  load `psapi.dll` explicitly and report `None` instead of a misleading `0` when
  the call fails.
  *Provenance:* these smoke figures predate the final batch-size fix (each
  forward now carries `batch_size` samples instead of one), so a rerun would
  report different timing/memory. No new smoke run was made after that fix, per
  instruction; the fixed loop was re-verified by the 18-test suite instead.
- Quick prepare with warm pool cache: **4.4 s** (elapsed_s from manifest).
- Cold prepare bottleneck fixed: language-ID now classifies a bounded prefix;
  re-runs reuse per-source pool caches. A full-size cold run is still slow by
  nature (100k+ docs), but cached re-runs are seconds, not hours.

## 11. Blockers / decisions needing approval

- **Full-size run not started** (per instruction). Approve before running
  `python train.py` without `--quick`.
- **CUDA unverified here** — no GPU on this machine. The RTX 4060 path must be
  confirmed on the target machine via the step-5 check in the README.
- Dolly CC BY-SA 3.0 share-alike applicability to trained weights is unsettled;
  documented, not resolved.

## Known limitations

- 1.68M params is a very small model. Fine-tuning improves domain fit, not
  general capability. No ChatGPT-level behavior should be expected or implied.
- English-only is a property of the training data selection, not a guarantee the
  model can never emit non-English tokens.
