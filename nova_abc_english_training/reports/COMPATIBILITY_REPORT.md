# Model & Tokenizer Compatibility Report

Verified by loading the real baseline checkpoint (not assumed). Figures below are
measured on this machine, not copied from documentation.

## Baseline identity

- **Run:** `../runs/real_tiny_B_20261012_222702`, checkpoint `best`
- **Checkpoint sha256:** `68cd689715181a47b8fd71d0a3accae2c2624ef46d7501df632d04989880b53e`
  (re-verified byte-identical after all work in this package — see
  `tests/test_16_baseline_artifacts_unmodified`)
- **Parameters:** 1,676,032 (`param_count` from a strict `load_state_dict`)
- **Model config:** `d_model=128, n_layers=4, n_heads=4, n_kv_heads=4, ff_mult=2.0,
  max_seq_len=256, dropout=0.0, tie_embeddings=true, rope_theta=10000.0`
- **Architecture:** decoder-only transformer (Nova `NovaLM`), RoPE, pre-norm, learned
  token + positional embeddings. No adapter/LoRA support built in.

## Tokenizer

- **Vocab size:** 7,965
- **Fingerprint:** `4ad206e309760ba77c9d204d4392f9ee721549cc56f2c79e67553d9af9c44af1`
- **Type:** `NovaTokenizer` — a trained subword tokenizer persisted to `tokenizer.json`
  in the run dir.
- **Special tokens:** the chat template in `nova/chat_template.py` relies on
  role/turn-delimiter tokens. `abc_data.load_baseline` **refuses** to run if the
  checkpoint's `vocab_size` or `tokenizer_fingerprint` does not match the tokenizer
  on disk. The vocabulary is never resized and a different tokenizer is never
  silently substituted.

## Fine-tuning compatibility (Part 5 conclusion)

**Compatible.** Fine-tuning reuses the existing components unchanged:

| Component | Reused as-is | Where |
|-----------|--------------|-------|
| Model class `NovaLM` + config | yes | `abc_data.load_baseline` |
| Baseline weights | yes (read-only load) | `abc_data.load_baseline` |
| Tokenizer + fingerprint check | yes | `abc_data.baseline_tokenizer` |
| Chat template / loss masking | yes | `chat_template.build_training_example` via `abc_data.render_masked` |

**What is NEW (and why):** `train.py` ships a fresh `AbcTrainer` loop rather than
reusing `nova_assistant/src/nova/training/engine.py`. The existing engine trains
on unmasked random raw-text windows; it cannot apply assistant-only loss masking,
which is the whole point of a conversational fine-tune. The new trainer keeps the
same checkpoint format (`save`/`_state` mirror the engine's schema) so downstream
`Generator.from_checkpoint` consumes a fine-tuned run unchanged.

No existing file was edited to make this work. The baseline run directory is
opened read-only; `load_baseline` writes nothing.

## Loss masking

`chat_template.build_training_example` labels only assistant tokens (up to and
including the turn delimiter) with real ids; every user/system/unlabeled position
is `-100` (ignored by cross-entropy). Verified in
`tests/test_07_chat_template_loss_masks`. Window sampling
(`abc_data.sample_windows`) rejects any window with zero labeled tokens, so loss
is never NaN.

## No compatibility blockers

There is no tokenizer/vocab/shape incompatibility, so no "STOP and propose a
separately-reviewed solution" condition (Part 5) was triggered.
