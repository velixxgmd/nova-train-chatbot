# Nova Assistant — Architecture

Status: design for the new `nova_assistant/` effort. This is a separate project from the
research baseline under `src/Final model/` (untouched) and from `src/nova_chat.py` (CLI proof
of concept; it stays as-is for evaluating the old character model).

## 1. Why a new stack

The Nova V4 model is a ~200K-parameter, character-level, NumPy transformer trained on
~400 KB of text. It cannot reasonably carry a dialogue: the context is 32 characters, the
corpus is a tiny synthetic/curated corpus, and the tokenizer has no notion of words or
roles. It remains a *benchmark*: the old artifacts (model/optimizerJSON checkpoints,
learning curve, evaluation baseline) stay frozen and untouched.

The new project targets a **real modern LM stack** with honest expectations about the
development laptop (Dell Latitude 5400, i5-8365U, 8 GB RAM, no GPU assumed):

- Maintainable, tested, extensible code (typed Python, PyTorch backend).
- BPE subword tokenizer, configurable model dims, standard decoder-only attention
  (RoPE, RMSNorm, SwiGLU), grouped-parameter efficiency (weight tying).
- Small default configs train in minutes-to-hours on CPU; larger configs exist as
  *configuration*, not assumption, so a GPU machine can pick them up.
- Fine-tune path for instruction data; inference/safety/chat plumbing around it.

## 2. Component map

```
nova_assistant/
  configs/          YAML model+training configs (defaults tiny, scale via separate files)
  docs/             architecture, data provenance, roadmap, hardware notes
  src/nova/
    tokenization/   BPE tokenizer (training, save/load, encode/decode, special tokens)
    models/         causal decoder (configurable), parameter counting, device utils
    training/       training loop with fresh/resume/finetune modes, checkpoints, logs
    data/           corpus loading, dedup, splits, dataset fingerprinting, prosevenance
    inference/      generation loop (streaming, stop tokens, temperature, repetition ctl)
    api/            FastAPI server (stream + chat endpoints, error handling, CORS)
    app/            chat UI (static HTML/JS frontend hitting the API; no build step)
    evaluation/     held-out loss/perplexity, fixed prompt suite, instruction eval
    safety/         input/output filtering boundaries, run-code prohibition
    tools/          empty for now: reserved for later, clearly separated tool use
  tests/            pytest suite (tokenizer, model shapes, masking, loss, checkpoints, e2e)
  scripts/          entrypoints: download_data.py, train.py, chat_server.py, eval.py
  data/             raw and processed corpora (with provenance manifest)
  runs/             one directory per training run (checkpoints/ logs/ run_config.json)
  checkpoints/      default parent of runs/*/checkpoints
```

## 3. Model

A decoder-only Transformer with modern defaults:

- **Tokenizer**: byte-level BPE (via `tokenizers` lib) with configurable vocab, byte
  fallback for any Unicode. Chosen so small vocab + robust character coverage survives
  tiny corpora (no `<unk>`).
- **Attention**: causal self-attention with RoPE (rotary position embeddings). RoPE
  extrapolates better than learned absolute positions and works with any context length.
- **Normalization**: RMSNorm (pre-norm), stable in fp32 on CPU.
- **FFN**: SwiGLU with configurable multiplier (default ≈ 8/3·d_model rounded).
- **Attention**: multi-head with rotary embeddings; flash/scaled-dot-product via
  PyTorch's `F.scaled_dot_product_attention` (works fine on CPU and uses fused kernels
  when cuda is available).
- **Tied embeddings**: input and output projection share one embedding matrix — saves
  ~vocab×d_model parameters, standard practice at small scales.
- **Precision**: fp32 for CPU; the code has explicit device detection and later can
  enable AMP via a flag (`mixed_precision=true` requires CUDA; guard-checked, never
  assumed).

Config knobs (all in `configs/*.yaml`): `vocab_size`, `d_model`, `n_layers`, `n_heads`,
`n_kv_heads` (GQA-ready), `ff_mult`, `max_seq_len`, `dropout`, `tie_embeddings`.

Default "nano" config: d_model=192, 4 layers, 6 heads, ff_mult=2.0, seq_len=256,
vocab≈8k → around **10–12 M parameters**, ~45 MB fp32 for weights, ~90 MB optimizer
state (AdamW m+v). That trains on 8 GB RAM in a few hours for a small corpus and is a
sensible "first real model" scale. A "tiny" 2M config exists for CI tests and
parameter-shape verification without much memory.

## 4. Tokenizer

`positional=rope`, `tokenizer="bpe"`:

- Byte-level BPE (from HuggingFace `tokenizers`) — no unknown tokens ever.
- Special tokens: `<|bos|>`, `<|eos|>`, `<|user|>`, `<|assistant|>`, `<|system|>`,
  `<|pad|>` (id 0), all fixed at training time.
- Trained on a corpus snapshot; **must be frozen and shipped with each run** (the
  checkpoint records tokenizer `vocab_size` + a hash of the `tokenizer.json` to prevent
  mismatched loads).

## 5. Training pipeline

- `fresh` (default) / `resume <run-dir>` / `inference <run-dir>` / `finetune <run-dir>`
  modes mirror the proven conventions from `src/nova_v4_upgraded.py`.
- Checkpoints: fused `checkpoint.pt` (torch.save dict of model-state, optimizer-state,
  scheduler, step, epoch, RNG state) + sidecar `checkpoint.json` (config, tokenizer
  fingerprint, dataset fingerprint, loss history). `best*.pt` only written on new-best
  val loss; never overwritten by worse results — same rules as the NumPy upgrade.
- Atomic-ish saves: save to `<name>.pt.tmp-<pid>-<n>`; fsync; `os.replace`; verify by
  reloading **before** declaring success.
- Logging: per-epoch `metrics.jsonl`, per-N-step `step_metrics.jsonl`, human log, and a
  live same-line progress display every 10 steps (carriage-return, configurable).
- Gradient accumulation supported (`accum_steps`), useful on 8 GB RAM. Mixed precision
  behind an explicit flag that only activates when the device is CUDA.

## 6. Inference & conversation

`inference/generator.py`:

- Streaming generation: yields decoded chunks token-by-token; supports stop tokens
  (assistant end turn), max_new_tokens, temperature, top-k/top-p, repetition penalty.
- Conversation state: the *chat template* wraps history into
  `system\n<|user|>…<|assistant|>…` and trims the oldest turns to fit the context window;
  a `Conversation` class owns history, reset, and context-budget accounting.
- The API is the same object regardless of which backend strength loads the model:
  `nova.inference.Generator.from_checkpoint(dir)` returns a streaming generator.

## 7. Chat application

`scripts/chat_server.py` runs FastAPI (serving both `/api/chat` (SSE streaming) and the
static single-page UI under `src/nova/app/ui/`). The frontend is a small, dependency-free
HTML/JS page: conversation list, streaming bubble, markdown (tiny renderer), stop button,
temperature/max-tokens controls, and a model indicator. Config (backend selection,
port, models dir) lives in `configs/server.yaml`.

## 8. Safety

- All generation goes through `safety/boundaries.py`: a deny-list check (self-harm,
  weapons, malware, etc.) that can prepend a refusal pattern. It is honest about being a
  keyword layer, not a classifier — a placeholder for a trained safety head later.
- No tool execution in v1. The `tools/` package exists but stays empty; the app must not
  execute model-generated code.

## 9. Hardware expectations (honest)

| Config | Params | fp32 weights | AdamW state | Activations @bs=8, 512 ctx | Train 10k steps @64×256 | Machine |
|---|---|---|---|---|---|---|
| tiny | ~2 M | 8 MB | 16 MB | <1 GB | ~40 min | this laptop |
| nano | ~10 M | 40 MB | 80 MB | ~2 GB | ~3–6 h | this laptop (small batches) |
| base | ~50 M | 200 MB | 400 MB | ~6 GB | days | needs 16+ GB RAM or GPU |

These estimates are *rough* — verify before launching. On the dev laptop we will validate
the whole pipeline with `tiny`, not nano/base.

## 10. Data

- Stage A/B: TinyStories-syle public corpus (dedup, license checked in
  `docs/DATA_PROVENANCE.md`) — coherent, easy LM target before instruction data.
  The repo ships none by default; `scripts/download_data.py` fetches into `data/`.
- Stage C: instruction/dialogue — only sources with clear licenses (e.g., subsets of
  public instruct datasets). Documented per-file with URL, license, download date,
  sha256 in the manifests.

## 11. Explicitly not (yet)

- Tool/function calling: the infrastructure is reserved but unimplemented.
- Long-term memory beyond the context window: not in v1.
- Web search/retrieval: not in v1.
- Any claim of ChatGPT/Claude-class capability: not until Stage C/D evaluations say so.
