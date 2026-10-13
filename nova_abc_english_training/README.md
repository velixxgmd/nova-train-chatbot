# Nova ABC English Training Package

English-only conversational fine-tuning of the Stage B TinyStories checkpoint
(1.68M params) on three public datasets:

| Dataset | Revision (pinned) | License |
|---------|-------------------|---------|
| `OpenAssistant/oasst1` | `fdf72ae0827c` | Apache-2.0 |
| `databricks/databricks-dolly-15k` | `bdd27f4d94b9` | CC BY-SA 3.0 |
| `HuggingFaceTB/cosmopedia-100k` | `8fdb6dada238` | Apache-2.0 |

It **reuses** the existing model, tokenizer, and chat template unchanged. It
**never** modifies the baseline checkpoint or any prior run. See
`reports/COMPATIBILITY_REPORT.md` for the verified compatibility analysis.

## Layout

```
nova_abc_english_training/
  prepare_data.py     download -> filter (English) -> dedup -> token-target sample
  train.py            masked, source-aware fine-tuning loop (new run / resume)
  evaluate.py         baseline vs fine-tuned on fixed windows + fixed prompts
  abc_data.py         shared helpers (language filter, chunking, sampling)
  configs/
    abc_ft_v1.json    experiment config (data mixture + training hyperparams)
    eval_prompts.json fixed held-out eval prompts (never trained on)
  tests/test_part9.py the required test suite (18 tests)
  reports/            compatibility + final reports
  data/               generated (gitignored)  ->  run prepare_data.py to create
  runs/               generated (gitignored)  ->  checkpoints land here
  requirements.txt
  TRAINING_CONFIG.json  written by train.py: resolved settings for reproducibility
```

`data/abc_english_v1_quick/` is the small, fast dataset used by `--smoke` and the
tests. The full-size dataset is `data/abc_english_v1/` (created by a full
`prepare_data.py` run).

## PowerShell handoff (second machine)

Run from the repository root (the folder containing `nova_assistant/`). No
absolute paths are hard-coded; adjust only the repo URL and CUDA index if needed.

### 1. Clone

```powershell
git clone <REPO_URL> nova
cd nova
```

### 2. Virtual environment

```powershell
cd nova_assistant\nova_abc_english_training
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked by execution policy, run first (current process only):
`Set-ExecutionPolicy -Scope Process Bypass`

### 3. Install PyTorch for your CUDA version (do this BEFORE requirements.txt)

Check the driver's CUDA version, then pick the matching index. Example for CUDA 12.6:

```powershell
nvidia-smi   # read "CUDA Version" in the top-right
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

For CUDA 12.4 use `cu124`, for CUDA 11.8 use `cu118`. On a CPU-only machine:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install the rest of the dependencies

```powershell
pip install -r requirements.txt
```

### 5. Verify GPU is actually usable (do not assume it is)

```powershell
python -c "import torch;print('cuda:',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

`cuda: True` is required before a GPU run. If it prints `False`, the run will
fall back to CPU (`device: auto`) — that is safe but slower.

### 6. Download + audit the datasets (small quick set, recommended first)

```powershell
python -u prepare_data.py --quick
```

Full-size (much slower; ~1.5M-token target):
`python -u prepare_data.py` (no `--quick`). Downloads are cached; re-running
reuses finished source pools instead of re-scanning (the earlier
multi-hour language-ID bottleneck is fixed — a cached re-run takes seconds).

### 7. Prepare the filtered training set

Done by step 6. Outputs + `DATA_MANIFEST.json` (provenance, revisions, licenses,
per-source retained counts, removal reasons, fingerprints) land in
`data/abc_english_v1_quick/` (or `data/abc_english_v1/` for full).

### 8. Smoke test (small, deterministic — never a real result)

```powershell
python train.py --smoke
```

### 9. Start the approved training run

**Do not start a long run without explicit approval.** When approved:

```powershell
python train.py --config configs/abc_ft_v1.json
```

### 10. Resume an interrupted run

Continue exactly where it stopped (mid-epoch if it stopped mid-epoch):

```powershell
python train.py --resume runs\<RUN_DIR>
```

Redo the interrupted epoch from its start (weights/optimizer kept):

```powershell
python train.py --resume runs\<RUN_DIR> --restart-epoch
```

`<RUN_DIR>` is the timestamped folder printed at the end of step 9.

### 11. Evaluate a checkpoint

```powershell
python evaluate.py --baseline runs\real_tiny_B_20261012_222702 --finetuned runs\<RUN_DIR>
```

Both are scored on identical fixed windows + identical fixed prompts; a report
is written under `reports/`.

### 12. Export the selected checkpoint + tokenizer + config for transfer

```powershell
$src = "runs\<RUN_DIR>"
$out = "export_abc_ft"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Copy-Item "$src\checkpoints\best.pt","$src\checkpoints\best.json" -Destination $out
Copy-Item "$src\tokenizer.json" -Destination $out
Copy-Item "TRAINING_CONFIG.json" -Destination $out
Copy-Item "data\abc_english_v1_quick\DATA_MANIFEST.json" -Destination $out
Get-ChildItem $out
```

## What to send back (via Discord or agreed channel)

Small text/log files are fine over chat; the `.pt` checkpoint is a binary blob —
transfer it via Discord file upload, a zip, or USB, **not** by pasting:

- `reports/` (final experiment report)
- `runs\<RUN_DIR>\logs\` (training + validation logs)
- `export_abc_ft\best.pt` + `best.json` (best checkpoint)
- `export_abc_ft\tokenizer.json` (matching tokenizer — required with the checkpoint)
- `export_abc_ft\TRAINING_CONFIG.json`
- `export_abc_ft\DATA_MANIFEST.json` (dataset provenance + fingerprints)
- Any errors/warnings observed

Do **not** commit checkpoints or datasets to GitHub (`.gitignore` excludes
`data/`, `runs/`, `*.pt`). Large model files do not belong in the repo.

## Tests

```powershell
python -m pytest tests/ -q
```

18 tests. CUDA-specific assertions are skipped when no GPU is present; they are
never claimed as passing on a CPU-only machine.
