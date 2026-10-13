# Data Provenance & Licensing

Every dataset used by Nova Assistant must be listed here with source, license, and date.

## Currently used data

| Dataset | Source | License | Used for | Added |
|---|---|---|---|---|
| synthetic_corpus | generated in-repo by `scripts/build_corpus.py` (seed 7, deterministic templates) | N/A (authored for this project) | pipeline validation only | 2026-10-12 |

**The default corpus is synthetic.** It exists to validate training plumbing (loss
goes down, checkpoints work, generation is coherent *relative to that corpus*). It
confers no general language competence and must not be used to claim capabilities.

## Rules for future datasets

1. Public/free sources only (TinyStories, Wikipedia subsets, public-domain books,
   permissively licensed instruction sets). Record the exact URL and the license text
   (or SPDX identifier) in `data/raw/<name>/LICENSE` at download time.
2. Never ship copyrighted text in the repo. Datasets live in `data/raw/` (gitignored)
   and are referenced by manifest.
3. Dedup (already part of `nova.data.corpus`) plus a Minimum Count Filter pass for
   Stage B.
4. Evaluation sets are split off BEFORE any training run that uses the remaining
   data, and are never mixed back in.
5. Keep a per-file SHA-256 in the manifest; training refuses a run if the fingerprint
   of the configured corpus changes mid-run (same discipline as the NumPy upgrade).
