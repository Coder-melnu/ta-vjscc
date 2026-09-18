# TA-VJSCC

Research code and frozen experimental records for the TA-VJSCC video DeepJSCC attribution study.

## Research continuity handover

**Handover date:** 18 September 2026
**Repository:** https://github.com/Coder-melnu/ta-vjscc.git
**Branch:** `main`
**Verified environment snapshot:** commit `4523b09023d2a3cc6645c68587be3e2fc1c90ffe`
**Canonical environment:** Conda `ta-vjscc`, Python 3.10.20, PyTorch 2.5.1+cu121, CUDA 12.1, RTX 3060 12 GB.

The Overleaf manuscript is shared separately with Dr. Amir and Than Than Nu as editors. Manuscript sources are not stored here. Canonical rows are therefore identified by stable `result_id` values rather than mutable Overleaf table numbers.

### Authoritative records

| Purpose | File |
|---|---|
| Baseline, control, and exact top-k results | `canonical_results.csv` |
| UEP diagnostic results | `canonical_uep_diagnostics.csv` |
| Result row to checkpoints and evaluation script | `checkpoint_result_map.csv` |
| Canonical result summaries | `canonical_result_sources.txt` |
| Canonical checkpoints | `canonical_checkpoint_sources.txt` |
| Checkpoint hashes | `canonical_checkpoint_SHA256SUMS.txt` |
| Complete result-package inventory | `canonical_result_artifact_sources.txt` |
| Result-package hashes | `canonical_result_artifact_SHA256SUMS.txt` |
| Handover metadata hashes | `HANDOVER_SHA256SUMS.txt` |
| Experiment environment | `environment_experiment.txt` |
| Conda environment export | `environment_experiment_conda.yml` |
| Python package snapshot | `environment_experiment_pip_freeze.txt` |

`checkpoint_result_map.csv` is the authoritative mapping for which checkpoint feeds each manuscript row. Join its `result_id` with the corresponding row in `canonical_results.csv`. Every mapping records the source summary, model checkpoint and SHA-256, evaluator checkpoint and SHA-256, and evaluation script.

### Row groups and checkpoint rules

| Manuscript row group | `family` | Canonical checkpoint |
|---|---|---|
| Video DeepJSCC baselines | `reconstruction_only_c4`, `reconstruction_only_c8` | Reconstruction-trained `best.pt` |
| Fine-tuning attribution | `fine_tune_only_c8` | Primary `best_joint_loss.pt` |
| Native matched-rate control | `fine_tuned_native_c4` | Seed-specific `best_joint_loss.pt` |
| Hard-budget comparison | `exact_topk` | Seed-specific exact top-k `best.pt` |
| UEP diagnostics | `canonical_uep_diagnostics.csv` | Diagnostic summaries; not an official-test table |

Locked evaluator checkpoints:

- TSN: `downstream/action_recognition/weights/tsn_ucf101_head_locked_split_best.pt`
- R(2+1)D-18: `downstream/action_recognition/weights/r2plus1d_ucf101_layer4_locked_split_best.pt`

The canonical fine-tune-only c=8 result uses the primary minimum-validation-joint-loss checkpoint, `best_joint_loss.pt`. The earlier secondary `best_top1.pt` result is excluded from the canonical inventory.

### Regeneration and verification

Run from the repository root after `conda activate ta-vjscc`:

```bash
python scripts/generate_uep_diagnostics.py
python scripts/generate_canonical_results.py
python scripts/generate_checkpoint_result_map.py

sha256sum -c canonical_checkpoint_SHA256SUMS.txt
sha256sum -c canonical_result_artifact_SHA256SUMS.txt
```

Expected records:

- `canonical_uep_diagnostics.csv`: 52 rows
- `canonical_results.csv`: 109 rows (49 validation and 60 official-test)
- `checkpoint_result_map.csv`: 109 mappings
- Checkpoint inventory: 29 files
- Result-artifact inventory: 364 files from 64 canonical result packages

Large checkpoints, per-clip CSVs, manifests, configurations, and result packages remain at their recorded lab-server paths and are intentionally excluded from Git. The tracked inventories and SHA-256 records detect missing or modified artifacts.

Example row lookup:

```bash
rg 'test_exact_topk_learned_topk_tsn_seed42_AWGN_c8_snr13' checkpoint_result_map.csv
```

If Overleaf table numbers change, match rows using `family`, `mode`, `evaluator`, `training_seed`, `channel`, `c`, and `snr_db`; do not alter canonical values or result identities.

## Acknowledgements

Base code adapted from [Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch).
