# MIL Bench

This repo provides multiple slide-level models for whole-slide image (WSI) analysis, including classic MIL architectures and pre-trained slide encoders that operate on patch features.

## Models

**MIL models**
- abmil
- gated_abmil
- transmil
- amdmil
- clam_sb
- clam_mb
- dsmil
- 2dmamba
- retmil
- wikg
- ilramil

**Pre-trained slide encoders**
- titan
- prism
- gigapath
- madeleine
- chief
- feather_uni_v1
- feather_uni_v2
- feather_conch_v1_5

## Usage

Run the demo to verify all models:
```bash
python demo_test_model.py
```

Create a model in Python:
```python
from slide_encoder_models.model_registry import create_slide_encoder, list_models

print(list_models())
model = create_slide_encoder("abmil", dim_in=1024, dim_hidden=512, num_classes=2)
```

## Benchmark (MIL downstream tasks)

### Task definitions

- **Classification tasks**: `downstream_task_jsons/`
- **Survival tasks**: `surv_downstream_task_jsons/`

### Run all benchmarks

Edit `run_benchmark_all.sh` first:

- **DATA root**: set `DATA_ROOT="YOUR_DATA_ROOT"` to your dataset root.
- **Models**: set `PFM_NAMES=(...)`, `SLIDE_NAME=...`, `DTYPE=...`
- **Paths** (usually no change): `JSON_DIR=./downstream_task_jsons`, `SURV_JSON_DIR=./surv_downstream_task_jsons`, `JOB_DIR=./results`

Then run:

```bash
bash run_benchmark_all.sh
```

### What gets executed

- **Single-split classification**: per-dataset, 5 seeds (`run_five_seeds`).
- **10-fold CV classification**: `cptac_*` fold tasks, 10 folds × 5 seeds per fold.
- **Survival prediction (OS)**: per-cohort folds × 5 seeds (`run_five_seeds_surv`).

### Outputs

Metrics are written under `results/` following the script arguments (dataset / pfm_name / slide_name / seed / benchmark), e.g.:

- `results/<dataset>/<pfm_name>/<slide_name>/<seed>/benchmark/all_test_metrics.json`
- For fold tasks: `results/<dataset>/<task>/fold<k>/<pfm_name>/<slide_name>/<seed>/benchmark/all_test_metrics.json`

---

*Jiawen Li, <jw-li24@mails.tsinghua.edu.cn>*