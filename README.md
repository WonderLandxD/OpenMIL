# OpenMIL

This repo provides:

- Open available slide-level models for whole-slide image (WSI) analysis, including classic MIL architectures and pre-trained slide encoders that operate on patch features. 

- Open available dataset `.json` lists for WSI analysis, including classification and survival prediction datasets. You can find the lists in `downstream_task_jsons/` and `surv_downstream_task_jsons/`.

- Easy-to-use benchmark scripts for downstream tasks (classification and survival prediction).


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
- care
- feather_uni_v1
- feather_uni_v2
- feather_conch_v1_5
- tangle_v2

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

## Benchmark (pre-trained slide encoders)

This repo also provides lightweight benchmarks for **pre-trained slide encoders** (WSI encoders) that take **patch features** as input.

### KNN benchmark

Script: `knn_benchmark.py` + `run_knn_benchmark.sh`

- **Input**: `*.pth` patch-feature files referenced by `downstream_task_jsons/*.json` (with `<PFM_NAME>` placeholder in the path).
- **Model**: `--slide-encoder` (pre-trained slide encoders only).
- **Output**: `knn_results/<pfm_name>/<dataset>/<slide_encoder>/k{k}_{metric}/`

Run:

```bash
# 1) Edit DATA_ROOT in the script
bash run_knn_benchmark.sh
```

### Linear probe benchmark

Script: `linear_probe_benchmark.py` + `run_linear_benchmark.sh`

- **Protocol**: StratifiedKFold on train split; early-stop on val balanced-acc; evaluate on held-out test.
- **Output**: `linear_probe_results/<pfm_name>/<dataset>/<slide_encoder>/lp_f{num_folds}_pat{patience}/`

Run:

```bash
# 1) Edit DATA_ROOT in the script
bash run_linear_benchmark.sh
```

### Patch encoder is fixed for each slide encoder

For these pre-trained slide encoders, the required patch encoder is fixed (do not mix), and the benchmark scripts enforce this mapping:

- **titan**: `conch_v1_5`
- **care**: `conch_v1_5`
- **madeleine**: `conch_v1`
- **chief**: `ctranspath`
- **prism**: `virchow_1`
- **gigapath**: `prov_gigapath`
- **feather_uni_v1**: `uni_v1`
- **feather_uni_v2**: `uni_v2`
- **feather_conch_v1_5**: `conch_v1_5`
- **tangle_v2**: `uni_v1`

*Jiawen Li, <jw-li24@mails.tsinghua.edu.cn>*