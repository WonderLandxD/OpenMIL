#!/usr/bin/env bash
set -euo pipefail

# ===== Fixed settings (edit these two) =====
# PFM_NAMES=(conch_v1 conch_v1_5 ctranspath gpfm h_optimus_0 pathorchestra plip prov_gigapath uni_v1 virchow_1 virchow_2) 
PFM_NAMES=(conch_v1 gpfm h_optimus_0 pathorchestra plip prov_gigapath uni_v1 virchow_2)
 # Add more: ("conch_v1_5" "conch_v2" "other_model")
SLIDE_NAME="abmil"
DTYPE="fp32"

# ===== Paths =====
PROJECT_DIR="./"
BENCHMARK_PY="${PROJECT_DIR}/benchmark.py"
JSON_DIR="${PROJECT_DIR}/small_medium_large_huge_data_jsons"
JOB_DIR="${PROJECT_DIR}/results"

# ===== Data roots =====
DATA_ROOT_A="${PROJECT_DIR}/mini_trident_datasets"
DATA_ROOT_B="${PROJECT_DIR}/mini_trident_datasets"

run_one() {
  local data_dir="$1"
  local json_path="$2"
  local seed="$3"

  python "${BENCHMARK_PY}" \
    --data_dir "${data_dir}" \
    --json_path "${json_path}" \
    --pfm_name "${PFM_NAME}" \
    --slide_name "${SLIDE_NAME}" \
    --job_dir "${JOB_DIR}" \
    --seed "${seed}" \
    --gpu_id 6 \
    --dtype "${DTYPE}" \
    --best_metrics "bal_accuracy" \
    --epochs 50 \
    --batch_size 1 \
    --num_workers 4 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_accum_steps 1 \
    --early_stop_patience 5
}

run_five_seeds() {
  local data_dir="$1"
  local json_path="$2"

  run_one "${data_dir}" "${json_path}" 2077
  run_one "${data_dir}" "${json_path}" 2078
  run_one "${data_dir}" "${json_path}" 2079
  run_one "${data_dir}" "${json_path}" 2080
  run_one "${data_dir}" "${json_path}" 2081
}

# ===== Run each PFM in turn =====
for PFM_NAME in "${PFM_NAMES[@]}"; do
  echo "===== PFM: ${PFM_NAME} ====="
  # ===== bracs / cptac / ebrains -> DATA_ROOT_A =====
  # run_five_seeds "${DATA_ROOT_A}/bracs" "${JSON_DIR}/bracs_coarse.json"
  # run_five_seeds "${DATA_ROOT_A}/bracs" "${JSON_DIR}/bracs_fine.json"
  run_five_seeds "${DATA_ROOT_A}/cptac_organ" "${JSON_DIR}/cptac_organ.json"
  run_five_seeds "${DATA_ROOT_A}/ebrains" "${JSON_DIR}/ebrains_coarse.json"
  run_five_seeds "${DATA_ROOT_A}/ebrains" "${JSON_DIR}/ebrains_fine.json"
  run_five_seeds "${DATA_ROOT_A}/ebrains" "${JSON_DIR}/ebrains_idh.json"

  # # ===== panda / tissuenet / camelyon -> DATA_ROOT_B =====
  # run_five_seeds "${DATA_ROOT_B}/panda" "${JSON_DIR}/panda_grading.json"
  # run_five_seeds "${DATA_ROOT_B}/panda" "${JSON_DIR}/panda_screen.json"
  # run_five_seeds "${DATA_ROOT_B}/tissuenet" "${JSON_DIR}/tissuenet.json"
  # run_five_seeds "${DATA_ROOT_B}/camelyon" "${JSON_DIR}/camelyon_coarse.json"
  # run_five_seeds "${DATA_ROOT_B}/camelyon" "${JSON_DIR}/camelyon_fine.json"
done
