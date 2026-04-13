#!/usr/bin/env bash
set -euo pipefail

# ===== Fixed settings (edit these two) =====
# PFM_NAMES=(conch_v1 conch_v1_5 ctranspath gpfm h_optimus_0 pathorchestra plip prov_gigapath uni_v1 virchow_1 virchow_2) 
# PFM_NAMES=(conch_v1 gpfm h_optimus_0 pathorchestra plip prov_gigapath uni_v1 virchow_2)
PFM_NAMES=(uni_v2)
 # Add more: ("conch_v1_5" "conch_v2" "other_model")
SLIDE_NAME="abmil"
DTYPE="fp32"

# ===== Paths =====
PROJECT_DIR="."
BENCHMARK_PY="${PROJECT_DIR}/benchmark.py"
BENCHMARK_SURV_PY="${PROJECT_DIR}/benchmark_surv.py"
JSON_DIR="${PROJECT_DIR}/downstream_task_jsons"
SURV_JSON_DIR="${PROJECT_DIR}/surv_downstream_task_jsons"
JOB_DIR="${PROJECT_DIR}/results"

# ===== Data roots =====
DATA_ROOT="YOUR_DATA_ROOT"  # NOTE: edit this

run_one() {
  local data_dir="$1"
  local json_path="$2"
  local seed="$3"
  local dataset_name="${4:-}"

  local ds
  if [[ -n "${dataset_name}" ]]; then
    ds="${dataset_name}"
  else
    ds="$(basename "${json_path}" .json)"
  fi
  local metrics_file="${JOB_DIR}/${ds}/${PFM_NAME}/${SLIDE_NAME}/${seed}/benchmark/all_test_metrics.json"
  if [[ -f "${metrics_file}" ]]; then
    echo "[skip] already done: ${metrics_file}"
    return 0
  fi

  local ds_name_arg=""
  if [[ -n "${dataset_name}" ]]; then
    ds_name_arg="--dataset_name ${dataset_name}"
  fi

  python "${BENCHMARK_PY}" \
    --data_dir "${data_dir}" \
    --json_path "${json_path}" \
    --pfm_name "${PFM_NAME}" \
    --slide_name "${SLIDE_NAME}" \
    --job_dir "${JOB_DIR}" \
    --seed "${seed}" \
    --gpu_id 4 \
    --dtype "${DTYPE}" \
    --best_metrics "bal_accuracy" \
    --epochs 50 \
    --batch_size 1 \
    --num_workers 4 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_accum_steps 1 \
    --early_stop_patience 5 \
    ${ds_name_arg}
}

run_five_seeds() {
  local data_dir="$1"
  local json_path="$2"
  local dataset_name="${3:-}"

  run_one "${data_dir}" "${json_path}" 2077 "${dataset_name}"
  run_one "${data_dir}" "${json_path}" 2078 "${dataset_name}"
  run_one "${data_dir}" "${json_path}" 2079 "${dataset_name}"
  run_one "${data_dir}" "${json_path}" 2080 "${dataset_name}"
  run_one "${data_dir}" "${json_path}" 2081 "${dataset_name}"
}

run_one_surv() {
  local data_dir="$1"
  local json_path="$2"
  local seed="$3"
  local dataset_name="${4:-}"

  local ds
  if [[ -n "${dataset_name}" ]]; then
    ds="${dataset_name}"
  else
    ds="$(basename "${json_path}" .json)"
  fi
  local metrics_file="${JOB_DIR}/${ds}/${PFM_NAME}/${SLIDE_NAME}/${seed}/benchmark/all_test_metrics.json"
  if [[ -f "${metrics_file}" ]]; then
    echo "[skip] already done: ${metrics_file}"
    return 0
  fi

  local ds_name_arg=""
  if [[ -n "${dataset_name}" ]]; then
    ds_name_arg="--dataset_name ${dataset_name}"
  fi

  python "${BENCHMARK_SURV_PY}" \
    --data_dir "${data_dir}" \
    --json_path "${json_path}" \
    --pfm_name "${PFM_NAME}" \
    --slide_name "${SLIDE_NAME}" \
    --job_dir "${JOB_DIR}" \
    --seed "${seed}" \
    --gpu_id 4 \
    --dtype "${DTYPE}" \
    --best_metrics "c_index" \
    --epochs 50 \
    --batch_size 1 \
    --num_workers 4 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_accum_steps 1 \
    --early_stop_patience 5 \
    ${ds_name_arg}
}

run_five_seeds_surv() {
  local data_dir="$1"
  local json_path="$2"
  local dataset_name="${3:-}"

  run_one_surv "${data_dir}" "${json_path}" 2077 "${dataset_name}"
  run_one_surv "${data_dir}" "${json_path}" 2078 "${dataset_name}"
  run_one_surv "${data_dir}" "${json_path}" 2079 "${dataset_name}"
  run_one_surv "${data_dir}" "${json_path}" 2080 "${dataset_name}"
  run_one_surv "${data_dir}" "${json_path}" 2081 "${dataset_name}"
}

# ===== Run each PFM in turn =====
for PFM_NAME in "${PFM_NAMES[@]}"; do
  echo "===== PFM: ${PFM_NAME} ====="
  # ===== Classification tasks (single split, 5 seeds) =====

  # --- bcnb ---
  run_five_seeds "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_er.json"
  run_five_seeds "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_pr.json"
  run_five_seeds "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_her2.json"

  # --- bracs ---
  run_five_seeds "${DATA_ROOT}/bracs" "${JSON_DIR}/bracs_coarse.json"
  run_five_seeds "${DATA_ROOT}/bracs" "${JSON_DIR}/bracs_fine.json"

  # --- cptac_organ (non-fold tasks) ---
  run_five_seeds "${DATA_ROOT}/cptac_organ" "${JSON_DIR}/cptac_pancancer.json"
  run_five_seeds "${DATA_ROOT}/cptac_organ" "${JSON_DIR}/cptac_nsclc.json"

  # --- kidrare ---
  run_five_seeds "${DATA_ROOT}/kidrare" "${JSON_DIR}/kidrare_fine.json"
  run_five_seeds "${DATA_ROOT}/kidrare" "${JSON_DIR}/kidrare_coarse.json"

  # --- ebrains ---
  run_five_seeds "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_coarse.json"
  run_five_seeds "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_fine.json"
  run_five_seeds "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_idh.json"

  # --- mut_het_rcc ---
  run_five_seeds "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_bap1_mutation.json"
  run_five_seeds "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_pbrm1_mutation.json"
  run_five_seeds "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_setd2_mutation.json"

  # --- rcc_dhmc ---
  run_five_seeds "${DATA_ROOT}/rcc_dhmc" "${JSON_DIR}/dhmc_rcc_subtyping.json"

  # --- panda ---
  run_five_seeds "${DATA_ROOT}/panda" "${JSON_DIR}/panda_grading.json"
  run_five_seeds "${DATA_ROOT}/panda" "${JSON_DIR}/panda_screen.json"

  # --- tissuenet ---
  run_five_seeds "${DATA_ROOT}/tissuenet" "${JSON_DIR}/tissuenet_screen.json"

  # --- camelyon ---
  run_five_seeds "${DATA_ROOT}/camelyon" "${JSON_DIR}/camelyon_coarse.json"
  run_five_seeds "${DATA_ROOT}/camelyon" "${JSON_DIR}/camelyon_fine.json"
done

# ===== Classification tasks (10-fold CV, 5 seeds per fold) =====

# ===== cptac_ccrcc tasks (10-fold cross-validation for small sample tasks) =====
# Tasks: BAP1_mutation, Immune_class, PBRM1_mutation, VHL_mutation (excluding OS)
# Each fold has train:val:test split maintained at ~0.64:0.16:0.2 via stratification
# Results saved to: results/cptac_ccrcc/{task_name}/fold{fold_idx}/{pfm_name}/abmil/{seed}/benchmark/
# Usage: Uncomment the lines below to run 10-fold CV (5 seeds per fold) for cptac_ccrcc tasks
#
for FOLD in $(seq 0 9); do
  echo "=== cptac_ccrcc Fold ${FOLD} ==="
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_ccrcc/BAP1_mutation_fold${FOLD}.json" \
    "cptac_ccrcc/BAP1_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_ccrcc/Immune_class_fold${FOLD}.json" \
    "cptac_ccrcc/Immune_class/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_ccrcc/PBRM1_mutation_fold${FOLD}.json" \
    "cptac_ccrcc/PBRM1_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_ccrcc/VHL_mutation_fold${FOLD}.json" \
    "cptac_ccrcc/VHL_mutation/fold${FOLD}"
done

# ===== cptac_brca (10-fold cross-validation for small sample tasks) =====
  # Note: cptac_brca tasks use pre-defined 10 folds with train:val:test = 0.64:0.16:0.2 via label-stratified split
  # Sample size is small (~100), so 10 folds with 5 seeds each provides more robust evaluation
  # Results saved to: results/cptac_brca/{task_name}/fold{fold_idx}/{pfm_name}/abmil/{seed}/benchmark/
  # Uncomment below to run all 3 tasks (Immune_class, PIK3CA_mutation, TP53_mutation) across 10 folds
  #
  for FOLD in $(seq 0 9); do
    echo "  --- cptac_brca Fold ${FOLD} ---"
    run_five_seeds "${DATA_ROOT}/cptac_organ" \
      "${JSON_DIR}/cptac_brca/Immune_class_fold${FOLD}.json" \
      "cptac_brca/Immune_class/fold${FOLD}"
    run_five_seeds "${DATA_ROOT}/cptac_organ" \
      "${JSON_DIR}/cptac_brca/PIK3CA_mutation_fold${FOLD}.json" \
      "cptac_brca/PIK3CA_mutation/fold${FOLD}"
    run_five_seeds "${DATA_ROOT}/cptac_organ" \
      "${JSON_DIR}/cptac_brca/TP53_mutation_fold${FOLD}.json" \
      "cptac_brca/TP53_mutation/fold${FOLD}"
  done

# ===== cptac_coad tasks (10-fold cross-validation for small sample tasks) =====
for FOLD in $(seq 0 9); do
  echo "=== cptac_coad Fold ${FOLD} ==="
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/ACVR2A_mutation_fold${FOLD}.json" \
    "cptac_coad/ACVR2A_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/APC_mutation_fold${FOLD}.json" \
    "cptac_coad/APC_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/ARID1A_mutation_fold${FOLD}.json" \
    "cptac_coad/ARID1A_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/Immune_class_fold${FOLD}.json" \
    "cptac_coad/Immune_class/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/KRAS_mutation_fold${FOLD}.json" \
    "cptac_coad/KRAS_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/MSI_H_fold${FOLD}.json" \
    "cptac_coad/MSI_H/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/PIK3CA_mutation_fold${FOLD}.json" \
    "cptac_coad/PIK3CA_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/SETD1B_mutation_fold${FOLD}.json" \
    "cptac_coad/SETD1B_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_coad/TP53_mutation_fold${FOLD}.json" \
    "cptac_coad/TP53_mutation/fold${FOLD}"
done

# ===== cptac_gbm tasks (10-fold cross-validation for small sample tasks) =====
for FOLD in $(seq 0 9); do
  echo "=== cptac_gbm Fold ${FOLD} ==="
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_gbm/EGFR_mutation_fold${FOLD}.json" \
    "cptac_gbm/EGFR_mutation/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_gbm/Immune_class_fold${FOLD}.json" \
    "cptac_gbm/Immune_class/fold${FOLD}"
  run_five_seeds "${DATA_ROOT}/cptac_organ" \
    "${JSON_DIR}/cptac_gbm/TP53_mutation_fold${FOLD}.json" \
    "cptac_gbm/TP53_mutation/fold${FOLD}"
done

# ===== Survival prediction (5 folds × 5 seeds) =====
# Path in json will be patched to use patch_features/<PFM_NAME>/... before running.
# Results saved to: results/{dataset}/OS/fold{fold_idx}/{pfm_name}/abmil/{seed}/benchmark/
#
for fold_idx in {0..4}; do
  # --- cptac_ccrcc OS ---
  echo "=== surv cptac_ccrcc OS fold ${fold_idx} ==="
  run_five_seeds_surv "${DATA_ROOT}/cptac_organ" \
    "${SURV_JSON_DIR}/cptac_ccrcc/cptac_ccrcc_os_fold${fold_idx}.json" \
    "cptac_ccrcc/OS/fold${fold_idx}"
done

for fold_idx in {0..4}; do
  # --- cptac_hnsc OS ---
  echo "=== surv cptac_hnsc OS fold ${fold_idx} ==="
  run_five_seeds_surv "${DATA_ROOT}/cptac_organ" \
    "${SURV_JSON_DIR}/cptac_hnsc/cptac_hnsc_os_fold${fold_idx}.json" \
    "cptac_hnsc/OS/fold${fold_idx}"
done

for fold_idx in {0..2}; do
  # --- cptac_luad OS ---
  echo "=== surv cptac_luad OS fold ${fold_idx} ==="
  run_five_seeds_surv "${DATA_ROOT}/cptac_organ" \
    "${SURV_JSON_DIR}/cptac_luad/cptac_luad_os_fold${fold_idx}.json" \
    "cptac_luad/OS/fold${fold_idx}"
done

for fold_idx in {0..4}; do
  # --- cptac_pda OS ---
  echo "=== surv cptac_pda OS fold ${fold_idx} ==="
  run_five_seeds_surv "${DATA_ROOT}/cptac_organ" \
    "${SURV_JSON_DIR}/cptac_pda/cptac_pda_os_fold${fold_idx}.json" \
    "cptac_pda/OS/fold${fold_idx}"
done

