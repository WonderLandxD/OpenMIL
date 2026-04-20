#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PROJECT_DIR="."
JSON_DIR="${PROJECT_DIR}/downstream_task_jsons"
KNN_PY="${PROJECT_DIR}/knn_benchmark.py"

# ===== Edit =====
SLIDE_ENCODERS=(titan)
# SLIDE_ENCODERS=(titan prism gigapath madeleine chief feather_uni_v2)

DATA_ROOT="YOUR_DATA_ROOT"
SAVE_ROOT="${PROJECT_DIR}/knn_results"
DEVICE="cuda:0"
DTYPE="fp32"
K=5
METRIC="cosine"
BACKEND="auto"
TRAIN_SPLIT="train"
TEST_SPLIT="test"

run_knn() {
  local data_dir="$1"
  local json_path="$2"
  local enc
  for enc in "${SLIDE_ENCODERS[@]}"; do
    local PFM_NAME
    case "${enc}" in
      titan|feather_conch_v1_5) PFM_NAME="conch_v1_5" ;;
      care) PFM_NAME="conch_v1_5" ;;
      tangle_v2) PFM_NAME="uni_v1" ;;
      madeleine) PFM_NAME="conch_v1" ;;
      chief) PFM_NAME="ctranspath" ;;
      prism) PFM_NAME="virchow_1" ;;
      gigapath) PFM_NAME="prov_gigapath" ;;
      feather_uni_v1) PFM_NAME="uni_v1" ;;
      feather_uni_v2) PFM_NAME="uni_v2" ;;
      *) echo "[error] unknown slide encoder: ${enc}" >&2; exit 1 ;;
    esac

    local metrics="${SAVE_ROOT}/${PFM_NAME}/$(basename "${json_path}" .json)/${enc}/k${K}_${METRIC}/metrics.json"
    if [[ -f "${metrics}" ]]; then
      echo "[skip] ${metrics}"
      continue
    fi

    echo "[run] PFM=${PFM_NAME} enc=${enc} json=${json_path}"
    python "${KNN_PY}" \
      --data-dir "${data_dir}" \
      --json-path "${json_path}" \
      --pfm-name "${PFM_NAME}" \
      --save-root "${SAVE_ROOT}" \
      --slide-encoder "${enc}" \
      --train-split "${TRAIN_SPLIT}" \
      --test-split "${TEST_SPLIT}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --k "${K}" \
      --metric "${METRIC}" \
      --backend "${BACKEND}" \
      --reuse-features
  done
}

echo "===== KNN benchmark ====="

  # --- bcnb ---
  run_knn "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_er.json"
  run_knn "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_pr.json"
  run_knn "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_her2.json"

  # --- bracs ---
  run_knn "${DATA_ROOT}/bracs" "${JSON_DIR}/bracs_coarse.json"
  run_knn "${DATA_ROOT}/bracs" "${JSON_DIR}/bracs_fine.json"

  # --- cptac_organ ---
  run_knn "${DATA_ROOT}/cptac_organ" "${JSON_DIR}/cptac_pancancer.json"
  run_knn "${DATA_ROOT}/cptac_organ" "${JSON_DIR}/cptac_nsclc.json"

  # --- kidrare ---
  run_knn "${DATA_ROOT}/kidrare" "${JSON_DIR}/kidrare_fine.json"
  run_knn "${DATA_ROOT}/kidrare" "${JSON_DIR}/kidrare_coarse.json"

  # --- ebrains ---
  run_knn "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_coarse.json"
  run_knn "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_fine.json"
  run_knn "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_idh.json"

  # --- mut_het_rcc ---
  run_knn "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_bap1_mutation.json"
  run_knn "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_pbrm1_mutation.json"
  run_knn "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_setd2_mutation.json"

  # --- rcc_dhmc ---
  run_knn "${DATA_ROOT}/rcc_dhmc" "${JSON_DIR}/dhmc_rcc_subtyping.json"

  # --- panda ---
  run_knn "${DATA_ROOT}/panda" "${JSON_DIR}/panda_grading.json"
  run_knn "${DATA_ROOT}/panda" "${JSON_DIR}/panda_screen.json"

  # --- tissuenet ---
  run_knn "${DATA_ROOT}/tissuenet" "${JSON_DIR}/tissuenet_screen.json"

  # --- camelyon ---
  run_knn "${DATA_ROOT}/camelyon" "${JSON_DIR}/camelyon_plus_coarse.json"
  run_knn "${DATA_ROOT}/camelyon" "${JSON_DIR}/camelyon_plus_fine.json"
