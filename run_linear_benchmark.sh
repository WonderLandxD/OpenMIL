#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PROJECT_DIR="."
JSON_DIR="${PROJECT_DIR}/downstream_task_jsons"
LP_PY="${PROJECT_DIR}/linear_probe_benchmark.py"

# ===== Edit =====
SLIDE_ENCODERS=(titan)
# SLIDE_ENCODERS=(titan prism gigapath madeleine chief feather_uni_v2)

DATA_ROOT="YOUR_DATA_ROOT"
SAVE_ROOT="${PROJECT_DIR}/linear_probe_results"
DEVICE="cuda:0"
DTYPE="fp32"
EPOCHS=50
PATIENCE=10
NUM_FOLDS=5
LP_TAG="lp_f${NUM_FOLDS}_pat${PATIENCE}"
LR=1e-2
WEIGHT_DECAY=1e-4
TRAIN_BATCH_SIZE=1024
SEED=42
TRAIN_SPLIT="train"
TEST_SPLIT="test"

run_linear() {
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

    local metrics="${SAVE_ROOT}/${PFM_NAME}/$(basename "${json_path}" .json)/${enc}/${LP_TAG}/metrics.json"
    if [[ -f "${metrics}" ]]; then
      echo "[skip] ${metrics}"
      continue
    fi

    echo "[run] PFM=${PFM_NAME} enc=${enc} json=${json_path}"
    python "${LP_PY}" \
      --data-dir "${data_dir}" \
      --json-path "${json_path}" \
      --pfm-name "${PFM_NAME}" \
      --save-root "${SAVE_ROOT}" \
      --slide-encoder "${enc}" \
      --train-split "${TRAIN_SPLIT}" \
      --test-split "${TEST_SPLIT}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --num-folds "${NUM_FOLDS}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --train-batch-size "${TRAIN_BATCH_SIZE}" \
      --seed "${SEED}" \
      --reuse-features
  done
}

echo "===== Linear probe benchmark ====="

  # --- bcnb ---
  run_linear "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_er.json"
  run_linear "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_pr.json"
  run_linear "${DATA_ROOT}/bcnb" "${JSON_DIR}/bcnb_her2.json"

  # --- bracs ---
  run_linear "${DATA_ROOT}/bracs" "${JSON_DIR}/bracs_coarse.json"
  run_linear "${DATA_ROOT}/bracs" "${JSON_DIR}/bracs_fine.json"

  # --- cptac_organ ---
  run_linear "${DATA_ROOT}/cptac_organ" "${JSON_DIR}/cptac_pancancer.json"
  run_linear "${DATA_ROOT}/cptac_organ" "${JSON_DIR}/cptac_nsclc.json"

  # --- kidrare ---
  run_linear "${DATA_ROOT}/kidrare" "${JSON_DIR}/kidrare_fine.json"
  run_linear "${DATA_ROOT}/kidrare" "${JSON_DIR}/kidrare_coarse.json"

  # --- ebrains ---
  run_linear "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_coarse.json"
  run_linear "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_fine.json"
  run_linear "${DATA_ROOT}/ebrains" "${JSON_DIR}/ebrains_idh.json"

  # --- mut_het_rcc ---
  run_linear "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_bap1_mutation.json"
  run_linear "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_pbrm1_mutation.json"
  run_linear "${DATA_ROOT}/mut_het_rcc" "${JSON_DIR}/mut_het_rcc_setd2_mutation.json"

  # --- rcc_dhmc ---
  run_linear "${DATA_ROOT}/rcc_dhmc" "${JSON_DIR}/dhmc_rcc_subtyping.json"

  # --- panda ---
  run_linear "${DATA_ROOT}/panda" "${JSON_DIR}/panda_grading.json"
  run_linear "${DATA_ROOT}/panda" "${JSON_DIR}/panda_screen.json"

  # --- tissuenet ---
  run_linear "${DATA_ROOT}/tissuenet" "${JSON_DIR}/tissuenet_screen.json"

  # --- camelyon ---
  run_linear "${DATA_ROOT}/camelyon" "${JSON_DIR}/camelyon_plus_coarse.json"
  run_linear "${DATA_ROOT}/camelyon" "${JSON_DIR}/camelyon_plus_fine.json"
