#!/usr/bin/env bash
# Extract BrainIAC features for all ADNI cohorts.
#
# Runs (from src/):
#   python get_brainiac_features.py \
#     --checkpoint .../BrainIAC.ckpt \
#     --input_csv  .../BrainIAC_processed/<cohort>/<cohort>.csv \
#     --output_csv .../BrainIAC_features/<cohort>_features.csv \
#     --root_dir   .../BrainIAC_processed/<cohort>/images
#
# Usage:
#   bash scripts/run_adni_features.sh
#
# Background:
#   mkdir -p logs
#   nohup bash scripts/run_adni_features.sh > logs/adni_features.out 2>&1 &
#   tail -f logs/adni_features.out
#
# Overrides:
#   COHORTS="adni1 adni2" BATCH_SIZE=2 bash scripts/run_adni_features.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
cd "${SRC_DIR}"

CONDA_ENV="${CONDA_ENV:-brainiac}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/polus2/donyapourn2/mambaforge-pypy3/envs/${CONDA_ENV}/bin}"
PYTHON="${PYTHON:-${CONDA_ENV_BIN}/python}"

DATA_ROOT="${DATA_ROOT:-/polus2/donyapourn2/data/ADNI_dataset}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${DATA_ROOT}/BrainIAC_processed}"
FEATURES_ROOT="${FEATURES_ROOT:-${DATA_ROOT}/BrainIAC_features}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/BrainIAC_checkpoints/BrainIAC.ckpt}"
COHORTS="${COHORTS:-adni1 adni2 adni3 adnigo}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "${LOG_DIR}" "${FEATURES_ROOT}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${SRC_DIR}/get_brainiac_features.py" ]]; then
  echo "ERROR: missing ${SRC_DIR}/get_brainiac_features.py" >&2
  exit 1
fi

echo "Checking imports..."
"${PYTHON}" - <<'PY'
import torch
from load_brainiac import load_brainiac  # noqa: F401
from dataset import BrainAgeDataset  # noqa: F401
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)}")
PY

echo "============================================================"
echo "Repo:         ${REPO_ROOT}"
echo "Cwd:          ${SRC_DIR}"
echo "Python:       ${PYTHON}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Processed:    ${PROCESSED_ROOT}"
echo "Features:     ${FEATURES_ROOT}"
echo "Cohorts:      ${COHORTS}"
echo "Batch size:   ${BATCH_SIZE}"
echo "GPU:          CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

failed=0
for cohort in ${COHORTS}; do
  input_csv="${PROCESSED_ROOT}/${cohort}/${cohort}.csv"
  root_dir="${PROCESSED_ROOT}/${cohort}/images"
  output_csv="${FEATURES_ROOT}/${cohort}_features.csv"
  log_file="${LOG_DIR}/features_${cohort}_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -f "${input_csv}" ]]; then
    echo "WARNING: skipping ${cohort}: missing CSV ${input_csv}" >&2
    continue
  fi
  if [[ ! -d "${root_dir}" ]]; then
    echo "WARNING: skipping ${cohort}: missing images dir ${root_dir}" >&2
    continue
  fi

  echo
  echo ">>> [${cohort}] START $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "    csv:    ${input_csv}"
  echo "    images: ${root_dir}"
  echo "    out:    ${output_csv}"
  echo "    log:    ${log_file}"

  if ! "${PYTHON}" -u get_brainiac_features.py \
      --checkpoint "${CHECKPOINT}" \
      --input_csv "${input_csv}" \
      --output_csv "${output_csv}" \
      --root_dir "${root_dir}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      2>&1 | tee "${log_file}"; then
    echo "ERROR: [${cohort}] feature extraction failed (see ${log_file})" >&2
    failed=1
    continue
  fi

  echo ">>> [${cohort}] DONE $(date --iso-8601=seconds 2>/dev/null || date)"
done

echo
if [[ "${failed}" -ne 0 ]]; then
  echo "Finished with errors." >&2
  exit 1
fi
echo "All requested cohorts finished successfully."
