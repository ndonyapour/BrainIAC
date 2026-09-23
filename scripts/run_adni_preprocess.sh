#!/usr/bin/env bash
# Preprocess ADNI cohorts with mri_preprocess_3d_simple_parallel.py
#
# Writes:
#   <OUTPUT_ROOT>/<cohort>/images/   # skull-stripped MRI
#   <OUTPUT_ROOT>/<cohort>/masks/    # HD-BET masks
#
# Usage (on the compute host):
#   bash scripts/run_adni_preprocess.sh
#
# Background:
#   mkdir -p logs
#   nohup bash scripts/run_adni_preprocess.sh > logs/adni_preprocess.out 2>&1 &
#   tail -f logs/adni_preprocess.out
#
# Overrides:
#   NUM_WORKERS=32 COHORTS="adni1 adni2 adni3 adnigo" bash scripts/run_adni_preprocess.sh
#   PYTHON=/path/to/python bash scripts/run_adni_preprocess.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-brainiac}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/polus2/donyapourn2/mambaforge-pypy3/envs/${CONDA_ENV}/bin}"
PYTHON="${PYTHON:-${CONDA_ENV_BIN}/python}"

NUM_WORKERS="${NUM_WORKERS:-32}"
DATA_ROOT="${DATA_ROOT:-/polus2/donyapourn2/data/ADNI_dataset}"
INPUT_ROOT="${INPUT_ROOT:-${DATA_ROOT}/selected}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/brainaic}"
TEMP_IMG="${TEMP_IMG:-${REPO_ROOT}/src/preprocessing/atlases/temp_head.nii.gz}"
COHORTS="${COHORTS:-adni1 adni2 adni3 adnigo}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}/src/preprocessing${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at: ${PYTHON}" >&2
  echo "Set PYTHON=/path/to/brainiac/bin/python or create the conda env." >&2
  exit 1
fi

if [[ ! -f "${TEMP_IMG}" ]]; then
  echo "ERROR: template not found: ${TEMP_IMG}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/mri_preprocess_3d_simple_parallel.py" ]]; then
  echo "ERROR: missing ${REPO_ROOT}/mri_preprocess_3d_simple_parallel.py" >&2
  exit 1
fi

echo "Checking imports..."
"${PYTHON}" - <<'PY'
import SimpleITK  # noqa: F401
import torch
from HD_BET.hd_bet import hd_bet  # noqa: F401
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)}")
PY

echo "============================================================"
echo "Repo:         ${REPO_ROOT}"
echo "Python:       ${PYTHON}"
echo "Template:     ${TEMP_IMG}"
echo "Workers:      ${NUM_WORKERS}"
echo "GPU:          CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Cohorts:      ${COHORTS}"
echo "Input root:   ${INPUT_ROOT}"
echo "Output root:  ${OUTPUT_ROOT}/<cohort>/{images,masks}"
echo "============================================================"

failed=0
for cohort in ${COHORTS}; do
  input_dir="${INPUT_ROOT}/${cohort}"
  output_dir="${OUTPUT_ROOT}/${cohort}"
  log_file="${LOG_DIR}/preprocess_${cohort}_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -d "${input_dir}" ]]; then
    echo "WARNING: skipping missing input dir: ${input_dir}" >&2
    continue
  fi

  mkdir -p "${output_dir}/images" "${output_dir}/masks"

  echo
  echo ">>> [${cohort}] START $(date --iso-8601=seconds 2>/dev/null || date -Iseconds 2>/dev/null || date)"
  echo "    input:  ${input_dir}"
  echo "    output: ${output_dir}/images"
  echo "            ${output_dir}/masks"
  echo "    log:    ${log_file}"

  if ! "${PYTHON}" -u mri_preprocess_3d_simple_parallel.py \
      --temp_img "${TEMP_IMG}" \
      --input_dir "${input_dir}" \
      --output_dir "${output_dir}" \
      --num_workers "${NUM_WORKERS}" \
      2>&1 | tee "${log_file}"; then
    echo "ERROR: [${cohort}] preprocessing failed (see ${log_file})" >&2
    failed=1
    continue
  fi

  n_img="$(find "${output_dir}/images" -maxdepth 1 -type f -name '*.nii.gz' ! -name '*_mask*' | wc -l)"
  n_mask="$(find "${output_dir}/masks" -maxdepth 1 -type f -name '*_mask*.nii.gz' | wc -l)"
  echo ">>> [${cohort}] DONE images=${n_img} masks=${n_mask}"
done

echo
if [[ "${failed}" -ne 0 ]]; then
  echo "Finished with errors." >&2
  exit 1
fi
echo "All requested cohorts finished successfully."
