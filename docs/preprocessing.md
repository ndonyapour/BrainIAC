# MRI Preprocessing

## Overview

BrainIAC expects structural MRI volumes that are bias-field corrected, rigidly registered to a template, and skull-stripped. This repository provides a SimpleITK + HD-BET pipeline that performs those steps.

The pipeline:

1. **N4 bias field correction** (SimpleITK)
2. **Rigid registration** to an atlas / template image (1 mm isotropic resampling of the template)
3. **Brain extraction / skull stripping** with [HD-BET](https://github.com/MIC-DKFZ/HD-BET)

Two entry points are available:

| Script | Location | Notes |
|--------|----------|--------|
| `mri_preprocess_3d_simple.py` | `src/preprocessing/` (also mirrored at repo root) | Sequential registration |
| `mri_preprocess_3d_simple_parallel.py` | repo root | Parallel registration across CPU workers |

Prefer the parallel script for larger cohorts.

## Installation

### Environment

Use the project conda environment (Python 3.9+, CUDA recommended for HD-BET):

```bash
conda create -n brainiac python=3.9
conda activate brainiac
# Install PyTorch for your CUDA (example for CUDA 11.8)
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Preprocessing-specific dependencies already covered by `requirements.txt` include:

- `SimpleITK`
- `torch` (via your PyTorch install; needed for HD-BET device selection)
- `tqdm`
- `nibabel` / related stack used elsewhere in the repo

### HD-BET package path

The scripts import HD-BET as:

```python
from HD_BET.hd_bet import hd_bet
```

The package lives under `src/preprocessing/HD_BET/`. Run the script from `src/preprocessing/`, or add that directory to `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/preprocessing"
```

HD-BET model weights are expected in `src/preprocessing/hd-bet_params/` (e.g. `0.model`). If missing, HD-BET can download them from Zenodo on first run (`https://zenodo.org/record/2540695`).

### Atlas template

Provide a NIfTI atlas via `--temp_img`. The repo includes templates under:

```text
src/preprocessing/atlases/
├── temp_head.nii.gz
├── nihpd_asym_13.0-18.5_t1w.nii
└── nihpd_asym_13.0-18.5_t2w.nii
```

For the default BrainIAC preprocessing path, use `temp_head.nii.gz` (also referenced in `src/quickstart.ipynb`).

## Input requirements

- `--input_dir` may be a flat folder or a nested layout (e.g. `patient_id/*.nii`). The parallel script searches recursively for `.nii` only (not `.nii.gz`).
- Symlinked NIfTI files are supported (targets are followed when reading).
- Files whose names contain `_mask` are skipped.
- If your data are DICOM, convert to NIfTI first:

```bash
cd src
python ./preprocessing/dicomtonifti_2.py -i ./data/dicom -o ./data/unprocessed
```

## Run preprocessing

### Sequential (canonical path from quickstart)

From `src/`:

```bash
cd src
python ./preprocessing/mri_preprocess_3d_simple.py \
  --temp_img ./preprocessing/atlases/temp_head.nii.gz \
  --input_dir ./data/sample/unprocessed \
  --output_dir ./data/sample/processed
```

### Parallel (recommended for many subjects)

From the repo root (with `PYTHONPATH` set so `HD_BET` imports resolve):

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/preprocessing"

python mri_preprocess_3d_simple_parallel.py \
  --temp_img ./src/preprocessing/atlases/temp_head.nii.gz \
  --input_dir /path/to/unprocessed \
  --output_dir /path/to/processed \
  --num_workers 32
```

Example for ADNI-style patient folders:

```bash
python mri_preprocess_3d_simple_parallel.py \
  --temp_img ./src/preprocessing/atlases/temp_head.nii.gz \
  --input_dir /polus2/donyapourn2/data/ADNI_dataset/selected/adni1 \
  --output_dir /polus2/donyapourn2/data/ADNI_dataset/brainaic/adni1_processed \
  --num_workers 32
```

To run ADNI2 / ADNI3 / ADNIGO in sequence on a remote host (SSH):

```bash
cd /polus2/donyapourn2/projects/BrainIAC
bash scripts/run_adni_preprocess.sh

# or detach with nohup
mkdir -p logs
nohup bash scripts/run_adni_preprocess.sh > logs/adni_preprocess.out 2>&1 &
tail -f logs/adni_preprocess.out
```

The script uses the `brainiac` conda Python directly (no need to `conda activate` first). Outputs go to `/polus2/donyapourn2/data/ADNI_dataset/brainaic/<cohort>/{images,masks}/`.

The parallel script uses `--num_workers` CPU processes for registration (default: `cpu_count - 2`), then runs HD-BET once on the registered batch. CUDA device `0` is used when available; otherwise CPU. On large hosts, prefer ~24–32 workers rather than the default.

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--temp_img` | yes | Path to atlas / template NIfTI |
| `--input_dir` | yes | Root directory of input `.nii` images (searched recursively) |
| `--output_dir` | yes | Directory for skull-stripped outputs |
| `--num_workers` | no | CPU workers for registration (default: `cpu_count - 2`) |

## Outputs

- Intermediate registered volumes are written as `{ID}_0000.nii.gz` (HD-BET naming), then removed after brain extraction for the parallel script.
- Final skull-stripped images (and HD-BET masks when enabled) are saved under `--output_dir`.
- These preprocessed NIfTI volumes are the expected inputs for BrainIAC feature extraction and downstream tasks (typically resized to 96×96×96 at train/inference time).

## Next steps

- Feature extraction: see `src/quickstart.ipynb` and `src/get_brainiac_features.py`
- Downstream tasks: see [docs/downstream_tasks/](./downstream_tasks/)
