# BrainIAC preprocessing & features

Scripts in this folder support ADNI MRI preprocessing and BrainIAC feature extraction
using [BrainIAC (`parallelization` branch)](https://github.com/ndonyapour/BrainIAC/tree/parallelization).

Use a **conda** environment for BrainIAC / ANTs-style preprocessing and feature extraction
(see also the root `README.md`). Use **uv** for the rest of this re-identification repo.

## Install (conda)

```bash
# Clone the parallelization branch
git clone -b parallelization https://github.com/ndonyapour/BrainIAC.git
cd BrainIAC

# Create and activate env
conda create -n brainiac python=3.9
conda activate brainiac

# Install PyTorch for your CUDA version (example: CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining deps
pip install -r requirements.txt
```

### Checkpoint

Download `BrainIAC.ckpt` (see the BrainIAC README Dropbox link) and place it under:

```bash
mkdir -p src/checkpoints
# cp ~/Downloads/BrainIAC.ckpt src/checkpoints/
```

Or use the local ADNI checkpoint, for example:

`/polus2/donyapourn2/data/ADNI_dataset/BrainIAC_checkpoints/BrainIAC.ckpt`

## Scripts in this folder

| Script | Purpose |
|--------|---------|
| `create_adni_feature_csvs.py` | Build per-cohort `pat_id,label` CSVs from `images/` |
| `run_adni_preprocess.sh` | Run parallel MRI preprocess (skull-strip + masks) |
| `run_adni_features.sh` | Extract BrainIAC features for ADNI cohorts |

These shell wrappers expect a BrainIAC checkout layout (`src/`, preprocess scripts, etc.).
Copy or symlink them into your BrainIAC clone, or set paths via the env vars below.

### Generate feature CSVs

```bash
conda activate brainiac
python create_adni_feature_csvs.py \
  --root /polus2/donyapourn2/data/ADNI_dataset/BrainIAC_processed
```

Writes e.g. `BrainIAC_processed/adni1/adni1.csv` with columns `pat_id,label`.

### Preprocess

```bash
conda activate brainiac
bash run_adni_preprocess.sh
```

### Extract features

```bash
conda activate brainiac
bash run_adni_features.sh

# background
mkdir -p logs
nohup bash run_adni_features.sh > logs/adni_features.out 2>&1 &
```

Defaults (override with env vars):

| Variable | Default |
|----------|---------|
| `CHECKPOINT` | `/polus2/donyapourn2/data/ADNI_dataset/BrainIAC_checkpoints/BrainIAC.ckpt` |
| `PROCESSED_ROOT` | `/polus2/donyapourn2/data/ADNI_dataset/BrainIAC_processed` |
| `FEATURES_ROOT` | `/polus2/donyapourn2/data/ADNI_dataset/BrainIAC_features` |
| `COHORTS` | `adni1 adni2 adni3 adnigo` |

```bash
COHORTS="adni1" BATCH_SIZE=2 bash run_adni_features.sh
```

## References

- BrainIAC repo: https://github.com/ndonyapour/BrainIAC/tree/parallelization
- Upstream / paper details: see that repository’s README
