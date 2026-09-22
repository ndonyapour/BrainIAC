# BrainIAC: A foundation model for generalized Brain MRI analysis

<p align="center">
  <img src="pngs/brainiac.jpeg" width="800" alt="BrainIAC_V2 Logo"/>
</p>

## Overview

BrainIAC (Brain Imaging Adaptive Core) is vision based foundation model for generalized structural Brain MRI analysis. This repository provides the BrainIAC and downstream model checkpoints, with training/inference pipeline across all downstream tasks. Checkout the [Paper](https://www.nature.com/articles/s41593-026-02202-6)


## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.0+

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/AIM-KannLab/BrainIAC.git
cd BrainIAC

# Create conda environment
conda create -n brainiac python=3.9
conda activate brainiac

# Install PyTorch for your CUDA version (example: CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Model Checkpoints

Download the official BrainIAC backbone and downstream task weights from Dropbox:

**[Download checkpoints](https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&st=b9cnvwh8&dl=0)**

1. Open the Dropbox folder and download the checkpoint files (at minimum `BrainIAC.ckpt` for feature extraction).
2. Place them under `./src/checkpoints/` (or any path you pass via `--checkpoint`):

```bash
mkdir -p src/checkpoints
# after download, for example:
# cp ~/Downloads/BrainIAC.ckpt src/checkpoints/
```

Typical files in the Dropbox folder include:

| File | Use |
|------|-----|
| `BrainIAC.ckpt` | Foundation backbone (feature extraction) |
| `brainage.ckpt` | Brain age downstream head |
| `idh_model.pt` / `mci_model.pt` / `os_model.pt` / `stroke_model.pt` / `sequenceclassification_model.pt` | Other downstream models |

For feature extraction you only need `BrainIAC.ckpt`.


## Quick Start

See [quickstart.ipynb](./src/quickstart.ipynb) to get started on how to preprocess data, load BrainIAC to extract features, generate and visualize saliency maps. We provide data samples from publicly available [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) [License](https://creativecommons.org/licenses/by/4.0/) (with no modifications to the provided preprocessed images) and the [Pixar](https://openneuro.org/datasets/ds000228/versions/1.1.1)  [License](https://creativecommons.org/public-domain/cc0/) dataset in the [sample_data](src/data/sample/processed/) directory.

For install and CLI usage of the MRI preprocessing scripts (including the parallel pipeline), see [MRI Preprocessing](./docs/preprocessing.md).


## Feature Extraction

Preprocessed inputs should be skull-stripped NIfTI volumes (see [MRI Preprocessing](./docs/preprocessing.md)). The feature script expects a CSV and an image folder.

### CSV format

`src/get_brainiac_features.py` uses `BrainAgeDataset`, which loads:

```text
<root_dir>/<pat_id>.nii.gz
```

So the CSV must have columns `pat_id,label`, where `pat_id` is the filename **without** `.nii.gz`:

```csv
pat_id,label
ADNI_002_S_0295_I123685_0000,0.0
I10307487_0000,0.0
```

For pure feature extraction, `label` can be a dummy value (e.g. `0.0`).

### Generate CSVs from processed image folders

If each cohort has an `images/` directory (e.g. `BrainIAC_processed/adni1/images/*.nii.gz`), generate one CSV per cohort with:

```bash
python scripts/create_adni_feature_csvs.py \
  --root /path/to/BrainIAC_processed
```

This writes:

```text
BrainIAC_processed/adni1/adni1.csv
BrainIAC_processed/adni2/adni2.csv
BrainIAC_processed/adni3/adni3.csv
BrainIAC_processed/adnigo/adnigo.csv
```

### Run feature extraction (single cohort)

From `src/`:

```bash
conda activate brainiac
cd src

python get_brainiac_features.py \
  --checkpoint ./checkpoints/BrainIAC.ckpt \
  --input_csv /path/to/BrainIAC_processed/adni1/adni1.csv \
  --output_csv /path/to/BrainIAC_features/adni1_features.csv \
  --root_dir /path/to/BrainIAC_processed/adni1/images
```

### Run feature extraction for all ADNI cohorts

```bash
bash scripts/run_adni_features.sh

# background
mkdir -p logs
nohup bash scripts/run_adni_features.sh > logs/adni_features.out 2>&1 &
```

Defaults (override with env vars as needed):

| Variable | Default |
|----------|---------|
| `CHECKPOINT` | `/polus2/donyapourn2/data/ADNI_dataset/BrainIAC_checkpoints/BrainIAC.ckpt` |
| `PROCESSED_ROOT` | `/polus2/donyapourn2/data/ADNI_dataset/BrainIAC_processed` |
| `FEATURES_ROOT` | `/polus2/donyapourn2/data/ADNI_dataset/BrainIAC_features` |
| `COHORTS` | `adni1 adni2 adni3 adnigo` |

Example override:

```bash
COHORTS="adni1" BATCH_SIZE=2 bash scripts/run_adni_features.sh
```


## Train and Infer Downstream Models

- [Brain Age Prediction](./docs/downstream_tasks/brain_age_prediction.md)
- [IDH Mutation Classification](./docs/downstream_tasks/idh_mutation_classification.md)
- [Mild Cognitive Impairment Classification](./docs/downstream_tasks/mild_cognitive_impairment_classification.md)
- [Diffuse Glioma Overall Survival Prediction](./docs/downstream_tasks/diffuse_glioma_overall_survival.md)
- [MR Sequence Classification](./docs/downstream_tasks/MR_sequence_classification.md)
- [Time to Stroke Prediction](./docs/downstream_tasks/timetostroke_prediction.md)
- [Tumor Segmentation](./docs/downstream_tasks/tumor_segmentation.md)


## Brainiac Platform

BrainIAC and all the downstream models are hosted at [**Brainiac Platform**](https://www.brainiac-platform.com/) for inference. The platform provides an interface for uploading the structural brain MRI data and running inference on the models including BrainIAC, brain age, MCI classification, time since stroke prediction etc.



## Citation

```bibtex
@article{tak2026generalizable,
  title={A generalizable foundation model for analysis of human brain MRI},
  author={Tak, Divyanshu and Gormosa, B.A. and Zapaishchykova, A. and others},
  journal={Nature Neuroscience},
  year={2026},
  publisher={Springer Nature},
  doi={10.1038/s41593-026-02202-6},
  url={https://doi.org/10.1038/s41593-026-02202-6}
}
```


## License

This project is licensed for non-commercial academic research use only.
Commercial use is not permitted without a separate license from
Mass General Brigham.

For commercial licensing inquiries, please contact the
Mass General Brigham Office of Technology Development. See [LICENSE](LICENSE) for details.
