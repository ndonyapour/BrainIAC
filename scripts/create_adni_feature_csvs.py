#!/usr/bin/env python3
"""Create BrainIAC feature-extraction CSVs for ADNI cohorts.

For each cohort folder under ``--root`` that contains an ``images/`` directory,
writes a CSV with columns ``pat_id,label`` into that cohort folder.

``pat_id`` is the image filename without the ``.nii.gz`` suffix, matching
``BrainAgeDataset`` path construction: ``root_dir / (pat_id + ".nii.gz")``.

Example:
    python scripts/create_adni_feature_csvs.py \\
        --root /polus2/donyapourn2/data/ADNI_dataset/BrainIAC_processed

Outputs (default):
    <root>/adni1/adni1.csv
    <root>/adni2/adni2.csv
    ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _stem_without_nii_gz(path: Path) -> str:
    """Return filename stem for ``*.nii.gz`` (not just the first suffix)."""
    name = path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    return path.stem


def collect_pat_ids(images_dir: Path) -> list[str]:
    """List skull-stripped image IDs under ``images_dir``."""
    pat_ids = []
    for path in sorted(images_dir.glob('*.nii.gz')):
        if '_mask' in path.name:
            continue
        pat_ids.append(_stem_without_nii_gz(path))
    return pat_ids


def write_cohort_csv(
    cohort_dir: Path,
    default_label: float = 0.0,
    csv_name: str | None = None,
) -> Path | None:
    """Write ``pat_id,label`` CSV for one cohort folder.

    Args:
        cohort_dir: Cohort directory containing ``images/``.
        default_label: Label value written for every row.
        csv_name: Optional CSV filename; defaults to ``<cohort>.csv``.

    Returns:
        Path to the written CSV, or None if skipped.
    """
    images_dir = cohort_dir / 'images'
    if not images_dir.is_dir():
        print(f'Skip {cohort_dir.name}: no images/ directory')
        return None

    pat_ids = collect_pat_ids(images_dir)
    if not pat_ids:
        print(f'Skip {cohort_dir.name}: no .nii.gz images found')
        return None

    out_name = csv_name or f'{cohort_dir.name}.csv'
    out_path = cohort_dir / out_name
    df = pd.DataFrame({'pat_id': pat_ids, 'label': default_label})
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(df)} rows)')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create pat_id,label CSVs for BrainIAC feature extraction.',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/polus2/donyapourn2/data/ADNI_dataset/BrainIAC_processed'),
        help='Root directory containing cohort folders (adni1, adni2, ...).',
    )
    parser.add_argument(
        '--cohorts',
        nargs='*',
        default=None,
        help='Cohort folder names (default: all subdirs of --root with images/).',
    )
    parser.add_argument(
        '--label',
        type=float,
        default=0.0,
        help='Default label value for all rows (default: 0.0).',
    )
    parser.add_argument(
        '--csv-name',
        type=str,
        default=None,
        help='CSV filename inside each cohort folder (default: <cohort>.csv).',
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f'Root not found: {root}')

    if args.cohorts:
        cohort_dirs = [root / name for name in args.cohorts]
    else:
        cohort_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    written = 0
    for cohort_dir in cohort_dirs:
        if not cohort_dir.is_dir():
            print(f'Skip missing cohort dir: {cohort_dir}')
            continue
        if write_cohort_csv(cohort_dir, default_label=args.label, csv_name=args.csv_name):
            written += 1

    print(f'Done. Wrote {written} CSV file(s) under {root}')


if __name__ == '__main__':
    main()
