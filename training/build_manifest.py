"""One-off script that produced Dataset500_manifest.json from a local
Dataset500 layout (symlinks into the organisers' train_label tree), whose
download tree mirrors the HuggingFace dataset structure. Kept for provenance,
not needed for reproduction.

Usage:
    python build_manifest.py \
        --dataset /path/to/nnUNet_raw/Dataset500_FLAREPanCancer \
        --download-root /path/to/flare26_downloads \
        --out Dataset500_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def to_hf_path(abs_path: str, root: str) -> str:
    """Convert <root>/train_label/... into train_label/... (HF-relative)."""
    if not abs_path.startswith(root):
        raise RuntimeError(f"unexpected path outside the download root: {abs_path}")
    return abs_path[len(root):]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Dataset500_manifest.json.")
    ap.add_argument(
        "--dataset", required=True, type=Path,
        help="nnU-Net raw Dataset500 folder (imagesTr/, labelsTr/, dataset.json)")
    ap.add_argument(
        "--download-root", required=True, type=Path,
        help="root of the organisers' download tree; its train_label/ subtree "
             "mirrors the HuggingFace layout")
    ap.add_argument(
        "--out", required=True, type=Path,
        help="output manifest path")
    args = ap.parse_args()

    ds = args.dataset
    root = str(args.download_root.resolve())
    if not root.endswith("/"):
        root += "/"

    cases: list[dict] = []
    for img in sorted((ds / "imagesTr").glob("*_0000.nii.gz")):
        stem = img.name[:-len("_0000.nii.gz")]
        lbl = ds / "labelsTr" / f"{stem}.nii.gz"
        if not lbl.exists():
            raise RuntimeError(f"missing label for {stem}")
        cases.append({
            "stem": stem,
            "image_hf_path": to_hf_path(str(img.resolve()), root),
            "label_hf_path": to_hf_path(str(lbl.resolve()), root),
        })

    with (ds / "dataset.json").open() as f:
        ds_json = json.load(f)

    payload = {
        "repo_id": "FLARE-MedFM/PancancerCTSeg",
        "repo_type": "dataset",
        "n_cases": len(cases),
        "dataset_json": ds_json,
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {args.out}  ({len(cases)} cases)")


if __name__ == "__main__":
    main()