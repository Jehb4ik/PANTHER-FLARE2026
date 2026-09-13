"""Assemble Dataset500_FLAREPanCancer (nnU-Net raw layout) from HuggingFace.

Downloads the 8762 image+label pairs listed in Dataset500_manifest.json
(~350 GB) into <root>/_flare26_hf_cache and links them into
<root>/nnUNet_raw/Dataset500_FLAREPanCancer/{imagesTr,labelsTr,dataset.json}.

Usage: python reproduce_dataset500.py --root /some/path [--copy]
Requires huggingface_hub and access to FLARE-MedFM/PancancerCTSeg (HF_TOKEN).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "Dataset500_manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="destination root (needs plenty of disk)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--symlink", action="store_true", default=True,
                    help="link into nnUNet_raw instead of copying (default)")
    ap.add_argument("--copy", dest="symlink", action="store_false",
                    help="hard-copy the files into nnUNet_raw (uses 2x space)")
    args = ap.parse_args()

    with MANIFEST.open() as f:
        m = json.load(f)

    cache = args.root / "_flare26_hf_cache"
    ds = args.root / "nnUNet_raw" / "Dataset500_FLAREPanCancer"
    cache.mkdir(parents=True, exist_ok=True)
    (ds / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds / "labelsTr").mkdir(parents=True, exist_ok=True)

    patterns = []
    for row in m["cases"]:
        patterns.append(row["image_hf_path"])
        patterns.append(row["label_hf_path"])

    print(f"downloading {len(patterns)} files "
          f"({m['n_cases']} image+label pairs) from HuggingFace ...")
    snapshot_download(
        repo_id=m["repo_id"],
        repo_type=m["repo_type"],
        local_dir=str(cache),
        allow_patterns=patterns,
        max_workers=args.workers,
    )

    import os
    import shutil
    linked = 0
    for row in m["cases"]:
        stem = row["stem"]
        src_img = cache / row["image_hf_path"]
        src_lbl = cache / row["label_hf_path"]
        dst_img = ds / "imagesTr" / f"{stem}_0000.nii.gz"
        dst_lbl = ds / "labelsTr" / f"{stem}.nii.gz"
        for src, dst in [(src_img, dst_img), (src_lbl, dst_lbl)]:
            if not src.exists():
                raise RuntimeError(f"expected file missing after download: {src}")
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.symlink:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
        linked += 1

    (ds / "dataset.json").write_text(json.dumps(m["dataset_json"], indent=1))
    print(f"Dataset500 ready at {ds}: {linked} cases "
          f"({'symlinks' if args.symlink else 'copies'})")
    print(f"\nNext: ./preprocess_dataset500.sh {args.root}  (uses the shipped nnUNetPlans_tr.json)")


if __name__ == "__main__":
    main()
