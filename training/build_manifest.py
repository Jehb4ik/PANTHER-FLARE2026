"""One-off script that produced Dataset500_manifest.json from our internal
Dataset500 layout (symlinks into the organisers' train_label tree). Paths below
are specific to our server; kept for provenance, not needed for reproduction.
"""
from __future__ import annotations

import json
from pathlib import Path

DS = Path("/home/jovyan/shares/SR006.nfs3/zhenya/SUMMER/flare26/nnUNet_raw/Dataset500_FLAREPanCancer")
NFS = "/home/jovyan/shares/SR006.nfs1/flare26/"
OUT = Path("/home/jovyan/shares/SR006.nfs3/zhenya/SUMMER/flare26/downloads/dataset500_reproducer/Dataset500_manifest.json")


def nfs_to_hf(nfs_path: str) -> str:
    """Convert /home/.../nfs1/flare26/train_label/... into train_label/... on HF."""
    if not nfs_path.startswith(NFS):
        raise RuntimeError(f"unexpected non-nfs1 target: {nfs_path}")
    return nfs_path[len(NFS):]


cases: list[dict] = []
for img in sorted((DS / "imagesTr").glob("*_0000.nii.gz")):
    stem = img.name[:-len("_0000.nii.gz")]
    lbl = DS / "labelsTr" / f"{stem}.nii.gz"
    if not lbl.exists():
        raise RuntimeError(f"missing label for {stem}")
    img_hf = nfs_to_hf(str(img.resolve()))
    lbl_hf = nfs_to_hf(str(lbl.resolve()))
    cases.append({"stem": stem, "image_hf_path": img_hf, "label_hf_path": lbl_hf})

with (DS / "dataset.json").open() as f:
    ds_json = json.load(f)

payload = {
    "repo_id": "FLARE-MedFM/PancancerCTSeg",
    "repo_type": "dataset",
    "n_cases": len(cases),
    "dataset_json": ds_json,
    "cases": cases,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    json.dump(payload, f, indent=1)
print(f"wrote {OUT}  ({len(cases)} cases)")
