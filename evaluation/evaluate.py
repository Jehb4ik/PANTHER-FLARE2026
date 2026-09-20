"""Lesion DSC and NSD (2 mm tolerance) for predicted masks versus ground truth.

Follows the FLARE metric definition: masks are binarised (>0), DSC is the
Dice coefficient, NSD is the normalized surface Dice with a 2 mm tolerance
computed from the physical voxel spacing. An empty prediction on an empty
ground truth scores 1.0 for both metrics.

Usage: python evaluate.py --gt <label dir> --pred <mask dir> [--tolerance 2.0] [--csv per_case.csv]
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(gt: np.ndarray, pr: np.ndarray) -> float:
    s = gt.sum() + pr.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(gt, pr).sum() / s


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def nsd(gt: np.ndarray, pr: np.ndarray, spacing, tol: float) -> float:
    if not gt.any() and not pr.any():
        return 1.0
    if not gt.any() or not pr.any():
        return 0.0
    sg, sp = surface(gt), surface(pr)
    dist_to_gt = distance_transform_edt(~sg, sampling=spacing)
    dist_to_pr = distance_transform_edt(~sp, sampling=spacing)
    pr_ok = (dist_to_gt[sp] <= tol).sum()
    gt_ok = (dist_to_pr[sg] <= tol).sum()
    return (pr_ok + gt_ok) / (sp.sum() + sg.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=2.0, help="NSD tolerance in mm")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for gp in sorted(args.gt.glob("*.nii.gz")):
        pp = args.pred / gp.name
        if not pp.exists():
            print(f"missing prediction for {gp.name}, scored as empty")
        g_im = nib.load(str(gp))
        gt = np.asanyarray(g_im.dataobj) > 0
        pr = np.asanyarray(nib.load(str(pp)).dataobj) > 0 if pp.exists() else np.zeros_like(gt)
        if pr.shape != gt.shape:
            raise SystemExit(f"{gp.name}: prediction shape {pr.shape} != ground truth {gt.shape}")
        spacing = tuple(float(s) for s in g_im.header.get_zooms()[:3])
        rows.append((gp.name[:-7], dice(gt, pr), nsd(gt, pr, spacing, args.tolerance)))
        print(f"{rows[-1][0]}  DSC {rows[-1][1]:.4f}  NSD {rows[-1][2]:.4f}")

    d = np.array([r[1] for r in rows]); n = np.array([r[2] for r in rows])
    print(f"\n{len(rows)} cases  mean DSC {d.mean():.4f}  mean NSD {n.mean():.4f}")
    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["case", "dsc", "nsd"]); w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
