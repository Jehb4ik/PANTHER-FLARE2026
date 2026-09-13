"""Showcase figure: the N best cases by Dice (ground truth vs prediction).

Computes Dice for every case that has a prediction in --pred and a label in
--labels, picks the top N, and draws one row per case: CT with the ground-truth
contour (green) and the same slice with the prediction contour (red). The slice
is the one with the largest ground-truth area. Window/level 400/40.

Usage: python make_showcase.py --images <CT dir> --labels <GT dir> --pred <mask dir> [--top 4] [--out showcase.png]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_opening, label

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


def dice(a, b):
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def to_axial(vol, im):
    vol = nib.as_closest_canonical(nib.Nifti1Image(vol, im.affine)).get_fdata()
    return np.transpose(vol, (2, 1, 0))[:, ::-1, ::-1]


def body_box(sl, pad=6):
    """Square crop around the largest connected component above -500 HU (the body)."""
    lab, n = label(binary_opening(sl > -500, iterations=3))
    if n == 0:
        return 0, sl.shape[0], 0, sl.shape[1]
    body = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    ys, xs = np.where(body)
    y0, y1, x0, x1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    side = max(y1 - y0, x1 - x0)
    y0 = int(np.clip((y0 + y1) // 2 - side // 2, 0, sl.shape[0] - side))
    x0 = int(np.clip((x0 + x1) // 2 - side // 2, 0, sl.shape[1] - side))
    return y0, y0 + side, x0, x0 + side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("showcase.png"))
    args = ap.parse_args()

    scores = []
    for lp in sorted(args.labels.glob("*.nii.gz")):
        pp = args.pred / lp.name
        if pp.exists():
            g = np.asanyarray(nib.load(str(lp)).dataobj)
            p = np.asanyarray(nib.load(str(pp)).dataobj)
            scores.append((dice(g, p), lp.name[:-7]))
    scores.sort(reverse=True)
    print("cases with predictions:", len(scores))
    for d, c in scores:
        print(f"  {c}  DSC {d:.3f}")
    best = scores[:args.top]

    # one row: (ground truth, prediction) per case
    fig, axes = plt.subplots(1, 2 * len(best), figsize=(3.0 * 2 * len(best), 3.4))
    axes = np.atleast_1d(axes)
    for r, (d, case) in enumerate(best):
        im = nib.load(str(args.images / f"{case}_0000.nii.gz"))
        ct = to_axial(np.asanyarray(im.dataobj).astype(np.float32), im)
        gt = to_axial(np.asanyarray(nib.load(str(args.labels / f"{case}.nii.gz")).dataobj).astype(np.float32), im)
        pr = to_axial(np.asanyarray(nib.load(str(args.pred / f"{case}.nii.gz")).dataobj).astype(np.float32), im)
        z = int(np.argmax(gt.reshape(gt.shape[0], -1).sum(1)))
        y0, y1, x0, x1 = body_box(ct[z], pad=2)
        win = np.clip((ct[z][y0:y1, x0:x1] - (40 - 200)) / 400, 0, 1)
        for k, (mask, color, title) in enumerate([(gt[z], "lime", f"{case[-4:]} · ground truth"),
                                                  (pr[z], "red", f"{case[-4:]} · prediction, DSC {d:.3f}")]):
            ax = axes[2 * r + k]
            ax.imshow(win, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if mask[y0:y1, x0:x1].max() > 0:
                ax.contour(mask[y0:y1, x0:x1] > 0, levels=[0.5], colors=color, linewidths=1.8)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
    plt.tight_layout(w_pad=0.3)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
