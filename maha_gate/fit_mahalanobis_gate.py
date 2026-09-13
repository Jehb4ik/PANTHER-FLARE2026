"""Fit the Mahalanobis shape gate on ensemble predictions.

Features per predicted mask: volume [ml], number of connected components
>= 27 voxels, largest-component volume [ml], largest-component fraction,
surface-to-volume ratio (mean/std HU inside the mask are stored in the CSV but
not used by the gate). Features are log1p-transformed; mean and inverse
covariance (1e-3 ridge) are fitted on the non-empty healthy masks. Empty masks
count as distance 0. Prints AUROC, a tau sweep, the cancer cases lost at
tau = 7 and a 5-fold cross-validation of the Youden optimum; writes the gate
JSON consumed by docker/predict.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label, sum as ndsum
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

FEATURES = ["pred_ml", "n_cc", "largest_ml", "largest_frac", "surf_to_vol"]
TAU_GRID = (5, 7, 10, 14, 20)
TAU_PICK = 7.0


def features(args: tuple) -> dict:
    cohort, mask_path, ct_path = args
    m_im = sitk.ReadImage(str(mask_path))
    m = (sitk.GetArrayFromImage(m_im) > 0.5).astype(np.uint8)
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path)))
    vox_ml = float(np.prod(m_im.GetSpacing())) / 1000.0
    case = mask_path.name[:-len(".nii.gz")]
    total_vox = int(m.sum())
    if total_vox == 0:
        return dict(cohort=cohort, case=case, pred_ml=0.0, n_cc=0, largest_ml=0.0,
                    largest_frac=0.0, hu_mean_in=np.nan, hu_std_in=np.nan, surf_to_vol=np.nan)
    lab, n_cc = cc_label(m)
    sizes = ndsum(m, lab, index=np.arange(1, n_cc + 1))
    largest = float(sizes.max())
    # 6-neighbour boundary voxels
    dz = np.zeros_like(m, dtype=bool)
    dz[1:] |= m[1:] != m[:-1]; dz[:-1] |= m[1:] != m[:-1]
    dy = np.zeros_like(m, dtype=bool)
    dy[:, 1:] |= m[:, 1:] != m[:, :-1]; dy[:, :-1] |= m[:, 1:] != m[:, :-1]
    dx = np.zeros_like(m, dtype=bool)
    dx[..., 1:] |= m[..., 1:] != m[..., :-1]; dx[..., :-1] |= m[..., 1:] != m[..., :-1]
    surf = int(((dz | dy | dx) & (m > 0)).sum())
    hu_in = ct[m.astype(bool)]
    return dict(cohort=cohort, case=case,
                pred_ml=round(total_vox * vox_ml, 3),
                n_cc=int((sizes >= 27).sum()),
                largest_ml=round(largest * vox_ml, 3),
                largest_frac=round(largest / max(1, total_vox), 4),
                hu_mean_in=round(float(hu_in.mean()), 1),
                hu_std_in=round(float(hu_in.std()), 1),
                surf_to_vol=round(surf / max(1, total_vox), 4))


def collect(cohort: str, pred_dir: Path, ct_dir: Path) -> list[tuple]:
    tasks = []
    for m in sorted(pred_dir.glob("*.nii.gz")):
        stem = m.name[:-len(".nii.gz")]
        img = ct_dir / f"{stem}_0000.nii.gz"
        if not img.exists():
            img = ct_dir / f"{stem}.nii.gz"
        if img.exists():
            tasks.append((cohort, m, img))
    return tasks


def maha(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    d = x - mu[None, :]
    return np.einsum("ij,jk,ik->i", d, inv_cov, d)


def fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-3
    return mu, np.linalg.inv(cov)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-healthy", type=Path, required=True, help="ensemble masks for the healthy scans")
    ap.add_argument("--ct-healthy", type=Path, required=True, help="healthy CT scans")
    ap.add_argument("--pred-public", type=Path, required=True, help="ensemble masks for the public validation scans")
    ap.add_argument("--ct-public", type=Path, required=True, help="public validation CT scans")
    ap.add_argument("--out-gate", type=Path, default=Path("maha_gate.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("maha_gate_features.csv"))
    ap.add_argument("--dice-csv", type=Path, default=None,
                    help="optional per-case CSV with columns case,dice (public set) to annotate lost cases")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = collect("HEALTHY", args.pred_healthy, args.ct_healthy) + collect("CANCER", args.pred_public, args.ct_public)
    print(f"processing {len(tasks)} cases")
    with mp.Pool(args.workers) as pool:
        rows = pool.map(features, tasks)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    healthy = [r for r in rows if r["cohort"] == "HEALTHY"]
    cancer = [r for r in rows if r["cohort"] == "CANCER"]
    nonempty = lambda rs: [r for r in rs if r["pred_ml"] > 0]
    xh = np.log1p(np.array([[r[k] for k in FEATURES] for r in nonempty(healthy)], dtype=float))
    cancer_ne = nonempty(cancer)
    xc = np.log1p(np.array([[r[k] for k in FEATURES] for r in cancer_ne], dtype=float))
    n_h, n_h_empty = len(healthy), len(healthy) - len(xh)
    print(f"healthy non-empty {len(xh)}/{n_h}, cancer non-empty {len(xc)}/{len(cancer)}")

    mu, inv_cov = fit(xh)
    dh, dc = maha(xh, mu, inv_cov), maha(xc, mu, inv_cov)
    auroc = roc_auc_score(np.r_[np.zeros(len(dh)), np.ones(len(dc))], np.r_[dh, dc])
    auroc_full = roc_auc_score(np.r_[np.zeros(n_h), np.ones(len(dc))], np.r_[np.zeros(n_h_empty), dh, dc])
    print(f"AUROC non-empty healthy vs cancer {auroc:.3f}; all healthy (empty = 0) vs cancer {auroc_full:.3f}")

    print(f"{'tau':>5} {'healthy empty':>14} {'cancer kept':>12} {'FPR':>6} {'TPR':>6}")
    for tau in TAU_GRID:
        blocked = n_h_empty + int((dh <= tau).sum())
        kept = int((dc > tau).sum())
        print(f"{tau:>5} {blocked:>7}/{n_h:<6} {kept:>6}/{len(dc):<5} {1 - blocked / n_h:>6.3f} {kept / len(dc):>6.3f}")

    dice = {}
    if args.dice_csv and args.dice_csv.exists():
        with args.dice_csv.open() as f:
            dice = {r["case"]: float(r["dice"]) for r in csv.DictReader(f)}
    lost = [(cancer_ne[i]["case"], float(dc[i])) for i in range(len(dc)) if dc[i] <= TAU_PICK]
    print(f"cancer cases zeroed at tau={TAU_PICK}: {len(lost)}")
    for case, d2 in sorted(lost, key=lambda x: x[1]):
        print(f"  {case:<20} d2={d2:6.2f} dice={dice.get(case, float('nan')):.3f}")

    # 5-fold CV of the Youden optimum on the non-empty healthy vectors
    taus = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(xh):
        mu_k, inv_k = fit(xh[tr])
        dh_te, dc_te = maha(xh[te], mu_k, inv_k), maha(xc, mu_k, inv_k)
        empty_te = n_h_empty * len(te) / len(xh)          # empty healthy share of this fold
        best = max(np.linspace(1.0, 30.0, 60),
                   key=lambda t: (dc_te > t).mean() - (1 - ((dh_te <= t).sum() + empty_te) / (len(te) + empty_te)))
        taus.append(best)
    print(f"Youden tau* across folds: {np.mean(taus):.2f} +- {np.std(taus):.2f}")

    gate = {"feature_names": FEATURES, "log1p": True, "mean": mu.tolist(), "inv_cov": inv_cov.tolist(),
            "tau_youden_full": TAU_PICK, "n_healthy_train": int(len(xh)), "n_cancer_probe": int(len(xc)),
            "auroc_full": float(auroc_full),
            "note": "Fit on healthy CT predictions (log1p shape features). Apply: if distance^2 <= tau then zero the mask."}
    args.out_gate.write_text(json.dumps(gate, indent=1))
    print(f"wrote {args.out_gate}")


if __name__ == "__main__":
    main()
