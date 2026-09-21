"""Five-fold nnU-Net ensemble inference for the FLARE 2026 Task 1 submission.

Reads `<case>_0000.nii.gz` from `/workspace/inputs`, writes `<case>.nii.gz` to
`/workspace/outputs`. Pipeline: direction-matrix repair -> in-process
preprocessing and sliding-window inference (five folds averaged) -> optional
gates -> Mahalanobis shape gate -> mask in the original geometry.

Measured on the 50 public validation cases (RTX 3090, `-m 28G`, default
`--shm-size`): 5.6-29.3 s per case, peak GPU memory 2614 MB (nvidia-smi, 1 Hz).

Robustness: a non-orthonormal direction matrix (13 of 8762 training volumes)
is fixed by polar decomposition; a per-case failure writes an empty mask so
that one broken file does not zero out the remaining cases.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

IN = Path(os.environ.get("FLARE_INPUTS", "/workspace/inputs"))
OUT = Path(os.environ.get("FLARE_OUTPUTS", "/workspace/outputs"))
MODEL = Path(os.environ.get("FLARE_MODEL", "/workspace/model"))
FOLDS = tuple(int(x) for x in os.environ.get("FLARE_FOLDS", "0,1,2,3,4").split(","))
TILE_STEP = float(os.environ.get("FLARE_TILE_STEP", "0.7"))
# Volume gate, off by default: every threshold tested (2000/5000/20000 mm3)
# lowered hidden-validation DSC (75.38 -> 71.81/70.43/69.51).
VOLUME_GATE_MM3 = float(os.environ.get("FLARE_VOLUME_GATE", "0"))

# Mahalanobis shape gate (on by default). Five log1p mask-shape features;
# squared Mahalanobis distance to the distribution fitted on 172 healthy CTs;
# d^2 <= tau zeroes the mask. AUROC 0.974, tau = 7 (Youden J). See
# ../maha_gate/. Disable with FLARE_MAHA_GATE=0.
MAHA_GATE_PATH = Path(os.environ.get(
    "FLARE_MAHA_GATE_JSON", "/workspace/maha_gate.json"))
MAHA_GATE_TAU = float(os.environ.get("FLARE_MAHA_TAU", "7.0"))
MAHA_GATE_ENABLED = bool(int(os.environ.get("FLARE_MAHA_GATE", "1")))

# Component-confidence gate, off by default: drops connected components with
# mean softmax <= CC_CONF_TAU. +0.010 DSC on public validation, no change on
# hidden validation, and the softmax export doubles the time on large volumes
# (case 0029: 69.6 s vs 29.3 s). Enable with FLARE_CC_CONF_GATE=1.
CC_CONF_TAU = float(os.environ.get("FLARE_CC_CONF_TAU", "0.65"))
CC_CONF_MIN_VOX = int(os.environ.get("FLARE_CC_CONF_MIN_VOX", "5"))
CC_CONF_GATE_ENABLED = bool(int(os.environ.get("FLARE_CC_CONF_GATE", "0")))


def orthonormalize(aff: np.ndarray) -> np.ndarray:
    """Nearest affine with orthonormal directions (polar decomposition); scale and
    translation are preserved."""
    out = aff.copy()
    m = aff[:3, :3]
    scale = np.linalg.norm(m, axis=0)
    u, _, vt = np.linalg.svd(m / scale)
    out[:3, :3] = (u @ vt) * scale
    return out


def sanitize(path: Path, tmp: Path) -> Path:
    """Return `path`, or a temporary copy with an orthonormalised direction matrix
    if SimpleITK refuses to open the original. Voxels are left untouched."""
    import SimpleITK as sitk
    try:
        sitk.ReadImage(str(path))
        return path
    except Exception:                                    # noqa: BLE001
        pass
    import nibabel as nib
    im = nib.load(path)
    aff = orthonormalize(im.affine)
    fixed = nib.Nifti1Image(np.asanyarray(im.dataobj), aff, im.header)
    fixed.set_sform(aff, code=1)
    fixed.set_qform(aff, code=1)
    dst = tmp / path.name
    nib.save(fixed, dst)
    print(f"  direction matrix repaired: {path.name}", flush=True)
    return dst


def apply_component_confidence_gate(mask_path: Path, probs_path: Path,
                                    tau: float, min_vox: int) -> int:
    """Drop components with mean class-1 softmax <= tau; return how many were dropped."""
    if not (mask_path.exists() and probs_path.exists()):
        return 0
    import SimpleITK as sitk
    from scipy.ndimage import label as _cc

    m_im = sitk.ReadImage(str(mask_path))
    m = (sitk.GetArrayFromImage(m_im) > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return 0
    probs_data = np.load(probs_path)
    key = "probabilities" if "probabilities" in probs_data.files else probs_data.files[0]
    probs = probs_data[key][1].astype(np.float32)  # class 1
    if probs.shape != m.shape:                             # sanity, must match
        return 0
    lab, n = _cc(m)
    if n == 0:
        return 0
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    dropped = 0
    for cid in range(1, n + 1):
        comp = lab == cid
        vox = int(comp.sum())
        if vox < min_vox:
            keep[cid] = False
            dropped += 1
            continue
        if float(probs[comp].mean()) <= tau:
            keep[cid] = False
            dropped += 1
    if dropped == 0:
        return 0
    m2 = keep[lab].astype(np.uint8)
    out = sitk.GetImageFromArray(m2)
    out.CopyInformation(m_im)
    sitk.WriteImage(out, str(mask_path), True)
    return dropped


def apply_maha_gate(mask_path: Path, ct_path: Path, gate: dict) -> bool:
    """Zero the mask if its shape features fall within the healthy distribution.

    Features (same as in the fit): volume [ml], number of components >= 27 vox,
    largest-component volume and fraction, surface-to-volume ratio; log1p, then
    squared Mahalanobis distance with the mean and inverse covariance from `gate`.
    Returns True if the mask was zeroed.
    """
    if not mask_path.exists():
        return False
    import SimpleITK as sitk
    from scipy.ndimage import label as _cc, sum as _ndsum
    m_im = sitk.ReadImage(str(mask_path))
    m = (sitk.GetArrayFromImage(m_im) > 0.5).astype(np.uint8)
    total = int(m.sum())
    if total == 0:                                       # already empty
        return False

    sp = m_im.GetSpacing()
    vox_ml = float(np.prod(sp)) / 1000.0
    pred_ml = total * vox_ml

    lab, n_cc = _cc(m)
    if n_cc:
        sizes = _ndsum(m, lab, index=np.arange(1, n_cc + 1))
        n_cc_big = int((sizes >= 27).sum())
        largest = float(sizes.max())
    else:
        n_cc_big = 0
        largest = 0.0
    largest_ml = largest * vox_ml
    largest_frac = largest / max(1, total)

    dz = np.zeros_like(m, dtype=bool)
    dz[1:] |= m[1:] != m[:-1]; dz[:-1] |= m[1:] != m[:-1]
    dy = np.zeros_like(m, dtype=bool)
    dy[:, 1:] |= m[:, 1:] != m[:, :-1]; dy[:, :-1] |= m[:, 1:] != m[:, :-1]
    dx = np.zeros_like(m, dtype=bool)
    dx[..., 1:] |= m[..., 1:] != m[..., :-1]; dx[..., :-1] |= m[..., 1:] != m[..., :-1]
    surf = int(((dz | dy | dx) & (m > 0)).sum())
    surf_to_vol = surf / max(1, total)

    raw = {
        "pred_ml": pred_ml,
        "n_cc": n_cc_big,
        "largest_ml": largest_ml,
        "largest_frac": largest_frac,
        "surf_to_vol": surf_to_vol,
    }
    x = np.array([raw[k] for k in gate["feature_names"]], dtype=float)
    if gate.get("log1p", True):
        x = np.log1p(x)
    mu = np.array(gate["mean"], dtype=float)
    inv_cov = np.array(gate["inv_cov"], dtype=float)
    diff = x - mu
    d2 = float(diff @ inv_cov @ diff)
    if d2 > MAHA_GATE_TAU:
        return False

    out = sitk.GetImageFromArray(np.zeros_like(m, dtype=np.uint8))
    out.CopyInformation(m_im)
    sitk.WriteImage(out, str(mask_path), True)
    return True


def apply_volume_gate(path: Path) -> bool:
    """Zero the mask if its volume is below VOLUME_GATE_MM3; return True if zeroed."""
    if VOLUME_GATE_MM3 <= 0 or not path.exists():
        return False
    import SimpleITK as sitk
    im = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(im)
    vox = float(np.prod(im.GetSpacing()))
    if float((a > 0.5).sum()) * vox >= VOLUME_GATE_MM3:
        return False
    out = sitk.GetImageFromArray(np.zeros_like(a, dtype=np.uint8))
    out.CopyInformation(im)
    sitk.WriteImage(out, str(path), True)
    return True


def write_empty(src: Path, dst: Path) -> None:
    """Write an empty mask in the geometry of `src`."""
    import nibabel as nib
    im = nib.load(src)
    z = np.zeros(im.shape[:3], np.uint8)
    nib.save(nib.Nifti1Image(z, im.affine, im.header), dst)


def main() -> int:
    """Predict every volume in the input directory; return 0, or 1 if there is no input."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(IN.glob("*.nii.gz"))
    if not files:
        print(f"no .nii.gz files in input dir {IN}", flush=True)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(tile_step_size=TILE_STEP, use_gaussian=True,
                                use_mirroring=False, perform_everything_on_device=True,
                                device=device, verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(MODEL), use_folds=FOLDS,
                                                   checkpoint_name="checkpoint_final.pth")
    print(f"ensemble folds: {len(FOLDS)} ({FOLDS})", flush=True)
    # Same reader nnU-Net used at preprocessing (dataset.json: SimpleITKIO).
    image_io = SimpleITKIO()

    maha_gate = None
    if MAHA_GATE_ENABLED and MAHA_GATE_PATH.exists():
        import json as _json
        with MAHA_GATE_PATH.open() as _f:
            maha_gate = _json.load(_f)
        print(f"Mahalanobis gate active: {MAHA_GATE_PATH} (tau={MAHA_GATE_TAU})",
              flush=True)
    elif MAHA_GATE_ENABLED:
        print(f"Mahalanobis gate requested but JSON not found: {MAHA_GATE_PATH}",
              flush=True)

    t_start, failed = time.time(), []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for f in files:
            # FLARE feeds files as <case>_0000.nii.gz; the output expects <case>.nii.gz.
            case = f.name[:-len(".nii.gz")]
            if case.endswith("_0000"):
                case = case[:-len("_0000")]
            t0 = time.time()
            try:
                src = sanitize(f, tmp)
                need_probs = CC_CONF_GATE_ENABLED
                # In-process preprocessing + inference: no worker processes, no
                # /dev/shm dependency (64 MB by default in Docker), identical masks.
                data, props = image_io.read_images([str(src)])
                predictor.predict_single_npy_array(
                    data, props, output_file_truncated=str(OUT / case),
                    save_or_return_probabilities=need_probs)
            except Exception:                            # noqa: BLE001
                failed.append(case)
                traceback.print_exc()
                try:
                    write_empty(f, OUT / f"{case}.nii.gz")
                    print(f"{case}: FAILED, empty mask written", flush=True)
                except Exception:                        # noqa: BLE001
                    print(f"{case}: FAILED, could not write empty mask", flush=True)
                continue
            # Gates run outside the inference try: if one of them fails, the
            # predicted mask is kept as written instead of being replaced by
            # an empty one.
            cc_dropped, gated, maha_zeroed = 0, False, False
            tags = []
            try:
                if need_probs:
                    probs_path = OUT / f"{case}.npz"
                    cc_dropped = apply_component_confidence_gate(
                        OUT / f"{case}.nii.gz", probs_path,
                        CC_CONF_TAU, CC_CONF_MIN_VOX)
                    # nnU-Net also writes {case}.pkl next to the .npz
                    for ext in (".npz", ".pkl"):
                        p = OUT / f"{case}{ext}"
                        p.unlink(missing_ok=True)
                gated = apply_volume_gate(OUT / f"{case}.nii.gz")
                if maha_gate is not None:
                    maha_zeroed = apply_maha_gate(
                        OUT / f"{case}.nii.gz", f, maha_gate)
            except Exception:                            # noqa: BLE001
                traceback.print_exc()
                tags.append("gate failed, mask kept unfiltered")
            if cc_dropped:
                tags.append(f"dropped {cc_dropped} low-confidence components")
            if gated:
                tags.append("zeroed by volume gate")
            if maha_zeroed:
                tags.append("zeroed by Mahalanobis gate")
            suffix = f" [{' + '.join(tags)}]" if tags else ""
            print(f"{case}: {time.time() - t0:.1f} s{suffix}", flush=True)
    # nnU-Net drops its own auxiliary files next to the masks. The organizers
    # read the whole output directory, so the extras are removed.
    for junk in ("dataset.json", "plans.json", "predict_from_raw_data_args.json"):
        (OUT / junk).unlink(missing_ok=True)

    peak_mb = 0.0
    peak_gpu_mb = 0.0
    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:                                    # noqa: BLE001
        pass
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_reserved() / 2 ** 20
    print(f"done: {len(files)} scans in {time.time() - t_start:.1f} s, "
          f"failures {len(failed)}, peak RAM {peak_mb:.0f} MB, "
          f"peak VRAM {peak_gpu_mb:.0f} MiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
