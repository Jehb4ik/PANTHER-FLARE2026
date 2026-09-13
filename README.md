# PANTHER: training and inference code for FLARE 2026 Task 1 (team jehb4ik)

![PANTHER results: hidden-validation DSC/NSD, false-positive rate on healthy CTs, per-scan inference time and GPU memory](docs/hero.png)

Code for the paper *PANTHER: PAN-cancer CT Segmentation with Two-tier
Healthy-scan Ensemble Rejection*. The submitted Docker image is a 5-fold plain
U-Net ensemble (nnU-Net v2.8.1) trained on the per-source capped
`Dataset500_FLAREPanCancer` subset (8762 CT scans) with a post-hoc Mahalanobis
mask-shape gate fitted on 172 healthy CT scans.

Hidden validation: DSC 75.38 %, NSD 70.09 %. False-positive rate on 172 healthy
CT scans: 100 % (single fold) -> 69.8 % (ensemble) -> 8.1 % (ensemble + gate).
Inference on the 50 public validation cases (RTX 3090, `-m 28G`, default
`--shm-size`): 5.6-29.3 s per case, peak GPU memory 2614 MB.

## Qualitative examples

Best public validation cases by Dice: ground truth (green) versus the prediction
of the submitted image (red), axial slice with the largest lesion area, window/level 400/40.

![Best cases: FLARE23Ts_0057 (DSC 0.950) and FLARE23Ts_0001 (DSC 0.927)](docs/showcase.png)

Regenerate with `docs/make_showcase.py --images <CT dir> --labels <GT dir> --pred <mask dir> --top N`;
`docs/make_hero.py` rebuilds the summary figure from `sanity.cast` and the nvidia-smi log.

## Layout

```
code/
├── training/    # Dataset500 manifest and downloader, nnU-Net plan, trainer, preprocessing
├── maha_gate/   # Mahalanobis shape-gate fit and the fitted parameters
└── docker/      # Submitted inference container and the sanity-test script
```

## Reproduction

1. `training/README.md`: download Dataset500, preprocess with the shipped plan,
   train five folds (about 72 h per fold on one H100 80 GB).
2. `maha_gate/README.md`: predict the 172 healthy and 50 public validation
   scans with the ensemble, fit mu, Sigma^-1 and tau with
   `fit_mahalanobis_gate.py`. The fitted parameters used in the submission are
   `maha_gate/maha_gate.fitted.json`.
3. `docker/README.md`: put the five `checkpoint_final.pth` files into
   `docker/model/fold_{0..4}/`, build the image, run `run_sanity.sh`.

Not included: trained checkpoints (5 x 236 MB, hosted on HuggingFace, see
`docker/README.md`; place them in `docker/model/`) and the FLARE 2026 data (registration on
the `FLARE-MedFM/PancancerCTSeg` HuggingFace dataset is required; the manifest
lists the exact 8762 files).

## Citation

Przhezdzetskaia E., Baranov V., Khazova M., Gombolevskiy V. PANTHER: PAN-cancer
CT Segmentation with Two-tier Healthy-scan Ensemble Rejection. MICCAI 2026
FLARE Challenge, Task 1 submission.

## License

Apache License 2.0 (see `LICENSE`). Built on nnU-Net v2.8.1 (Apache 2.0).
