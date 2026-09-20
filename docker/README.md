# Submitted inference container (jehb4ik)

5-fold plain U-Net ensemble + Mahalanobis shape gate. This is the exact code
of the image submitted to FLARE 2026 Task 1.

| File | Purpose |
|---|---|
| `Dockerfile` | `pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime`, nnunetv2 2.8.1 |
| `predict.py` | inference: in-process preprocessing, sliding window (step 0.7, no TTA), five folds averaged, Mahalanobis gate |
| `predict.sh` | entrypoint run by the organizers as `/bin/bash -c "sh predict.sh"` |
| `custom_trainer.py` | stand-in trainer class needed to load the checkpoints |
| `maha_gate.json` | fitted gate parameters (mean, inverse covariance, tau = 7) |
| `build.sh`, `test.sh` | build the image, run it on a local folder |
| `run_sanity.sh` | timed run with a 1 Hz GPU-memory log, the organizers' docker flags (the numbers in the paper) |
| `model/` | put `plans.json`, `dataset.json` and `fold_{0..4}/checkpoint_final.pth` here |

## Weights

The five checkpoints (236 MB each), `plans.json` and `dataset.json` are hosted
on HuggingFace at https://huggingface.co/jehb4ik/PANTHER-FLARE2026:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('jehb4ik/PANTHER-FLARE2026', local_dir='model')"
```

Expected layout: `model/plans.json`, `model/dataset.json`, `model/fold_{0..4}/checkpoint_final.pth`.

## Build and test

```bash
./build.sh jehb4ik:latest            # add "save" to also write a tar.gz
./test.sh  jehb4ik:latest /path/in /path/out
./run_sanity.sh /path/in /path/out jehb4ik:latest   # timed run, logs in /path/sanity_logs/
```

Input `<case>_0000.nii.gz`, output `<case>.nii.gz`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `FLARE_FOLDS` | `0,1,2,3,4` | folds in the ensemble |
| `FLARE_TILE_STEP` | `0.7` | sliding-window step |
| `FLARE_MAHA_GATE` / `FLARE_MAHA_TAU` | `1` / `7.0` | Mahalanobis gate on/off and threshold |
| `FLARE_CC_CONF_GATE` / `FLARE_CC_CONF_TAU` | `0` / `0.65` | component-confidence gate (off: doubles time on large volumes) |
| `FLARE_VOLUME_GATE` | `0` | zero masks below this volume in mm3 (off) |

## Measured on the 50 public validation cases

RTX 3090, `docker run --gpus device=0 -m 28G` with the default `--shm-size`:
553 s in total, 5.6 / 9.2 / 11.1 / 29.3 s per case (min / median / mean / max),
peak GPU memory 2614 MB (nvidia-smi, 1 Hz), peak RAM 6.0 GB, no failures.
