---
license: apache-2.0
tags:
- medical-imaging
- image-segmentation
- nnU-Net
- CT
- pan-cancer
- FLARE2026
---

# PANTHER: FLARE 2026 Task 1 weights (team jehb4ik)

Five-fold plain U-Net ensemble (nnU-Net v2.8.1, 3d_fullres, patch 96x160x160,
spacing 2.0x1.5x1.5 mm, 30.9 M parameters per fold) for pan-cancer lesion
segmentation on CT, trained on the per-source capped `Dataset500` subset
(8762 scans) of the FLARE 2026 training data. Used together with a Mahalanobis
mask-shape gate fitted on 172 healthy CT scans.

Paper: *PANTHER: PAN-cancer CT Segmentation with Two-tier Healthy-scan Ensemble
Rejection* (MICCAI 2026 FLARE Challenge, Task 1). Code: https://github.com/Jehb4ik/PANTHER-FLARE2026.

| Metric | Value |
|---|---|
| Lesion DSC, hidden validation | 75.38 % |
| Lesion NSD, hidden validation | 70.09 % |
| False-positive rate on 172 healthy CTs (with gate) | 8.1 % |
| Median / max time per scan (RTX 3090) | 9.2 s / 29.3 s |
| Peak GPU memory (nvidia-smi) | 2614 MB |

## Files

- `plans.json`, `dataset.json`: nnU-Net plan and label definitions.
- `fold_{0..4}/checkpoint_final.pth`: the five checkpoints (236 MB each).

## Usage

Place the files in `docker/model/` of the code repository and build the
container, or load them with `nnUNetPredictor.initialize_from_trained_model_folder`
(the trainer class `nnUNetTrainer_Epoch5000_Lr1e3` from the repository must be
importable).

## License

Apache License 2.0. Trained only on data provided by the FLARE 2026 organisers;
no external pre-trained weights.
