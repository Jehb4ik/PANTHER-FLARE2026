# Mahalanobis shape gate

Post-hoc abstention gate applied to the final ensemble mask. Five shape
features of the binary mask (volume in ml, number of connected components
>= 27 voxels, largest-component volume, largest-component fraction,
surface-to-volume ratio) are transformed with `log1p`; the squared Mahalanobis
distance to the distribution fitted on the 172 healthy CT scans is thresholded
at tau = 7. AUROC 0.974 on 172 healthy vs 50 public cancer scans; false-positive
rate on healthy scans 69.8 % -> 8.1 %; no mask changed on hidden validation.

## Files

- `fit_mahalanobis_gate.py`: computes the features on the ensemble predictions,
  fits mu and Sigma^-1 on the non-empty healthy masks (covariance with a 1e-3
  ridge), prints the threshold sweep and a 5-fold cross-validation of the
  Youden optimum, and writes `maha_gate.json`.
- `maha_gate.fitted.json`: the parameters shipped in the submitted image
  (identical to `../docker/maha_gate.json`).

## Usage

```bash
# predictions of the 5-fold ensemble for both cohorts are required first
python fit_mahalanobis_gate.py \
  --pred-healthy /data/pred/healthy_ens5 --ct-healthy /data/flare26/validation/HealthyImages-noLesion \
  --pred-public  /data/pred/public_ens5  --ct-public  /data/flare26/validation/Validation-Public-Images \
  --out-gate maha_gate.json --out-csv maha_gate_features.csv
```

Empty predictions count as distance 0 (already abstained). The final tau = 7
is taken from the sweep over {5, 7, 10, 14, 20}; cross-validated Youden optima
on our predictions were 8.77 +- 2.05.
