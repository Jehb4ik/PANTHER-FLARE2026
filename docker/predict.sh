#!/bin/bash
# FLARE 2026 Task 1 submission entrypoint.
# The organizers run it as: /bin/bash -c "sh predict.sh"
set -e

# CPU threads are capped on purpose: the eval host has 12 threads, and going
# beyond that only adds context-switch overhead.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
# Otherwise nnU-Net spends seconds checking environment variables we do not set.
export nnUNet_raw=/workspace/nnunet_dummy
export nnUNet_preprocessed=/workspace/nnunet_dummy
export nnUNet_results=/workspace/nnunet_dummy

python3 /workspace/predict.py
