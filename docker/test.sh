#!/bin/bash
# Test the image on local test data.
#
# Usage:
#   ./test.sh <tag> <inputs_dir> <outputs_dir>
#
# Example:
#   mkdir -p /tmp/flare_in /tmp/flare_out
#   cp /path/to/*_0000.nii.gz /tmp/flare_in/
#   ./test.sh flare-ens5:v1 /tmp/flare_in /tmp/flare_out
set -e
TAG=${1:?"provide the image tag"}
IN=${2:?"provide the input dir with *_0000.nii.gz"}
OUT=${3:?"provide the output dir"}

mkdir -p "$OUT"
echo ">> running $TAG"
echo "   input:  $IN"
echo "   output: $OUT"

# Flags mirror how the FLARE organizers run it:
#   --gpus "device=0"     one GPU, as on the eval host
#   --shm-size=8gb        nnU-Net DataLoader needs shm
#   -e nnUNet_*=...       env vars are also set by predict.sh, duplicated here
#                         for running without the sh wrapper
docker run --rm --gpus "device=0" --shm-size=8gb \
  -v "$IN:/workspace/inputs:ro" \
  -v "$OUT:/workspace/outputs" \
  "$TAG" /bin/bash -c "sh predict.sh"

echo
echo ">> result"
ls -la "$OUT"
