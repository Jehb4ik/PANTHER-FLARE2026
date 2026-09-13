#!/bin/bash
# Preprocess Dataset500_FLAREPanCancer with the shipped plan nnUNetPlans_tr.
#
# Assumes reproduce_dataset500.py has already assembled
# <ROOT>/nnUNet_raw/Dataset500_FLAREPanCancer/{imagesTr,labelsTr,dataset.json}.
#
# Steps:
#   1. Set nnU-Net environment variables to <ROOT>.
#   2. Extract dataset fingerprint (reads every image header, 30-60 min).
#   3. Copy our nnUNetPlans_tr.json into the preprocessed dir.
#   4. Run preprocess on 3d_fullres with -p nnUNetPlans_tr (30-60 min).
#
# Result:
#   <ROOT>/nnUNet_preprocessed/Dataset500_FLAREPanCancer/
#     dataset_fingerprint.json
#     nnUNetPlans_tr.json           # shipped plan
#     nnUNetPlans_tr_3d_fullres/    # ~120-180 GB of preprocessed .npy/.npz
#
# Usage:
#   ./preprocess_dataset500.sh /path/to/root
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <ROOT>"
  echo "  <ROOT> must contain nnUNet_raw/Dataset500_FLAREPanCancer/"
  exit 1
fi

ROOT="$1"
HERE=$(cd "$(dirname "$0")" && pwd)

if [ ! -d "$ROOT/nnUNet_raw/Dataset500_FLAREPanCancer" ]; then
  echo "expected $ROOT/nnUNet_raw/Dataset500_FLAREPanCancer to exist"
  echo "run reproduce_dataset500.py --root $ROOT first"
  exit 1
fi

export nnUNet_raw="$ROOT/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/nnUNet_preprocessed"
export nnUNet_results="$ROOT/nnUNet_results"
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

echo "[1/3] extract_fingerprint (~30-60 min, one pass over 8762 files)"
nnUNetv2_extract_fingerprint -d 500 -np 16

DST_PP="$nnUNet_preprocessed/Dataset500_FLAREPanCancer"
if [ ! -f "$DST_PP/nnUNetPlans_tr.json" ]; then
  echo "[2/3] copying plan nnUNetPlans_tr.json into preprocessed dir"
  cp "$HERE/nnUNetPlans_tr.json" "$DST_PP/nnUNetPlans_tr.json"
else
  echo "[2/3] $DST_PP/nnUNetPlans_tr.json already exists, keeping it"
fi

echo "[3/3] preprocess 3d_fullres with plan nnUNetPlans_tr (~30-60 min)"
nnUNetv2_preprocess -d 500 -c 3d_fullres -plans_name nnUNetPlans_tr -np 16

echo
echo "done. next: train a fold, for example on GPU 0"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo "  nnUNetv2_train 500 3d_fullres 0 \\"
echo "    -tr nnUNetTrainer_Epoch5000_Lr1e3 -p nnUNetPlans_tr --npz"
echo
echo "install the trainer once (once per Python env):"
echo "  cp $HERE/nnUNetTrainer_Epoch5000_Lr1e3.py \\"
echo "    \$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/variants/training_length/"
