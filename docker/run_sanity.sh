#!/bin/bash
# Timed sanity test of the submission image: per-case inference time, a 1 Hz
# GPU memory-time log, peak GPU memory, total wall time and peak host RAM.
# Runs the container exactly as the FLARE organizers do (one GPU, -m 28G,
# default --shm-size). Run under asciinema to record the screen.
#
# Usage: ./run_sanity.sh <inputs_dir> <outputs_dir> [image]
#   <inputs_dir>   folder with <case>_0000.nii.gz
#   <outputs_dir>  masks are written here; logs go to <outputs_dir>/../sanity_logs/
#   [image]        docker image tag, default jehb4ik:latest
set -eo pipefail

IN=${1:?"usage: $0 <inputs_dir> <outputs_dir> [image]"}
OUT=${2:?"usage: $0 <inputs_dir> <outputs_dir> [image]"}
IMG=${3:-jehb4ik:latest}
LOGDIR=$(cd "$(dirname "$OUT")" && pwd)/sanity_logs
NAME=sanity_$$

# Use plain docker if the current shell is in the docker group, else sudo.
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

mkdir -p "$OUT" "$LOGDIR"; rm -f "$OUT"/*.nii.gz 2>/dev/null || true
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

echo "############################################################"
echo "# FLARE 2026 Task 1 sanity test - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "############################################################"
echo "## Host GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "## Docker: $($DOCKER --version)"
echo "## Cases: $(ls -1 "$IN"/*.nii.gz | wc -l)"
echo

# Background 1 Hz GPU memory sampler -> CSV (the GPU memory-time curve).
TS=$(date +%s)
LOG="$LOGDIR/gpu_mem_${TS}.csv"
echo "time_s,mem_used_MiB,util_pct" > "$LOG"
( while true; do
    printf "%s,%s\n" "$(date +%s.%N)" \
      "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    sleep 1
  done ) >> "$LOG" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

echo ">>> running inference (per-case timing streams below from the container) ..."
echo
/usr/bin/time -v $DOCKER run --rm -t --gpus device=0 -m 28G --name "$NAME" \
  -e PYTHONUNBUFFERED=1 \
  -v "$IN":/workspace/inputs/ \
  -v "$OUT":/workspace/outputs/ \
  "$IMG" /bin/bash -c "sh predict.sh" 2>&1 | tee "$LOGDIR/stdout_${TS}.log"

kill $SAMPLER 2>/dev/null; trap - EXIT
sleep 1

echo
echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
PEAK=$(awk -F, 'NR>1 && $2+0>m {m=$2} END{print m}' "$LOG")
NOUT=$(ls -1 "$OUT"/*.nii.gz 2>/dev/null | wc -l)
NIN=$(ls -1 "$IN"/*.nii.gz | wc -l)
echo "Peak GPU memory used : ${PEAK} MiB   (FLARE tolerance 4096 MiB)"
echo "Output masks written : ${NOUT} / ${NIN}"
echo "GPU memory-time log  : ${LOG}  (columns: time_s,mem_used_MiB,util_pct)"
echo "Container stdout log : ${LOGDIR}/stdout_${TS}.log"
echo "Per-case seconds are printed above by the container; each must be < 60 s."
echo "Total wall time + peak host RAM are in the '/usr/bin/time -v' block above."
