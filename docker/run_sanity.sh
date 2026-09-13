#!/bin/bash
# Recorded sanity test for the FLARE 2026 Task1 submission (jehb4ik).
# Streams on-screen proof: per-case inference time, GPU memory-time curve, peak VRAM,
# total wall time and peak host RAM. Run this under asciinema to produce the video.
set -eo pipefail

IN=/home/ubuntu/Validation-Public-Images
OUT=/home/ubuntu/jehb4ik_outputs
IMG=jehb4ik:latest

# Use plain docker if the current shell is in the docker group, else sudo.
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

mkdir -p "$OUT"; rm -f "$OUT"/* 2>/dev/null || true
$DOCKER rm -f jehb4ik >/dev/null 2>&1 || true

echo "############################################################"
echo "# FLARE 2026 Task1 sanity test - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "############################################################"
echo "## Host GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "## Docker: $($DOCKER --version)"
echo "## Cases: $(ls -1 "$IN"/*.nii.gz | wc -l)"
echo

# Background 1 Hz GPU memory sampler -> CSV (the GPU memory-time curve).
TS=$(date +%s)
LOG=/home/ubuntu/gpu_mem_${TS}.csv
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
# Flags mirror the FLARE evaluation protocol: one GPU, 28G RAM cap, default shm.
/usr/bin/time -v $DOCKER run --rm -t --gpus device=0 -m 28G --name jehb4ik \
  -e PYTHONUNBUFFERED=1 \
  -v "$IN":/workspace/inputs/ \
  -v "$OUT":/workspace/outputs/ \
  "$IMG" /bin/bash -c "sh predict.sh" 2>&1 | tee "/home/ubuntu/jehb4ik_stdout_${TS}.log"

kill $SAMPLER 2>/dev/null; trap - EXIT
sleep 1

echo
echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
PEAK=$(awk -F, 'NR>1 && $2+0>m {m=$2} END{print m}' "$LOG")
NOUT=$(ls -1 "$OUT"/*.nii.gz 2>/dev/null | wc -l)
NIN=$(ls -1 "$IN"/*.nii.gz | wc -l)
echo "Peak GPU memory used : ${PEAK} MiB   (FLARE gives a 4096 MiB tolerance)"
echo "Output masks written : ${NOUT} / ${NIN}"
echo "GPU memory-time log  : ${LOG}  (columns: time_s,mem_used_MiB,util_pct)"
echo "Per-case seconds are printed above by the container; each must be < 60 s."
echo "Total wall time + peak host RAM are in the '/usr/bin/time -v' block above."
