#!/bin/bash
# Local build of the submission image. Run on a machine with docker installed.
#
# Usage:
#   ./build.sh              build image flare-ens5:v1
#   ./build.sh <tag>        set a custom tag
#   ./build.sh <tag> save   additionally save to flare-ens5.tar.gz
set -e
TAG=${1:-flare-ens5:v1}
SAVE=${2:-}

echo ">> building image $TAG"
docker build -t "$TAG" .

echo
echo ">> image size"
docker image ls "$TAG"

if [ "$SAVE" = "save" ]; then
  OUT=$(echo "$TAG" | tr ':/' '__').tar.gz
  echo
  echo ">> saving to $OUT (a few minutes)"
  docker save "$TAG" | gzip > "$OUT"
  ls -lh "$OUT"
fi
