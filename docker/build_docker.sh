#!/usr/bin/env bash

set -e

TAG=$1

docker build --pull --build-arg CACHEBUST="$(date +%s)" -t "${TAG}" .
docker push "${TAG}"
