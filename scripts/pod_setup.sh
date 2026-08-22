#!/usr/bin/env bash
# Pod-side setup + run for the embedding build. Copied to the pod and executed
# by scripts/remote_embed.sh — not meant to run on the laptop.
set -euo pipefail
cd "$(dirname "$0")"

[ -d venv ] || python3 -m venv venv
PIP=./venv/bin/pip

$PIP install --quiet numpy
$PIP install --quiet fastembed-gpu || $PIP install --quiet fastembed

# CUDA userspace libraries. onnxruntime's CUDA provider dlopens this fixed set:
# cudart, cublas, cudnn 9, cufft, curand. NVIDIA has renamed the pip packages
# across packaging eras and left failing stubs behind, so try variants in order.
try_install() {
  local pkg
  for pkg in "$@"; do
    if $PIP install --quiet "$pkg" 2>/dev/null; then
      echo "[setup] installed $pkg"
      return 0
    fi
  done
  echo "[setup] WARNING: could not install any of: $*" >&2
}

try_install nvidia-cuda-runtime nvidia-cuda-runtime-cu13 nvidia-cuda-runtime-cu12
try_install nvidia-cublas nvidia-cublas-cu13 nvidia-cublas-cu12
try_install nvidia-cudnn-cu13 nvidia-cudnn-cu12
try_install nvidia-cufft nvidia-cufft-cu13 nvidia-cufft-cu12
try_install nvidia-curand nvidia-curand-cu13 nvidia-curand-cu12

# If an earlier CPU-fastembed fallback ran, plain onnxruntime overwrote the GPU
# build's files (same import name). Reassert the GPU build if it's installed.
if $PIP show onnxruntime-gpu >/dev/null 2>&1; then
  $PIP install --quiet --force-reinstall --no-deps onnxruntime-gpu
  echo "[setup] onnxruntime-gpu reasserted"
fi

./venv/bin/python embed_cards_remote.py cards.jsonl.gz --out-dir out
