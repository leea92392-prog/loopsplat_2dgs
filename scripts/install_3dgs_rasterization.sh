#!/bin/bash
# Build and install diff-gaussian-rasterization (3DGS) from thirdparty.
# Uses the w-pose fork (rmurai0610/diff-gaussian-rasterization-w-pose) which
# supports native depth/opacity rendering and optional SE(3) pose gradients.
# Run from repo root. Requires CUDA, PyTorch with CUDA, and ninja (optional).
# If you see CUDA arch errors, set e.g.: export TORCH_CUDA_ARCH_LIST="8.0;8.6"

set -e
cd "$(dirname "$0")/.."
echo "Initializing diff-gaussian-rasterization submodules (e.g. third_party/glm) ..."
(cd thirdparty/diff-gaussian-rasterization && git submodule update --init --recursive)
echo "Installing diff-gaussian-rasterization from thirdparty/diff-gaussian-rasterization ..."
pip install --no-build-isolation ./thirdparty/diff-gaussian-rasterization
echo "Done. Use renderer: \"3dgs\" in config to enable 3DGS rendering."
