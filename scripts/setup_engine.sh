#!/usr/bin/env bash
# Installs the diffusion engine (ComfyUI and the node packs at the commits) and SAM 2.1
#   bash scripts/setup_engine.sh            # installs into ./engine
#   FIREFACADE_ENGINE=/path bash scripts/setup_engine.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${FIREFACADE_ENGINE:-$ROOT/engine}
PY=${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
PY=${PY:-python3}
echo "installing into $($PY -c 'import sys; print(sys.executable)')"
$PY -m pip install -q -U pip setuptools wheel

clone_at() {
    local url=$1 dir=$2 commit=$3
    if [ ! -d "$dir/.git" ]; then
        git clone --quiet "$url" "$dir"
    fi
    git -C "$dir" checkout --quiet "$commit" 2>/dev/null || {
        git -C "$dir" fetch --quiet origin "$commit"
        git -C "$dir" checkout --quiet "$commit"
    }
    echo "$(basename "$dir") at $(git -C "$dir" rev-parse --short HEAD)"
}

mkdir -p "$ENGINE"
clone_at https://github.com/Comfy-Org/ComfyUI.git "$ENGINE/ComfyUI" b4d3652d88927a341f22a35252471562f1f25f1b
NODES=$ENGINE/ComfyUI/custom_nodes
clone_at https://github.com/cubiq/ComfyUI_IPAdapter_plus.git "$NODES/ComfyUI_IPAdapter_plus" e5c9744b232ac812b5430fbdc4ebe9f7789bd8d2
clone_at https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git "$NODES/ComfyUI-Advanced-ControlNet" b66cd70c9845a109a85b4a0ef13cefda41ca6039
clone_at https://github.com/Fannovel16/comfyui_controlnet_aux.git "$NODES/comfyui_controlnet_aux" 1e9eac6377c882da8bb360c7544607036904362c
clone_at https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git "$NODES/ComfyUI-Custom-Scripts" f2838ed5e59de4d73cde5c98354b87a8d3200190
clone_at https://github.com/ThatGlennD/ComfyUI-Image-Analysis-Tools.git "$NODES/ComfyUI-Image-Analysis-Tools" 2db611da6370d1af2660b0da29cb6514066edbe1

PIP="$PY -m pip install -c $ROOT/requirements.txt"
$PIP -r "$ENGINE/ComfyUI/requirements.txt"
$PIP -r "$NODES/comfyui_controlnet_aux/requirements.txt"
$PIP -r "$NODES/ComfyUI-Image-Analysis-Tools/requirements.txt"

# SAM 2.1 at the commit used for the paper, without the CUDA extension: the masks were made
# without it and its post-processing would change them.
SAM2_BUILD_CUDA=0 $PIP --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

mkdir -p "$ENGINE/ComfyUI/models/"{checkpoints,controlnet,ipadapter,clip_vision,sam2}
echo "engine installed in $ENGINE"
