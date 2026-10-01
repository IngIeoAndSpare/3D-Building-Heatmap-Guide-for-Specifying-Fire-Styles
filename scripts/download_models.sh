#!/usr/bin/env bash
# Downloads the model weights into engine/ComfyUI under the names the pipeline expects and
# verifies every file by SHA-256. Files that are already in place are not downloaded again.
#   bash scripts/download_models.sh            # download what is missing, then verify
#   bash scripts/download_models.sh --check    # verify only
# Set CIVITAI_TOKEN (an API key from the Civitai account settings) if Civitai asks for a login.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${FIREFACADE_ENGINE:-$ROOT/engine}
COMFY=$ENGINE/ComfyUI
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

HF=https://huggingface.co
# destination under engine/ComfyUI | sha256 | download URL
MODELS=(
  "models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors|33e58e86686f6b386c526682b5da9228ead4f91d994abd4b053442dc5b42719e|https://civitai.com/api/download/models/782002"
  "models/controlnet/t2i-adapter-lineart-sdxl-1.0.fp16.safetensors|c19e9e1038c7c53b238185d8dc9fad87f95aa3124d211e19f39f1165bfb7230f|$HF/TencentARC/t2i-adapter-lineart-sdxl-1.0/resolve/5f1f33b049b96fa599f0aa3d77b31dbef5095e97/diffusion_pytorch_model.fp16.safetensors"
  "models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors|3f5062b8400c94b7159665b21ba5c62acdcd7682262743d7f2aefedef00e6581|$HF/h94/IP-Adapter/resolve/018e402774aeeddd60609b4ecdb7e298259dc729/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
  "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors|6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030|$HF/h94/IP-Adapter/resolve/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder/model.safetensors"
  "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/sk_model.pth|c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6|$HF/lllyasviel/Annotators/resolve/982e7edaec38759d914a963c48c4726685de7d96/sk_model.pth"
  "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/sk_model2.pth|30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0|$HF/lllyasviel/Annotators/resolve/982e7edaec38759d914a963c48c4726685de7d96/sk_model2.pth"
  "models/sam2/sam2.1_hiera_small.pt|6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38|$HF/facebook/sam2.1-hiera-small/resolve/ee5bba1d82bb8749febdf90f45e84b687142ba03/sam2.1_hiera_small.pt"
)

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# token...

fetch() {
  local url=$1 dest=$2 type
  if [[ $url == https://civitai.com/* && -n ${CIVITAI_TOKEN:-} ]]; then
    url="$url?token=$CIVITAI_TOKEN"
  fi
  mkdir -p "$(dirname "$dest")"
  type=$(curl -fL --retry 3 --progress-bar -o "$dest.part" -w '%{content_type}' "$url") || { rm -f "$dest.part"; return 1; }
  if [[ $type == text/html* ]]; then
    rm -f "$dest.part"
    return 1
  fi
  mv "$dest.part" "$dest"
}

[ -d "$COMFY" ] || { echo "$COMFY not found; run scripts/setup_engine.sh first"; exit 1; }
status=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r rel sum url <<< "$entry"
  dest=$COMFY/$rel
  if [ ! -f "$dest" ] && [ $CHECK -eq 0 ]; then
    if [ -n "$url" ]; then
      echo "downloading $rel"
      fetch "$url" "$dest" || echo "download failed: $url"
    else
      echo "place by hand: $dest (see the README)"
    fi
  fi
  if [ ! -f "$dest" ]; then
    echo "missing   $rel"
    status=1
  elif [ "$(sha256 "$dest")" = "$sum" ]; then
    echo "ok        $rel"
  else
    echo "MISMATCH  $rel (delete it and run this script again)"
    status=1
  fi
done
exit $status
