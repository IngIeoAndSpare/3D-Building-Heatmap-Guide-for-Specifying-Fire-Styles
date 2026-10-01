import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGINE_DIR = os.path.abspath(os.path.expanduser(os.environ.get("FIREFACADE_ENGINE") or os.path.join(ROOT, "engine")))
COMFY_DIR = os.path.join(ENGINE_DIR, "ComfyUI")
MODELS_DIR = os.path.join(COMFY_DIR, "models")
SAM2_CHECKPOINT = os.path.join(MODELS_DIR, "sam2", "sam2.1_hiera_small.pt")
STYLE_IMAGE = os.path.join(ROOT, "styles", "firewall_style.png")
GRAPH_TEMPLATE = os.path.join(ROOT, "workflows", "stylize_api.json")

SEED = 1093557031939530
SCALE = 1.0
ENGINE_PORT = 8188

# Weights the pipeline expects, relative to ComfyUI; scripts/download_models.sh downloads and
# verifies them.
MODELS = [
    "models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors",
    "models/controlnet/t2i-adapter-lineart-sdxl-1.0.fp16.safetensors",
    "models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors",
    "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/sk_model.pth",
    "custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/sk_model2.pth",
    "models/sam2/sam2.1_hiera_small.pt",
]


def missing_weights():
    return [p for p in MODELS if not os.path.exists(os.path.join(COMFY_DIR, p))]
