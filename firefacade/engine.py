import io
import os
import sys
import copy
import json
import time
import uuid
import subprocess

import requests
from PIL import Image

from . import config


class Engine:
    def __init__(self, url=f"http://127.0.0.1:{config.ENGINE_PORT}", template=config.GRAPH_TEMPLATE):
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.template = json.load(open(template))

    def alive(self, timeout=3):
        try:
            return requests.get(f"{self.url}/system_stats", timeout=timeout).ok
        except requests.RequestException:
            return False

    def upload(self, path, name=None):
        name = name or os.path.basename(path)
        with open(path, "rb") as fh:
            r = requests.post(f"{self.url}/upload/image", files={"image": (name, fh, "image/png")},
                              data={"overwrite": "true"}, timeout=120)
        r.raise_for_status()
        return r.json()["name"]

    def build(self, facade, mask, heatmap, style, seed=config.SEED):
        """Graph with the uploaded file names filled in."""
        g = copy.deepcopy(self.template)
        g["31"]["inputs"]["image"] = facade
        g["44"]["inputs"]["image"] = mask
        g["99"]["inputs"]["image"] = heatmap
        g["12"]["inputs"]["image"] = style
        g["16"]["inputs"]["seed"] = int(seed)
        g["23"]["inputs"]["filename_prefix"] = f"firefacade/{uuid.uuid4().hex}"
        return g

    def run(self, graph, timeout=1800):
        r = requests.post(f"{self.url}/prompt", json={"prompt": graph, "client_id": self.client_id}, timeout=60)
        res = r.json()
        if "error" in res:
            raise RuntimeError(f"engine rejected the graph: {res['error']} {res.get('node_errors', '')}")
        pid = res["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            hist = requests.get(f"{self.url}/history/{pid}", timeout=30).json()
            if pid in hist:
                item = hist[pid]
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise RuntimeError(f"engine error: {msgs[0][1].get('exception_message') if msgs else status}")
                info = item["outputs"]["23"]["images"][0]
                r = requests.get(f"{self.url}/view", params={"filename": info["filename"], "subfolder": info.get("subfolder", ""),
                                                             "type": info.get("type", "output")}, timeout=120)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            time.sleep(0.5)
        raise TimeoutError("engine did not finish in time")

    def stylize(self, facade_png, mask_png, heatmap_png, style_png, seed=config.SEED):
        names = [self.upload(p) for p in (facade_png, mask_png, heatmap_png, style_png)]
        return self.run(self.build(*names, seed=seed))


# run comfyui
def start_server(port=config.ENGINE_PORT, comfy_dir=config.COMFY_DIR, log_path=None, extra_args=()):
    main = os.path.join(comfy_dir, "main.py")
    if not os.path.exists(main):
        raise FileNotFoundError(f"{main} not found; run scripts/setup_engine.sh first")
    args = [sys.executable, main, "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch", *extra_args]
    log_path = log_path or os.path.join(comfy_dir, "firefacade_engine.log")
    proc = subprocess.Popen(args, cwd=comfy_dir, stdout=open(log_path, "a"), stderr=subprocess.STDOUT)
    proc.log_path = log_path
    return proc


def wait_ready(engine, proc=None, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if engine.alive():
            return True
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"run error {proc.returncode}; {getattr(proc, 'log_path', 'its log')}")
        time.sleep(2)
    raise TimeoutError(f"time out {getattr(proc, 'log_path', 'its log')}")


def ensure_engine(url=None, port=config.ENGINE_PORT, log_path=None):
    if url:
        engine = Engine(url)
        if not engine.alive():
            raise RuntimeError(f"no engine at {url}")
        return engine, None
    engine = Engine(f"http://127.0.0.1:{port}")
    if engine.alive():
        print(f"using the engine already running at {engine.url}")
        return engine, None
    proc = start_server(port, log_path=log_path)
    print(f"starting the engine on port {port}, log {proc.log_path}")
    wait_ready(engine, proc)
    return engine, proc
