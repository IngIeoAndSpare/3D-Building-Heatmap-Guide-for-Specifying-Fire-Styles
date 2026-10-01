"""Browser demo: upload a textured building model and get it back with fire appearance
applied to its facades.

  run cli example 
  python demo.py 
  python demo.py --gpu 1 --port 7861 # setting gpu worker index, multi gpu not cover :( sorry
"""

import os
import sys
import glob
import time
import argparse
import traceback
import gradio as gr

# module
from firefacade import config
from firefacade.engine import ensure_engine
from firefacade.masks import WallWindowMasker
from firefacade.pipeline import Pipeline


ROOT = os.path.dirname(os.path.abspath(__file__))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="address the UI listens on; 0.0.0.0 makes it reachable from other machines")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--gpu", type=int, default=None, help="CUDA device index; default: the current CUDA_VISIBLE_DEVICES")
    ap.add_argument("--engine-url", default=None, help="use a running engine instead of starting one")
    ap.add_argument("--engine-port", type=int, default=None, help="port of the engine this script starts (default 8188)")
    ap.add_argument("--runs", default="runs", help="where results are written")
    ap.add_argument("--examples", default=os.path.join(ROOT, "examples"), help="folder of example buildings")
    ap.add_argument("--share", action="store_true", help="also publish a temporary public Gradio link")
    return ap.parse_args()

# gradio building model finder
def find_examples(root, cache_dir):
    """[(name, obj, mtl or None, textures, thumbnail)] for every folder of root with one OBJ."""
    import cv2
    from firefacade.obj_io import load_building
    from firefacade.preview import render_thumbnail
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        objs = glob.glob(os.path.join(d, "*.obj"))
        if not os.path.isdir(d) or len(objs) != 1:
            continue
        mtls = glob.glob(os.path.join(d, "*.mtl"))
        textures = sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))
                          and os.path.basename(p) != "thumbnail.png")
        name = os.path.basename(d)
        thumb = os.path.join(d, "thumbnail.png")
        if not os.path.exists(thumb):
            thumb = os.path.join(cache_dir, f"{name}.png")
            if not os.path.exists(thumb):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    cv2.imwrite(thumb, render_thumbnail(load_building(objs[0], mtls[0] if mtls else None, textures)))
                except Exception as e:
                    print(f"example {name} skipped: {e}")
                    continue
        label = name.split("_", 1)[1] if name.split("_", 1)[0].isdigit() and "_" in name else name
        out.append((label, objs[0], mtls[0] if mtls else None, textures, thumb))
    return out


def main():
    args = parse()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    missing = config.missing_weights()
    if missing:
        print("missing weights (see README):\n  " + "\n  ".join(missing))
        return 1
    engine, proc = ensure_engine(args.engine_url, args.engine_port or config.ENGINE_PORT)
    masker = WallWindowMasker(config.SAM2_CHECKPOINT)
    pipe = Pipeline(engine, masker)
    examples = find_examples(args.examples, os.path.join(args.runs, "_thumbnails"))

    def run(model_files, texture_files, progress=gr.Progress()):
        model_files = [f if isinstance(f, str) else f.name for f in (model_files or [])]
        texture_files = [f if isinstance(f, str) else f.name for f in (texture_files or [])]
        objs = [f for f in model_files if f.lower().endswith(".obj")]
        mtls = [f for f in model_files if f.lower().endswith(".mtl")]
        if len(objs) != 1:
            raise gr.Error("Upload exactly one .obj file.")
        if not texture_files:
            raise gr.Error("Upload the texture atlas image(s) of the model.")
        name = os.path.splitext(os.path.basename(objs[0]))[0]
        out_dir = os.path.join(args.runs, time.strftime("%Y%m%d_%H%M%S") + "_" + name)
        try:
            result = pipe.run(objs[0], mtls[0] if mtls else None, texture_files, out_dir,
                              progress=lambda msg: progress(0, desc=msg))
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(str(e))
        failed = [f["facade"] for f in result["facades"] if f["status"] == "failed"]
        if failed:
            gr.Warning("Kept the original texture for: " + ", ".join(failed))
        return result.get("original_glb"), result.get("stylized_glb"), result["archive"]

    with gr.Blocks(title="Facade fire appearance stylization") as ui:
        gr.Markdown("## Heatmap-guided fire appearance stylization of building facade textures\n"
                    "Upload a textured building model and its texture atlas. The facades are extracted, stylized "
                    "with a fire appearance anchored at a reference window, and written back into the atlas.")
        with gr.Row():
            model_in = gr.File(label="Building model (.obj)", file_count="multiple", file_types=[".obj", ".mtl"])
            tex_in = gr.File(label="Texture atlas (.png or .jpg)", file_count="multiple", file_types=["image"])
        btn = gr.Button("Apply fire appearance", variant="primary")
        with gr.Row():
            original = gr.Model3D(label="Input model")
            stylized = gr.Model3D(label="Stylized model")
        archive = gr.File(label="Stylized model package (zip)")
        btn.click(run, [model_in, tex_in], [original, stylized, archive])

        if examples:
            gr.Markdown("### Example building models\nClick a building to load it and run the stylization.")
            gallery = gr.Gallery(value=[(e[4], e[0]) for e in examples], columns=min(len(examples), 5), height=260,
                                 object_fit="contain", allow_preview=False, show_label=False)

            def load_example(evt: gr.SelectData):
                _, obj, mtl, textures, _ = examples[evt.index]
                return [obj] + ([mtl] if mtl else []), textures

            gallery.select(load_example, None, [model_in, tex_in]).then(run, [model_in, tex_in], [original, stylized, archive])

    try:
        ui.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                          allowed_paths=[os.path.abspath(args.runs), os.path.abspath(args.examples)])
    finally:
        if proc is not None:
            proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
