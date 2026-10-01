import os
import json
import time
import shutil

import cv2
import numpy as np
from PIL import Image

from . import config
from .obj_io import load_building, write_building
from .rectify import rectify, dihedral
from .heatmap import fire_heatmap, save_change_map, overlay
from .bake import bake
from .preview import to_glb


class Pipeline:
    def __init__(self, engine, masker, style_image=config.STYLE_IMAGE, scale=config.SCALE, seed=config.SEED, min_side=64):
        self.engine = engine
        self.masker = masker
        self.style_image = style_image
        self.scale = scale
        self.seed = seed
        self.min_side = min_side

    def run(self, obj_path, mtl_path, texture_paths, out_dir, progress=None):
        report = progress or (lambda msg: None)
        os.makedirs(out_dir, exist_ok=True)
        work = os.path.join(out_dir, "facades")
        os.makedirs(work, exist_ok=True)

        timing = {}
        t = time.perf_counter()
        building = load_building(obj_path, mtl_path, texture_paths)
        original_dir = os.path.join(out_dir, "original")
        write_building(building, original_dir, building.textures)
        facades = rectify(building)
        timing["rectify_s"] = time.perf_counter() - t
        report(f"{len(facades)} facades extracted")

        styled, records = {}, []
        for i, fac in enumerate(facades):
            rec = {"facade": fac.name, "width": fac.size[0], "height": fac.size[1]}
            if min(fac.size) < self.min_side:
                rec["status"] = "kept (too small)"
                records.append(rec)
                continue
            try:
                report(f"[{i + 1}/{len(facades)}] {fac.name}: wall-window mask")
                rgb = np.ascontiguousarray(fac.image[:, :, ::-1])
                t = time.perf_counter()
                mask, minfo = self.masker(np.ascontiguousarray(fac.raw_image[:, :, ::-1]))
                mask = dihedral(mask, fac.rot_k, fac.mirror)
                rec["mask_s"] = time.perf_counter() - t
                t = time.perf_counter()
                heat, hinfo = fire_heatmap(mask, self.scale)
                rec["heatmap_s"] = time.perf_counter() - t
                base = os.path.join(work, fac.name)
                cv2.imwrite(base + ".png", fac.image)
                cv2.imwrite(base + "_mask.png", mask)
                save_change_map(heat, base + "_heatmap.png")
                Image.fromarray(overlay(rgb, heat)).save(base + "_heatmap_overlay.png")

                report(f"[{i + 1}/{len(facades)}] {fac.name}: stylization")
                t = time.perf_counter()
                out = self.engine.stylize(base + ".png", base + "_mask.png", base + "_heatmap.png", self.style_image, self.seed)
                rec["stylize_s"] = time.perf_counter() - t
                out = np.asarray(out)[:, :, ::-1]
                h, w = fac.image.shape[:2]
                if out.shape[:2] != (h // 8 * 8, w // 8 * 8):
                    raise RuntimeError(f"unexpected output size {out.shape[:2]}")
                canvas = fac.image.copy()
                y0, x0 = (h % 8) // 2, (w % 8) // 2
                canvas[y0:y0 + out.shape[0], x0:x0 + out.shape[1]] = out
                cv2.imwrite(base + "_stylized.png", canvas)
                styled[fac.name] = canvas
                rec.update({"status": "stylized", "mask": minfo, "heatmap": hinfo})
            except Exception as e:
                report(f"{fac.name}: failed ({e}); original texture kept")
                rec.update({"status": "failed", "error": str(e)})
            records.append(rec)

        report("atlas reconstruction")
        t = time.perf_counter()
        textures = bake(building, facades, styled)
        model_dir = os.path.join(out_dir, "stylized")
        obj_out = write_building(building, model_dir, textures)
        timing["bake_s"] = time.perf_counter() - t
        result = {"building": building.name, "facades": records, "timing_s": timing, "stylized_obj": obj_out,
                  "original_obj": os.path.join(original_dir, os.path.basename(obj_path))}
        try:
            result["stylized_glb"] = to_glb(obj_out, os.path.join(out_dir, f"{building.name}_stylized.glb"))
            result["original_glb"] = to_glb(result["original_obj"], os.path.join(out_dir, f"{building.name}_original.glb"))
        except Exception as e:
            result["preview_error"] = str(e)
        archive = shutil.make_archive(os.path.join(out_dir, f"{building.name}_stylized"), "zip", model_dir)
        result["archive"] = archive
        json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
        return result
