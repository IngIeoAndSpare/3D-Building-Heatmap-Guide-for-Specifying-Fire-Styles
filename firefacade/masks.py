import warnings

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class WallWindowMasker:
    def __init__(self, checkpoint, device=None, points_per_side=64, pred_iou_thresh=0.7,
                 stability_score_thresh=0.85, min_mask_region_area=50):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint, device=self.device)
        self.generator = SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
                                                    stability_score_thresh=stability_score_thresh,
                                                    min_mask_region_area=min_mask_region_area)

    def segments(self, image_rgb):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="(?s).*Skipping the post-processing step", category=UserWarning)
            anns = self.generator.generate(np.ascontiguousarray(image_rgb))
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return anns

    def __call__(self, image_rgb, max_dim=1024):
        h, w = image_rgb.shape[:2]
        image = image_rgb
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        building = cv2.morphologyEx((gray < 250).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        building = cv2.morphologyEx(building, cv2.MORPH_OPEN, kernel).astype(bool)
        total = image.shape[0] * image.shape[1]
        min_area, max_area = int(total * 0.0001), int(total * 0.15)
        anns = self.segments(image)
        windows = []
        for a in anns:
            seg = a["segmentation"]
            n = seg.sum()
            overlap = (seg & building).sum() / n if n > 0 else 0
            if overlap < 0.5:
                continue
            if min_area < a["area"] < max_area and gray[seg].mean() < 180:
                windows.append(seg)
        mask = building.astype(np.uint8) * 255
        for seg in windows:
            mask[seg] = 0
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask, {"segments": len(anns), "windows": len(windows)}
