import cv2
import numpy as np
from PIL import Image


def window_boxes(mask, min_area=50):
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    h, w = thresh.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i][:5]
        if area < min_area:
            continue
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if area > 0.15 * h * w or (touches and area > 0.005 * h * w):
            continue
        out.append({"box": (y, x, y + bh, x + bw), "center": (int(centroids[i][0]), int(centroids[i][1]))})
    return out


def reference_position(mask, windows):
    h, w = mask.shape
    if windows:
        lower = [win for win in windows if win["center"][1] >= h * 2.0 / 3.0]
        best = min(lower or windows, key=lambda win: abs(win["center"][0] - w / 2))
        y1, x1, _, x2 = best["box"]
        return (x1 + x2) // 2, y1
    return w // 2, int(h * 0.8)


def fire_heatmap(mask, scale=1.0, reference=None, base_sigma_ratio=0.40):
    mask = np.asarray(mask)
    h, w = mask.shape
    windows = window_boxes(mask)
    x0, y0 = reference if reference is not None else reference_position(mask, windows)
    w_ref = None
    if windows:
        near = min(windows, key=lambda win: (win["center"][0] - x0) ** 2 + (win["center"][1] - y0) ** 2)
        w_ref = near["box"][3] - near["box"][1]
    if not w_ref or w_ref <= 0:
        w_ref = int(h * 0.08)

    l_s = scale * h
    sigma_0 = base_sigma_ratio * w_ref
    heat = np.zeros((h, w), dtype=np.float32)
    x_diff_sq = (np.arange(w, dtype=np.float32) - x0) ** 2
    if y0 >= 0:
        n_up = y0 + 1
        dy = (y0 - np.arange(n_up, dtype=np.float32)).reshape(-1, 1)
        sig = sigma_0 + scale * dy
        heat[:n_up] = (np.exp(-dy / (l_s + 1e-8)) * np.exp(-x_diff_sq / (2.0 * sig ** 2 + 1e-8))).astype(np.float32)
    down_limit = min(h, y0 + int(w_ref * 0.15) + 1)
    n_down = down_limit - (y0 + 1)
    if n_down > 0:
        dy = np.arange(1, n_down + 1, dtype=np.float32).reshape(-1, 1)
        heat[y0 + 1:down_limit] = (np.exp(-dy * 4.0 / (l_s + 1e-8))
                                   * np.exp(-x_diff_sq / (2.0 * sigma_0 ** 2 + 1e-8))).astype(np.float32)
    heat = cv2.GaussianBlur(heat, (25, 25), 0)
    heat = heat * (mask > 127).astype(np.float32)
    heat = np.clip(heat, 0.0, 1.0).astype(np.float32)
    info = {"reference": (int(x0), int(y0)), "reference_window_width": int(w_ref), "windows": len(windows),
            "scale": float(scale)}
    return heat, info


def save_change_map(heat, path):
    v = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    rgba = np.zeros(heat.shape + (4,), np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = v
    rgba[..., 3] = np.clip((1.0 - heat) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgba).save(path)


def overlay(image_rgb, heat, alpha=0.5):
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (image_rgb * (1 - alpha) + colored * alpha).astype(np.uint8)
