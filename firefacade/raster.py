import cv2
import numpy as np


def triangle_area(p):
    p = np.asarray(p, dtype=np.float64)
    return 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))


def bbox_of(pts, shape, pad=1):
    p = np.asarray(pts, dtype=np.float64)
    h, w = shape
    x0 = min(max(int(np.floor(p[:, 0].min())) - pad, 0), w)
    y0 = min(max(int(np.floor(p[:, 1].min())) - pad, 0), h)
    x1 = min(max(int(np.ceil(p[:, 0].max())) + pad + 1, 0), w)
    y1 = min(max(int(np.ceil(p[:, 1].max())) + pad + 1, 0), h)
    return y0, y1, x0, x1


def mask_fill(pts, box):
    y0, y1, x0, x1 = box
    sub = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    if sub.size:
        cv2.fillConvexPoly(sub, np.int32(np.float32(pts)) - np.int32([x0, y0]), 1)
    return sub == 1


def mask_center(pts, box):
    y0, y1, x0, x1 = box
    p = np.asarray(pts, dtype=np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx, cy = xx + 0.5, yy + 0.5

    def edge(a, b):
        return (b[0] - a[0]) * (cy - a[1]) - (b[1] - a[1]) * (cx - a[0])

    e0, e1, e2 = edge(p[0], p[1]), edge(p[1], p[2]), edge(p[2], p[0])
    return ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))


def raster(pts, shape, rule):
    box = bbox_of(pts, shape)
    if box[1] <= box[0] or box[3] <= box[2]:
        return box, np.zeros((0, 0), dtype=bool)
    return box, (mask_center(pts, box) if rule == "center" else mask_fill(pts, box))


def warp_into(src, dst, src_pts, dst_pts, interp, rule, border=cv2.BORDER_CONSTANT):
    box, mk = raster(dst_pts, dst.shape[:2], rule)
    if mk.size == 0 or not mk.any():
        return 0
    y0, y1, x0, x1 = box
    M = cv2.getAffineTransform(np.float32(src_pts) - 0.5, np.float32(dst_pts) - 0.5).astype(np.float64)
    M[0, 2] -= x0
    M[1, 2] -= y0
    warped = cv2.warpAffine(src, M, (x1 - x0, y1 - y0), flags=interp, borderMode=border)
    if warped.ndim == 2:
        warped = warped[:, :, None]
    np.copyto(dst[y0:y1, x0:x1], warped.reshape(dst[y0:y1, x0:x1].shape), where=mk[:, :, None])
    return int(mk.sum())


def coverage(triangles, shape):
    cov = np.zeros(shape, dtype=bool)
    for pts in triangles:
        box, mk = raster(pts, shape, "center")
        if mk.size:
            y0, y1, x0, x1 = box
            cov[y0:y1, x0:x1] |= mk
    return cov


def pad_nearest(img, cov):
    if cov.all() or not cov.any():
        return img
    from scipy.ndimage import distance_transform_edt
    iy, ix = distance_transform_edt(~cov, return_distances=False, return_indices=True)
    return img[iy, ix]
