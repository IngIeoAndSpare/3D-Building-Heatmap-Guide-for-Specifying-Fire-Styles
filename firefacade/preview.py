import cv2
import numpy as np
import trimesh

from .raster import warp_into, triangle_area
from .rectify import estimate_up


def to_glb(obj_path, out_path):
    scene = trimesh.load(obj_path, force="scene", process=False)
    scene.export(out_path)
    return out_path


def _up_to_z(up):
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def render_thumbnail(building, size=320, azimuth=35.0, elevation=18.0):
    F = building.F
    tri_idx = F[:, :, 0] - 1
    uv_idx = np.clip(F[:, :, 1] - 1, 0, max(len(building.VT) - 1, 0))
    uv = building.VT[uv_idx]
    tex_of = [building.texture_of(t) for t in range(len(F))]
    wh = np.array([building.textures[t].shape[1::-1] if t else (1, 1) for t in tex_of], dtype=np.float64)
    uvpx = np.stack([uv[:, :, 0] * wh[:, None, 0], (1.0 - uv[:, :, 1]) * wh[:, None, 1]], -1)
    V = (building.V - building.V.mean(0)) @ _up_to_z(estimate_up(building.V, tri_idx, uvpx)).T
    az, el = np.radians(azimuth), np.radians(elevation)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    P = V @ (Rx @ Rz).T
    xy = np.c_[P[:, 0], -P[:, 2]]
    lo, hi = xy.min(0), xy.max(0)
    scale = (size * 0.9) / max((hi - lo).max(), 1e-9)
    xy = (xy - (lo + hi) / 2) * scale + size / 2
    img = np.full((size, size, 3), 255, np.uint8)
    tri = P[tri_idx]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for f in np.argsort(-tri[:, :, 1].mean(1)):
        if n[f, 1] >= 0 or tex_of[f] is None:
            continue
        dst = xy[tri_idx[f]]
        if triangle_area(dst) < 0.3 or triangle_area(uvpx[f]) < 1e-6:
            continue
        warp_into(building.textures[tex_of[f]][:, :, :3], img, uvpx[f], dst, cv2.INTER_AREA, "center")
    return img
