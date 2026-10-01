import numpy as np

from .raster import coverage

RESOLUTION = 1024
DIRECTIONS = (("+X", "right", 0), ("-X", "left", 0), ("+Y", "back", 1), ("-Y", "front", 1))


class Facade:
    def __init__(self, name, image, cov, triangles, rot_k, mirror, raw_image):
        self.name = name
        self.image = image
        self.coverage = cov
        self.triangles = triangles
        self.rot_k = rot_k
        self.mirror = mirror
        self.raw_image = raw_image

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]


def face_normals(V, tri_idx):
    v0, v1, v2 = V[tri_idx[:, 0]], V[tri_idx[:, 1]], V[tri_idx[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(cross, axis=1) / 2
    normal = np.where(area[:, None] > 1e-6, cross / np.maximum(area[:, None] * 2, 1e-30), np.array([0.0, 0.0, 1.0]))
    return normal, area

# Rotate the vertices into the building frame
def align(V, tri_idx):
    T = np.eye(3)
    normal, area = face_normals(V, tri_idx)
    keep = area > 1e-6
    if keep.any():
        n, w = normal[keep], area[keep] / area[keep].sum()
        centered = n - np.average(n, weights=w, axis=0)
        cov = (centered * w[:, None]).T @ centered
        _, vec = np.linalg.eigh(cov)
        axes = vec[:, ::-1].copy()
        if np.linalg.det(axes) < 0:
            axes[:, 2] = -axes[:, 2]
        V = V @ axes
        T = axes.T @ T
    size = V.max(0) - V.min(0)
    up_axis = int(np.argmax(size))
    if up_axis == 1:
        a = -np.pi / 2
        rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        V = V @ rot.T
        T = rot @ T
    elif up_axis == 0:
        a = np.pi / 2
        rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        V = V @ rot.T
        T = rot @ T
    normal, area = face_normals(V, tri_idx)
    horizontal = (area > 1e-6) & (np.abs(normal[:, 2]) > 0.7)
    cz = V[tri_idx][:, :, 2].mean(1)
    top, bottom = horizontal & (normal[:, 2] > 0), horizontal & (normal[:, 2] <= 0)
    if top.any() and bottom.any():
        top_z = (cz[top] * area[top]).sum() / area[top].sum()
        bottom_z = (cz[bottom] * area[bottom]).sum() / area[bottom].sum()
        if top_z < bottom_z:
            V = V * np.array([1.0, 1.0, -1.0])
            T = np.diag([1.0, 1.0, -1.0]) @ T
    V = V - np.array([0.0, 0.0, V[:, 2].min()])
    normal, area = face_normals(V, tri_idx)
    wall = (np.abs(normal[:, 2]) < 0.3) & (area > 1e-6)
    xy = normal[:, :2]
    norm = np.linalg.norm(xy, axis=1)
    wall &= norm > 0.1
    if wall.any():
        w = area[wall] / area[wall].sum()
        ang = np.mod(np.arctan2(xy[wall, 1], xy[wall, 0]), np.pi)
        bins = np.linspace(0, np.pi, 5)
        hist, _ = np.histogram(ang, bins=bins, weights=w)
        d = int(np.argmax(hist))
        sel = (ang >= bins[d]) & (ang < bins[d + 1])
        if sel.any():
            dominant = np.average(ang[sel], weights=w[sel])
            targets = [0.0, np.pi / 2]
            r = targets[int(np.argmin([abs(dominant - t) for t in targets]))] - dominant
            rot = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
            V = V @ rot.T
            T = rot @ T
    return V, T


def _texture_up_frac(tri, uvpx, nn, a, untex, d):
    E = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], -1)
    D = np.stack([uvpx[:, 1] - uvpx[:, 0], uvpx[:, 2] - uvpx[:, 0]], -1)
    det = np.linalg.det(D)
    ok = np.abs(det) > 1e-14
    J = np.zeros((len(tri), 3, 2))
    J[ok] = E[ok] @ np.linalg.inv(D[ok])
    tup = -J[:, :, 1]
    tup /= np.maximum(np.linalg.norm(tup, axis=1, keepdims=True), 1e-12)
    wall = (np.abs(nn @ d) < 0.2) & ok & ~untex
    return float((a[wall] * (tup[wall] @ d > 0.9)).sum() / max(a[wall].sum(), 1e-12))


def _refine_up(u, nn, a, thr=0.25, iters=3):
    for _ in range(iters):
        wl = np.abs(nn @ u) < thr
        if a[wl].sum() <= 0:
            break
        _, v = np.linalg.eigh((nn[wl] * a[wl, None]).T @ nn[wl])
        un = v[:, 0] * np.sign(v[:, 0] @ u + 1e-15)
        if un @ u < 0.9:
            break
        u = un
    return u

# Up vector of the model in its own coordinates
def estimate_up(V, tri_idx, uvpx):
    tri = V[tri_idx]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    nn = n / np.maximum(a2[:, None], 1e-15)
    a = a2 / 2.0
    A = a.sum()
    m = (nn * a[:, None]).sum(0) / A
    d = nn - m
    _, W = np.linalg.eigh((d * a[:, None]).T @ d / A)
    N = (nn * a[:, None]).sum(0) / A
    uva = 0.5 * np.abs((uvpx[:, 1, 0] - uvpx[:, 0, 0]) * (uvpx[:, 2, 1] - uvpx[:, 0, 1])
                       - (uvpx[:, 2, 0] - uvpx[:, 0, 0]) * (uvpx[:, 1, 1] - uvpx[:, 0, 1]))
    good = a > 1e-9
    dens = np.where(good, uva / np.maximum(a, 1e-12), 0)
    med = float(np.median(dens[good])) if good.any() else 0.0
    untex = dens < 0.05 * med
    opened = float(np.linalg.norm(N)) > 0.01
    best, best_e = None, -np.inf
    for j in range(3):
        for s in (1.0, -1.0):
            cand = s * W[:, j]
            S = (nn @ (-cand)) > 0.9
            af = float(a[S].sum() / A)
            fl = float(a[S & untex].sum() / max(a[S].sum(), 1e-12)) if af > 0.01 else 0.0
            e = float(N @ cand) if opened else fl + (0.0 if opened else _texture_up_frac(tri, uvpx, nn, a, untex, cand))
            if e > best_e:
                best, best_e = cand, e
    zc = max(range(3), key=lambda j: abs(W[2, j]))
    zprior = W[:, zc] * np.sign(W[2, zc] + 1e-15)
    u0 = best if (opened or best_e >= 0.3) else zprior
    return _refine_up(u0, nn, a)


def classify(normal, threshold=0.7):
    a = np.abs(normal)
    k = a.argmax(1)
    strong = a[np.arange(len(a)), k] >= threshold
    sign = np.where(normal[np.arange(len(a)), k] > 0, "+", "-")
    label = np.char.add(sign, np.array(["X", "Y", "Z"])[k])
    return np.where(strong, label, "other")


def raster_triangle(texture, uv, pts, canvas):
    h, w = canvas.shape[:2]
    th, tw = texture.shape[:2]
    x0, x1 = max(0, int(np.floor(pts[:, 0].min()))), min(w - 1, int(np.ceil(pts[:, 0].max())))
    y0, y1 = max(0, int(np.floor(pts[:, 1].min()))), min(h - 1, int(np.ceil(pts[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return
    p0, p1, p2 = pts
    area = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(area) < 1e-6:
        return
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    px, py = xs + 0.5, ys + 0.5
    w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / area
    w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / area
    w2 = 1 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return
    u = w0 * uv[0, 0] + w1 * uv[1, 0] + w2 * uv[2, 0]
    v = w0 * uv[0, 1] + w1 * uv[1, 1] + w2 * uv[2, 1]
    tx = np.trunc(u * tw).astype(np.int64) % tw
    ty = np.trunc((1 - v) * th).astype(np.int64) % th
    canvas[ys[inside], xs[inside]] = texture[ty[inside], tx[inside]]


def dihedral(img, k, mirror):
    a = np.rot90(img, k)
    return np.ascontiguousarray(a[:, ::-1] if mirror else a)


def dihedral_inverse(img, k, mirror):
    a = img[:, ::-1] if mirror else img
    return np.ascontiguousarray(np.rot90(a, -k))


def upright_transform(up, x_axis, y_axis, view):
    if abs(up @ view) > 0.7:
        return None
    frames = {0: (x_axis, y_axis), 1: (y_axis, -x_axis), 2: (-x_axis, -y_axis), 3: (-y_axis, x_axis)}
    k = min(frames, key=lambda i: frames[i][1] @ up)
    nx = frames[k][0]
    mirror = bool(nx @ np.cross(up, view) < 0)
    return k, mirror


def rectify(building, resolution=RESOLUTION):
    F = building.F
    tri_idx = F[:, :, 0] - 1
    uv_idx = np.clip(F[:, :, 1] - 1, 0, max(len(building.VT) - 1, 0))
    V, T = align(building.V.astype(np.float64), tri_idx)
    normal, _ = face_normals(V, tri_idx)
    centroid = V[tri_idx].mean(1)
    label = classify(normal)
    uv_all = building.VT[uv_idx]
    tex_sizes = np.array([building.textures[building.texture_of(t)].shape[1::-1] if building.texture_of(t) else (1, 1)
                          for t in range(len(F))], dtype=np.float64)
    uvpx = np.stack([uv_all[:, :, 0] * tex_sizes[:, None, 0], (1.0 - uv_all[:, :, 1]) * tex_sizes[:, None, 1]], -1)
    up = T @ estimate_up(building.V.astype(np.float64), tri_idx, uvpx)

    facades = []
    for direction, name, depth in DIRECTIONS:
        idx = np.where(label == direction)[0]
        if len(idx) == 0:
            continue
        cols = [1, 2] if depth == 0 else [0, 2]
        P = V[tri_idx[idx]][:, :, cols]
        lo, hi = P.reshape(-1, 2).min(0), P.reshape(-1, 2).max(0)
        size = np.where(hi - lo < 1e-6, 1.0, hi - lo)
        aspect = size[0] / size[1]
        if aspect > 1:
            width, height = resolution, max(1, int(resolution / aspect))
        else:
            width, height = max(1, int(resolution * aspect)), resolution
        order = np.argsort(centroid[idx, depth] * (1 if direction[0] == "+" else -1), kind="stable")
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        triangles = []
        for j in order:
            t = idx[j]
            tex = building.texture_of(t)
            if tex is None:
                continue
            texture = building.textures[tex]
            th, tw = texture.shape[:2]
            uv = building.VT[uv_idx[t]]
            pts = (P[j] - lo) / size * [width, height]
            pts[:, 1] = height - pts[:, 1]
            triangles.append({"texture_name": tex, "texture_coords": np.stack([uv[:, 0] * tw, (1 - uv[:, 1]) * th], -1),
                              "facade_coords": pts})
            raster_triangle(texture[:, :, 2::-1], uv, pts, canvas)
        if not triangles:
            continue
        axes = np.eye(3)
        x_axis, y_axis = axes[cols[0]], -axes[2]
        view = axes[depth] * (1 if direction[0] == "+" else -1)
        tr = upright_transform(up, x_axis, y_axis, view)
        if tr is None:
            continue
        k, mirror = tr
        raw = np.ascontiguousarray(canvas[:, :, ::-1])
        cov = coverage([tri["facade_coords"] for tri in triangles], raw.shape[:2])
        facades.append(Facade(f"{building.name}_{name}", dihedral(raw, k, mirror), cov, triangles, k, mirror, raw))
    return facades
