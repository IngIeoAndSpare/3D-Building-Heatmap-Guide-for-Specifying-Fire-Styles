import os
import shutil

import cv2
import numpy as np


class Building:
    def __init__(self, name, V, VT, F, face_material, materials, textures, obj_path, mtl_path, mtllib=None):
        self.name = name
        self.mtllib = mtllib
        self.V, self.VT, self.F = V, VT, F
        self.face_material = face_material
        self.materials = materials
        self.textures = textures
        self.obj_path, self.mtl_path = obj_path, mtl_path

    def texture_of(self, face):
        return self.materials.get(self.face_material[face])


def read_mtl(path):
    maps, current = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.split()
        if not s:
            continue
        if s[0] == "newmtl":
            current = s[1]
        elif s[0] == "map_Kd" and current is not None:
            # options such as -s or -o may precede the file name
            tokens = [t for t in s[1:] if not t.startswith("-")]
            maps[current] = os.path.basename(tokens[-1].replace("\\", "/"))
    return maps


def read_obj(path):
    v, vt, f, mat = [], [], [], []
    mtllib, current = None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            v.append([float(x) for x in s[1:4]])
        elif s[0] == "vt":
            vt.append([float(x) for x in s[1:3]])
        elif s[0] == "mtllib":
            mtllib = " ".join(s[1:])
        elif s[0] == "usemtl":
            current = s[1] if len(s) > 1 else None
        elif s[0] == "f":
            idx = []
            for tok in s[1:]:
                parts = tok.split("/")
                vi = int(parts[0])
                ti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                if vi <= 0 or ti < 0:
                    raise ValueError("only positive OBJ indices are supported...")
                idx.append((vi, ti))
            for k in range(1, len(idx) - 1):
                f.append([idx[0], idx[k], idx[k + 1]])
                mat.append(current)
    V = np.array(v, dtype=np.float64).reshape(-1, 3)
    VT = np.array(vt, dtype=np.float64).reshape(-1, 2)
    F = np.array(f, dtype=np.int64).reshape(-1, 3, 2)
    return V, VT, F, mat, mtllib


def load_building(obj_path, mtl_path=None, texture_paths=None):
    V, VT, F, face_material, mtllib = read_obj(obj_path)
    if VT.size == 0 or (F[:, :, 1] == 0).all():
        raise ValueError("the model has no texture coordinates...")
    if mtl_path is None:
        cand = os.path.join(os.path.dirname(obj_path), mtllib) if mtllib else None
        if cand and os.path.exists(cand):
            mtl_path = cand
    materials = read_mtl(mtl_path) if mtl_path else {}
    by_name = {os.path.basename(p): p for p in (texture_paths or [])}
    if not materials:
        if len(by_name) != 1:
            raise ValueError("no MTL given and the texture is ambiguous..")
        materials = {m: next(iter(by_name)) for m in set(face_material)}
    textures = {}
    names = set(materials.values())
    for tex in names:
        p = by_name.get(tex) or os.path.join(os.path.dirname(obj_path), tex)
        if not os.path.exists(p) and len(names) == 1 and len(by_name) == 1:
            # a single texture given under another file name than the MTL uses
            p = next(iter(by_name.values()))
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"texture {tex} not found")
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        elif img.dtype != np.uint8:
            raise ValueError(f"texture {tex}: unsupported pixel type {img.dtype}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        textures[tex] = img
    name = os.path.splitext(os.path.basename(obj_path))[0]
    return Building(name, V, VT, F, face_material, materials, textures, obj_path, mtl_path, mtllib)


def write_building(building, out_dir, textures):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(building.obj_path, os.path.join(out_dir, os.path.basename(building.obj_path)))
    if building.mtl_path:
        shutil.copyfile(building.mtl_path, os.path.join(out_dir, os.path.basename(building.mtl_path)))
    elif building.mtllib:
        with open(os.path.join(out_dir, os.path.basename(building.mtllib)), "w") as fh:
            for material, tex in building.materials.items():
                if material is not None:
                    fh.write(f"newmtl {material}\nKd 1.0 1.0 1.0\nmap_Kd {tex}\n")
    for tex, img in textures.items():
        if not cv2.imwrite(os.path.join(out_dir, tex), img):
            raise IOError(f"could not write {tex}")
    return os.path.join(out_dir, os.path.basename(building.obj_path))
