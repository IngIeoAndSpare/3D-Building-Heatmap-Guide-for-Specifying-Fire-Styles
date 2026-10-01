import cv2
import numpy as np

from .raster import warp_into, pad_nearest
from .rectify import dihedral_inverse


def bake(building, facades, styled):
    out = {tex: img.copy() for tex, img in building.textures.items()}
    for fac in facades:
        img = styled.get(fac.name)
        if img is None:
            continue
        src = pad_nearest(dihedral_inverse(img[:, :, :3], fac.rot_k, fac.mirror), fac.coverage)
        for t in fac.triangles:
            canvas = out[t["texture_name"]]
            rgb = np.ascontiguousarray(canvas[:, :, :3])
            warp_into(src, rgb, t["facade_coords"], t["texture_coords"], cv2.INTER_NEAREST, "fill",
                      cv2.BORDER_REPLICATE)
            canvas[:, :, :3] = rgb
    return out
