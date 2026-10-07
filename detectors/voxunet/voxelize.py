"""Mesh -> solid occupancy volume (the stack of layers a printer would build)."""
import zlib

import numpy as np
import gpytoolbox as gpy

from ..utils import sanitize


def load_mesh(path):
    """Read + sanitize. Returns (V, F) or None if nothing usable survives."""
    try:
        V, F = gpy.read_mesh(path)
    except Exception:
        return None
    if V is None or F is None:
        return None
    V, F = sanitize(V, F)
    if len(F) == 0 or len(V) < 3:
        return None
    return np.asarray(V, float), np.asarray(F, np.int64)


def path_seed(path, base=0):
    """Deterministic per-file seed (so eval is reproducible and pose varies per file)."""
    return (zlib.crc32(str(path).encode()) + base) % (2 ** 31)


def voxelize(V, F, res=64, R=None, pad=0.04, seed=0, zoom=1.0, ref=None):
    """Mesh -> (res, res, res) bool occupancy. Axis 2 is the build (z) direction.

    Inside/outside comes from the generalized winding number (robust to the
    non-watertight, messy meshes adversaries produce); voxels the surface passes
    through are also marked so thin walls and open sheets are not lost.

    R     optional rotation applied first (pose randomization / augmentation).
    zoom  > 1 shrinks the object inside the grid at a random offset (train-time
          scale augmentation: a device fused to a big carrier ends up SMALL).
    ref   vertices of another mesh that define the grid instead of V: used to
          voxelize a device into exactly the grid of the fused mesh it belongs to.
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    Vr = V if ref is None else np.asarray(ref, float)
    if R is not None:
        mu = Vr.mean(0)
        V, Vr = (V - mu) @ R.T, (Vr - mu) @ R.T
    lo, hi = Vr.min(0), Vr.max(0)
    c, s = (lo + hi) / 2, float((hi - lo).max()) * (1 + 2 * pad)
    if not np.isfinite(s) or s <= 0:
        return None
    if zoom > 1.0:
        rng = np.random.default_rng(seed + 7)
        c = c + rng.uniform(-1, 1, 3) * (zoom - 1) * s / 2
        s *= zoom
    x = (np.arange(res) + 0.5) / res - 0.5
    G = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3) * s + c
    W = np.abs(gpy.fast_winding_number(G, V, F)).reshape(res, res, res)
    occ = W > 0.5
    P = gpy.random_points_on_mesh(V, F, 4 * res * res, rng=np.random.default_rng(seed))
    idx = np.clip(((P - c) / s + 0.5) * res, 0, res - 1).astype(np.int64)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ
