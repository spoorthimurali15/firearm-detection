"""Fused concealment: merge the device into a larger benign carrier.

The "disconnected" adversary places the device next to separate junk
components. Per-component detectors handle that easily: they split the mesh
into connected components and score each one on its own. This adversary takes
that option away. It joins the device to one larger benign carrier with a
Boolean union, so the two end up as a single connected component that still
prints as one object. Simply overlapping the meshes would not be enough,
because connectivity comes from shared vertices, not from shapes touching in
space.

The carrier has to overlap the device only partly. If it fully contained the
device, the union would erase the device altogether. To avoid that, the carrier
is moved along a random direction, and an offset is accepted only when a
limited fraction of the device's vertices falls inside the carrier (checked
with the generalized winding number).

Carriers come from the benign shapes that have already been generated, as in
"disconnected". Each one is scaled to be larger than the device and rotated
at random, and the fused result is given a random final pose.

Note: the Boolean union uses "gpytoolbox.copyleft.mesh_boolean" (libigl +
CGAL), which is GPL-licensed.
"""
import os

import numpy as np
import gpytoolbox as gpy
from gpytoolbox.copyleft import mesh_boolean

from ..base import Adversary, _bbox_diagonal
from ..utils import random_rotation


def _n_components(F):
    return len(np.unique(gpy.connected_components(F)))


def _clean(V, F):
    """Minimal sanitize: finite vertices, in-range non-repeated faces, no unreferenced vertices."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3 or len(V) == 0:
        return V[:0], F[:0]
    finite = np.isfinite(V).all(1)
    F = F[((F >= 0) & (F < len(V))).all(1)]
    F = F[finite[F].all(1)]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    if len(F) == 0:
        return V[:0], F[:0]
    V, F = gpy.remove_unreferenced(V, F)
    return np.asarray(V, float), np.asarray(F, np.int64)


class Fused(Adversary):
    """Join the device to a benign carrier that only partly overlaps it.

    Parameters
    ----------
    carrier_dir : str or None
        Folder of benign ".obj" files to use as carriers (for example "data/train/benign").
    carriers : list of str or None
        Paths to specific carrier files. If given, these are used instead of "carrier_dir".
    size_lo, size_hi : float
        Range for the carrier's size, measured as its bounding-box diagonal
        relative to the device's.
    overlap_lo, overlap_hi : float
        Range for how much of the device may sit inside the carrier, as a
        fraction of the device's vertices.
    tries : int
        How many placements and unions to attempt before giving up. If all of
        them fail, an error is raised and the pipeline skips this shape.
    """

    def __init__(self, carrier_dir=None, carriers=None, size_lo=1.2, size_hi=2.5,
                 overlap_lo=0.05, overlap_hi=0.45, tries=8,
                 max_carrier_faces=20000, n_probe=1500):
        self.carrier_dir = carrier_dir
        self._pool = list(carriers) if carriers is not None else None
        self.size_lo, self.size_hi = size_lo, size_hi
        self.overlap_lo, self.overlap_hi = overlap_lo, overlap_hi
        self.tries = tries
        self.max_carrier_faces = max_carrier_faces
        self.n_probe = n_probe

    def _carriers(self):
        if self._pool is None:
            d = self.carrier_dir
            self._pool = ([os.path.join(d, f) for f in sorted(os.listdir(d))
                           if f.lower().endswith(".obj")] if d and os.path.isdir(d) else [])
        return self._pool

    def _carrier(self, rng, diag):
        """A single-component benign shape, centered, scaled larger than the device, rotated."""
        pool = self._carriers()
        for _ in range(6):
            if not pool:
                break
            p = pool[int(rng.integers(len(pool)))]
            try:
                v, f = _clean(*gpy.read_mesh(p))
            except Exception:
                continue
            if 0 < len(f) <= self.max_carrier_faces and _n_components(f) == 1:
                break
        else:
            raise RuntimeError("fused: no usable carrier")
        d = _bbox_diagonal(v) or 1.0
        v = (v - v.mean(0)) * (rng.uniform(self.size_lo, self.size_hi) * diag / d)
        return v @ random_rotation(rng).T, f

    def __call__(self, vertices, faces, rng):
        Vu, Fu, _, _ = self.fuse(vertices, faces, rng)
        return Vu, Fu

    def fuse(self, vertices, faces, rng):
        """Like ``__call__`` but also returns the device mesh under the SAME final
        transform, so a detector can build per-voxel device labels for training."""
        V, F = _clean(vertices, faces)
        if len(F) == 0:
            raise RuntimeError("fused: empty device")
        diag = _bbox_diagonal(V)
        n_dev = _n_components(F)
        probe = V[rng.choice(len(V), min(len(V), self.n_probe), replace=False)]
        center = (V.min(0) + V.max(0)) / 2
        for _ in range(self.tries):
            try:
                Vc, Fc = self._carrier(rng, diag)
            except Exception:
                continue
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            r = 0.5 * (diag + _bbox_diagonal(Vc))
            good = []   # offsets along u where the carrier only partly overlaps the device
            for t in np.linspace(0.0, 1.0, 17):
                Vt = Vc + center + t * r * u
                w = np.abs(gpy.fast_winding_number(probe, Vt, Fc))
                if self.overlap_lo <= float((w > 0.5).mean()) <= self.overlap_hi:
                    good.append(Vt)
            if not good:
                continue
            Vt = good[int(rng.integers(len(good)))]
            try:
                Vu, Fu = mesh_boolean(V, F.astype(np.int32), Vt, Fc.astype(np.int32),
                                      boolean_type="union")
            except Exception:
                continue
            Vu, Fu = np.asarray(Vu, float), np.asarray(Fu, np.int64)
            if len(Fu) == 0 or _n_components(Fu) > n_dev:   # carrier must merge in
                continue
            R, mu = random_rotation(rng), Vu.mean(0)          # random final pose
            return (Vu - mu) @ R.T, Fu, (V - mu) @ R.T, F
        raise RuntimeError("fused: no valid placement/union found")
