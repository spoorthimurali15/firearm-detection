#!/usr/bin/env python3
"""Populate the train/test/eval dataset for the regulated-firearm-part detection benchmark.

Run from the repository root:

    python load_data.py [--seed S] [--n-benign N [N N]]
                        [--n-malign-per-adversary N [N N]] [--timeout SECONDS]

Data is laid out split-first, ``data/<split>/<class>/``::

    data/train/benign  data/train/malign
    data/test/benign   data/test/malign
    data/eval/benign   data/eval/malign

* ``train`` is what ``Detector.train()`` fits on; ``test`` is a
  validation split for hyperparameter tuning; ``eval`` is held out and is the
  only split the leaderboard is computed on.
* Benign shapes are a seeded random sample that is **half** TetWild-cleaned
  Thingi10K (loaded via the ``thingi10k`` pip API, downloaded and cached once)
  and **half** ABC CAD models (loaded from a locally extracted ABC chunk; see
  the "ABC-dataset mesh pool" helpers below for where the pool lives). Files are
  named ``t10k_<id>.obj`` / ``abc_<stem>.obj`` so their source is visible.
* Malign shapes are seeded procedural variants of every base shape in the active
  base-shape folder. Each variant is produced by exactly one adversary
  (adversaries are never stacked); the count is given *per adversary*
  (``--n-malign-per-adversary``), so every adversary contributes the same number
  of shapes and adding one never shrinks the others' share. The real device
  templates live (untracked) in ``data/base-malign-shapes/``; the repo instead
  ships ABC-derived *proxy* base shapes (``proxy_abc_*.obj``) in the separate,
  tracked ``data/proxy-malign-shapes/`` so this step runs without the private
  data. The pipeline uses the real folder if it holds any ``*.obj`` and otherwise
  falls back to the proxies -- the two folders have opposite git status, so no
  tracked file is ever deleted to switch between them. When the proxies are in
  use, each one's ABC source model is excluded from the benign-ABC half so the two
  classes stay disjoint; a real-data run excludes nothing and reproduces the
  report's benign sample exactly. Every shape is generated in a worker process
  under a per-shape ``--timeout``, so a wedged native mesher (e.g. fTetWild) is
  skipped and logged instead of hanging the run.

Re-running with the same ``--seed`` reproduces the same dataset, up to fTetWild's
intrinsic (unseeded) meshing noise in the ``tetwild`` adversary.
"""

import argparse
import glob
import hashlib
import multiprocessing as mp
import os
import shutil
import time

import gpytoolbox as gpy
import numpy as np

# NOTE: wildmeshing (fTetWild, used by the tetwild adversary) ships native
# libraries that clash with thingi10k's when thingi10k is imported first -- the
# clash segfaults fTetWild at tetrahedralization time. Importing wildmeshing
# before thingi10k pins the safe native-load order, so keep this line above the
# thingi10k import (this is also why the adversary imports stay below it).
import wildmeshing  # noqa: F401  (imported for its native-load-order side effect)
import thingi10k

from adversaries.degenerate import Degenerate
from adversaries.disconnected import Disconnected
from adversaries.fused import Fused
from adversaries.faceswap import FaceSwap
from adversaries.jitter import Jitter
from adversaries.rigid import RigidTransform
from adversaries.tetwild import TetWild

DATA_ROOT = "data"
SPLITS = ("train", "test", "eval")
# The real (private) device templates live, untracked, in BASE_MALIGN_DIR; the
# repo instead ships ABC-derived proxy stand-ins in the separate, tracked
# PROXY_MALIGN_DIR. Keeping them in two folders with opposite git status means
# neither is ever emptied to switch data, so you can never accidentally commit a
# deletion of the proxies just to run on the real shapes (or vice versa).
BASE_MALIGN_DIR = os.path.join("data", "base-malign-shapes")
PROXY_MALIGN_DIR = os.path.join("data", "proxy-malign-shapes")


# ---------------------------------------------------------------------------
# Adversaries
#
# The atomic adversaries live in the ``adversaries/`` package (one module each,
# mirroring ``detectors/``); here we just list the ones that build the malign
# dataset. To add a new adversary, create it under ``adversaries/`` and append
# it to ``default_adversaries`` below.
# ---------------------------------------------------------------------------
def default_adversaries():
    """The adversaries that generate the malign set, one shape per adversary.

    Adversaries are **never stacked**: each malign shape is produced by exactly
    one adversary, and the count is given *per adversary*, so every adversary
    contributes the same number of analyzable variants and adding one never
    shrinks the others' share. If you want a complex modification that combines
    several effects, write it as a single new adversary rather than composing
    these.

    ``tetwild`` re-meshes the surface with the
    *same* fTetWild pipeline used to build the benign Thingi10K shapes
    (tetrahedralize, then keep the boundary surface), then perturbs the result
    with a random isometry and small vertex jitter, so benign and malign meshes
    are produced the identical way and a detector cannot cheat on the meshing
    pipeline or on a fixed pose; ``rigid`` applies a pure isometry; ``jitter``
    adds small Gaussian vertex noise; ``faceswap`` reorders faces so only the
    file bytes / any file hash change.

    ``degenerate`` appends loader-exploit garbage (NaN / out-of-range /
    repeated-index faces, dangling vertices) to the otherwise-intact device, so a
    detector that crashes on bad input and falls back to "benign" is fooled;
    ``disconnected`` hides the intact device among extra real shapes read at
    random from the already-generated benign set (half Thingi10K / half ABC),
    rotated and rescaled and added as separate connected components, so a detector
    that only looks at the mesh globally averages the device away. Both are caught by the
    input-sanitizing, per-component geometry detectors and are pure stress tests
    of that hardening.

    ``fused`` joins the device to one larger benign carrier with a Boolean union.
    The carrier comes from the train benign set, is scaled and rotated, and only
    partly overlaps the device, so the two become a single connected component
    that still prints as one object. Unlike ``disconnected``, per-component
    detectors can no longer separate the device from what surrounds it.

    ``remesh`` is intentionally omitted: it is parked on a separate draft branch
    until a gpytoolbox release ships a better isotropic remesher.
    """
    return [
        ("tetwild", TetWild()),
        ("rigid", RigidTransform()),
        ("jitter", Jitter(sigma_frac=1e-3)),
        ("faceswap", FaceSwap()),
        ("degenerate", Degenerate()),
        ("disconnected", Disconnected(benign_dir=os.path.join(DATA_ROOT, "train", "benign"))),
        ("fused", Fused(carrier_dir=os.path.join(DATA_ROOT, "train", "benign"))),
    ]


# ---------------------------------------------------------------------------
# ABC-dataset mesh pool (the benign-ABC half is sampled from this)
#
# The ABC dataset (https://deep-geometry.github.io/abc-dataset/) ships no pip
# package, so one 10k-model OBJ chunk is downloaded and extracted out of band
# into a directory of ``<id>/<id>_..._trimesh_NNN.obj`` files (see
# ignore/redteam/). These helpers locate that directory, list the usable meshes
# once (cached per process so seeded sampling is reproducible and the tree is
# walked once), and load one by path with light validation. The pool resolves
# from the ``ABC_POOL_DIR`` environment variable if set, else ``data/abc-pool``.
# ---------------------------------------------------------------------------
_ABC_LIST_CACHE = {}  # pool dir -> sorted [obj paths]; listed once per process


def _abc_pool_dir():
    """Directory holding the extracted ABC ``*.obj`` files (env override or default)."""
    return os.environ.get("ABC_POOL_DIR") or os.path.join(DATA_ROOT, "abc-pool")


def _abc_list_meshes(pool=None):
    """Sorted list of every ``*.obj`` under the pool (recursive), cached per process.

    Sorted + cached so seeded sampling on top of it is reproducible and the
    directory tree is walked only once per worker. Empty list if the pool is absent.
    """
    pool = pool or _abc_pool_dir()
    if pool not in _ABC_LIST_CACHE:
        _ABC_LIST_CACHE[pool] = (sorted(glob.glob(os.path.join(pool, "**", "*.obj"),
                                                  recursive=True))
                                 if os.path.isdir(pool) else [])
    return _ABC_LIST_CACHE[pool]


def _abc_load_mesh(path, max_faces=None):
    """Load one ABC OBJ as triangles via gpytoolbox. Returns ``(V, F)`` or ``None``.

    ``None`` for anything unusable -- unreadable, non-triangular, empty,
    non-finite, or above ``max_faces`` -- so callers can simply skip it and try
    the next sample. ABC ships pre-triangulated ``*_trimesh_*.obj`` files, so
    ``gpytoolbox.read_mesh`` (triangle-only) reads them directly with no extra
    dependency.
    """
    try:
        V, F = gpy.read_mesh(path)
    except Exception:
        return None
    if V is None or F is None:
        return None
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        return None
    if max_faces and len(F) > max_faces:
        return None
    if not np.all(np.isfinite(V)):
        return None
    return V, F


# Chunk 0000 (model ids 0000-9999) is the benign-ABC source; the proxy malign
# shapes come from chunk 0001, so the two stay disjoint. The download manifest
# lives on the dataset's gh-pages branch; chunk 0000 is bitstream 89085.
_ABC_CHUNK0_URL = "https://archive.nyu.edu/rest/bitstreams/89085/retrieve"


def download_abc_pool(dest=None, n_models=4000, archive=None):
    """Download + extract the ABC benign pool so ``load_data`` finds it automatically.

    One-time setup for the benign-ABC half. Fetches ABC OBJ *chunk 0000* (a
    ~6.5 GB ``.7z``) and extracts one triangulated mesh per model
    (``<id>/<id>_..._trimesh_000.obj``) into ``dest`` -- the pool dir, i.e.
    ``$ABC_POOL_DIR`` or ``data/abc-pool``. Run once via ``--download-abc``.
    The extracted CAD meshes are large, so ``n_models`` (default 4000, enough for
    the default benign split) costs tens of GB; set ``ABC_POOL_DIR`` to keep them
    off any synced folder. ``archive`` reuses an already-downloaded ``.7z``.
    """
    import tempfile
    import urllib.request
    try:
        import py7zr
    except ImportError:
        raise SystemExit("--download-abc requires py7zr (`pip install py7zr`).")

    dest = dest or _abc_pool_dir()
    os.makedirs(dest, exist_ok=True)
    arch = archive or os.path.join(tempfile.gettempdir(), "abc_0000_obj_v00.7z")
    if os.path.exists(arch):
        print(f"[abc] reusing existing archive {arch}", flush=True)
    else:
        print(f"[abc] downloading chunk 0000 (~6.5 GB) -> {arch}", flush=True)
        seen = [0.0]

        def _hook(blocks, bs, total):
            mb = blocks * bs / 1e6
            if mb - seen[0] >= 256 or (total > 0 and blocks * bs >= total):
                seen[0] = mb
                pct = f" ({100.0 * blocks * bs / total:.0f}%)" if total > 0 else ""
                print(f"[abc]   {mb:.0f} MB{pct}", flush=True)

        urllib.request.urlretrieve(_ABC_CHUNK0_URL, arch, reporthook=_hook)

    print("[abc] indexing archive ...", flush=True)
    with py7zr.SevenZipFile(arch, "r") as z:
        names = z.getnames()
    by_model = {}
    for n in names:
        if n.lower().endswith(".obj") and "trimesh" in n.lower():
            by_model.setdefault(n.split("/")[0], n)  # one trimesh per model id
    pick = [by_model[m] for m in sorted(by_model)[:n_models]]
    print(f"[abc] extracting {len(pick)} of {len(by_model)} models -> {dest}", flush=True)
    with py7zr.SevenZipFile(arch, "r") as z:
        z.extract(path=dest, targets=pick)
    got = len(glob.glob(os.path.join(dest, "**", "*.obj"), recursive=True))
    print(f"[abc] done: pool now holds {got} .obj under {dest}", flush=True)
    return got


# ---------------------------------------------------------------------------
# Benign loading (Thingi10K, TetWild variant)
# ---------------------------------------------------------------------------
def load_benign(data_root, splits, counts, rng, max_facets=None, exclude_abc_stems=None):
    """Sample distinct benign shapes per split, **half Thingi10K, half ABC**.

    Each split's count is split into a Thingi10K half (the ceil) and an ABC half
    (the floor); the two sources are sampled independently (from two child RNG
    streams, so a change to one source's sampling can't perturb the other) and
    disjointly within each source, so no benign shape appears in more than one
    split and the class is an even mix of organic Thingi10K scans and ABC CAD
    models. Provenance is recorded in the filenames (``t10k_<id>.obj`` /
    ``abc_<stem>.obj``).

    ``exclude_abc_stems`` is an optional set of ABC source stems to drop from the
    benign-ABC candidate pool, so a proxy malign shape and a benign shape can never
    come from the same ABC CAD model. It is empty whenever the real private bases
    are in use, leaving the benign sample identical to the report's.
    """
    t_counts = [c - c // 2 for c in counts]   # Thingi10K gets the ceil half
    a_counts = [c // 2 for c in counts]        # ABC gets the floor half
    rng_t, rng_a = rng.spawn(2)
    _load_benign_thingi10k(data_root, splits, t_counts, rng_t, max_facets)
    _load_benign_abc(data_root, splits, a_counts, rng_a, max_facets,
                     exclude_abc_stems=exclude_abc_stems)


def _load_benign_thingi10k(data_root, splits, counts, rng, max_facets=None):
    """Write the Thingi10K half: one disjoint random sample partitioned per split."""
    thingi10k.init(variant="tetwild")
    ds = thingi10k.dataset(num_facets=(None, max_facets) if max_facets else None)
    total = len(ds)
    n_total = sum(counts)
    if total < n_total:
        raise SystemExit(f"Only {total} Thingi10K entries match the filter; need {n_total}.")

    sample = rng.choice(total, size=n_total, replace=False)  # random, disjoint
    offset = 0
    for split, n in zip(splits, counts):
        out_dir = os.path.join(data_root, split, "benign")
        chunk = sorted(int(i) for i in sample[offset:offset + n])
        offset += n
        for i in chunk:
            entry = ds[i]
            vertices, facets = thingi10k.load_file(entry["file_path"])
            gpy.write_mesh(os.path.join(out_dir, f"t10k_{entry['file_id']}.obj"), vertices, facets)
        print(f"[benign/{split}] thingi10k: {len(chunk)} shapes -> {out_dir}", flush=True)


def _load_benign_abc(data_root, splits, counts, rng, max_facets=None, exclude_abc_stems=None):
    """Write the ABC half: stream the seeded-shuffled pool, keep the first
    ``sum(counts)`` meshes that load (triangulated, finite, within the facet cap),
    and partition them disjointly across the splits.

    Sampling streams a shuffled permutation rather than picking exact indices
    because some ABC files are unreadable or above the cap; streaming until enough
    pass keeps the per-split counts exact while staying reproducible from ``rng``.
    The cap defaults to 50k facets when ``max_facets`` is unset: ABC is heavily
    right-skewed (a measured chunk had a ~36k-facet median, only ~18% of files
    <= 20k but ~69% <= 50k), so a 20k cap would throw away most of the pool while
    50k keeps the bulk of it yet still rejects the occasional million-triangle CAD
    part that would bloat the benign set and slow the spectral/point detectors.

    ``exclude_abc_stems`` (ABC source stems backing the committed proxy malign
    shapes) is removed from the candidate pool up front so a benign shape and a
    proxy-malign shape never come from the same ABC model; it is empty whenever the
    real private base shapes are used.
    """
    n_total = sum(counts)
    if n_total == 0:
        return
    paths = _abc_list_meshes()
    if exclude_abc_stems:
        paths = [p for p in paths
                 if os.path.splitext(os.path.basename(p))[0] not in exclude_abc_stems]
    if not paths:
        raise SystemExit(
            f"ABC pool is empty at {_abc_pool_dir()}; need {n_total} ABC benign "
            f"shapes. Download+extract an ABC OBJ chunk and/or set ABC_POOL_DIR "
            f"(see the ABC-dataset mesh pool helpers above and ignore/redteam/).")
    cap = max_facets or 50000
    order = rng.permutation(len(paths))
    kept = []
    for idx in order:
        vf = _abc_load_mesh(paths[int(idx)], max_faces=cap)
        if vf is not None:
            kept.append((paths[int(idx)], vf))
            if len(kept) >= n_total:
                break
    if len(kept) < n_total:
        raise SystemExit(
            f"Only {len(kept)} usable ABC meshes (<= {cap} facets) in "
            f"{_abc_pool_dir()}; need {n_total}.")
    offset = 0
    for split, n in zip(splits, counts):
        out_dir = os.path.join(data_root, split, "benign")
        for path, (vertices, facets) in kept[offset:offset + n]:
            stem = os.path.splitext(os.path.basename(path))[0]
            gpy.write_mesh(os.path.join(out_dir, f"abc_{stem}.obj"), vertices, facets)
        offset += n
        print(f"[benign/{split}] abc: {n} shapes -> {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Malign generation (procedural variants of the base shapes)
#
# Every adversary runs inside a persistent worker subprocess so a single
# per-shape ``--timeout`` can bound it. A wedged native mesher (fTetWild has no
# timeout of its own and cannot be interrupted by signals or threads) is killed
# and the worker restarted; the offending shape is skipped and logged. The heavy
# native imports are paid once when the worker starts and reused across shapes.
# A ``spawn`` context is used (never ``fork``) to match the safe native-load
# order pinned at import time.
# ---------------------------------------------------------------------------
def _adversary_worker(conn):
    """Worker loop: apply (adversary, V, F, seed) messages until told to stop."""
    while True:
        msg = conn.recv()
        if msg is None:
            break
        adversary, V, F, seed = msg
        try:
            v, f = adversary(V, F, np.random.default_rng(seed))
            conn.send(("ok", (np.asarray(v), np.asarray(f))))
        except Exception as exc:  # report, let the parent decide to skip
            conn.send(("err", repr(exc)))
    conn.close()


class _TimeoutPool:
    """A single reusable spawn worker that applies one adversary at a time under
    a wall-clock timeout, killing and restarting itself if a call overruns."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.ctx = mp.get_context("spawn")
        self._spawn()

    def _spawn(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_adversary_worker, args=(child,), daemon=True)
        self.proc.start()
        child.close()  # parent keeps only its end of the pipe

    def _restart(self):
        try:
            self.proc.kill()
            self.proc.join()
        except Exception:
            pass
        self._spawn()

    def apply(self, adversary, V, F, seed):
        """Run ``adversary(V, F, rng(seed))`` in the worker. Returns
        ``("ok", (V, F))``, ``("timeout", None)``, or ``("error", message)``."""
        try:
            self.conn.send((adversary, V, F, seed))
        except (BrokenPipeError, OSError) as exc:
            self._restart()
            return "error", f"send failed: {exc!r}"
        if not self.conn.poll(self.timeout):
            self._restart()
            return "timeout", None
        try:
            status, payload = self.conn.recv()
        except (EOFError, OSError) as exc:
            self._restart()
            return "error", f"recv failed: {exc!r}"
        return ("ok", payload) if status == "ok" else ("error", payload)

    def close(self):
        try:
            self.conn.send(None)
            self.proc.join(timeout=5)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join()


def _expand_counts(values, flag):
    """Normalize a per-split count flag to one value per split.

    Accepts a single value (applied to every split) or exactly one value per
    split (in ``SPLITS`` order); anything else is a usage error.
    """
    counts = values if len(values) > 1 else values * len(SPLITS)
    if len(counts) != len(SPLITS):
        raise SystemExit(f"{flag} takes 1 or {len(SPLITS)} values "
                         f"({', '.join(SPLITS)}); got {len(values)}.")
    return counts


def _shape_seed(master_seed, stem, split, adversary_name, idx):
    """A reproducible per-shape ``SeedSequence`` keyed on the master seed and the
    (base shape, split, adversary, index), independent of generation order."""
    digest = hashlib.sha256(f"{stem}|{split}|{adversary_name}".encode()).digest()
    return np.random.SeedSequence([master_seed, 1, int.from_bytes(digest[:4], "big"), idx])


# Committed ABC-derived stand-ins for the private device templates live in
# ``proxy-malign-shapes/`` as ``proxy_abc_<abc_stem>.obj`` (the filename records
# the ABC source model), so the malign pipeline runs without the private data. The
# real templates go in the separate, untracked ``base-malign-shapes/``; the
# pipeline prefers that folder when populated, so neither folder is ever emptied to
# switch between proxy and real data.
_PROXY_ABC_PREFIX = "proxy_abc_"


def _has_obj(directory):
    """True iff ``directory`` exists and contains at least one ``*.obj`` file."""
    return (os.path.isdir(directory)
            and any(f.lower().endswith(".obj") for f in os.listdir(directory)))


def _active_malign_dir():
    """Folder that feeds the malign pipeline: the real private templates in
    ``BASE_MALIGN_DIR`` if it holds any ``*.obj``, else the committed proxy
    stand-ins in ``PROXY_MALIGN_DIR``. Both the malign generator and the
    benign-ABC exclusion key off this single choice, so a real-data run excludes
    nothing (reproducing the report's benign sample byte-for-byte) while a proxy
    run stays disjoint from the benign-ABC half."""
    return BASE_MALIGN_DIR if _has_obj(BASE_MALIGN_DIR) else PROXY_MALIGN_DIR


def _proxy_abc_stems(base_dir=None):
    """ABC source stems backing any ``proxy_abc_<stem>.obj`` shapes in ``base_dir``.

    Used to drop those exact ABC models from the benign-ABC sample so a proxy
    malign shape and a benign shape never come from the same ABC CAD model.
    ``base_dir`` defaults to the active malign folder, so this is empty whenever
    the real private bases are in use (they are not named ``proxy_abc_*``) and the
    benign sample then matches the report's exactly.
    """
    base_dir = base_dir if base_dir is not None else _active_malign_dir()
    stems = set()
    if os.path.isdir(base_dir):
        for f in os.listdir(base_dir):
            if f.startswith(_PROXY_ABC_PREFIX) and f.lower().endswith(".obj"):
                stems.add(os.path.splitext(f)[0][len(_PROXY_ABC_PREFIX):])
    return stems


def load_malign(base_dir, data_root, splits, counts, master_seed,
                adversaries=None, timeout=10.0):
    """Generate single-adversary malign variants of each base shape per split.

    ``counts`` is the number of variants **each adversary** produces per base for
    each split (in ``splits`` order), so the per-split malign total is
    ``counts[i] * len(adversaries)`` per base and scales with the adversary list
    -- adding an adversary never shrinks the others' share. Each variant is
    produced by exactly one adversary (adversaries are never stacked) and its
    filename ``<stem>_<adversary>_<idx>.obj`` records which one produced it. Every
    shape is generated in a worker subprocess under ``timeout`` seconds; overruns
    and errors are skipped and logged rather than aborting the run. Per-shape
    seeds are reproducible from ``master_seed`` regardless of generation order.
    """
    adversaries = adversaries if adversaries is not None else default_adversaries()
    bases = (sorted(f for f in os.listdir(base_dir) if f.lower().endswith(".obj"))
             if os.path.isdir(base_dir) else [])
    if not bases:
        print(f"No base malign shapes (*.obj) found in {base_dir}/ -- neither the "
              f"real private templates ({BASE_MALIGN_DIR}/) nor the committed proxy "
              f"stand-ins ({PROXY_MALIGN_DIR}/) are populated. The real device "
              f"templates are a PRIVATE dataset, not shipped with the repo; to "
              f"request access for legitimate research, email "
              f"silviasellan@cs.columbia.edu. Skipping malign generation.")
        return
    using_real = os.path.abspath(base_dir) == os.path.abspath(BASE_MALIGN_DIR)
    print(f"[malign] {len(bases)} base shape(s) from {base_dir}/ "
          f"({'real private templates' if using_real else 'committed ABC-derived proxies'})",
          flush=True)
    pool = _TimeoutPool(timeout)
    try:
        for base in bases:
            stem = os.path.splitext(base)[0]
            V, F = gpy.read_mesh(os.path.join(base_dir, base))
            for split, n in zip(splits, counts):
                out_dir = os.path.join(data_root, split, "malign")
                os.makedirs(out_dir, exist_ok=True)
                made = skipped = 0
                for name, adversary in adversaries:
                    for idx in range(n):
                        label = f"{stem}_{name}_{idx:04d}"
                        seed = _shape_seed(master_seed, stem, split, name, idx)
                        t0 = time.perf_counter()
                        status, payload = pool.apply(adversary, V, F, seed)
                        dt = time.perf_counter() - t0
                        if status == "ok":
                            v, f = payload
                            gpy.write_mesh(os.path.join(out_dir, f"{label}.obj"), v, f)
                            made += 1
                            print(f"[malign/{split}] {label}  ok ({dt:.2f}s)", flush=True)
                        elif status == "timeout":
                            skipped += 1
                            print(f"[malign/{split}] {label}  TIMEOUT after "
                                  f"{timeout:.0f}s -- skipped", flush=True)
                        else:
                            skipped += 1
                            print(f"[malign/{split}] {label}  ERROR {payload} -- "
                                  f"skipped ({dt:.2f}s)", flush=True)
                print(f"[malign/{split}] {stem}: {made} made, {skipped} skipped "
                      f"({n} per adversary x {len(adversaries)} adversaries) -> {out_dir}",
                      flush=True)
    finally:
        pool.close()


# ---------------------------------------------------------------------------
def _reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0, help="Master RNG seed (default: 0)")
    parser.add_argument("--n-benign", type=int, nargs="+", default=[4000, 500, 500],
                        help="Per-split benign shape counts. Pass one value for all splits, "
                             "or one per split in train/test/eval order. The default "
                             "(4000 500 500) reproduces the report; pass e.g. --n-benign 100 "
                             "for a quick smaller run.")
    parser.add_argument("--n-malign-per-adversary", type=int, nargs="+", default=[1000, 125, 125],
                        help="Per-split malign variants produced *by each adversary* per base "
                             "shape; the malign total scales with the number of adversaries, so "
                             "adding one never shrinks the others' share. Pass one value for all "
                             "splits, or one per split in train/test/eval order. The default "
                             "(1000 125 125) reproduces the report; pass e.g. "
                             "--n-malign-per-adversary 25 for a quick smaller run.")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Per-shape wall-clock budget in seconds for each adversary; "
                             "overruns are skipped and logged (default: 10)")
    parser.add_argument("--max-facets", type=int, default=None, help="Optional cap on benign mesh facet count")
    parser.add_argument("--skip-benign", action="store_true", help="Do not (re)generate benign data")
    parser.add_argument("--skip-malign", action="store_true", help="Do not (re)generate malign data")
    parser.add_argument("--download-abc", action="store_true",
                        help="One-time: download + extract the ABC benign pool into "
                             "$ABC_POOL_DIR (or data/abc-pool), then exit. Needs py7zr.")
    parser.add_argument("--abc-models", type=int, default=4000,
                        help="With --download-abc: number of distinct ABC models to "
                             "extract (default 4000; enough for the default benign split).")
    args = parser.parse_args()

    if args.download_abc:
        download_abc_pool(n_models=args.abc_models)
        return

    # Resolve the base-shape folder once: the real private templates if present,
    # else the committed proxies. Both the benign-ABC exclusion and the malign
    # generator key off this one choice, so a real-data run excludes nothing
    # (reproducing the report's benign sample) while a proxy run stays disjoint.
    active_malign_dir = _active_malign_dir()

    if not args.skip_benign:
        benign_counts = _expand_counts(args.n_benign, "--n-benign")
        # Stream index 0 of the master seed keeps the benign sample identical to
        # before regardless of the malign-side changes.
        rng_benign = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(2)[0])
        # Drop the active proxies' ABC sources from the benign-ABC pool so the
        # classes stay disjoint; empty (a no-op) whenever the real private bases
        # are in use, so the benign sample then matches the report's exactly.
        exclude_abc = _proxy_abc_stems(active_malign_dir)
        for split in SPLITS:
            _reset_dir(os.path.join(DATA_ROOT, split, "benign"))
        load_benign(DATA_ROOT, SPLITS, benign_counts, rng_benign,
                    max_facets=args.max_facets, exclude_abc_stems=exclude_abc)
    if not args.skip_malign:
        malign_counts = _expand_counts(args.n_malign_per_adversary, "--n-malign-per-adversary")
        for split in SPLITS:
            _reset_dir(os.path.join(DATA_ROOT, split, "malign"))
        load_malign(active_malign_dir, DATA_ROOT, SPLITS, malign_counts, args.seed, timeout=args.timeout)


if __name__ == "__main__":
    main()
