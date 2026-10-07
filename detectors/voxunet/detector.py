"""VoxUNet: find regulated components by looking at the print layer by layer.

The existing detectors, PointNet and the spectral signature, make one decision
per connected component. When a device is joined to a larger benign carrier
with a Boolean union (the "fused" adversary), it is no longer its own
component, so they can't judge it separately. VoxUNet works differently. It
treats the print the way a printer builds it, as a stack of layers that forms
a solid volume, and predicts for every voxel whether it belongs to a device.
The mesh score is the mean of the k highest voxel scores on or next to the
material, so a small device inside a large body is not averaged away.

Architecture: a 3D U-Net (see "model.py") with a bidirectional Mamba block in
the middle that reads the voxels in print order, layer by layer. If "mamba_ssm"
is not installed, for example on CPU or Apple Silicon, it uses a convolutional
block instead.

Training ("train") uses only the train and test splits it is given:
  1. Every train mesh is voxelized several times, with random rotations and a
     random zoom-out.
  2. Extra fused examples are created by joining train devices to train benign
     carriers with the "Fused" adversary. Since these shapes are built here,
     the exact device voxels are known, so they get a per-voxel loss
     (BCE + Dice). Benign shapes, with or without a carrier, get a target of
     "no device in any voxel".
  3. Every shape also gets a mesh-level loss based on its k highest voxel
     scores.
  4. The checkpoint with the best validation AUC is kept. Validation uses the
     test split plus fused test shapes, made from test devices and test
     carriers.

If a mesh can't be read, it is scored as malign (1.0) rather than passed as
benign.
"""
import concurrent.futures
import multiprocessing as mp
import os
import time

# Let ops MPS lacks fall back to CPU so the same code runs on Apple Silicon (as PointNet does).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import gpytoolbox as gpy
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

from ..base import Detector
from .model import VoxUNet as _VoxUNetModel, amp_enabled
from .voxelize import load_mesh, path_seed, voxelize


# --------------------------------------------------------------------------- helpers
def _resolve_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mamba_available():
    try:
        import mamba_ssm  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def _rotation(seed):
    from adversaries.utils import random_rotation
    return random_rotation(np.random.default_rng(seed))


def _vox_job(job):
    """Worker: voxelize one item. job = (kind, payload, seed, res, zoom_max).

    kind "path": payload = obj path                     -> (bits, None)
    kind "mesh": payload = (V, F, Vdev or None, Fdev)   -> (bits, mask bits or None)
    """
    kind, payload, seed, res, zoom_max = job
    try:
        if kind == "path":
            m, dev = load_mesh(payload), None
        else:
            V, F, Vd, Fd = payload
            m, dev = (V, F), (None if Vd is None else (Vd, Fd))
        if m is None:
            return None, None
        rng = np.random.default_rng(seed)
        zoom = float(np.exp(rng.uniform(0, np.log(zoom_max)))) if zoom_max > 1 else 1.0
        R = _rotation(seed)
        occ = voxelize(*m, res=res, R=R, seed=seed, zoom=zoom)
        if occ is None:
            return None, None
        mask = None
        if dev is not None:      # device voxelized into the fused mesh's own grid
            mask = voxelize(*dev, res=res, R=R, seed=seed, zoom=zoom, ref=m[0]) & occ
        return np.packbits(occ.ravel()), (None if mask is None else np.packbits(mask.ravel()))
    except Exception:
        return None, None


def _fuse_worker(conn):
    """Spawned worker applying Fused.fuse under a hard timeout (CGAL can hang)."""
    while True:
        msg = conn.recv()
        if msg is None:
            break
        adv, V, F, seed = msg
        try:
            out = adv.fuse(V, F, np.random.default_rng(seed))
            conn.send(("ok", tuple(np.asarray(x) for x in out)))
        except Exception as exc:
            conn.send(("err", repr(exc)))


class _FusePool:
    def __init__(self, timeout):
        self.timeout, self.ctx = timeout, mp.get_context("spawn")
        self._spawn()

    def _spawn(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_fuse_worker, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def apply(self, adv, V, F, seed):
        try:
            self.conn.send((adv, V, F, seed))
            if not self.conn.poll(self.timeout):
                self.proc.kill(); self.proc.join(); self._spawn()
                return None
            status, out = self.conn.recv()
            return out if status == "ok" else None
        except (EOFError, OSError, BrokenPipeError):
            self.proc.kill(); self.proc.join(); self._spawn()
            return None

    def close(self):
        try:
            self.conn.send(None)
            self.proc.join(timeout=5)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.kill()


def _seg_loss(vox, occ, mask, has):
    """BCE + soft Dice on voxels on/near the material, for samples with known voxel labels."""
    if not has.any():
        return vox.sum() * 0.0
    vox, occ, mask = vox[has], occ[has], mask[has]
    near = (Fn.max_pool3d(occ, 3, stride=1, padding=1) > 0).float()
    bce = Fn.binary_cross_entropy_with_logits(vox, mask, reduction="none")
    bce = (bce * near).sum((1, 2, 3, 4)) / near.sum((1, 2, 3, 4)).clamp(min=1)
    p = torch.sigmoid(vox) * near
    dice = 1 - (2 * (p * mask).sum((1, 2, 3, 4)) + 1) / (p.sum((1, 2, 3, 4)) + mask.sum((1, 2, 3, 4)) + 1)
    return (bce + dice).mean()


class _VoxSet(torch.utils.data.Dataset):
    def __init__(self, bits, y, mbits, has, res, augment):
        self.bits, self.y, self.mbits, self.has = bits, y, mbits, has
        self.res, self.augment = res, augment

    def __len__(self):
        return len(self.y)

    def _unpack(self, row):
        return np.unpackbits(row)[: self.res ** 3].reshape(self.res, self.res, self.res)

    def __getitem__(self, i):
        v, m = self._unpack(self.bits[i]), self._unpack(self.mbits[i])
        if self.augment:              # same flips / xy-transpose for volume + mask (z = build axis)
            for ax in range(3):
                if np.random.rand() < 0.5:
                    v, m = np.flip(v, ax), np.flip(m, ax)
            if np.random.rand() < 0.5:
                v, m = v.transpose(1, 0, 2), m.transpose(1, 0, 2)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))[None]
        return f(v), np.float32(self.y[i]), f(m), bool(self.has[i])


# --------------------------------------------------------------------------- detector
class VoxUNet(Detector):
    def __init__(self, res=64, epochs=30, batch_size=16, lr=1e-3, k=64,
                 train_rot=2, train_zoom=2.5, seg_weight=1.0, seg_benign=True,
                 n_fused=600, n_fused_benign=400, n_val_fused=90, n_val_fused_benign=60,
                 fuse_timeout=30.0, fuse_workers=None, vox_workers=None, loader_workers=None,
                 bottleneck="auto", device="auto", seed=0):
        super().__init__()
        self.res, self.epochs, self.batch_size, self.lr, self.k = res, epochs, batch_size, lr, k
        self.train_rot, self.train_zoom = train_rot, train_zoom
        self.seg_weight, self.seg_benign = seg_weight, seg_benign
        self.n_fused, self.n_fused_benign = n_fused, n_fused_benign
        self.n_val_fused, self.n_val_fused_benign = n_val_fused, n_val_fused_benign
        self.fuse_timeout = fuse_timeout
        ncpu = os.cpu_count() or 1
        self.fuse_workers = fuse_workers or max(1, min(4, ncpu))
        self.vox_workers = vox_workers or max(1, ncpu - 1)
        self.loader_workers = min(8, ncpu) if loader_workers is None else loader_workers
        self.bottleneck = bottleneck
        self.device = _resolve_device(device)
        self.seed = seed
        self._model = None
        self._cfg = None

    # ----------------------------------------------------------------- data
    def _voxelize_many(self, jobs, tag):
        n_bytes = (self.res ** 3 + 7) // 8
        bits = np.zeros((len(jobs), n_bytes), np.uint8)
        mbits = np.zeros((len(jobs), n_bytes), np.uint8)
        ok = np.zeros(len(jobs), bool)
        t0 = time.time()
        with mp.get_context("spawn").Pool(self.vox_workers) as pool:
            for i, (b, m) in enumerate(pool.imap(_vox_job, jobs, chunksize=4)):
                if b is not None:
                    bits[i], ok[i] = b, True
                    if m is not None:
                        mbits[i] = m
                if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                    print(f"[voxunet] voxelizing {tag}: {i + 1}/{len(jobs)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        return bits, mbits, ok

    @staticmethod
    def _device_candidates(paths):
        """Train/test devices usable as fusion templates: skip shapes that are already
        fused or carry junk (their 'device' would include benign geometry)."""
        skip = ("_fused_", "_disconnected_")
        return [p for p in paths if not any(s in os.path.basename(p) for s in skip)]

    def _make_fused(self, devices, carriers, benign, n_mal, n_ben, seed, tag):
        """Fuse devices / benign shapes into carriers. Returns lists of
        (V, F, Vdev, Fdev, label); Vdev is None for benign+carrier shapes."""
        from adversaries.fused import Fused
        if not devices or not carriers or (n_mal + n_ben) == 0:
            return []
        adv = Fused(carriers=carriers)
        rng = np.random.default_rng(seed)
        todo = ([(devices[int(rng.integers(len(devices)))], 1) for _ in range(n_mal)] +
                [(benign[int(rng.integers(len(benign)))], 0) for _ in range(n_ben)])
        seeds = rng.integers(2 ** 31, size=len(todo))
        pools = [_FusePool(self.fuse_timeout) for _ in range(self.fuse_workers)]
        out, t0, done = [], time.time(), 0

        def work(i):
            path, label = todo[i]
            m = load_mesh(path)
            if m is None:
                return None
            if label == 1 and len(np.unique(gpy.connected_components(m[1]))) != 1:
                return None   # single-component devices only, so the voxel labels are clean
            r = pools[i % len(pools)].apply(adv, m[0], m[1], int(seeds[i]))
            if r is None:
                return None
            Vu, Fu, Vd, Fd = r
            return (Vu, Fu, Vd if label == 1 else None, Fd, label)

        try:
            with concurrent.futures.ThreadPoolExecutor(len(pools)) as ex:
                # each pool processes the indices i with i % n == its id, sequentially
                for res_ in ex.map(lambda w: [work(i) for i in range(w, len(todo), len(pools))],
                                   range(len(pools))):
                    for r in res_:
                        done += 1
                        if r is not None:
                            out.append(r)
        finally:
            for p in pools:
                p.close()
        n_m = sum(1 for r in out if r[4] == 1)
        print(f"[voxunet] fused {tag}: {n_m}/{n_mal} devices, {len(out) - n_m}/{n_ben} benign+carrier "
              f"({time.time() - t0:.0f}s)", flush=True)
        return out

    def _dataset(self, path_items, fused_items, n_rot, zoom, seed0, augment):
        jobs, y, from_fused = [], [], []
        for i, (p, lab) in enumerate(path_items):
            for r in range(n_rot):
                jobs.append(("path", p, seed0 + 1009 * i + r, self.res, zoom))
                y.append(lab); from_fused.append(False)
        for i, (V, F, Vd, Fd, lab) in enumerate(fused_items):
            for r in range(n_rot):
                jobs.append(("mesh", (V, F, Vd, Fd), seed0 + 7 * 10 ** 6 + 1009 * i + r, self.res, zoom))
                y.append(lab); from_fused.append(True)
        bits, mbits, ok = self._voxelize_many(jobs, f"{len(jobs)} volumes")
        y, from_fused = np.array(y, np.float32), np.array(from_fused)
        # known voxel labels: fused shapes (device mask or all-zero), and benign shapes
        has = from_fused | ((y == 0) if self.seg_benign else np.zeros_like(from_fused))
        return _VoxSet(bits[ok], y[ok], mbits[ok], has[ok], self.res, augment)

    # ----------------------------------------------------------------- train
    def _build_model(self, bottleneck):
        return _VoxUNetModel(k=self.k, bottleneck=bottleneck).to(self.device)

    @torch.no_grad()
    def _predict(self, model, ds):
        model.eval()
        dl = torch.utils.data.DataLoader(ds, self.batch_size, num_workers=self.loader_workers)
        out = []
        for x, *_ in dl:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=amp_enabled(self.device)):
                logit, _ = model(x.to(self.device))
            out.append(torch.sigmoid(logit.float()).cpu())
        return torch.cat(out).numpy()

    def train(self, train, test=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        bottleneck = self.bottleneck
        if bottleneck == "auto":
            bottleneck = "mamba" if _mamba_available() else "conv"
        print(f"[voxunet] device={self.device} bottleneck={bottleneck} res={self.res}", flush=True)

        tr_items = [(p, 0) for p in train.benign] + [(p, 1) for p in train.malign]
        tr_fused = self._make_fused(self._device_candidates(train.malign), train.benign, train.benign,
                                    self.n_fused, self.n_fused_benign, self.seed + 1, "train")
        tr = self._dataset(tr_items, tr_fused, self.train_rot, self.train_zoom, self.seed * 10 ** 8, True)
        va = None
        if test is not None and test.benign and test.malign:
            va_fused = self._make_fused(self._device_candidates(test.malign), test.benign, test.benign,
                                        self.n_val_fused, self.n_val_fused_benign, self.seed + 2, "val")
            va_items = [(p, 0) for p in test.benign] + [(p, 1) for p in test.malign]
            va = self._dataset(va_items, va_fused, 1, 1.0, self.seed * 10 ** 8 + 5 * 10 ** 7, False)
        print(f"[voxunet] train {len(tr)} volumes ({int(tr.y.sum())} malign, {int(tr.has.sum())} with "
              f"voxel labels) | val {0 if va is None else len(va)}", flush=True)

        model = self._build_model(bottleneck)
        dl = torch.utils.data.DataLoader(tr, self.batch_size, shuffle=True, drop_last=True,
                                         num_workers=self.loader_workers,
                                         persistent_workers=self.loader_workers > 0)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.epochs * max(len(dl), 1))
        pos_w = torch.tensor((len(tr.y) - tr.y.sum()) / max(tr.y.sum(), 1), device=self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best, best_state = -1.0, None
        for ep in range(self.epochs):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for x, y, m, has in dl:
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=amp_enabled(self.device)):
                    logit, vox = model(x)
                loss = loss_fn(logit.float(), y)
                if self.seg_weight > 0:
                    loss = loss + self.seg_weight * _seg_loss(vox.float(), x, m.to(self.device), has.to(self.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                tot, n = tot + loss.item(), n + 1
            auc = (float(roc_auc_score(va.y, self._predict(model, va)))
                   if va is not None and len(set(va.y)) == 2 else float("nan"))
            print(f"[voxunet] epoch {ep + 1}/{self.epochs} loss={tot / max(n, 1):.4f} "
                  f"val_auc={auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if va is None or auc >= best:
                best = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._cfg = {"res": self.res, "k": self.k, "bottleneck": bottleneck, "seed": self.seed,
                     "best_val_auc": best}
        print(f"[voxunet] best val AUC {best:.4f}", flush=True)

    # ----------------------------------------------------------------- save / load / eval
    def save(self):
        if self._model is None:
            raise RuntimeError("VoxUNet.save called before train().")
        torch.save({"state": self._model.state_dict(), "cfg": self._cfg},
                   os.path.join(self.trained_dir, "model.pt"))

    def load(self):
        path = os.path.join(self.trained_dir, "model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No cached model at {path}; run `python evaluate.py --train` first.")
        ck = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ck["cfg"]
        if cfg["bottleneck"] == "mamba" and not _mamba_available():
            raise RuntimeError("This checkpoint uses the Mamba bottleneck; install mamba-ssm on a CUDA "
                               "machine, or retrain with bottleneck='conv'.")
        self.res, self.k = cfg["res"], cfg["k"]
        model = self._build_model(cfg["bottleneck"])
        model.load_state_dict(ck["state"])
        model.eval()
        self._model, self._cfg = model, cfg

    @torch.no_grad()
    def eval(self, obj_path):
        if self._model is None:
            return {"score": 0.0}
        m = load_mesh(obj_path)
        if m is None:
            return {"score": 1.0}                      # unreadable -> fail closed
        seed = path_seed(obj_path)
        occ = voxelize(*m, res=self.res, R=_rotation(seed), seed=seed)
        if occ is None:
            return {"score": 1.0}
        x = torch.from_numpy(occ.astype(np.float32))[None, None].to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=amp_enabled(self.device)):
            logit, _ = self._model(x)
        return {"score": float(torch.sigmoid(logit.float()).item())}
