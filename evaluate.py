import argparse
import concurrent.futures
import functools
import json
import os

from detectors.always_benign import AlwaysBenign
from detectors.always_malign import AlwaysMalign
from detectors.always_random import Random
from detectors.file_hash import FileHash
from detectors.laplacian_spectrum import LaplacianSpectrum
from detectors.pointnet import PointNet
from detectors.image_classifier import ImageClassifier
from detectors.voxunet import VoxUNet
# Deferred: the PSO solver builds and runs on macOS, but it runs a full
# 1500-particle x 200-iteration swarm per mesh (~2-3 min/shape, ~9h for the
# eval split), so it's left out of routine runs for now.
# from detectors.pso_fit import ParticleSwarm

DETECTORS = [
    ("always_benign", AlwaysBenign),
    ("always_malign", AlwaysMalign),
    ("always_random", Random),
    ("file_hash", FileHash),
    ("laplacian_spectrum", LaplacianSpectrum),
    ("pointnet", PointNet),
    ("image_classifier", ImageClassifier),
    ("voxunet", VoxUNet),
    # ("pso", ParticleSwarm),  # deferred: too slow for routine runs (see import above)
]

def _run(args, train=False, install=False):
    name, cls = args
    print(f"[{name}] {'Training + scoring' if train else 'Loading + scoring'}...", flush=True)
    try:
        instance = cls()
        n = instance.eval_method(os.path.join("output", "by_detector", name),
                                 train=train, install=install)
        print(f"[{name}] Done.", flush=True)
        return name, True, n
    except Exception as e:
        print(f"[{name}] Failed: {e}", flush=True)
        return name, False, str(e)


def _read_timing(name):
    """Return the timing dict written by ``eval_method`` for ``name``, or {}."""
    try:
        with open(os.path.join("output", "by_detector", name, "timing.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Score detectors in DETECTORS on the held-out eval split."
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Retrain (selected) detectors from scratch and cache the result to each "
             "detector's trained/ folder. Without this flag (the default) detectors "
             "load their cached artifacts and nothing is retrained -- run with "
             "--train once before the first plain `evaluate.py`.",
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Run each detector's one-time install() (build native code, fetch "
             "weights, ...) before training. Off by default, so installation is "
             "not repeated on every run; pass it once after adding/changing a detector.",
    )
    parser.add_argument(
        "--detectors", nargs="+", metavar="NAME", default=None,
        help="Only run these detectors (by name); default: all of "
             f"{[n for n, _ in DETECTORS]}. Lets you install/train/score a single "
             "detector without touching the others (e.g. --detectors image_classifier).",
    )
    args = parser.parse_args()

    selected = DETECTORS
    if args.detectors:
        by_name = dict(DETECTORS)
        unknown = [m for m in args.detectors if m not in by_name]
        if unknown:
            raise SystemExit(f"Unknown detector(s): {unknown}. "
                             f"Available: {[n for n, _ in DETECTORS]}")
        selected = [(n, by_name[n]) for n in args.detectors]

    os.makedirs(os.path.join("output", "by_detector"), exist_ok=True)

    run = functools.partial(_run, train=args.train, install=args.install)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run, selected))

    print("\n=== Evaluation Summary ===")
    for name, success, info in results:
        if not success:
            print(f"  {name:20s}  FAILED — {info}")
            continue
        t = _read_timing(name)
        bits = []
        if "install_seconds" in t:
            bits.append(f"install {t['install_seconds']:.1f}s")
        if "train_seconds" in t:
            bits.append(f"train {t['train_seconds']:.1f}s")
        if "eval_seconds_mean" in t:
            bits.append(f"eval {t['eval_seconds_mean'] * 1e3:.1f}ms/shape")
        extra = ("  [" + ", ".join(bits) + "]") if bits else ""
        print(f"  {name:20s}  scored {info} shapes{extra}")
    if not args.train and any(not s for _, s, _ in results):
        print("\nSome detectors failed to load a cache. Run `python evaluate.py --train` "
              "once to train and cache them first.")
    print("\nRun `python generate_report.py` to build the ROC comparison and leaderboard.")

if __name__ == "__main__":
    main()
