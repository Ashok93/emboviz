"""W2 equality gate: batched diagnostics must be numerically identical to
their pre-refactor (sequential) selves.

Strategy: run each refactored diagnostic against a deterministic MockVLA,
then run the git-HEAD (pre-batching) version of the SAME diagnostic against
the SAME mock + scene + calibration, and assert the DiagnosticResult is
bit-for-bit identical. MockVLA.predict is a pure function of (image bytes,
instruction) and uses the default loop predict_batch, so any divergence is a
real refactor bug (bad index/scatter/order), not model nondeterminism.

Run: uv run python scripts/dev/verify_w2_batching.py
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, is_dataclass
from enum import Enum

import numpy as np

from emboviz.calibration import ModelCalibration
from emboviz.core.types import Scene
from emboviz.models.mock import MockVLA
from emboviz.modality_pools import ModalityPool
from emboviz.perturb._target_detection import TargetDetection

REPO = "/Users/benutzer/projects/botsigil"


def load_old_class(relpath: str, classname: str):
    """exec the git-HEAD version of a module in a fresh namespace and return
    one of its classes. Absolute imports resolve against the live tree."""
    src = subprocess.check_output(["git", "show", f"HEAD:{relpath}"], cwd=REPO).decode()
    ns: dict = {"__name__": f"_old_{classname}", "__file__": relpath, "__builtins__": __builtins__}
    exec(compile(src, f"OLD::{relpath}", "exec"), ns)
    return ns[classname]


def normalize(x):
    """Recursively turn a result into comparable plain Python."""
    if is_dataclass(x) and not isinstance(x, type):
        return {k: normalize(v) for k, v in asdict(x).items()}
    if isinstance(x, Enum):
        return ("enum", x.__class__.__name__, x.value)
    if isinstance(x, np.ndarray):
        return ("ndarray", x.shape, x.astype(np.float64).round(10).tolist())
    if isinstance(x, (np.floating, np.integer)):
        return normalize(x.item())
    if isinstance(x, dict):
        return {k: normalize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [normalize(v) for v in x]
    if isinstance(x, float):
        return round(x, 10) if math.isfinite(x) else str(x)
    return x


def diff(a, b, path=""):
    """Yield human-readable mismatches between two normalized structures."""
    na, nb = normalize(a), normalize(b)
    out: list[str] = []

    def walk(x, y, p):
        if type(x) is not type(y):
            out.append(f"{p}: type {type(x).__name__} != {type(y).__name__}")
            return
        if isinstance(x, dict):
            if set(x) != set(y):
                out.append(f"{p}: keys {sorted(x)} != {sorted(y)}")
                return
            for k in x:
                walk(x[k], y[k], f"{p}.{k}")
        elif isinstance(x, list):
            if len(x) != len(y):
                out.append(f"{p}: len {len(x)} != {len(y)}")
                return
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{p}[{i}]")
        else:
            if x != y:
                out.append(f"{p}: {x!r} != {y!r}")

    walk(na, nb, path or "root")
    return out


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def make_scene(seed=0, size=48):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return Scene.from_image(img, instruction="pick up the red block", scene_id="s0")


def make_calib():
    # Deterministic mock → zero noise floor; unit typical magnitude.
    return ModelCalibration(
        noise_floor=0.0, typical_action_magnitude=1.0,
        n_noise_probes=3, n_baseline_frames=3, n_samples=1,
    )


class MaskDetector:
    """Tiny TargetDetector returning a fixed box mask (memorization needs a mask)."""
    def __init__(self, box):
        self.box = box

    def __call__(self, scene):
        arr = np.asarray(scene.observations.images["primary"].data)
        H, W = arr.shape[:2]
        m = np.zeros((H, W), dtype=bool)
        x0, y0, x1, y1 = self.box
        m[y0:y1, x0:x1] = True
        return TargetDetection(bbox=self.box, mask=m, label="target", confidence=0.9)


# ----------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------
def case_sensitivity():
    from emboviz.diagnostics.sensitivity_map import SensitivityMapDiagnostic as New
    Old = load_old_class("emboviz/diagnostics/sensitivity_map.py", "SensitivityMapDiagnostic")
    model = MockVLA(mode="noun_blind", action_dim=7, seed=0)
    scene, calib = make_scene(), make_calib()
    base = model.predict(scene)
    kw = dict(grid_side=4, calibration=calib)
    rn = New(**kw).run(model, scene, baseline=base)
    ro = Old(**kw).run(model, scene, baseline=base)
    return diff(rn, ro)


def case_memorization():
    from emboviz.diagnostics.memorization import MemorizationDiagnostic as New
    Old = load_old_class("emboviz/diagnostics/memorization.py", "MemorizationDiagnostic")
    model = MockVLA(mode="noun_blind", action_dim=7, seed=0)
    scene, calib = make_scene(), make_calib()
    base = model.predict(scene)
    kw = lambda: dict(target_detector=MaskDetector((8, 8, 40, 40)),
                      fill_modes=["channel_mean", "gaussian_blur"], calibration=calib)
    rn = New(**kw()).run(model, scene, baseline=base)
    ro = Old(**kw()).run(model, scene, baseline=base)
    return diff(rn, ro)


def case_modality_dropout():
    from emboviz.diagnostics.modality_dropout import ModalityDropoutDiagnostic as New
    Old = load_old_class("emboviz/diagnostics/modality_dropout.py", "ModalityDropoutDiagnostic")
    model = MockVLA(mode="noun_blind", action_dim=7, seed=0)
    scene, calib = make_scene(), make_calib()
    base = model.predict(scene)
    # Pool with image (noun_blind responds to image → exercises per-sample order)
    # and instruction samples (consumed but ignored → zero response, still valid).
    img_pool = [np.random.default_rng(s).integers(0, 256, (48, 48, 3), np.uint8)
                for s in (11, 22, 33, 44, 55)]
    pool = ModalityPool(
        instruction_samples=["open the drawer", "stack the cubes", "wipe the table"],
        image_samples={"primary": img_pool},
        ref_distance={"image:primary": 0.0, "instruction": 0.0},
    )
    kw = lambda: dict(pool=pool, calibration=calib, k_samples=4, seed=7)
    rn = New(**kw()).run(model, scene, baseline=base)
    ro = Old(**kw()).run(model, scene, baseline=base)
    return diff(rn, ro)


def main():
    cases = [
        ("sensitivity_map", case_sensitivity),
        ("memorization", case_memorization),
        ("modality_dropout", case_modality_dropout),
    ]
    failed = 0
    for name, fn in cases:
        try:
            diffs = fn()
        except Exception as e:
            import traceback
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
            continue
        if diffs:
            failed += 1
            print(f"[FAIL]  {name}: {len(diffs)} mismatch(es)")
            for d in diffs[:20]:
                print(f"          {d}")
        else:
            print(f"[OK]    {name}: batched == sequential (identical DiagnosticResult)")
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} CASE(S) FAILED'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
