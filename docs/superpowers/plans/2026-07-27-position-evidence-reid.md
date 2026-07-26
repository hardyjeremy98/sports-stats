# Position-Evidence Re-ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add position/occupancy evidence to tracklet re-ID as a calibrated log-likelihood-ratio channel that fuses with appearance, and measure whether it breaks the one-scalar plateau.

**Architecture:** Two new pure modules in `matchlab_core/reid/` (`occupancy.py` for footprint representation, `evidence.py` for LLR calibration and fusion), a FOOTPASS tactical loader in `matchlab_train/datasets/`, and experiment entry points that run the pre-registered Phase A falsification test and the Phase C fusion sweep. Nothing touches the shipped merge path until Phase C.

**Tech Stack:** Python 3.12, numpy, h5py, pydantic v2, pytest, uv.

**Spec:** [`docs/superpowers/specs/2026-07-27-position-evidence-reid.md`](../specs/2026-07-27-position-evidence-reid.md)

## Global Constraints

- Line length 100 (ruff, root `pyproject.toml`). Run `uv run ruff check packages` before each commit.
- Tests: `uv run pytest packages -q`. New tests live flat in `packages/matchlab_core/tests/test_*.py`.
- **Observability discipline:** occupancy footprints are built ONLY from frames where the player is observable (`ROI_X` non-NaN in FOOTPASS). Never use off-camera positions.
- **No role features as input.** FOOTPASS `ROLE_ID` is analysis-only; it must never enter a footprint, an LLR, or a merge decision.
- Pure modules take data in and return values — no file I/O, no config reads.
- Held-out SNMOT-124..127 stay untouched. All SoccerNet work is on tuning SNMOT-116..123.
- Pre-registered thresholds from the spec §3 are fixed before any arm runs and are not moved after seeing results.

---

### Task 1: FOOTPASS tactical loader

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/datasets/footpass.py`
- Test: `packages/matchlab_core/tests/test_footpass_loader.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `FootpassHalf` dataclass with fields `game_id: str`, `half: int`, `rows: np.ndarray` (N×14 or N×13); `load_half(path, key) -> FootpassHalf`; `observable_spans(half, player_id, max_gap_frames) -> list[tuple[int,int]]`; `COL` index constants (`COL.FRAME`, `COL.PLAYER_ID`, `COL.TEAM`, `COL.SHIRT`, `COL.ROLE`, `COL.X`, `COL.Y`, `COL.VX`, `COL.VY`, `COL.ROI_X`, `COL.ROI_Y`, `COL.ROI_W`, `COL.ROI_H`, `COL.CLS`).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from matchlab_train.datasets.footpass import COL, FootpassHalf, observable_spans


def _half(rows: list[list[float]]) -> FootpassHalf:
    return FootpassHalf(game_id="game_1", half=1, rows=np.asarray(rows, dtype=np.float32))


def test_observable_spans_splits_on_offcamera_gap():
    # Player 100 visible frames 0-2, off-camera 3-9, visible 10-11.
    nan = float("nan")
    rows = []
    for f in range(12):
        visible = f <= 2 or f >= 10
        rows.append([f, 100, 0, 7, 2, 0.1, 0.5, 0, 0, (1.0 if visible else nan), 0, 0, 0, 0])
    spans = observable_spans(_half(rows), player_id=100, max_gap_frames=2)
    assert spans == [(0, 2), (10, 11)]


def test_observable_spans_bridges_gap_within_tolerance():
    nan = float("nan")
    rows = []
    for f in range(6):
        visible = f != 3
        rows.append([f, 100, 0, 7, 2, 0.1, 0.5, 0, 0, (1.0 if visible else nan), 0, 0, 0, 0])
    spans = observable_spans(_half(rows), player_id=100, max_gap_frames=2)
    assert spans == [(0, 5)]


def test_col_indices_match_documented_schema():
    assert (COL.FRAME, COL.PLAYER_ID, COL.TEAM, COL.ROLE) == (0, 1, 2, 4)
    assert (COL.X, COL.Y, COL.ROI_X) == (5, 6, 9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_footpass_loader.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'matchlab_train.datasets.footpass'`

- [ ] **Step 3: Write minimal implementation**

```python
"""FOOTPASS tactical-data loader (SN-PCBAS-2026).

Schema and acquisition: docs/reference/footpass-setup.md. One HDF5 per split,
keyed `game_<id>_H<half>`; each value is an N x 14 float32 array of
per-player-per-frame rows (N x 13 for CHALLENGE, which drops CLS).

`ROI_X` is NaN when the player is not visible in frame. That flag is the whole
point of this loader for B2: a player's observable spans ARE their tracklets,
so exit/re-entry pairs come straight out of the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np


class COL:
    FRAME: Final = 0
    PLAYER_ID: Final = 1
    TEAM: Final = 2
    SHIRT: Final = 3
    ROLE: Final = 4
    X: Final = 5
    Y: Final = 6
    VX: Final = 7
    VY: Final = 8
    ROI_X: Final = 9
    ROI_Y: Final = 10
    ROI_W: Final = 11
    ROI_H: Final = 12
    CLS: Final = 13


ROLE_NAMES: Final[dict[int, str]] = {
    1: "GK", 2: "LB", 3: "LCB", 4: "MCB", 5: "RCB", 6: "LM", 7: "RM",
    8: "DM", 9: "AM", 10: "LW", 11: "RW", 12: "CF", 13: "RB",
}


@dataclass
class FootpassHalf:
    game_id: str
    half: int
    rows: np.ndarray

    @property
    def player_ids(self) -> list[int]:
        return sorted({int(v) for v in self.rows[:, COL.PLAYER_ID]})

    def player_rows(self, player_id: int) -> np.ndarray:
        return self.rows[self.rows[:, COL.PLAYER_ID] == player_id]

    def role_of(self, player_id: int) -> int:
        r = self.player_rows(player_id)
        return int(r[0, COL.ROLE]) if len(r) else 0

    def team_of(self, player_id: int) -> int:
        r = self.player_rows(player_id)
        return int(r[0, COL.TEAM]) if len(r) else -1


def load_half(path: str | Path, key: str) -> FootpassHalf:
    import h5py

    game_id, _, half = key.rpartition("_H")
    with h5py.File(str(path), "r") as f:
        rows = np.asarray(f[key][:], dtype=np.float32)
    return FootpassHalf(game_id=game_id, half=int(half), rows=rows)


def observable_spans(
    half: FootpassHalf, player_id: int, *, max_gap_frames: int = 2
) -> list[tuple[int, int]]:
    """Contiguous frame ranges where the player is visible in frame.

    Gaps of at most `max_gap_frames` are bridged (matching the GT-fragment
    harness's `gap_frames`, so FOOTPASS fragments are constructed the same way
    SoccerNet ones are). Returns inclusive (start_frame, end_frame) pairs.
    """
    rows = half.player_rows(player_id)
    if not len(rows):
        return []
    order = np.argsort(rows[:, COL.FRAME])
    rows = rows[order]
    frames = rows[:, COL.FRAME][~np.isnan(rows[:, COL.ROI_X])].astype(int)
    if not len(frames):
        return []
    spans: list[tuple[int, int]] = []
    start = prev = int(frames[0])
    for f in frames[1:]:
        f = int(f)
        if f - prev > max_gap_frames + 1:
            spans.append((start, prev))
            start = f
        prev = f
    spans.append((start, prev))
    return spans
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_footpass_loader.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Add a real-data smoke test (skipped when data absent)**

```python
FOOTPASS_VAL = Path("data/footpass/tactical/val_tactical_data.h5")


@pytest.mark.skipif(not FOOTPASS_VAL.exists(), reason="FOOTPASS tactical data not downloaded")
def test_real_val_half_matches_documented_schema():
    from matchlab_train.datasets.footpass import load_half

    half = load_half(FOOTPASS_VAL, "game_18_H1")
    assert half.rows.shape[1] == 14
    assert len(half.player_ids) >= 22
    spans = observable_spans(half, half.player_ids[0])
    assert len(spans) > 1, "a broadcast half must fragment the player at least once"
```

- [ ] **Step 6: Run and commit**

```bash
uv run pytest packages/matchlab_core/tests/test_footpass_loader.py -q
uv run ruff check packages
git add packages/matchlab_train/src/matchlab_train/datasets/footpass.py \
        packages/matchlab_core/tests/test_footpass_loader.py
git commit -m "feat(footpass): tactical loader with observability-based fragmentation"
```

---

### Task 2: Occupancy footprint representation

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/reid/occupancy.py`
- Test: `packages/matchlab_core/tests/test_reid_occupancy.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Footprint` dataclass (`grid: np.ndarray` shape `(gy, gx)`, `n_frames: int`); `build_footprint(xs, ys, *, grid=(12, 8), sigma=1.0) -> Footprint`; `js_distance(a: Footprint, b: Footprint) -> float`; `bimodality(fp: Footprint) -> float`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from matchlab_core.reid.occupancy import Footprint, bimodality, build_footprint, js_distance


def test_footprint_normalises_to_unit_mass():
    fp = build_footprint([0.1, 0.2, 0.3], [0.5, 0.5, 0.5])
    assert fp.grid.shape == (8, 12)
    assert np.isclose(fp.grid.sum(), 1.0)
    assert fp.n_frames == 3


def test_identical_footprints_have_zero_distance():
    fp = build_footprint([0.2] * 10, [0.5] * 10)
    assert js_distance(fp, fp) == 0.0


def test_opposite_corners_are_far_apart():
    left = build_footprint([0.05] * 10, [0.05] * 10)
    right = build_footprint([0.95] * 10, [0.95] * 10)
    assert js_distance(left, right) > 0.8


def test_blur_makes_adjacent_cells_nearer_than_distant_ones():
    a = build_footprint([0.10] * 10, [0.5] * 10)
    near = build_footprint([0.18] * 10, [0.5] * 10)
    far = build_footprint([0.90] * 10, [0.5] * 10)
    assert js_distance(a, near) < js_distance(a, far)


def test_bimodality_detects_two_separated_clusters():
    single = build_footprint([0.2] * 20, [0.5] * 20)
    fused = build_footprint([0.1] * 10 + [0.9] * 10, [0.5] * 20)
    assert bimodality(fused) > bimodality(single)


def test_empty_footprint_is_uniform_and_flagged():
    fp = build_footprint([], [])
    assert fp.n_frames == 0
    assert np.isclose(fp.grid.sum(), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_reid_occupancy.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'matchlab_core.reid.occupancy'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Occupancy footprints: where a tracklet's player spent their observable time.

A footprint is a blurred, normalised histogram of pitch positions over the
frames where the player was actually visible. Deliberately NOT a role label —
no role taxonomy is imposed, so roles emerge as footprint clusters and the
representation transfers to amateur football where "left back" may not mean
anything.

Pure: arrays in, values out. The blur (sigma in cells) is what makes adjacent
cells near without paying for an earth-mover distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_GRID = (12, 8)  # (x, y) cells; approximates the 105x68 m pitch aspect


@dataclass
class Footprint:
    grid: np.ndarray  # (gy, gx), sums to 1
    n_frames: int


def _blur(g: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return g
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    out = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, g)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, out)


def build_footprint(
    xs, ys, *, grid: tuple[int, int] = DEFAULT_GRID, sigma: float = 1.0
) -> Footprint:
    """Normalised occupancy over `grid` cells from normalised pitch coords.

    `xs`/`ys` are expected in [0, 1] and are clipped there — FOOTPASS carries
    slightly out-of-range values for players off the touchline.
    """
    gx, gy = grid
    g = np.zeros((gy, gx), dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = int(len(xs))
    if n:
        ix = np.clip((np.clip(xs, 0.0, 1.0) * gx).astype(int), 0, gx - 1)
        iy = np.clip((np.clip(ys, 0.0, 1.0) * gy).astype(int), 0, gy - 1)
        np.add.at(g, (iy, ix), 1.0)
        g = _blur(g, sigma)
        total = g.sum()
        g = g / total if total > 0 else np.full_like(g, 1.0 / g.size)
    else:
        g = np.full_like(g, 1.0 / g.size)
    return Footprint(grid=g, n_frames=n)


def js_distance(a: Footprint, b: Footprint) -> float:
    """Jensen-Shannon distance in [0, 1] (sqrt of the base-2 divergence)."""
    p, q = a.grid.ravel(), b.grid.ravel()
    m = 0.5 * (p + q)
    div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(0.0, min(1.0, div))))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))


def bimodality(fp: Footprint) -> float:
    """How much a footprint looks like two separated clusters rather than one.

    Mass-weighted spread about the centroid, normalised by grid diagonal. A
    footprint fusing two players' zones scores high — that is the self-training
    guard for the two-pass bootstrap.
    """
    g = fp.grid
    gy, gx = g.shape
    yy, xx = np.mgrid[0:gy, 0:gx]
    cy = float((g * yy).sum())
    cx = float((g * xx).sum())
    var = float((g * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum())
    return float(np.sqrt(var) / np.hypot(gy, gx))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_reid_occupancy.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/reid/occupancy.py \
        packages/matchlab_core/tests/test_reid_occupancy.py
git commit -m "feat(reid): occupancy footprint representation with JS distance"
```

---

### Task 3: Calibrated log-likelihood-ratio evidence module

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/reid/evidence.py`
- Test: `packages/matchlab_core/tests/test_reid_evidence.py`

**Interfaces:**
- Consumes: nothing (operates on plain distance/score floats).
- Produces: `LLRCalibrator` with `fit(same_scores, diff_scores) -> LLRCalibrator`, `llr(score) -> float`, `to_dict()/from_dict()`; `impostor_field_llr(score, field_scores, *, higher_is_better) -> float`; `fuse(llrs, weights=None) -> float`.

**Design note for the implementer:** the calibrator is a 1-D histogram-ratio estimator with Laplace smoothing, NOT a KDE or a logistic fit. Reason: the tail is what matters (spec §1), and a parametric fit smooths exactly the tail we are trying to measure. Bin edges are quantiles of the pooled data so bins are equally populated.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from matchlab_core.reid.evidence import LLRCalibrator, fuse, impostor_field_llr


def test_llr_is_positive_where_same_dominates_and_negative_where_diff_does():
    rng = np.random.default_rng(0)
    same = rng.normal(0.2, 0.05, 500)   # same players: small distance
    diff = rng.normal(0.8, 0.05, 500)   # different players: large distance
    cal = LLRCalibrator.fit(same, diff)
    assert cal.llr(0.2) > 1.0
    assert cal.llr(0.8) < -1.0


def test_llr_is_near_zero_where_distributions_overlap_completely():
    rng = np.random.default_rng(1)
    same = rng.normal(0.5, 0.1, 500)
    diff = rng.normal(0.5, 0.1, 500)
    cal = LLRCalibrator.fit(same, diff)
    assert abs(cal.llr(0.5)) < 0.5


def test_calibrator_round_trips_through_dict():
    cal = LLRCalibrator.fit([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    restored = LLRCalibrator.from_dict(cal.to_dict())
    assert restored.llr(0.2) == pytest.approx(cal.llr(0.2))


def test_impostor_field_llr_rewards_beating_the_field():
    # score 0.95 against a field topping out at 0.6 -> strong evidence
    strong = impostor_field_llr(0.95, [0.6, 0.5, 0.4], higher_is_better=True)
    # same score against a field with a 0.94 competitor -> weak evidence
    weak = impostor_field_llr(0.95, [0.94, 0.5, 0.4], higher_is_better=True)
    assert strong > weak


def test_impostor_field_llr_with_empty_field_is_neutral():
    assert impostor_field_llr(0.9, [], higher_is_better=True) == 0.0


def test_fuse_sums_channels():
    assert fuse([1.5, -0.5]) == pytest.approx(1.0)
    assert fuse([1.0, 1.0], weights=[1.0, 0.0]) == pytest.approx(1.0)


def test_fuse_ignores_none_channels_as_abstention():
    assert fuse([2.0, None]) == pytest.approx(2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_reid_evidence.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'matchlab_core.reid.evidence'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Calibrated evidence: turn a channel's raw score into a log-likelihood ratio.

Every prior merge experiment compared raw similarities against a hand-tuned
threshold, which is why channels could never be combined without inventing a
weight. An LLR is already in units of evidence, so channels sum.

The pair-dependence the design needs -- zone evidence strong for a left back vs
a right winger, weak for two centre backs -- is NOT engineered here. It falls
out of the denominator: informativeness is a property of the impostor
population, so a trait shared by the alternatives yields LLR ~ 0 on its own.

Histogram-ratio estimation, deliberately: a parametric fit smooths the
distribution tail, and the tail (the single most confident impostor) is exactly
what governs merge safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOG_CLAMP = 6.0  # +/- this many nats; keeps one empty bin from dominating a sum


@dataclass
class LLRCalibrator:
    edges: np.ndarray
    log_ratio: np.ndarray

    @classmethod
    def fit(cls, same_scores, diff_scores, *, bins: int = 20) -> LLRCalibrator:
        same = np.asarray(list(same_scores), dtype=np.float64)
        diff = np.asarray(list(diff_scores), dtype=np.float64)
        pooled = np.concatenate([same, diff])
        qs = np.linspace(0, 100, bins + 1)
        edges = np.unique(np.percentile(pooled, qs))
        if len(edges) < 3:
            edges = np.linspace(pooled.min() - 1e-6, pooled.max() + 1e-6, 3)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        hs, _ = np.histogram(same, bins=edges)
        hd, _ = np.histogram(diff, bins=edges)
        ps = (hs + 1.0) / (hs.sum() + len(hs))
        pd = (hd + 1.0) / (hd.sum() + len(hd))
        return cls(edges=edges, log_ratio=np.clip(np.log(ps / pd), -_LOG_CLAMP, _LOG_CLAMP))

    def llr(self, score: float) -> float:
        i = int(np.clip(np.searchsorted(self.edges, score, side="right") - 1,
                        0, len(self.log_ratio) - 1))
        return float(self.log_ratio[i])

    def to_dict(self) -> dict:
        return {"edges": self.edges.tolist(), "log_ratio": self.log_ratio.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> LLRCalibrator:
        return cls(edges=np.asarray(d["edges"]), log_ratio=np.asarray(d["log_ratio"]))


def impostor_field_llr(score: float, field_scores, *, higher_is_better: bool) -> float:
    """Local evidence: how far this score stands out from the alternatives
    actually competing for the same tracklet.

    This is the principled generalisation of the margin-over-runner-up rule that
    empirically governs merge quality (SPO-85e). An empty field is neutral --
    with no alternative there is nothing to discriminate against.
    """
    field = np.asarray(list(field_scores), dtype=np.float64)
    if not len(field):
        return 0.0
    runner_up = field.max() if higher_is_better else field.min()
    margin = (score - runner_up) if higher_is_better else (runner_up - score)
    spread = float(field.std()) if len(field) > 1 else abs(float(runner_up)) or 1.0
    return float(np.clip(margin / max(spread, 1e-6), -_LOG_CLAMP, _LOG_CLAMP))


def fuse(llrs, weights=None) -> float:
    """Sum calibrated channels. `None` is abstention (ADR 003: missing evidence
    is neutral, never a penalty)."""
    vals = [v for v in llrs if v is not None]
    if weights is None:
        return float(sum(vals))
    ws = [w for v, w in zip(llrs, weights, strict=True) if v is not None]
    return float(sum(v * w for v, w in zip(vals, ws, strict=True)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_reid_evidence.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/reid/evidence.py \
        packages/matchlab_core/tests/test_reid_evidence.py
git commit -m "feat(reid): calibrated log-LR evidence with impostor-field normalisation"
```

---

### Task 4: Phase A — the pre-registered falsification test

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/experiments/position_evidence.py`
- Test: `packages/matchlab_core/tests/test_position_evidence_experiment.py`
- Output: `docs/reports/2026-07-27-phase-a-occupancy-llr.md`

**Interfaces:**
- Consumes: `load_half`, `observable_spans`, `COL`, `ROLE_NAMES` (Task 1); `build_footprint`, `js_distance` (Task 2); `LLRCalibrator` (Task 3).
- Produces: `build_fragments(half, *, max_gap_frames, min_frames) -> list[Fragment]`; `Fragment` dataclass (`player_id`, `start`, `end`, `xs`, `ys`, `role`, `team`); `phase_a(val_path, train_path) -> dict` returning `{"auc": float, "by_role_pair": {...}, "h1_pass": bool, "h2_pass": bool}`.

**Pre-registered thresholds (spec §3, do not move after seeing results):**
- H1 pass: pooled AUC ≥ 0.70. Fail: < 0.60.
- H2 pass: mean `|LLR|` for distant-role impostors ≥ 1.5× same-role impostors, and same-role is the weakest bucket. Fail: ratio < 1.2.

- [ ] **Step 1: Write the failing test (synthetic, deterministic)**

```python
import numpy as np
from matchlab_train.experiments.position_evidence import Fragment, auc, role_distance


def test_auc_is_one_for_perfectly_separated_scores():
    assert auc(same=[0.1, 0.2], diff=[0.8, 0.9]) == 1.0


def test_auc_is_half_for_identical_distributions():
    assert auc(same=[0.5, 0.5], diff=[0.5, 0.5]) == 0.5


def test_role_distance_is_zero_for_same_role_and_large_across_the_pitch():
    assert role_distance(3, 3) == 0.0            # LCB vs LCB
    assert role_distance(2, 11) > role_distance(3, 5)  # LB vs RW > LCB vs RCB
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_position_evidence_experiment.py -q`
Expected: FAIL, module not found

- [ ] **Step 3: Implement fragments, AUC, role distance, and the phase_a driver**

Key implementation points (write these, do not stub):
- `build_fragments` uses `observable_spans` then pulls that span's rows, keeping `X_POS`/`Y_POS` **only for observable frames**.
- Labelled pairs: within a half, all fragment pairs; `same` iff `player_id` matches. Exclude pairs whose spans overlap (they cannot be a merge candidate) — mirrors `TemporalOverlapGate`.
- `role_distance(r1, r2)`: Euclidean distance between canonical role anchor positions on a normalised pitch, defined as a literal table in the module (GK 0.05/0.5, LB 0.25/0.85, LCB 0.15/0.65, MCB 0.15/0.5, RCB 0.15/0.35, LM 0.5/0.85, RM 0.5/0.15, DM 0.35/0.5, AM 0.6/0.5, LW 0.8/0.85, RW 0.8/0.15, CF 0.9/0.5, RB 0.25/0.15). Analysis-only.
- Calibrator fitted on `train` halves, evaluated on `val` halves — never the same half.
- `auc` computed by rank statistic (Mann-Whitney), no sklearn dependency.

- [ ] **Step 4: Run tests, then run the real experiment**

```bash
uv run pytest packages/matchlab_core/tests/test_position_evidence_experiment.py -q
uv run python -m matchlab_train.experiments.position_evidence --phase a
```

- [ ] **Step 5: Write the report with the verdict**

`docs/reports/2026-07-27-phase-a-occupancy-llr.md` must state H1/H2 pass/fail against the
pre-registered thresholds, the AUC, the by-role-pair table, and — if failed — say so plainly
and scope the negative to the representation and rule tested.

- [ ] **Step 6: Commit**

```bash
git add packages/matchlab_train/src/matchlab_train/experiments/position_evidence.py \
        packages/matchlab_core/tests/test_position_evidence_experiment.py \
        docs/reports/2026-07-27-phase-a-occupancy-llr.md
git commit -m "feat(reid)+docs: Phase A occupancy-LLR falsification test on FOOTPASS"
```

**GATE:** If H1 fails, stop here. Report the negative and do not implement Tasks 5-8.

---

### Task 5: Prep — enable calibration in the re-ID configs, triage the smoke failures

**Files:**
- Modify: `configs/pipeline.tdlp-full-reid-oracle.yaml:37-39`, `configs/pipeline.gt-tracklets-reid.yaml:43-45`
- Create: `docs/reports/2026-07-27-smoke-merge-triage.md`

- [ ] **Step 1: Triage the wrong merges on the two smoke runs**

Read `association.json` + `reid_detail.json` from the systest-116 / systest-120 run dirs. For
each wrong merge, record: same-team or cross-team, gap length, crop heights, and the GT roles
if resolvable. **The question that matters: are wrong merges same-team pairs?** Same-team
confirms position evidence is aimed at the right target; cross-team means the team gate is the
defect instead.

- [ ] **Step 2: Enable `pnlcalib` and measure calibration coverage**

Switch the `calibrate` slot from `static-demo`/disabled to `pnlcalib`, run the two tuning
sequences, and report per-frame calibration confidence coverage.

- [ ] **Step 3: Commit**

```bash
git add configs/ docs/reports/2026-07-27-smoke-merge-triage.md
git commit -m "fix(configs)+docs: enable pitch calibration in re-ID configs; triage smoke merge failures"
```

---

### Task 6: Phase B — appearance LLR correctness check

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/reid/evidence.py` (add `appearance_llr_channel`)
- Test: `packages/matchlab_core/tests/test_reid_evidence.py` (extend)

- [ ] **Step 1: Write the failing test** — the registered prediction is that ranking by
  impostor-field LLR over appearance reproduces the mutual-best + margin admission set on a
  synthetic pool. Assert set equality.
- [ ] **Step 2: Run, verify fail. Step 3: implement. Step 4: verify pass. Step 5: commit.**

A large divergence here means the machinery is wrong, not that something was discovered.

---

### Task 7: Phase C — fusion and the operating-point sweep

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/stages/associate/reid_engine.py` (add `position_evidence: bool` + `position_weight: float` params, default OFF)
- Modify: `packages/matchlab_core/src/matchlab_core/reid/merge.py` only if the similarity callable signature must widen — prefer passing a fused score through the existing `similarity` callable
- Test: `packages/matchlab_core/tests/test_associate_reid.py` (extend)

- [ ] **Step 1: Write the failing test** — a fused run merges a pair that appearance alone
  rejects, when the two fragments share a footprint and the impostor does not.
- [ ] **Steps 2-4: TDD cycle.**
- [ ] **Step 5: Sweep** across `position_weight` and the similarity floor; report **correct
  merges at matched wrong-merge budgets**, never a single operating point.
- [ ] **Step 6: Commit.**

---

### Task 8: Phase D — two-pass bootstrap

Only if Phase C shows a gain. Conservative pass 1 → footprints (bimodality-rejected,
time-windowed) → pass 2. Bounded iterations, convergence shown.

---

### Task 9: Report, docs, Linear

- `docs/reports/2026-07-27-position-evidence-reid.md` — full verdict.
- `docs/implementation-status.md` — capability rows + Known findings.
- Linear: close SPO-74 (premise falsified), move SPO-85 out of Backlog, file the new work.

## Self-Review

**Spec coverage:** §2.1 → Task 3; §2.2 → Tasks 3, 4; §2.4 → Task 2; §2.5 → Task 8; §2.6 →
Tasks 6, 7; §3 Phase A → Task 4; Phase B → Task 6; Phase C → Task 7; Phase D → Task 8; §6
deliverables → Tasks 1-4, 9. Risk "position collapses to the motion gate" → Task 5 enables the
gate so the baseline is honest.

**Placeholders:** Tasks 6-9 carry less code than 1-4 by design — they depend on Phase A's
verdict, and writing their internals now would be speculative. Their interfaces and tests are
specified; internals are written when the gate opens.

**Type consistency:** `Footprint`, `LLRCalibrator`, `Fragment`, `COL` used consistently across
Tasks 1-4. `build_footprint(xs, ys)` takes normalised coords everywhere.
