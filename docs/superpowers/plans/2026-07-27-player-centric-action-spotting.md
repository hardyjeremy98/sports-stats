# Player-Centric Action Spotting — Implementation Plan (Phases 0–1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the FOOTPASS TAAD→DST player-centric action-spotting baseline
inside MatchLab, scored on FOOTPASS VAL, as the foundation for swapping in
MatchDay-produced tracking, roles and identity.

**Architecture:** Two stages joined by a frozen `(9, 26, T)` fp16 contract. Stage 1 is
a shared-backbone video model that pools per-player features by `roi_align` off a
stride-8 feature map. Stage 2 is an encoder-decoder Transformer that translates a
30-second window of noisy per-slot logits into an ordered event list. Players are
indexed by **role slot** (`left_to_right * 13 + role_id`, 26 slots), never by jersey.

**Tech Stack:** Python 3.12, PyTorch 2.12 (cu130), pytorchvideo (X3D-S), h5py,
pydantic v2, pytest. FOOTPASS reference is Apache-2.0 — an in-repo dependency, no
isolated environment.

**Spec:** `docs/superpowers/specs/2026-07-27-player-centric-action-spotting-design.md`

## Global Constraints

- Line length 100 (`uv run ruff check packages` must pass).
- `from __future__ import annotations` at the top of every new module.
- Python pinned to 3.12; sync dependency groups together:
  `uv sync --group cv --group eval --group dev`.
- **Never use recursive grep.** Use `git grep -n <pattern> -- <pathspec>` or `Read`.
- Data lives in the **main checkout** at `/home/jeremy/code/MatchDay/lab/data`, not in
  the worktree. Reference it by absolute path or via `MATCHLAB_DATA_DIR`.
- Secrets (`HF_TOKEN`, `SOCCERNET_NDA_PASSWORD`) are in `/home/jeremy/code/MatchDay/lab/.env`.
  **Never** print them or commit them.
- Class indices are fixed and must never be reordered:
  `0 background, 1 drive, 2 pass, 3 cross, 4 throw-in, 5 shot, 6 header, 7 tackle, 8 block`.
- Role slot is always `left_to_right * 13 + (role_id - 1)`, range 0–25.
- Every measured claim records dataset, split, sequence count, code revision.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
  ```
- Work on `worktree-spo-action-spotting-prd`. Never on `main`.

## File Structure

| File | Responsibility |
|---|---|
| `packages/matchlab_core/src/matchlab_core/pcbas/__init__.py` | package marker |
| `packages/matchlab_core/src/matchlab_core/pcbas/schema.py` | column constants, class names, slot helpers — one source of truth |
| `packages/matchlab_core/src/matchlab_core/pcbas/eval.py` | greedy-matching macro-F1 metric |
| `packages/matchlab_train/src/matchlab_train/datasets/footpass.py` | HDF5 → our schemas + roster + event GT |
| `packages/matchlab_core/src/matchlab_core/pcbas/action_head.py` | X3D-S + FPN + roi_align + temporal head |
| `packages/matchlab_core/src/matchlab_core/pcbas/denoiser.py` | encoder-decoder Transformer |
| `packages/matchlab_train/src/matchlab_train/datasets/footpass_clips.py` | clip sampler / dataloader for the action head |
| `packages/matchlab_train/src/matchlab_train/experiments/pcbas_*.py` | training + inference experiments |
| `packages/matchlab_core/tests/test_pcbas_*.py`, `packages/matchlab_train/tests/test_footpass*.py` | tests |

---

# PHASE 0 — Ingest, metric, GT (no video required)

Everything here runs against `data/footpass/tactical/*.h5`, already on disk.

---

### Task 1: Column schema and slot helpers

The one place column indices and class names are defined. Every later task imports
from here, so a schema mistake is caught once rather than in five modules.

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/pcbas/__init__.py` (empty)
- Create: `packages/matchlab_core/src/matchlab_core/pcbas/schema.py`
- Test: `packages/matchlab_core/tests/test_pcbas_schema.py`

**Interfaces:**
- Produces: `CLASS_NAMES`, `N_CLASSES`, `N_ROLES`, `N_SLOTS`, column index constants,
  `slot_index(left_to_right, role_id) -> int`, `slot_to_role(slot) -> tuple[int, int]`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pytest

from matchlab_core.pcbas.schema import (
    CLASS_NAMES,
    N_CLASSES,
    N_SLOTS,
    CLS,
    FRAME,
    LEFT_TO_RIGHT,
    ROLE_ID,
    ROI_X,
    X_POS,
    slot_index,
    slot_to_role,
)


def test_class_order_is_the_reference_order():
    # Frozen: reordering silently corrupts every trained checkpoint and metric.
    assert CLASS_NAMES == [
        "background", "drive", "pass", "cross",
        "throw-in", "shot", "header", "tackle", "block",
    ]
    assert N_CLASSES == 9


def test_column_indices_match_tactical_data_format():
    # From the dataset's own tactical_data_format.txt
    assert (FRAME, LEFT_TO_RIGHT, ROLE_ID, X_POS, ROI_X, CLS) == (0, 2, 4, 5, 9, 13)


def test_slot_index_packs_team_and_role():
    assert slot_index(0, 1) == 0     # left team, role 1
    assert slot_index(0, 13) == 12
    assert slot_index(1, 1) == 13    # right team, role 1
    assert slot_index(1, 13) == 25
    assert N_SLOTS == 26


def test_slot_round_trips():
    for ltr in (0, 1):
        for role in range(1, 14):
            assert slot_to_role(slot_index(ltr, role)) == (ltr, role)


@pytest.mark.parametrize("ltr,role", [(0, 0), (0, 14), (2, 1), (-1, 5)])
def test_slot_index_rejects_out_of_range(ltr, role):
    with pytest.raises(ValueError):
        slot_index(ltr, role)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_pcbas_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: matchlab_core.pcbas`

- [ ] **Step 3: Write the module**

```python
"""FOOTPASS / PCBAS tactical-data schema -- the single source of truth.

Column layout is quoted verbatim from the dataset's own `tactical_data_format.txt`:

    frame, player_id, left_to_right, shirt_number, role_id,
    x, y, speed_x, speed_y, roi_x, roi_y, roi_width, roi_height, class

TRAIN/VAL carry 14 columns; the CHALLENGE/TEST split carries 13 -- it has no `class`,
so its labels are withheld and it cannot be scored locally.

`roi_*` are FULL-HD pixel coordinates and are NaN when the player is off-screen in
the broadcast (59% of rows). The reference model consumes 352x640 video and divides
roi_x by 3 and roi_y by 3.068181 to match.
"""

from __future__ import annotations

# --- column indices ---
FRAME = 0
PLAYER_ID = 1
LEFT_TO_RIGHT = 2
SHIRT_NUMBER = 3
ROLE_ID = 4
X_POS = 5
Y_POS = 6
X_SPEED = 7
Y_SPEED = 8
ROI_X = 9
ROI_Y = 10
ROI_WIDTH = 11
ROI_HEIGHT = 12
CLS = 13

N_COLUMNS_LABELLED = 14    # TRAIN / VAL
N_COLUMNS_UNLABELLED = 13  # CHALLENGE / TEST -- no `class`

# --- classes ---
# FROZEN ORDER. Index is the on-disk label value; reordering invalidates every
# checkpoint and every reported number.
CLASS_NAMES = [
    "background", "drive", "pass", "cross",
    "throw-in", "shot", "header", "tackle", "block",
]
N_CLASSES = len(CLASS_NAMES)
BACKGROUND = 0
ACTION_CLASSES = tuple(range(1, N_CLASSES))  # the 8 scored classes

# --- role slots ---
N_ROLES = 13
N_TEAMS = 2
N_SLOTS = N_ROLES * N_TEAMS  # 26

# Full-HD -> 352x640 scaling, matching the reference dataset exactly.
ROI_SCALE_X = 3.0
ROI_SCALE_Y = 3.068181


def slot_index(left_to_right: int, role_id: int) -> int:
    """Pack (team, role) into a 0..25 slot. `role_id` is 1-based on disk."""
    if left_to_right not in (0, 1):
        raise ValueError(f"left_to_right must be 0 or 1, got {left_to_right}")
    if not 1 <= role_id <= N_ROLES:
        raise ValueError(f"role_id must be 1..{N_ROLES}, got {role_id}")
    return left_to_right * N_ROLES + (role_id - 1)


def slot_to_role(slot: int) -> tuple[int, int]:
    """Inverse of `slot_index`: slot -> (left_to_right, role_id)."""
    if not 0 <= slot < N_SLOTS:
        raise ValueError(f"slot must be 0..{N_SLOTS - 1}, got {slot}")
    return slot // N_ROLES, (slot % N_ROLES) + 1
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_pcbas_schema.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/pcbas/ packages/matchlab_core/tests/test_pcbas_schema.py
git commit -m "feat(pcbas): FOOTPASS tactical-data schema + role-slot helpers"
```

---

### Task 2: FOOTPASS ingest adapter

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/datasets/footpass.py`
- Test: `packages/matchlab_train/tests/test_footpass_ingest.py`

**Interfaces:**
- Consumes: `matchlab_core.pcbas.schema` (Task 1); `matchlab_core.schemas`;
  `matchlab_core.event_gt.EventGroundTruth`.
- Produces:
  - `list_halves(h5_path) -> list[str]`
  - `load_half(h5_path, key) -> np.ndarray` — (N, 14)
  - `half_to_tracklets(arr) -> tuple[list[Tracklet], list[TeamAssignment]]`
  - `half_to_events(arr, key) -> EventGroundTruth`
  - `roster_lookup(arr) -> dict[tuple[int, int], int]` — `(frame, slot) -> shirt_number`
  - `PCBASEvent` pydantic model: `frame_idx, left_to_right, role_id, slot, shirt_number, class_id, score`

- [ ] **Step 1: Write the failing test**

Build a small synthetic (N,14) array rather than depending on the 8.8 GB TRAIN file —
the real file is exercised in Task 4's verification step.

```python
from __future__ import annotations

import numpy as np
import pytest

from matchlab_core.pcbas.schema import CLS, FRAME, LEFT_TO_RIGHT, ROLE_ID, ROI_X
from matchlab_train.datasets.footpass import (
    half_to_events,
    half_to_tracklets,
    roster_lookup,
)


def _row(frame, pid, ltr, shirt, role, x, y, vx, vy, roi, cls):
    r = np.full(14, np.nan, dtype=np.float64)
    r[0], r[1], r[2], r[3], r[4] = frame, pid, ltr, shirt, role
    r[5], r[6], r[7], r[8] = x, y, vx, vy
    if roi is not None:
        r[9], r[10], r[11], r[12] = roi
    r[13] = cls
    return r


def _arr():
    rows = []
    for f in range(4):
        rows.append(_row(f, 100, 0, 7, 1, 0.5, 0.5, 0.0, 0.0, (960, 540, 60, 120), 0))
        rows.append(_row(f, 101, 1, 9, 4, 0.6, 0.4, 0.1, 0.0, None, 0))  # off-screen
    rows[2][CLS] = 2  # frame 1, player 100 -> a pass
    return np.stack(rows)


def test_tracklets_only_include_observed_boxes():
    tracklets, _ = half_to_tracklets(_arr())
    by_id = {t.tracklet_id: t for t in tracklets}
    assert len(by_id[100].frames) == 4          # on-screen every frame
    assert 101 not in by_id or not by_id[101].frames  # never on-screen -> no boxes


def test_boxes_are_scaled_to_352x640():
    tracklets, _ = half_to_tracklets(_arr())
    box = {t.tracklet_id: t for t in tracklets}[100].frames[0].box
    assert box.x1 == pytest.approx(960 / 3.0, abs=1.0)
    assert box.y1 == pytest.approx(540 / 3.068181, abs=1.0)


def test_events_extracted_with_slot_and_shirt():
    gt = half_to_events(_arr(), "game_1_H1")
    assert len(gt.events) == 1
    ev = gt.events[0]
    assert ev.frame_idx == 1
    assert ev.class_id == 2          # pass
    assert ev.slot == 0              # left team, role 1
    assert ev.shirt_number == 7


def test_background_rows_are_not_events():
    arr = _arr()
    arr[:, CLS] = 0
    assert half_to_events(arr, "k").events == []


def test_roster_lookup_maps_frame_and_slot_to_shirt():
    lut = roster_lookup(_arr())
    assert lut[(0, 0)] == 7
    assert lut[(0, 13 + 3)] == 9     # right team, role 4 -> slot 16


def test_off_screen_rows_still_produce_roster_and_events():
    """The 17.5% of actions on off-screen players must NOT be dropped -- they are
    exactly the cases only the sequence stage can recover."""
    arr = _arr()
    arr[1][CLS] = 5                  # frame 0, player 101 (no bbox) -> a shot
    gt = half_to_events(arr, "k")
    assert any(e.class_id == 5 and e.slot == 16 for e in gt.events)


def test_unlabelled_split_is_rejected():
    with pytest.raises(ValueError, match="13 columns"):
        half_to_events(np.zeros((3, 13)), "challenge_key")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/matchlab_train/tests/test_footpass_ingest.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the adapter**

Key requirements the tests pin, restated so the implementer does not have to infer:

- Rows with NaN `roi_x` contribute **no** `TrackletFrame` but **do** contribute
  roster entries and events.
- Boxes are `(roi_x/3, roi_y/3.068181)` top-left with `roi_width/3`,
  `roi_height/3.068181` extent, matching the reference exactly.
- `half_to_events` raises `ValueError` mentioning "13 columns" when passed an
  unlabelled (CHALLENGE) array.
- `TeamAssignment.team` maps `left_to_right==0 → Team.HOME`, `1 → Team.AWAY`,
  mirroring `stages/team/oracle.py`. Record in a comment that this is an arbitrary
  but consistent convention — `left_to_right` is an attacking direction, not a
  home/away fact.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/matchlab_train/tests/test_footpass_ingest.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff check packages
git add packages/matchlab_train/src/matchlab_train/datasets/footpass.py \
        packages/matchlab_train/tests/test_footpass_ingest.py
git commit -m "feat(pcbas): FOOTPASS tactical HDF5 ingest adapter"
```

---

### Task 3: The PCBAS metric

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/pcbas/eval.py`
- Test: `packages/matchlab_core/tests/test_pcbas_eval.py`

**Interfaces:**
- Consumes: `PCBASEvent` (Task 2), `matchlab_core.pcbas.schema`.
- Produces:
  - `ClassMetrics` (pydantic): `class_id, class_name, n_gt, n_pred, tp, fp, fn, precision, recall, f1`
  - `PCBASReport` (pydantic): `delta, conf_thresh, per_class, macro_f1, micro_f1`
  - `score_events(gt, pred, *, delta=12, conf_thresh=0.15) -> PCBASReport`

**This is NOT the same metric as `action_spotting_eval.average_map`.** That one is
class+time avg-mAP for SoccerNet-ball. Both must coexist; never quote one for the
other's task.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from matchlab_core.pcbas.eval import score_events
from matchlab_train.datasets.footpass import PCBASEvent


def _ev(frame, slot, cls, score=1.0):
    return PCBASEvent(
        frame_idx=frame, left_to_right=slot // 13, role_id=(slot % 13) + 1,
        slot=slot, shirt_number=0, class_id=cls, score=score,
    )


def test_exact_match_is_a_true_positive():
    r = score_events([_ev(100, 0, 2)], [_ev(100, 0, 2)])
    assert r.macro_f1 == 1.0


def test_match_within_delta():
    r = score_events([_ev(100, 0, 2)], [_ev(111, 0, 2)], delta=12)
    assert r.macro_f1 == 1.0


def test_no_match_outside_delta():
    r = score_events([_ev(100, 0, 2)], [_ev(113, 0, 2)], delta=12)
    assert r.macro_f1 == 0.0


def test_wrong_class_is_not_a_match():
    r = score_events([_ev(100, 0, 2)], [_ev(100, 0, 3)])
    assert r.macro_f1 == 0.0


def test_wrong_slot_is_not_a_match():
    """Player-CENTRIC: right action, right time, wrong player is a miss.
    This is the whole difference from class+time avg-mAP."""
    r = score_events([_ev(100, 0, 2)], [_ev(100, 5, 2)])
    assert r.macro_f1 == 0.0


def test_low_confidence_predictions_are_discarded():
    r = score_events([_ev(100, 0, 2)], [_ev(100, 0, 2, score=0.10)], conf_thresh=0.15)
    assert r.per_class[2].tp == 0
    assert r.per_class[2].fn == 1


def test_one_gt_absorbs_only_one_prediction():
    """Two predictions on one GT -> 1 TP, 1 FP, never 2 TP."""
    r = score_events([_ev(100, 0, 2)], [_ev(100, 0, 2), _ev(102, 0, 2)])
    assert (r.per_class[2].tp, r.per_class[2].fp) == (1, 1)


def test_greedy_matching_prefers_the_nearest_prediction():
    r = score_events([_ev(100, 0, 2)], [_ev(108, 0, 2, 0.9), _ev(101, 0, 2, 0.8)])
    assert r.per_class[2].tp == 1


def test_macro_f1_averages_over_scored_classes_only():
    """Background is never scored, and absent classes must not silently count as 1.0."""
    r = score_events([_ev(1, 0, 2), _ev(50, 0, 5)], [_ev(1, 0, 2), _ev(50, 0, 5)])
    assert r.macro_f1 == 1.0
    assert 0 not in r.per_class


def test_empty_predictions_give_zero_not_a_crash():
    r = score_events([_ev(1, 0, 2)], [])
    assert r.macro_f1 == 0.0
    assert r.per_class[2].fn == 1


def test_per_class_counts_are_reported():
    """Rare classes (VAL has 26 tackles) must never be readable as a bare average."""
    r = score_events([_ev(1, 0, 7)], [_ev(1, 0, 7)])
    assert r.per_class[7].n_gt == 1
    assert r.per_class[7].class_name == "tackle"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_pcbas_eval.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the metric**

Matching rule, mirroring `utils/metric_utils.py`:
1. Discard predictions with `score < conf_thresh`.
2. Group GT and predictions by `(class_id, slot)`.
3. Within each group, sort predictions by **descending score**; for each, take the
   unmatched GT with the smallest `|Δframe|` where `|Δframe| <= delta`. Mark it used.
4. Matched → TP; unmatched prediction → FP; unmatched GT → FN.
5. Per-class P/R/F1; `macro_f1` averages F1 over the 8 action classes **that appear in
   GT** (a class with `n_gt == 0` is excluded, not scored as 1.0).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_pcbas_eval.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/pcbas/eval.py \
        packages/matchlab_core/tests/test_pcbas_eval.py
git commit -m "feat(pcbas): player-centric macro-F1 metric (class + time + slot)"
```

---

### Task 3b: Validate the metric against the reference's own shipped predictions — **do this before Task 4**

The cheapest correctness gate available: no video, no NDA, no GPU, no training. The
FOOTPASS repo ships `playbyplay_GT/playbyplay_val.json` and
`playbyplay_PRED/playbyplay_{TAAD,DST}_val.json` — final event lists. Scoring them with
our metric must reproduce the reference's own numbers.

**Targets, measured by running the reference's `evaluation.py` on its own artifacts
(δ=12, τ=0.15):**

| arm | micro-F1 | macro-F1 (8 classes) |
|---|---:|---:|
| TAAD | **41.0** | 24.45 |
| TAAD + DST | **71.86** | 49.26 |

Note: the reference computes **micro**-F1 and per-class precision/recall. It computes
no macro-F1 — that is our addition, and the macro column above was computed by the
reviewer from the reference's per-class outputs. **Micro-F1 is the exact-reproduction
target; macro-F1 is a consistency check on our own aggregation.**

- [ ] **Step 1:** Add `score_events(..., average="micro"|"macro")` returning both.
- [ ] **Step 2:** Write `test_pcbas_eval_reproduces_reference.py`, reading the two
      shipped prediction files and the GT file from the FOOTPASS clone.
      Mark `@pytest.mark.skipif` when the clone is absent, so CI stays green.
- [ ] **Step 3:** Assert micro-F1 **41.0** and **71.86** to 2 d.p. — **THE GATE.**
      A mismatch means our matching rule differs from the reference's, and every
      subsequent number would be measuring our metric bug rather than a model.
- [ ] **Step 4:** Commit.

---

### Task 4: Phase 0 gate — reproduce the published GT counts

A falsifiable check that the ingest is correct, using the only numbers the dataset
publishes about itself.

**Files:**
- Modify: `packages/matchlab_train/src/matchlab_train/cli.py` — add `footpass-stats`
- Create: `docs/reports/2026-07-27-pcbas-phase0-ingest.md`

- [ ] **Step 1: Add the CLI subcommand**

`matchlab-train footpass-stats --h5 <path> --out stats.json`, reporting per split:
number of halves, total rows, total events, per-class event counts, fraction of rows
and of events with a bounding box, and unique slots seen.

- [ ] **Step 2: Run it on VAL and TRAIN**

```bash
D=/home/jeremy/code/MatchDay/lab/data/footpass/tactical
uv run matchlab-train footpass-stats --h5 $D/val_tactical_data.h5   --out /tmp/val.json
uv run matchlab-train footpass-stats --h5 $D/train_tactical_data.h5 --out /tmp/train.json
```

- [ ] **Step 3: Check against published figures — THE GATE**

| | expected | source |
|---|---|---|
| VAL halves | 6 (3 matches) | README |
| VAL events | **6,070** | README |
| TRAIN events | **91,327** | README |
| VAL events with a bbox | ~82.5% | measured 2026-07-27 |
| slots seen | ⊆ 0..25 | schema |

**A mismatch on the event totals is a stop.** It means the ingest is
mis-reading the label column, and every downstream number would be wrong.

- [ ] **Step 4: Write the report and commit**

Record the measured table, the code revision, and any discrepancy. Then:

```bash
git add packages/matchlab_train/src/matchlab_train/cli.py docs/reports/2026-07-27-pcbas-phase0-ingest.md
git commit -m "feat(pcbas): footpass-stats CLI + Phase 0 ingest gate report"
```

---

# PHASE 1 — Reproduce the baseline (requires video)

**Blocked on:** `videos_352x640_{VAL,TRAIN}.zip` downloading to `data/footpass/`.
Extraction needs `SOCCERNET_NDA_PASSWORD` from `.env` — the zips are password
protected.

---

### Task 5: Video extraction and frame-alignment check — **ALREADY VERIFIED 2026-07-27**

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/datasets/footpass_video.py`
- Create: `docs/reports/2026-07-27-pcbas-video-alignment.md`

This was the single most important verification in the phase: if the tactical `frame`
column did not index the video as assumed, every clip would train on the wrong pixels
and nothing downstream would reveal it. **It was performed during planning and it
passes.** The findings below are facts, not steps to attempt — three of them
contradict what a reasonable reading of the reference would have assumed.

**Finding 1 — the zips are AES-encrypted; `unzip` cannot read them.**

```
skipping: game_24.mp4   unsupported compression method 99
```

Use `7z` (present at `/usr/bin/7z`):

```bash
set -a; . /home/jeremy/code/MatchDay/lab/.env; set +a
cd /home/jeremy/code/MatchDay/lab/data/footpass
7z x -y -p"$SOCCERNET_NDA_PASSWORD" -ovideos_352x640/ videos_352x640_VAL.zip
```

**Finding 2 — one mp4 per MATCH, not per half, and frame indices are continuous
across the halves.** VAL contains exactly `game_18.mp4`, `game_24.mp4`, `game_47.mp4`.
Measured tactical frame ranges:

| key | frame range |
|---|---|
| `game_18_H1` | 32 – 75,307 |
| `game_18_H2` | 75,525 – 149,181 |
| `game_24_H1` | 5,397 – 77,221 |
| `game_24_H2` | 79,189 – 157,023 |

H2 continues from H1 with **no reset**. So the `frame` column indexes directly into
the single per-match mp4, and `game_{N}.mp4` is the correct path for both halves —
matching `TAAD_Dataset.py:188` (`f'game_{gm_idx}.mp4'`). Any code that assumes a
per-half video, or that offsets H2 frames, is wrong.

**Finding 3 — video properties confirmed.** `game_18.mp4`: **640×352, 25.0 fps,
149,651 frames**, which covers the max tactical frame (149,181). Resolution matches
the model's `(B,3,T,352,640)` input with no resize needed.

**Finding 4 — ROI scaling is correct as specified.** 12 randomly sampled *action* rows
with bounding boxes were decoded, scaled by `x/3.0` and `y/3.068181`, and cropped.
Every crop contains a centred player. Montage retained at
`scratchpad/align_check.png`; reproduce with the snippet in the report.

- [ ] **Step 1: Write `footpass_video.py` encoding these four findings**

`match_video_path(root, half_key) -> Path` (strips the `_H1`/`_H2` suffix),
`open_match(path)` asserting 640×352 @ 25 fps, and `read_clip(cap, start, length)`.

- [ ] **Step 2: Write the alignment report** with the four findings and the montage.

- [ ] **Step 3: Extract TRAIN when its download completes**, then re-run the montage
  check on **one TRAIN match** — the finding above is verified on VAL only, and a
  per-split difference in encoding would be exactly the kind of thing that passes
  silently.

- [ ] **Step 4: Commit**

---

### Task 6: Action head model

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/pcbas/action_head.py`
- Test: `packages/matchlab_core/tests/test_pcbas_action_head.py`

**Interfaces:**
- Produces: `ActionHead(nn.Module)` with
  `forward(video: Tensor[B,3,T,352,640], rois: Tensor[B,M,T,5], masks: Tensor[B,M,T]) -> Tensor[B,9,M,T]`

- [ ] **Step 1: Write the failing shape tests**

```python
def test_output_shape_is_b_classes_m_t():
    m = ActionHead()
    out = m(torch.zeros(1, 3, 8, 352, 640), torch.zeros(1, 4, 8, 5), torch.ones(1, 4, 8))
    assert out.shape == (1, 9, 4, 8)


def test_masked_players_produce_zero_features():
    """A player with mask 0 must not leak features from the shared map."""


def test_accepts_full_26_slots_at_inference():
    out = ActionHead()(torch.zeros(1, 3, 4, 352, 640), torch.zeros(1, 26, 4, 5), torch.ones(1, 26, 4))
    assert out.shape == (1, 9, 26, 4)
```

Mark these `@pytest.mark.slow` — they download X3D-S weights on first run.

- [ ] **Step 2–4: Implement, per spec §1.1**

X3D-S from `torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)`;
blocks 0–4 with two upsample+concat merges to (B,192,T,44,80); `roi_align` at
`output_size=(4,2)`, `spatial_scale=0.125`; multiply by mask; `Conv1d(192→512)` +
BatchNorm + GELU; `Linear(512, 9)`.

The ROI batch-index arithmetic is the subtle part: ROIs arrive as (B,M,T,5) with a
per-clip frame number in column 0, and must be re-indexed to the flattened (B*T)
feature batch as `frame + batch_idx * T` before `roi_align`.

- [ ] **Step 5: Commit**

---

### Task 7: Clip dataset and training loop

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/datasets/footpass_clips.py`
- Create: `packages/matchlab_train/src/matchlab_train/experiments/pcbas_action_head.py`
- Test: `packages/matchlab_train/tests/test_footpass_clips.py`

Reference recipe, to be matched unless a deviation is recorded with a reason:
clip_length 50, nb_tracklets 4, max_nb_samples_per_class 500 (**resampled every
epoch**), batch 6, grad-accum 8, 20 epochs, AdamW with lr 5e-5 backbone / 1e-3 head,
50-step warmup, AMP fp16.

Tests cover the sampler, not the training: every sampled clip contains its anchor
event; class balance holds; ROIs are in-bounds after scaling; masks are 0 exactly
where the bbox was NaN.

- [ ] Steps: failing tests → sampler → training experiment → smoke run (2 epochs, 1 half) → commit

---

### Task 8: Inference to the `(9, 26, T)` contract

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/experiments/pcbas_infer_logits.py`
- Test: `packages/matchlab_train/tests/test_pcbas_logits_contract.py`

Sliding window of 50 frames at stride 25, averaging the overlap, writing
`(9, 26, T)` fp16 `.npy` per half. Tests assert dtype, shape, and that slots with no
observed player anywhere are all-background.

- [ ] Steps: failing tests → implementation → run on VAL → commit

---

### Task 9: Denoiser model and dataset

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/pcbas/denoiser.py`
- Create: `packages/matchlab_train/src/matchlab_train/datasets/footpass_windows.py`
- Test: `packages/matchlab_core/tests/test_pcbas_denoiser.py`

Encoder-decoder Transformer, framespan 750, d_model 512, 6+6 layers, 8 heads, dropout
0.1. Encoder input per frame is 364 = 26 slots × (5 kinematic + 9 logits). Decoder
tokens are `action(10) ‖ role(27) ‖ timestamp(752)`. Pitch-symmetry augmentation (X, Y,
XY) with role-slot remap tables.

Behind one interface, two encoders: `flat` (reference) and `attn` (PAVE-style
per-player attention over slots) — the Notion page identifies the flat encoding as
precisely what the 2026 winner replaced, so both must be measurable.

- [ ] Steps: failing shape/masking tests → model → dataset → autoregressive-decode test → commit

---

### Task 10: Phase 1 gate — the reproduction

**Files:**
- Create: `docs/reports/2026-07-27-pcbas-phase1-reproduction.md`

- [ ] **Step 1: Train the action head on TRAIN, infer logits on VAL**
- [ ] **Step 2: Train the denoiser on TRAIN logits, decode VAL**
- [ ] **Step 3: Score with `score_events(delta=12, conf_thresh=0.15)`**
- [ ] **Step 4: Compare — THE GATE**

| arm | reference on VAL | our bar |
|---|---:|---|
| TAAD alone | micro 41.0 / macro 24.45 | report, no bar |
| TAAD + DST | micro **71.86** / macro **49.26** | **micro-F1 ≥ 65 AND macro-F1 ≥ 42** |

**Do not use 46.41.** That figure appears nowhere in the reference repo, is not
reproducible from its shipped artifacts, and is a **challenge-split leaderboard**
number while we score on VAL — so a comparison against it can pass or fail for split
reasons alone. The bars above come from the reference's own VAL predictions, which is
the only comparand that isolates our implementation from split effects.

Report **per-class F1 with GT counts**, never a bare macro number. VAL has 26 tackles
(TRAIN ~390) and 67 shots (TRAIN ~1,000); only `tackle` is genuinely near-unlearnable.

**Missing the band is a stop-and-diagnose, not a proceed.** Diagnose in this order:
1. Task 3b — is the metric itself right? (It should already be pinned.)
2. Task 5 — is the ROI geometry right? (Verified on VAL; re-check on TRAIN.)
3. Class balance — was `max_nb_samples_per_class` resampling actually applied per epoch?
4. Only then suspect the model.

**Control run (per spec Risks):** because D2 reimplements rather than vendors, a miss
is otherwise unattributable between our reimplementation, our ingest and the setup.
Run the reference code **once, unmodified**, on one match to bound that ambiguity
before concluding anything about the architecture.

- [ ] **Step 5: Update `docs/implementation-status.md` and commit**

---

## Phases 2–4

Specified in the design doc §3, deliberately **not** decomposed here. Each depends on
Phase 1's measured outcome, and writing bite-sized steps now would be inventing detail
rather than recording it. Each gets its own plan when its predecessor's gate passes:

- **Phase 2** — **export-time** roster remap (ADR 008, *not* ADR 007's withdrawn
  bijection) + PAVE attention encoder
- **Phase 3** — side/half determination (gate: 100%), role assignment (gate: ≥70%,
  kill <50%), **slot stability** (gate: ≤1 switch/entity/window, purity ≥0.8), then
  end-to-end decay. See spec §1.2a2, §1.2d, §2.3.
- **Phase 4** — pipeline stage integration, artifacts, Lab visualisation

Two schema gaps must be resolved before Phase 4 and are not yet designed:
`GroundTruthEvent` has no player field, and `SpottedEvent` has no identity field —
the latter being the frozen external contract shared with the T-DEED CLI.

---

## Appendix A — Integration surface (verified against the codebase 2026-07-27)

Recorded now so Phase 4 does not have to rediscover it, and so Phase 0–1 module
boundaries are drawn to fit.

### A.1 `spot()` receives only `ctx`

```python
class EventSpotter(Stage):                      # interfaces.py L191-197
    @abstractmethod
    def spot(self, ctx: StageContext) -> list[Event]: ...
```

Unlike `PossessionEstimator.estimate(ctx, tracklets, teams, ball)`, a spotter gets no
upstream arguments. `ball-trajectory` reads what it needs back off `ctx.store`, and so
must ours. **Consequence for module design:** all modelling functions must take plain
data arguments and be callable without a `StageContext`, so the stage file stays a
thin adapter that loads artifacts and delegates. This is the `possession_denoise.py` /
`stages/possession/viterbi.py` split, and it is why Tasks 6 and 9 build models that
know nothing about the store.

### A.2 The dual-write contract

The runner (L186-192) does:

```python
spotted = self._exec(StageKind.SPOTTING, lambda s: s.spot(ctx)) or []
if not store.exists(ArtifactName.SPOTTING) and poss_spotted:
    store.write_json(ArtifactName.SPOTTING, poss_spotted)
```

So a spotting impl must **write `spotting.json` itself** (which suppresses the
possession fallback) **and return `list[Event]`** (merged into `events.json`,
rendered by annotate, counted in the timeline). Phase 4 follows `ball_trajectory.py`
exactly here.

### A.3 New stage slot for role assignment (spec D5)

Four edits, precedented by SPO-77's addition of `POSSESSION`:
1. `schemas/run.py::StageKind` — add `ROLE = "role"`
2. `runner.py::STAGE_ORDER` — insert after `FUSE`, before `POSSESSION`
3. `runner.py::_run_stages` — a new `_exec` call (note: the runner does **not**
   iterate `STAGE_ORDER`; the sequence is hand-written at L110-222)
4. `interfaces.py` — a `RoleAssigner(Stage)` ABC

Plus a `ROLE_ASSIGNMENT` artifact: `ArtifactName` (`schemas/run.py` L62-88),
`ARTIFACT_FILES` (`artifacts.py` L13-38), `web/src/lib/types.ts`, and **three
coordinated edits** in `web/src/lib/artifacts.ts` — the `RunArtifacts` field, the
`artifactQueries` entry, and the *positional* destructuring at L107-124 plus the
return object at L155-172. That destructure is positional over `results.map(...)`;
order must match `artifactQueries` exactly or artifacts silently swap.

### A.4 Conventions the new modules must follow

- **`self.params = Params(**params)`** on every stage — `runner._resolved_params`
  reads `getattr(stage, "params", None)` and dumps it only if it is a pydantic model.
  Skip it and provenance records `{}`.
- **`Event.attrs` values must be scalars or `None`** (`schemas/events.py` L39-51) — no
  nested dicts or lists. Attribution payloads (slot, role, shirt) must be flattened.
- **`SpottedEvent`** uses `class_: str = Field(alias="class")` with
  `serialize_by_alias=True`; construct with `class_=`, JSON always emits `"class"`.
  Emit the **native taxonomy verbatim** — never map to `EventType` inside a spotter.
- Torch goes in the **`cv` extra** of `packages/matchlab_core/pyproject.toml`
  (already contains `torch>=2.3`), with the heavy import lazy inside `prepare()`/`spot()`.
- New schema modules must be re-exported from `schemas/__init__.py` (imports + `__all__`).
- `matchlab_train.registry.register` has **no duplicate check** — a name collision
  silently overwrites. Grep before naming an experiment.
- Add training experiments to `matchlab_train/experiments/__init__.py` or they never
  register; same for stage modules in `matchlab_core/stages/__init__.py`.

### A.5 The metric must not be confused with the existing one

`action_spotting_eval.average_map` is class+time avg-mAP for SoccerNet-ball, with a
documented zero-GT-class rule and greedy descending-confidence matching. Our
`pcbas.eval.score_events` additionally requires the **identity slot** to match and
reports macro-F1, not mAP. Both stay. `class_ap(result, "PASS")` remains the mandated
honest single-class number for the possession track. Quoting one for the other's task
is a governance error, not a rounding difference.

### A.6 Config naming

Two configs per capability, per the established convention:
`configs/pipeline.<name>-smoke.yaml` (keyless, GPU-less, synthetic upstream, runnable
unattended) and `configs/pipeline.<name>-eval.yaml` (real weights, human-gated,
leading with a "HONEST MEASUREMENT NOTES" comment header stating exactly which number
is claimable). Only `configs/pipeline.*.yaml` is discovered by `list_configs`.
