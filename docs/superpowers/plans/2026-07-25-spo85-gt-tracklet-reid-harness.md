# SPO-85: GT-tracklet re-ID harness + five-way embedder comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a contamination-free re-ID measurement substrate — GT tracks fragmented at their natural gaps — and compare five embedders on it by gate-restricted rank-1 retrieval.

**Architecture:** A new `oracle` TRACK stage (the GT twin of `tdlp-full`) fragments the video's ground truth and emits `tracklets.json` + `frame_features.npz`. Features come from either an in-repo `BodyEmbedder` (osnet/solider/dinov2, P=1) or the existing external CAMELTrack bridge (kpr/prtreid, P=6). A `matchlab_train` experiment then scores each arm by retrieval metrics over gate-passing candidate pools, with no engine run.

**Tech Stack:** Python 3.12, pydantic v2, numpy, OpenCV, pytest. External bridge = CAMELTrack venv (tracklab + KPR), reached by subprocess exactly as `tdlp-full` does today.

## Global Constraints

- Spec of record: `docs/superpowers/specs/2026-07-25-prtreid-gt-tracklet-harness-design.md`. Pre-registration: Linear SPO-85.
- Line length 100; `uv run ruff check packages` must pass. Config in root `pyproject.toml`.
- Full suite `uv run pytest packages -q` must stay green (baseline: 750 passed, 5 skipped).
- Tests assert external behaviour against hand-computed expectations on small synthetic inputs. Never assert internal call structure.
- Stages must not touch the filesystem outside `ctx.store` (`StageContext` invariant).
- Artifacts index by **source video `frame_idx`**, stride-independent.
- Registered fragmentation values, frozen: `gap_frames: 2`, `min_fragment_frames: 1`. Do not change these after any arm runs.
- Development and selection use tuning sequences **SNMOT-116–123 only**. Held-out SNMOT-124–127 must not be read by any step in this plan.
- A missing GT is a loud error, never silent empty output (convention from `stages/detect/oracle.py`).
- Never silently substitute a different model for a named arm — a mislabelled benchmark row is worse than a crash.

---

### Task 0: Quarantine held-out artifacts

**Files:**
- Move: `data/runs/recon-SNMOT-124`, `data/runs/recon-SNMOT-125`, `data/runs/recon-SNMOT-126` → `data/heldout-quarantine/`

- [ ] **Step 1: Move the surviving held-out run dirs out of reach**

```bash
cd /home/jeremy/code/MatchDay/lab
mkdir -p data/heldout-quarantine
mv data/runs/recon-SNMOT-124 data/runs/recon-SNMOT-125 data/runs/recon-SNMOT-126 \
   data/heldout-quarantine/
ls data/runs/
```

Expected: only the `spo73-view-*` dirs remain. These are inspection views of held-out sequences too — leave them, but no step in this plan may read them.

- [ ] **Step 2: Confirm nothing in this plan references them**

```bash
grep -rn "recon-SNMOT" configs/ packages/ || echo "no references — good"
```

Expected: `no references — good`.

No commit (data/ is gitignored).

---

### Task 1: `gt_fragments` — pure fragmentation

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/gt_fragments.py`
- Test: `packages/matchlab_core/tests/test_gt_fragments.py`

**Interfaces:**
- Consumes: `matchlab_core.gt.GroundTruth`, `GroundTruthTrack`, `GroundTruthFrame`; `matchlab_core.schemas.Tracklet`, `TrackletFrame`; `matchlab_core.schemas.detections.DetectionClass`
- Produces:
  - `fragment_tracks(gt: GroundTruth, *, gap_frames: int = 2, min_fragment_frames: int = 1, include_roles: frozenset[str] = DEFAULT_ROLES) -> FragmentResult`
  - `@dataclass FragmentResult: tracklets: list[Tracklet]; gt_track_by_fragment: dict[int, int]; jersey_by_fragment: dict[int, str | None]; team_by_fragment: dict[int, str | None]`
  - `DEFAULT_ROLES: frozenset[str] = frozenset({"player", "goalkeeper", "referee"})`

- [ ] **Step 1: Write the failing tests**

Create `packages/matchlab_core/tests/test_gt_fragments.py`:

```python
from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.gt_fragments import fragment_tracks
from matchlab_core.schemas.detections import DetectionClass


def _track(track_id: int, frame_idxs: list[int], **kw) -> GroundTruthTrack:
    return GroundTruthTrack(
        track_id=track_id,
        frames=[
            GroundTruthFrame(frame_idx=i, box={"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 20.0})
            for i in frame_idxs
        ],
        **kw,
    )


def _gt(tracks: list[GroundTruthTrack]) -> GroundTruth:
    return GroundTruth(fps=25.0, width=100, height=100, seq_length=100, tracks=tracks)


def test_contiguous_track_yields_one_fragment():
    res = fragment_tracks(_gt([_track(7, [0, 1, 2, 3])]))
    assert len(res.tracklets) == 1
    assert [f.frame_idx for f in res.tracklets[0].frames] == [0, 1, 2, 3]
    assert res.gt_track_by_fragment == {res.tracklets[0].tracklet_id: 7}


def test_gap_larger_than_threshold_splits():
    # frames 0,1 then 10,11 — a 9-frame gap, well over gap_frames=2
    res = fragment_tracks(_gt([_track(7, [0, 1, 10, 11])]), gap_frames=2)
    assert len(res.tracklets) == 2
    assert [f.frame_idx for f in res.tracklets[0].frames] == [0, 1]
    assert [f.frame_idx for f in res.tracklets[1].frames] == [10, 11]
    assert set(res.gt_track_by_fragment.values()) == {7}


def test_gap_exactly_at_threshold_does_not_split():
    # 0 -> 2 is a step of 2; gap_frames=2 means "split when step > 2"
    res = fragment_tracks(_gt([_track(7, [0, 2, 4])]), gap_frames=2)
    assert len(res.tracklets) == 1


def test_gap_one_over_threshold_splits():
    res = fragment_tracks(_gt([_track(7, [0, 3])]), gap_frames=2)
    assert len(res.tracklets) == 2


def test_min_fragment_frames_drops_slivers():
    res = fragment_tracks(_gt([_track(7, [0, 1, 2, 20])]), gap_frames=2, min_fragment_frames=2)
    assert len(res.tracklets) == 1
    assert [f.frame_idx for f in res.tracklets[0].frames] == [0, 1, 2]


def test_roles_map_to_detection_classes_and_other_is_excluded():
    gt = _gt([
        _track(1, [0, 1], role="player"),
        _track(2, [0, 1], role="goalkeeper"),
        _track(3, [0, 1], role="referee"),
        _track(4, [0, 1], role="other"),
        _track(5, [0, 1], role="ball"),
    ])
    res = fragment_tracks(gt)
    classes = sorted(t.cls.value for t in res.tracklets)
    assert classes == sorted([
        DetectionClass.PLAYER.value,
        DetectionClass.GOALKEEPER.value,
        DetectionClass.REFEREE.value,
    ])


def test_fragment_ids_are_unique_and_stable_across_calls():
    gt = _gt([_track(7, [0, 1, 10, 11]), _track(8, [0, 1])])
    a = fragment_tracks(gt)
    b = fragment_tracks(gt)
    ids = [t.tracklet_id for t in a.tracklets]
    assert len(set(ids)) == len(ids)
    assert ids == [t.tracklet_id for t in b.tracklets]


def test_jersey_and_team_are_carried_per_fragment():
    gt = _gt([_track(7, [0, 1, 10, 11], jersey="9", team="left")])
    res = fragment_tracks(gt)
    assert set(res.jersey_by_fragment.values()) == {"9"}
    assert set(res.team_by_fragment.values()) == {"left"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/matchlab_core/tests/test_gt_fragments.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'matchlab_core.gt_fragments'`

- [ ] **Step 3: Implement the module**

Create `packages/matchlab_core/src/matchlab_core/gt_fragments.py`:

```python
"""Fragment ground-truth tracks into a re-ID merge task.

A GT track is a complete per-player trajectory, so it poses no merging
question. Splitting it at its *natural* gaps — the frames where the player
left view or was occluded — produces fragments that are pure by construction
and whose correct grouping is known exactly, with the real crop degradation
at each re-entry preserved. That is the substrate the merge layer should be
measured on: tracker-produced tracklets carry their own impurity, so a wrong
merge cannot be told apart from an inherited swap.

Pure: GroundTruth in, Tracklets + exact fragment→GT-track map out. No video,
no model, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from matchlab_core.gt import GroundTruth
from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.tracks import TrackletFrame

# GT roles that become fragments. "other" (staff, photographers) and "ball"
# are not players and never enter a player-association task.
DEFAULT_ROLES: frozenset[str] = frozenset({"player", "goalkeeper", "referee"})

_ROLE_TO_CLASS: dict[str, DetectionClass] = {
    "player": DetectionClass.PLAYER,
    "goalkeeper": DetectionClass.GOALKEEPER,
    "referee": DetectionClass.REFEREE,
}


@dataclass
class FragmentResult:
    """Fragments plus the exact provenance the analyzer needs.

    `gt_track_by_fragment` is what makes this substrate better than a
    tracker's: correct pairs are known, not inferred by GT-argmax attribution.
    """

    tracklets: list[Tracklet] = field(default_factory=list)
    gt_track_by_fragment: dict[int, int] = field(default_factory=dict)
    jersey_by_fragment: dict[int, str | None] = field(default_factory=dict)
    team_by_fragment: dict[int, str | None] = field(default_factory=dict)


def fragment_tracks(
    gt: GroundTruth,
    *,
    gap_frames: int = 2,
    min_fragment_frames: int = 1,
    include_roles: frozenset[str] = DEFAULT_ROLES,
) -> FragmentResult:
    """Split each GT track wherever consecutive annotated frames step by more
    than `gap_frames`. Fragment ids are assigned in (track_id, start_frame)
    order so a given GroundTruth always fragments identically."""
    result = FragmentResult()
    next_id = 1
    for track in sorted(gt.tracks, key=lambda t: t.track_id):
        if track.role not in include_roles:
            continue
        cls = _ROLE_TO_CLASS[track.role]
        frames = sorted(track.frames, key=lambda f: f.frame_idx)
        if not frames:
            continue
        runs: list[list] = [[frames[0]]]
        for prev, cur in zip(frames, frames[1:], strict=False):
            if cur.frame_idx - prev.frame_idx > gap_frames:
                runs.append([cur])
            else:
                runs[-1].append(cur)
        for run in runs:
            if len(run) < min_fragment_frames:
                continue
            result.tracklets.append(
                Tracklet(
                    tracklet_id=next_id,
                    cls=cls,
                    frames=[
                        TrackletFrame(frame_idx=f.frame_idx, box=f.box, confidence=1.0)
                        for f in run
                    ],
                )
            )
            result.gt_track_by_fragment[next_id] = track.track_id
            result.jersey_by_fragment[next_id] = track.jersey
            result.team_by_fragment[next_id] = track.team
            next_id += 1
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/matchlab_core/tests/test_gt_fragments.py -q`
Expected: `9 passed`

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/gt_fragments.py \
        packages/matchlab_core/tests/test_gt_fragments.py
git commit -m "feat(reid): fragment GT tracks at natural gaps (SPO-85)"
```

---

### Task 2: `oracle` TRACK stage with the in-repo feature backend

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/stages/track/oracle.py`
- Modify: `packages/matchlab_core/src/matchlab_core/stages/track/__init__.py` (register the module)
- Test: `packages/matchlab_core/tests/test_track_oracle.py`

**Interfaces:**
- Consumes: `fragment_tracks`, `FragmentResult` (Task 1); `matchlab_core.stages.associate.embedders.get_embedder`; `StageContext`, `Tracker`, `register`, `ArtifactName`, `FrameFeatures`
- Produces: stage `oracle` in `StageKind.TRACK`; `Params` with `gt_path`, `gap_frames`, `min_fragment_frames`, `features_backend`, `features_model`, `max_frames_per_fragment`

- [ ] **Step 1: Write the failing tests**

Create `packages/matchlab_core/tests/test_track_oracle.py`:

```python
import json

import numpy as np
import pytest

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.registry import get_stage
from matchlab_core.schemas.run import ArtifactName, StageKind


def test_stage_is_registered():
    cls = get_stage(StageKind.TRACK, "oracle")
    assert cls is not None


def test_missing_gt_is_a_loud_error(tmp_path, synthetic_ctx):
    ctx = synthetic_ctx(gt=None)
    stage = get_stage(StageKind.TRACK, "oracle")(features_backend="none")
    with pytest.raises(RuntimeError, match="ground truth"):
        stage.track(ctx, [])


def test_emits_fragments_and_feature_rows(synthetic_ctx, toy_gt):
    ctx = synthetic_ctx(gt=toy_gt)
    stage = get_stage(StageKind.TRACK, "oracle")(features_backend="none")
    tracklets = stage.track(ctx, [])
    assert len(tracklets) == 2  # toy_gt: one track with one gap
    written = json.loads(
        (ctx.store.run_dir / "tracklets.json").read_text()
    ) if (ctx.store.run_dir / "tracklets.json").exists() else None
    assert written is None  # the runner writes tracklets.json, not the stage


def test_none_backend_writes_zero_dim_features(synthetic_ctx, toy_gt):
    ctx = synthetic_ctx(gt=toy_gt)
    stage = get_stage(StageKind.TRACK, "oracle")(features_backend="none")
    stage.track(ctx, [])
    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert len(feats) == 0
    assert feats.meta["backend"] == "none"


def test_in_repo_backend_writes_one_row_per_fragment_frame(synthetic_ctx, toy_gt, fake_embedder):
    ctx = synthetic_ctx(gt=toy_gt)
    stage = get_stage(StageKind.TRACK, "oracle")(
        features_backend="in-repo", features_model="fake"
    )
    tracklets = stage.track(ctx, [])
    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    expected_rows = sum(len(t.frames) for t in tracklets)
    assert len(feats) == expected_rows
    assert feats.embeddings.shape[1] == 1  # P=1 for single-vector embedders
    assert np.allclose(feats.visibility, 1.0)
    assert feats.meta["model"] == "fake"


def test_max_frames_per_fragment_subsamples(synthetic_ctx, toy_gt, fake_embedder):
    ctx = synthetic_ctx(gt=toy_gt)
    stage = get_stage(StageKind.TRACK, "oracle")(
        features_backend="in-repo", features_model="fake", max_frames_per_fragment=1
    )
    tracklets = stage.track(ctx, [])
    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert len(feats) == len(tracklets)
```

Fixtures go in the same file (the repo has no shared conftest for stages):

```python
@pytest.fixture
def toy_gt():
    from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack

    frames = [
        GroundTruthFrame(frame_idx=i, box={"x1": 1.0, "y1": 1.0, "x2": 9.0, "y2": 19.0})
        for i in [0, 1, 2, 8, 9]
    ]
    return GroundTruth(
        fps=25.0, width=32, height=32, seq_length=10,
        tracks=[GroundTruthTrack(track_id=1, role="player", jersey="9", team="left",
                                 frames=frames)],
    )


@pytest.fixture
def fake_embedder():
    """Register a deterministic embedder so the backend can be tested without torch."""
    import numpy as np

    from matchlab_core.stages.associate.embedders.base import EMBEDDERS, BodyEmbedder

    class FakeEmbedder(BodyEmbedder):
        name = "fake"
        dim = 4

        def prepare(self, device: str) -> None:
            return None

        def embed(self, crops):
            if not crops:
                return np.zeros((0, self.dim), np.float32), None
            v = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (len(crops), 1))
            return v, None

    EMBEDDERS["fake"] = FakeEmbedder
    yield
    EMBEDDERS.pop("fake", None)
```

`synthetic_ctx` builds a `StageContext` over a small generated video. Follow the existing pattern in `packages/matchlab_core/tests/` for building a `StageContext` (search for an existing test that constructs one and copy its helper verbatim rather than inventing a new one).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/matchlab_core/tests/test_track_oracle.py -q`
Expected: FAIL — stage `oracle` not registered for `StageKind.TRACK`

- [ ] **Step 3: Implement the stage**

Create `packages/matchlab_core/src/matchlab_core/stages/track/oracle.py`:

```python
"""Oracle tracker: emits the video's ground-truth tracks, fragmented at their
natural gaps, as this run's TRACK output — the GT twin of `tdlp-full` and the
missing member of the oracle family (`detect` and `team` already have one).

Why it exists: every association experiment in this repo has measured the
merge layer on tracker-produced tracklets, which carry their own residual
impurity, so "the engine merged wrongly" could never be separated from "the
tracker handed it a contaminated tracklet". Fragmenting GT gives tracklet
purity 1.0 by construction and exactly known correct pairs, with no tracker
run at all.

Features: the re-ID engine's input layer is `frame_features.npz`, so this
stage generates it too. `in-repo` embeds GT crops with a registered
BodyEmbedder (single-vector models; P=1). `external` is the CAMELTrack bridge
path added in SPO-85 Task 3 (part-based KPR/PRTreID; P=6). `none` writes an
empty artifact for tests and tracklet-only runs.

This stage is a measurement substrate, never a production tracker: it consumes
ground truth, so it is meaningless without it and says so loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.gt import GroundTruth
from matchlab_core.gt_fragments import fragment_tracks
from matchlab_core.interfaces import StageContext, Tracker
from matchlab_core.provenance import LicenseAxes, ModelProvenance
from matchlab_core.registry import register
from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.schemas.run import ArtifactName, StageKind


class Params(BaseModel):
    gt_path: str | None = None
    gap_frames: int = 2
    min_fragment_frames: int = 1
    # "none" | "in-repo" | "external"
    features_backend: str = "none"
    # in-repo: an EMBEDDERS name (osnet/solider/dinov2). external: kpr/prtreid.
    features_model: str = "osnet"
    # Cap crops embedded per fragment (None = every frame).
    max_frames_per_fragment: int | None = None


def _load_gt(gt_path: str | None, video_path: str) -> GroundTruth:
    candidates = [Path(gt_path)] if gt_path else [Path(video_path).with_suffix(".gt.json")]
    for path in candidates:
        if path.exists():
            return GroundTruth.model_validate_json(path.read_text())
    raise RuntimeError(
        "Oracle tracker: no ground truth found (tried "
        f"{[str(c) for c in candidates]}). An oracle run without GT is meaningless."
    )


@register(StageKind.TRACK, "oracle")
class OracleTracker(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)

    def provenance(self) -> list[ModelProvenance]:
        return [
            ModelProvenance(
                architecture="oracle-tracklets (GT fragmented at natural gaps)",
                revision=f"gap_frames={self.params.gap_frames}",
                lineage="ground truth; not a model",
                license=LicenseAxes(code="n/a", weights="n/a", training_data="n/a"),
            )
        ]

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        p = self.params
        gt = _load_gt(p.gt_path, ctx.video.path)
        res = fragment_tracks(
            gt, gap_frames=p.gap_frames, min_fragment_frames=p.min_fragment_frames
        )
        ctx.progress(StageKind.TRACK, 0.5, f"oracle: {len(res.tracklets)} fragments")

        if p.features_backend == "in-repo":
            feats = self._embed_in_repo(ctx, res.tracklets)
        elif p.features_backend == "external":
            from matchlab_core.stages.track.oracle_external import embed_external

            feats = embed_external(ctx, res.tracklets, model=p.features_model)
        else:
            feats = _empty_features(backend="none")
        feats.meta.update(
            {
                "gap_frames": p.gap_frames,
                "min_fragment_frames": p.min_fragment_frames,
                "gt_track_by_fragment": res.gt_track_by_fragment,
                "jersey_by_fragment": res.jersey_by_fragment,
                "team_by_fragment": res.team_by_fragment,
            }
        )
        feats.save(ctx.store.path(ArtifactName.FRAME_FEATURES))
        ctx.progress(StageKind.TRACK, 1.0, f"oracle: {len(feats)} feature rows")
        return res.tracklets

    def _embed_in_repo(self, ctx: StageContext, tracklets: list[Tracklet]) -> FrameFeatures:
        from matchlab_core.stages.associate.embedders import get_embedder

        p = self.params
        embedder = get_embedder(p.features_model)
        embedder.prepare(ctx.device)

        wanted: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
        for t in tracklets:
            frames = t.frames
            if p.max_frames_per_fragment is not None:
                step = max(1, len(frames) // p.max_frames_per_fragment)
                frames = frames[::step][: p.max_frames_per_fragment]
            for f in frames:
                wanted.setdefault(f.frame_idx, []).append(
                    (t.tracklet_id, (f.box.x1, f.box.y1, f.box.x2, f.box.y2))
                )

        tids: list[int] = []
        fidxs: list[int] = []
        vecs: list[np.ndarray] = []
        for frame in ctx.frames():
            picks = wanted.get(frame.frame_idx)
            if not picks:
                continue
            h, w = frame.image.shape[:2]
            crops = []
            keep = []
            for tid, (x1, y1, x2, y2) in picks:
                xa, ya = max(0, int(x1)), max(0, int(y1))
                xb, yb = min(w, int(x2)), min(h, int(y2))
                if xb - xa < 2 or yb - ya < 2:
                    continue
                crops.append(frame.image[ya:yb, xa:xb])
                keep.append(tid)
            if not crops:
                continue
            emb, _ = embedder.embed(crops)
            for tid, v in zip(keep, emb, strict=True):
                tids.append(tid)
                fidxs.append(frame.frame_idx)
                vecs.append(v)

        if not vecs:
            return _empty_features(backend="in-repo", model=p.features_model)
        arr = np.stack(vecs).astype(np.float32)[:, None, :]  # (N, 1, D)
        n = arr.shape[0]
        return FrameFeatures(
            tracklet_ids=np.array(tids, dtype=np.int64),
            frame_idxs=np.array(fidxs, dtype=np.int64),
            embeddings=arr,
            visibility=np.ones((n, 1), dtype=np.float32),
            keypoints_xyc=np.zeros((n, 1, 3), dtype=np.float32),
            keypoints_conf=np.zeros((n,), dtype=np.float32),
            meta={"backend": "in-repo", "model": p.features_model, "stage": "oracle"},
        )


def _empty_features(*, backend: str, model: str | None = None) -> FrameFeatures:
    return FrameFeatures(
        tracklet_ids=np.zeros((0,), dtype=np.int64),
        frame_idxs=np.zeros((0,), dtype=np.int64),
        embeddings=np.zeros((0, 1, 1), dtype=np.float32),
        visibility=np.zeros((0, 1), dtype=np.float32),
        keypoints_xyc=np.zeros((0, 1, 3), dtype=np.float32),
        keypoints_conf=np.zeros((0,), dtype=np.float32),
        meta={"backend": backend, "model": model, "stage": "oracle"},
    )
```

- [ ] **Step 4: Register the module**

Modify `packages/matchlab_core/src/matchlab_core/stages/track/__init__.py` — add `oracle` to the import list alongside the existing track stages, following the exact style already in that file.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/matchlab_core/tests/test_track_oracle.py -q`
Expected: all pass

- [ ] **Step 6: Lint, full suite, commit**

```bash
uv run ruff check packages
uv run pytest packages -q
git add packages/matchlab_core/src/matchlab_core/stages/track/oracle.py \
        packages/matchlab_core/src/matchlab_core/stages/track/__init__.py \
        packages/matchlab_core/tests/test_track_oracle.py
git commit -m "feat(track): oracle tracker — GT fragments + in-repo feature backend (SPO-85)"
```

---

### Task 3: External feature backend (KPR / PRTreID)

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/stages/track/oracle_external.py`
- Modify: `../external-trackers/bridge/gen_features.py` (add `--reid-model`, `--no-pose`)
- Test: `packages/matchlab_core/tests/test_track_oracle_external.py`

**Interfaces:**
- Consumes: `matchlab_core.stages.track.tdlp_full.bridge` — `stage_sequence`, `run_external`, `join_features_to_tracklets` (already IoU-based, so GT boxes match their own det.txt rows exactly and need no new join logic)
- Produces: `embed_external(ctx: StageContext, tracklets: list[Tracklet], *, model: str) -> FrameFeatures`

**Do this first:** confirm PRTreID weight acquisition before writing code. Run the sn-gamestate baseline once so TrackLab downloads its models, or fetch from Zenodo. If acquisition fails, **stop and record the failure** — per the spec's fallback, the experiment reduces to KPR + in-repo controls and the PRTreID question stays open. Do not substitute another model under the `prtreid` label.

- [ ] **Step 1: Verify PRTreID acquisition**

```bash
ls ../external-trackers/CAMELTrack/pretrained_models/reid/
../external-trackers/CAMELTrack/.venv/bin/python -c \
  "from tracklab.wrappers.reid import prtreid_api; print('prtreid_api available')"
```

Expected: either `prtreid_api available`, or an ImportError that tells you the wrapper must be installed into that venv. Record which.

- [ ] **Step 2: Write the failing test**

Create `packages/matchlab_core/tests/test_track_oracle_external.py`:

```python
import pytest

from matchlab_core.stages.track.oracle_external import _resolve_weights


def test_unknown_model_is_a_loud_error():
    with pytest.raises(ValueError, match="Unknown external re-ID model"):
        _resolve_weights("not-a-model", external_root="/tmp")


def test_missing_weights_names_the_acquisition_step(tmp_path):
    with pytest.raises(RuntimeError, match="acquire"):
        _resolve_weights("prtreid", external_root=str(tmp_path))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_track_oracle_external.py -q`
Expected: FAIL — module does not exist

- [ ] **Step 4: Implement `oracle_external.py`**

```python
"""External feature backend for the oracle tracker: part-based re-ID models
(KPR, PRTreID) that live in the isolated CAMELTrack venv, reached through the
same subprocess bridge `tdlp-full` already uses.

`gen_features.py` takes `--img-dir` + `--det-file` and embeds whatever boxes
it is handed, so GT fragment boxes need no new external machinery — and
because `join_features_to_tracklets` joins by IoU, a fragment box matches its
own det.txt row at IoU 1.0.

A named arm must never silently run a different model: an unavailable
checkpoint raises, naming the acquisition step.
"""

from __future__ import annotations

from pathlib import Path

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.interfaces import StageContext
from matchlab_core.schemas import Tracklet
from matchlab_core.stages.track.tdlp_full import bridge

_WEIGHTS: dict[str, str] = {
    "kpr": (
        "CAMELTrack/pretrained_models/reid/"
        "kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"
    ),
    "prtreid": "CAMELTrack/pretrained_models/reid/prtreid_soccernet.pth.tar",
}

_ACQUIRE = {
    "prtreid": (
        "acquire it by running the sn-gamestate baseline once so TrackLab "
        "downloads PRTreID, or fetch the checkpoint from the SoccerNet Zenodo "
        "release, then place it at the path above"
    ),
    "kpr": "acquire it from the CAMELTrack pretrained_models release",
}


def _resolve_weights(model: str, external_root: str) -> Path:
    if model not in _WEIGHTS:
        raise ValueError(
            f"Unknown external re-ID model {model!r}. Known: {sorted(_WEIGHTS)}."
        )
    path = Path(external_root) / _WEIGHTS[model]
    if not path.exists():
        raise RuntimeError(
            f"External re-ID weights for {model!r} not found at {path} — {_ACQUIRE[model]}. "
            "Refusing to run this arm with a different model."
        )
    return path
```

Then add `embed_external(ctx, tracklets, *, model)`: call `bridge.stage_sequence` to lay out frames + a det.txt built from the fragment boxes, `bridge.run_external` on `gen_features.py` with `--reid-model <model> --no-pose`, and `bridge.join_features_to_tracklets` to produce the `FrameFeatures`. Mirror the parameter plumbing in `stages/track/tdlp_full/stage.py` lines 100–140 and 225–246 exactly; do not invent a second convention.

- [ ] **Step 5: Extend `gen_features.py`**

Add `--reid-model {kpr,prtreid}` (default `kpr`) selecting between `build_kpr` and a new `build_prtreid`, and `--no-pose` which skips RTMPose and writes zero keypoints at confidence 0. The no-pose path is safe for this harness: no module under `matchlab_core/reid/` reads keypoints (verified by `git grep keypoints`). Keep `--weights` working as the checkpoint path.

- [ ] **Step 6: Run tests, lint, commit**

```bash
uv run pytest packages/matchlab_core/tests/test_track_oracle_external.py -q
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/stages/track/oracle_external.py \
        packages/matchlab_core/tests/test_track_oracle_external.py
git commit -m "feat(track): external KPR/PRTreID feature backend for the oracle tracker (SPO-85)"
```

---

### Task 4: `reid-retrieval` experiment

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/experiments/reid_retrieval.py`
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/__init__.py`
- Test: `packages/matchlab_train/tests/test_reid_retrieval.py`

**Interfaces:**
- Consumes: `FrameFeatures.load`, `Tracklet`, `matchlab_core.reid.representation.build_representations` + `pair_similarity_breakdown`, `matchlab_core.reid.gates` (`TemporalOverlapGate`, `TeamConsistencyGate`, `MotionFeasibilityGate`), `matchlab_train.experiments.base.Experiment`
- Produces: `retrieval_metrics(tracklets, feats, gt_track_by_fragment, gates, *, fps) -> dict` and the registered experiment task `reid-retrieval`

The metric definitions are pre-registered and must be implemented exactly:

- **rank-1**: for each fragment with ≥1 gate-passing same-GT-track partner, 1 if its top-1 gate-passing candidate shares its GT track (any partner counts), else 0. Denominator = fragments with ≥1 gate-passing same-track partner. Fragments with no partner are excluded and **counted separately in the output** so the base is visible.
- **mAP**: standard average precision over the gate-passing ranking, averaged over the same denominator.
- **separation**: mean/median/p10/p90 of same-track and different-track affinities.
- **top-1 margin**, split by whether top-1 is correct.
- **breakdowns** by gap length to the matched partner, fragment length, and mean crop height.

- [ ] **Step 1: Write the failing test**

Create `packages/matchlab_train/tests/test_reid_retrieval.py` with a hand-built three-fragment case whose rank-1 and mAP are computable by hand:

```python
def test_rank1_counts_only_gate_passing_candidates():
    # Fragments A(frames 0-2) and B(frames 10-12) are the same GT track and
    # must not be blocked; C(frames 1-3) overlaps A so the temporal gate must
    # exclude it from A's pool even though its embedding is nearest.
    ...
    assert metrics["rank1"] == 1.0
    assert metrics["n_scored"] == 2
    assert metrics["n_no_partner"] == 1
```

Write the fixture with explicit embeddings so the nearest neighbour is known by construction (e.g. C's vector identical to A's, B's at cosine 0.9). This test is the one that proves the gate restriction is real — without it, a regression to unrestricted ranking would silently inflate every arm.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/matchlab_train/tests/test_reid_retrieval.py -q`
Expected: FAIL — module missing

- [ ] **Step 3: Implement `retrieval_metrics` + the experiment**

Reuse `build_representations` and `pair_similarity_breakdown` from `matchlab_core.reid.representation` so affinities are computed by exactly the code the engine uses — a second similarity implementation would make the comparison meaningless. Build the gate stack as `[TemporalOverlapGate(tolerance_frames=2), TeamConsistencyGate(team_by_tracklet), MotionFeasibilityGate(fps=fps, camera_motion=None)]`; a pair is in the pool if every gate returns `None`.

Team labels come from the run's `teams.json` (the oracle TEAM stage output), read through the artifact store.

The experiment iterates the dataset manifest's tuning sequences, runs the pipeline config once per (sequence, arm), then aggregates. Follow `experiments/benchmark.py` for manifest loading, per-row provenance, and result aggregation — including its refusal on provenance inconsistency.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/matchlab_train/tests/test_reid_retrieval.py -q`

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check packages
uv run pytest packages -q
git add packages/matchlab_train/src/matchlab_train/experiments/reid_retrieval.py \
        packages/matchlab_train/src/matchlab_train/experiments/__init__.py \
        packages/matchlab_train/tests/test_reid_retrieval.py
git commit -m "feat(bench): reid-retrieval experiment — gate-restricted rank-1/mAP (SPO-85)"
```

---

### Task 5: Configs + end-to-end smoke

**Files:**
- Create: `configs/pipeline.gt-tracklets-reid.yaml`
- Create: `configs/train/reid-retrieval-tuning.yaml`
- Test: `packages/matchlab_core/tests/test_pipeline_gt_tracklets_smoke.py`

- [ ] **Step 1: Write the pipeline config**

`configs/pipeline.gt-tracklets-reid.yaml` — `detect: oracle`, `track: oracle` (with the registered `gap_frames: 2`, `min_fragment_frames: 1`, `features_backend`/`features_model` overridden per arm), `team: oracle`, `associate: per-tracklet`, `calibrate`/`fuse`/`events`/`spotting`/`annotate` disabled. Header comment must state that `gap_frames` is pre-registered and must not be tuned.

- [ ] **Step 2: Write the experiment config**

`configs/train/reid-retrieval-tuning.yaml` — `dataset_manifest: configs/datasets/soccernet.json`, `roles: [tuning]`, five arms overriding `stages.track.params.features_backend` / `features_model`: `kpr`+`prtreid` (external), `osnet`+`solider`+`dinov2` (in-repo).

- [ ] **Step 3: Smoke test on one tuning sequence**

```bash
uv run matchlab-run --video data/videos/soccernet/SNMOT-116.mp4 \
  --config configs/pipeline.gt-tracklets-reid.yaml --device cuda --run-id smoke-gt-116
```

Expected: completes; `data/runs/smoke-gt-116/` contains `tracklets.json` and `frame_features.npz`; fragment count > track count (gaps were found).

- [ ] **Step 4: Verify purity is 1.0 by construction**

```bash
uv run python -c "
from matchlab_core.gt import GroundTruth
from matchlab_core.gt_fragments import fragment_tracks
gt = GroundTruth.model_validate_json(open('data/videos/soccernet/SNMOT-116.gt.json').read())
r = fragment_tracks(gt)
print('gt tracks', len(gt.tracks), 'fragments', len(r.tracklets))
print('multi-fragment tracks', sum(1 for v in set(r.gt_track_by_fragment.values())
      if list(r.gt_track_by_fragment.values()).count(v) > 1))
"
```

Expected: fragments ≥ tracks, and a non-zero count of multi-fragment tracks — if zero, the sequence has no re-entries and cannot contribute to rank-1.

- [ ] **Step 5: Commit**

```bash
git add configs/pipeline.gt-tracklets-reid.yaml configs/train/reid-retrieval-tuning.yaml \
        packages/matchlab_core/tests/test_pipeline_gt_tracklets_smoke.py
git commit -m "feat(configs): GT-tracklet re-ID harness pipeline + retrieval experiment (SPO-85)"
```

---

## Self-review notes

- **Spec coverage:** `gt_fragments` → Task 1; oracle TRACK stage → Task 2; in-repo backend → Task 2; external backend + `gen_features` flags → Task 3; `reid-retrieval` metrics → Task 4; configs → Task 5; held-out quarantine → Task 0. Error-handling requirements are folded into the task whose code raises them.
- **Deferred deliberately:** running the five arms and reporting results is *execution* of the pre-registered protocol, not implementation, and is gated on Task 3's weight acquisition. The engine-confirmation run for the winner is likewise post-measurement.
- **Known soft spot:** Task 3 Step 4 and Task 4 Step 3 say "mirror the existing pattern" rather than reproducing 200 lines of bridge plumbing verbatim. That is a deliberate call — those files are the authority on their own conventions and duplicating them here would create a second source of truth that can drift.
