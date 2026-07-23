# Frame-Exit Exemption for Persistent ID Switches — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A player who completely leaves the frame and returns under a new tracker ID no longer
tallies in the persistent ID-switch counts; the exempted switches are reported under their own
key instead of vanishing.

**Architecture:** Extend `persistent_switch_counts` (in
`packages/matchlab_core/src/matchlab_core/evaluation.py`, added by commit `024c115` per
`docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md`) so each ID sequence
carries source frame indices, and each transition between surviving runs is classified by its
gap: a **frame exit** requires (a) the GT track has *no boxes at all* strictly inside the gap,
(b) the last GT box before and first GT box after the gap both touch the image border, and
(c) the absence lasts at least 0.2 s. Everything else — full occlusion mid-pitch, tracker
losing a visible player, instant handoffs — still counts, because silent player swaps are the
product's worst failure and must not be excused. Raw motmetrics IDsw is untouched.

**Tech Stack:** Python (motmetrics event stream already consumed), pytest, TypeScript (one
type widened; no UI component changes — headline keys keep their names and now simply exclude
frame exits).

## Global Constraints

- Raw IDsw (`num_switches`, `idsw_tracklet`, `idsw_entity`) must not change — it is the
  literature-comparable number.
- Headline keys keep their exact names (`idsw_persistent_tracklet`, `idsw_persistent_entity`)
  and their meaning becomes "persistent switches excluding frame exits". No UI component
  (dashboard/benchmark/diff) changes; only `web/src/lib/types.ts` widens.
- Exempted switches are **reported, never silently dropped**: each level's block gains a
  `frame_exit` sub-dict with the same `t_0.5s/t_1s/t_2s` keys.
- Fail-safe direction: when frame dimensions are unknown (`gt.width`/`gt.height` == 0) or GT
  boxes are unavailable, **no exemption is applied** — the switch counts. Abstain from
  excusing, never from charging.
- Exemption thresholds: border margin = 2 % of the respective image dimension
  (`_FRAME_EXIT_BORDER_FRAC = 0.02`); minimum absence = 0.2 s
  (`_FRAME_EXIT_MIN_ABSENCE_S = 0.2`).
- Run tests with `uv run python -m pytest` (NOT bare `pytest`/`uv run pytest` — this repo's
  venv has stale console-script shebangs). Lint: `uv run ruff check packages` (line length
  100). Web typecheck: `cd web && npm run build`.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Frame-aware sequences + exemption logic in the pure function

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/evaluation.py` (the
  `persistent_switch_counts`, `_matched_id_sequences` region, currently near the
  `_switch_instances` helper)
- Test: `packages/matchlab_core/tests/test_gt_eval.py` (the
  `persistent_switch_counts` test block at the end of the file)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (Task 2 relies on these exact signatures):
  - `persistent_switch_counts(id_sequences: dict[int, list[tuple[int, int]]], seconds_per_frame: float, thresholds_s: tuple[float, ...] = _PERSISTENCE_THRESHOLDS_S, *, stride: int = 1, gt_boxes: dict[int, dict[int, list[float]]] | None = None, frame_size: tuple[int, int] | None = None) -> dict`
    — `id_sequences` maps GT track id → ordered `(source_frame_idx, hyp_id)` pairs;
    `gt_boxes` maps GT track id → `{source_frame_idx: [x, y, w, h]}`; returns
    `{"t_0.5s": int, "t_1s": int, "t_2s": int, "frame_exit": {"t_0.5s": int, "t_1s": int, "t_2s": int}}`.
  - `_matched_id_sequences(acc) -> dict[int, list[tuple[int, int]]]` (was `list[int]`).

- [ ] **Step 1: Update the existing pure-function tests to the tuple-based sequence shape and write the new failing exemption tests**

In `packages/matchlab_core/tests/test_gt_eval.py`, replace the block from the
`# --- persistent_switch_counts` banner comment to the end of
`test_persistent_headline_none_for_legacy_payload` with:

```python
# --- persistent_switch_counts (flicker-insensitive IDsw; spec:
# docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md) -------

SPF_25 = 1 / 25.0  # seconds per frame at 25 fps, stride 1

NO_EXITS = {"t_0.5s": 0, "t_1s": 0, "t_2s": 0}


def _seq(ids: list[int], start: int = 0) -> list[tuple[int, int]]:
    """Consecutive source frames starting at `start`, one hyp id per frame."""
    return [(start + i, hid) for i, hid in enumerate(ids)]


def test_persistent_flicker_revert_counts_zero():
    from matchlab_core.evaluation import persistent_switch_counts

    # A for 2 s, B for 0.2 s, back to A for 2 s: raw IDsw would be 2; the
    # flicker and its reversion both vanish at every threshold.
    counts = persistent_switch_counts({7: _seq([1] * 50 + [2] * 5 + [1] * 50)}, SPF_25)
    assert counts == {"t_0.5s": 0, "t_1s": 0, "t_2s": 0, "frame_exit": NO_EXITS}


def test_persistent_flicker_then_handoff_counts_one():
    from matchlab_core.evaluation import persistent_switch_counts

    # A (2 s), brief B (0.2 s), then C (2 s): identity genuinely moved via a
    # brief intermediary -> exactly one persistent switch at every threshold.
    counts = persistent_switch_counts({7: _seq([1] * 50 + [2] * 5 + [3] * 50)}, SPF_25)
    assert counts == {"t_0.5s": 1, "t_1s": 1, "t_2s": 1, "frame_exit": NO_EXITS}


def test_persistent_boundary_run_survives():
    from matchlab_core.evaluation import persistent_switch_counts

    # Two runs of exactly 1.0 s (25 frames at 25 fps): >= threshold survives,
    # so t_1s counts the transition; t_2s drops both runs.
    counts = persistent_switch_counts({7: _seq([1] * 25 + [2] * 25)}, SPF_25)
    assert counts == {"t_0.5s": 1, "t_1s": 1, "t_2s": 0, "frame_exit": NO_EXITS}


def test_persistent_stride_normalized():
    from matchlab_core.evaluation import persistent_switch_counts

    # The same 2 s + 2 s real-time handoff sampled at stride 1 (50+50 frames,
    # 0.04 s/frame) and stride 2 (25+25 frames, 0.08 s/frame) must agree.
    stride1 = persistent_switch_counts({7: _seq([1] * 50 + [2] * 50)}, 1 / 25.0)
    stride2 = persistent_switch_counts(
        {7: [(2 * i, hid) for i, hid in enumerate([1] * 25 + [2] * 25)]},
        2 / 25.0,
        stride=2,
    )
    assert stride1 == stride2
    assert stride1["t_1s"] == 1


def test_persistent_sums_over_gt_tracks():
    from matchlab_core.evaluation import persistent_switch_counts

    handoff = [1] * 50 + [2] * 50
    counts = persistent_switch_counts({7: _seq(handoff), 8: _seq(handoff)}, SPF_25)
    assert counts["t_1s"] == 2


def test_persistent_unknown_fps_abstains():
    from matchlab_core.evaluation import persistent_switch_counts

    # seconds_per_frame 0 (fps unknown): every run is dropped -> 0 everywhere,
    # never a fabricated count.
    counts = persistent_switch_counts({7: _seq([1] * 50 + [2] * 50)}, 0.0)
    assert counts["t_1s"] == 0


def test_persistent_headline_none_for_legacy_payload():
    from matchlab_core.evaluation import _persistent_headline

    # eval.json written before the metric existed -> None, not a crash or 0.
    assert _persistent_headline({}, "tracklet") is None


# --- frame-exit exemption ---------------------------------------------------
# A switch across a gap where the player genuinely left the frame (no GT boxes
# during the gap; edge boxes touch the image border; absence >= 0.2 s) is not
# charged to t_* -- it is tallied under "frame_exit" instead. Everything the
# exemption cannot positively verify still counts (fail-safe direction).

W, H = 1920, 1080
FRAME = (W, H)
BORDER_BOX = [0.0, 500.0, 40.0, 120.0]  # x1 == 0: touches the left border
MID_BOX = [900.0, 500.0, 40.0, 120.0]  # nowhere near any border


def _exit_fixture(edge_box, comeback_box):
    """GT track 7: seen frames 0-49 (run A), absent 50-99 (2 s), seen 100-149
    (run B under a new tracker id). GT boxes exist only on the seen frames."""
    seq = _seq([1] * 50) + _seq([2] * 50, start=100)
    boxes = {f: list(edge_box) for f in range(0, 50)}
    boxes.update({f: list(comeback_box) for f in range(100, 150)})
    return {7: seq}, {7: boxes}


def test_frame_exit_switch_is_exempt_and_reported():
    from matchlab_core.evaluation import persistent_switch_counts

    seqs, boxes = _exit_fixture(BORDER_BOX, BORDER_BOX)
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 0  # not charged
    assert counts["frame_exit"]["t_1s"] == 1  # but not silently dropped


def test_occlusion_gap_mid_pitch_still_counts():
    from matchlab_core.evaluation import persistent_switch_counts

    # Same absence, but the player vanished mid-pitch (full occlusion): the
    # edge boxes are nowhere near the border, so the switch still counts.
    seqs, boxes = _exit_fixture(MID_BOX, MID_BOX)
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0


def test_lost_while_visible_still_counts():
    from matchlab_core.evaluation import persistent_switch_counts

    # GT boxes exist during the match gap (tracker lost a visible player,
    # even though the player stood at the border): no exemption.
    seqs, boxes = _exit_fixture(BORDER_BOX, BORDER_BOX)
    boxes[7].update({f: list(BORDER_BOX) for f in range(50, 100)})
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0


def test_instant_border_handoff_still_counts():
    from matchlab_core.evaluation import persistent_switch_counts

    # Adjacent runs (no real absence) at the border: min-absence gate keeps
    # a same-moment handoff between two border-adjacent players countable.
    seq = _seq([1] * 50 + [2] * 50)
    boxes = {7: {f: list(BORDER_BOX) for f in range(0, 100)}}
    counts = persistent_switch_counts(
        {7: seq}, SPF_25, gt_boxes=boxes, frame_size=FRAME
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0


def test_unknown_frame_size_never_exempts():
    from matchlab_core.evaluation import persistent_switch_counts

    # gt.width/height == 0 (e.g. SoccerTrack CSVs without dims): the exemption
    # cannot be verified, so the switch counts -- abstain from excusing.
    seqs, boxes = _exit_fixture(BORDER_BOX, BORDER_BOX)
    counts = persistent_switch_counts(
        seqs, SPF_25, gt_boxes=boxes, frame_size=(0, 0)
    )
    assert counts["t_1s"] == 1
    assert counts["frame_exit"]["t_1s"] == 0
```

- [ ] **Step 2: Run the tests to verify the new ones fail and the updated ones error**

Run: `uv run python -m pytest packages/matchlab_core/tests/test_gt_eval.py -q -k persistent or frame_exit or occlusion or lost_while or instant_border or unknown_frame`
(quote the `-k` expression). Expected: the five new frame-exit tests FAIL (unexpected
keyword `gt_boxes` / missing `frame_exit` key); the updated existing tests fail on the
`frame_exit` key until Step 3 lands.

- [ ] **Step 3: Implement the exemption in `evaluation.py`**

Replace the block from `_PERSISTENCE_THRESHOLDS_S` through `_matched_id_sequences`
(inclusive) with:

```python
# Persistence thresholds for the flicker-insensitive switch count; 1 s is the
# headline (see docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md).
_PERSISTENCE_THRESHOLDS_S = (0.5, 1.0, 2.0)
_PERSISTENCE_HEADLINE_S = 1.0
# Frame-exit exemption: a switch across an absence where the player left the
# image does not tally (reported under "frame_exit" instead). "Left the image"
# requires border-touching GT boxes on both sides of a >= 0.2 s gap with no GT
# boxes inside it -- full occlusion mid-pitch and losing a visible player are
# NOT exempt (silent swaps are the product's worst failure; abstain from
# excusing, never from charging).
_FRAME_EXIT_BORDER_FRAC = 0.02
_FRAME_EXIT_MIN_ABSENCE_S = 0.2


def persistent_switch_counts(
    id_sequences: dict[int, list[tuple[int, int]]],
    seconds_per_frame: float,
    thresholds_s: tuple[float, ...] = _PERSISTENCE_THRESHOLDS_S,
    *,
    stride: int = 1,
    gt_boxes: dict[int, dict[int, list[float]]] | None = None,
    frame_size: tuple[int, int] | None = None,
) -> dict:
    """Flicker-insensitive ID-switch count. Raw motmetrics IDsw charges every
    matched-ID change, so a 3-frame occlusion flicker (A->B->A) costs 2 -- the
    same as a permanent handoff. Here each GT track's matched-ID sequence is
    segmented into runs of constant ID, runs shorter than the threshold are
    dropped as flicker, and only transitions between surviving runs with
    *different* IDs count -- so A->B->A with a short B is 0 (the reversion
    vanishes with the flicker), while A->B->C with a short B is 1 (identity
    genuinely moved, via a brief intermediary).

    `id_sequences` holds, per GT track, `(source_frame_idx, matched hyp id)`
    at each evaluated frame where a match existed, in frame order. Unmatched
    frames are simply absent: runs are compared across occlusion gaps, because
    a handoff across an occlusion is precisely the real failure.

    A run's duration is `frames x seconds_per_frame` (`stride / fps`), so
    counts are comparable across sampling strides. A boundary-length run
    (exactly the threshold) survives. `seconds_per_frame == 0` (unknown fps)
    drops every run: the result is 0 everywhere rather than a fabricated
    count -- raw IDsw remains the number to read in that case.

    Frame-exit exemption (`gt_boxes` = GT track id -> {source_frame_idx:
    [x, y, w, h]}, `frame_size` = (width, height)): a transition whose gap has
    no GT boxes strictly inside it, border-touching GT boxes at both edges,
    and an absence of at least `_FRAME_EXIT_MIN_ABSENCE_S` is tallied under
    the returned dict's "frame_exit" sub-dict instead of the `t_*` counts.
    When `gt_boxes`/`frame_size` are missing or dimensions are 0 the exemption
    is never applied -- unverifiable switches still count.
    """
    counts = {_threshold_key(t): 0 for t in thresholds_s}
    exits = {_threshold_key(t): 0 for t in thresholds_s}
    for gt_id, seq in id_sequences.items():
        runs: list[list] = []  # [hyp_id, n_frames, first_frame, last_frame]
        for frame_idx, hid in seq:
            if runs and runs[-1][0] == hid:
                runs[-1][1] += 1
                runs[-1][3] = frame_idx
            else:
                runs.append([hid, 1, frame_idx, frame_idx])
        track_boxes = (gt_boxes or {}).get(gt_id)
        for t in thresholds_s:
            surviving = [r for r in runs if r[1] * seconds_per_frame >= t]
            for a, b in zip(surviving, surviving[1:]):
                if a[0] == b[0]:
                    continue
                if _is_frame_exit_gap(
                    track_boxes, a[3], b[2], frame_size, seconds_per_frame, stride
                ):
                    exits[_threshold_key(t)] += 1
                else:
                    counts[_threshold_key(t)] += 1
    return {**counts, "frame_exit": exits}


def _is_frame_exit_gap(
    track_boxes: dict[int, list[float]] | None,
    gap_start: int,
    gap_end: int,
    frame_size: tuple[int, int] | None,
    seconds_per_frame: float,
    stride: int,
) -> bool:
    """True only when the gap between two surviving runs is positively a
    frame exit; anything unverifiable is False (the switch then counts)."""
    if not track_boxes or not frame_size or not frame_size[0] or not frame_size[1]:
        return False
    if stride <= 0 or seconds_per_frame <= 0:
        return False
    absence_s = (gap_end - gap_start) * seconds_per_frame / stride
    if absence_s < _FRAME_EXIT_MIN_ABSENCE_S:
        return False
    if any(gap_start < f < gap_end for f in track_boxes):
        return False  # the player was annotated (visible) during the gap
    exit_box = track_boxes.get(gap_start)
    entry_box = track_boxes.get(gap_end)
    if exit_box is None or entry_box is None:
        return False
    return _touches_border(exit_box, frame_size) and _touches_border(entry_box, frame_size)


def _touches_border(box: list[float], frame_size: tuple[int, int]) -> bool:
    x, y, w, h = box
    width, height = frame_size
    mx, my = _FRAME_EXIT_BORDER_FRAC * width, _FRAME_EXIT_BORDER_FRAC * height
    return x <= mx or x + w >= width - mx or y <= my or y + h >= height - my


def _threshold_key(t: float) -> str:
    return f"t_{t:g}s"


def _matched_id_sequences(acc) -> dict[int, list[tuple[int, int]]]:
    """Per GT track, `(source_frame_idx, matched hyp id)` at each evaluated
    frame with a match, in frame order -- from the same motmetrics event
    stream raw IDsw is computed from, so the two never disagree about
    matching."""
    seqs: dict[int, list[tuple[int, int]]] = {}
    for (frame_idx, _), ev in acc.mot_events.iterrows():
        if ev["Type"] in ("MATCH", "SWITCH"):
            seqs.setdefault(int(ev["OId"]), []).append((int(frame_idx), int(ev["HId"])))
    return seqs
```

Note: `_gt_composition_of_tracklet`, `merge_quality`, and everything else in the file is
untouched. `_persistent_headline` needs no change — `ps[level][_threshold_key(...)]` still
resolves because `t_*` keys stay top-level in each level's dict.

- [ ] **Step 4: Run the pure-function tests**

Run: `uv run python -m pytest packages/matchlab_core/tests/test_gt_eval.py -q`
Expected: FAIL only in `test_evaluate_run_association_gain` (integration asserts the old
per-level shape `{"t_0.5s": 0, ...}` without `frame_exit` — Task 2 fixes the wiring and that
assertion). All pure-function tests PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/matchlab_core/src/matchlab_core/evaluation.py packages/matchlab_core/tests/test_gt_eval.py
git commit -m "feat(eval): frame-exit exemption in persistent_switch_counts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Wire GT boxes and frame size through `evaluate_run`

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/evaluation.py` (inside `evaluate_run`:
  the GT-indexing block near the top and the per-level loop)
- Test: `packages/matchlab_core/tests/test_gt_eval.py`
  (`test_evaluate_run_association_gain` + one new integration test)

**Interfaces:**
- Consumes: Task 1's `persistent_switch_counts` keyword signature
  (`stride=`, `gt_boxes=`, `frame_size=`).
- Produces: each level of `eval.json`'s `persistent_switches` block now reads
  `{"t_0.5s": N, "t_1s": N, "t_2s": N, "frame_exit": {"t_0.5s": M, "t_1s": M, "t_2s": M}}`.
  Task 3's type mirrors this exactly.

- [ ] **Step 1: Update the integration assertion and add the exit-scenario integration test**

In `test_evaluate_run_association_gain`, replace:

```python
    for level in ("tracklet", "entity"):
        assert ps[level] == {"t_0.5s": 0, "t_1s": 0, "t_2s": 0}
```

with:

```python
    for level in ("tracklet", "entity"):
        assert ps[level] == {
            "t_0.5s": 0, "t_1s": 0, "t_2s": 0,
            "frame_exit": {"t_0.5s": 0, "t_1s": 0, "t_2s": 0},
        }
```

Then append at the end of the file (the synthetic GT builder `_write_soccernet_seq` pins
`imWidth=1920`; track 1's boxes sit at x=100 — mid-frame — so this test builds its own GT
with border boxes):

```python
def test_evaluate_run_frame_exit_not_charged(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    # 300-frame sequence: GT track 1 sits at the left border for frames
    # 0-99, is absent (out of frame) for 100-199 (4 s), and returns at the
    # border for 200-299. The tracker covers it with two different ids.
    seq = tmp_path / "SNMOT-002"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-002\nimDir=img1\nframeRate=25\nseqLength=300\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )
    (seq / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-002\nnum_tracklets=1\n"
        "trackletID_1= player team left;10\n"
    )
    rows = []
    for frame in list(range(1, 101)) + list(range(201, 301)):  # 1-based
        rows.append(f"{frame},1,0,500,40,120,1,-1,-1,-1")  # x=0: left border
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))
    gt = load_soccernet_sequence(seq)

    tracklets = [
        _tracklet(10, [(f, 0, 500) for f in range(0, 100)]),
        _tracklet(11, [(f, 0, 500) for f in range(200, 300)]),
    ]
    players = [
        {"player_id": 1, "tracklet_ids": [10]},
        {"player_id": 2, "tracklet_ids": [11]},
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {"video": {"fps": 25.0, "frame_count": 300, "sample_stride": 1}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps(players))

    result = evaluate_run(run_dir, gt)

    ps = result["persistent_switches"]
    # Raw IDsw still charges the re-entry break; the persistent count exempts
    # it as a frame exit and reports it under frame_exit instead.
    assert result["levels"]["tracklet"]["num_switches"] == 1
    for level in ("tracklet", "entity"):
        assert ps[level]["t_1s"] == 0
        assert ps[level]["frame_exit"]["t_1s"] == 1
```

- [ ] **Step 2: Run to verify the new test fails**

Run: `uv run python -m pytest packages/matchlab_core/tests/test_gt_eval.py -q`
Expected: `test_evaluate_run_frame_exit_not_charged` FAILS
(`ps[level]["frame_exit"]["t_1s"] == 0` — the wiring doesn't pass `gt_boxes` yet, so no
exemption fires); `test_evaluate_run_association_gain` PASSES (its per-level dicts now carry
an all-zero `frame_exit`, matching Task 1's return shape).

- [ ] **Step 3: Wire the call site**

In `evaluate_run`, directly after the existing `gt_by_frame` loop
(`for t in scored_tracks: for f in t.frames: gt_by_frame.setdefault(...)`), add:

```python
    # Per-track GT boxes ({source frame -> xywh}) + image dims feed the
    # persistent-switch frame-exit exemption; width/height of 0 (unknown)
    # disables exemption rather than guessing.
    gt_track_boxes: dict[int, dict[int, list[float]]] = {
        t.track_id: {f.frame_idx: _xywh(f.box.model_dump()) for f in t.frames}
        for t in scored_tracks
    }
    gt_frame_size = (gt.width, gt.height)
```

and change the per-level call from:

```python
        persistent_levels[level] = persistent_switch_counts(
            _matched_id_sequences(acc), seconds_per_frame
        )
```

to:

```python
        persistent_levels[level] = persistent_switch_counts(
            _matched_id_sequences(acc),
            seconds_per_frame,
            stride=stride,
            gt_boxes=gt_track_boxes,
            frame_size=gt_frame_size,
        )
```

- [ ] **Step 4: Run the full core suite and lint**

Run: `uv run python -m pytest packages/matchlab_core -q && uv run ruff check packages`
Expected: all PASS, lint clean.

- [ ] **Step 5: Commit**

```bash
git add packages/matchlab_core/src/matchlab_core/evaluation.py packages/matchlab_core/tests/test_gt_eval.py
git commit -m "feat(eval): wire GT boxes/frame size into frame-exit exemption

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Type mirror + docs

**Files:**
- Modify: `web/src/lib/types.ts` (the `persistent_switches` member of `EvalResult`)
- Modify: `docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md`
- Modify: `docs/implementation-status.md` (the persistent-switches bullet)

**Interfaces:**
- Consumes: Task 2's eval.json shape.
- Produces: nothing downstream; UI components are untouched (LabDiff reads
  `["t_1s"]` which still resolves).

- [ ] **Step 1: Widen the TS type**

In `web/src/lib/types.ts`, replace the `persistent_switches` member with:

```ts
  // Flicker-insensitive ID switches (keys "t_0.5s" | "t_1s" | "t_2s");
  // threshold_headline_s names the threshold runs.metrics headlines use.
  // frame_exit tallies switches exempted because the player verifiably left
  // the frame (border-touching GT boxes both sides of a GT-empty gap) --
  // excluded from the t_* counts but never silently dropped.
  persistent_switches?: {
    threshold_headline_s: number;
    tracklet: PersistentSwitchLevel;
    entity: PersistentSwitchLevel;
  };
```

and add directly above `EvalResult`:

```ts
export interface PersistentSwitchLevel {
  "t_0.5s": number;
  t_1s: number;
  "t_2s": number;
  frame_exit?: { "t_0.5s": number; t_1s: number; "t_2s": number };
}
```

- [ ] **Step 2: Typecheck/build**

Run: `cd web && npm run build`
Expected: clean build (LabDiff's `ev.persistent_switches?.tracklet["t_1s"]` still typechecks).

- [ ] **Step 3: Update the spec and implementation-status**

Append to `docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md`:

```markdown
## Amendment (2026-07-23): frame-exit exemption

A transition between surviving runs is exempted from the `t_*` counts — and tallied under a
per-level `frame_exit` sub-dict instead — only when the gap is positively a frame exit: the
GT track has no boxes strictly inside the gap, its last box before and first box after the
gap both touch the image border (within 2 % of the respective dimension), and the absence
lasts ≥ 0.2 s. Full occlusion mid-pitch, losing a visible player, and anything unverifiable
(unknown frame dimensions) still count — abstain from excusing, never from charging. Raw
IDsw is unchanged and still charges re-entry breaks.
```

In `docs/implementation-status.md`, inside the "Persistent (flicker-insensitive) ID
switches" bullet, insert after the sentence ending "(a handoff across occlusion is the real
failure);":

```markdown
  a verified frame exit (no GT boxes inside the gap, border-touching GT boxes at both edges,
  absence ≥ 0.2 s) is exempted from the counts and tallied under a per-level `frame_exit`
  key instead — occlusion and unverifiable gaps still count;
```

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/types.ts docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md docs/implementation-status.md
git commit -m "docs+types: frame-exit exemption for persistent ID switches

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Re-score the TDLP-full runs and report the shift

**Files:**
- Use (already exists, no repo changes):
  `/tmp/claude-1000/-home-jeremy-code-MatchDay-lab/8ddff85d-0f98-4833-9883-8a0bbf6958ed/scratchpad/rescore_tdlp.py`
  — re-runs `evaluate_run` for the seven `tdlpfull*` runs, rewrites each run dir's
  `eval.json`, and merges `headline_metrics` into `runs.metrics` in
  `data/matchlab.db`. If the scratchpad has been cleaned, recreate it from the listing in
  the session transcript or re-derive: it is a ~40-line loop over
  `SELECT id, run_dir, video_id, metrics FROM runs WHERE id LIKE 'tdlpfull%'` joining
  `videos.gt_path`, calling `evaluate_run(run_dir, GroundTruth.model_validate_json(...))`.

**Interfaces:**
- Consumes: Tasks 1–2 (new eval shape).
- Produces: refreshed `eval.json` + `runs.metrics` for the 7 imported runs; the numbers for
  the final report.

- [ ] **Step 1: Extend the script's report line to show exits, then run it**

In the script, replace the final `print(...)` block with:

```python
    ps = result["persistent_switches"]["tracklet"]
    raw = result["levels"]["tracklet"]["num_switches"]
    print(
        f"{run['id']}: raw idsw {raw:>4}  ->  "
        f"persistent 1s {ps['t_1s']:>3} (+{ps['frame_exit']['t_1s']} frame-exit exempt)"
    )
```

Run: `uv run python <scratchpad>/rescore_tdlp.py`
Expected: seven lines; SoccerNet rows' persistent counts drop only where absences are
genuine border exits (SNMOT GT tracks have up to 13.5 s holes, so nonzero exemptions are
plausible); SportsMOT rows change little.

- [ ] **Step 2: Restart the dev API so served metrics refresh**

Run (from the repo root — the API resolves `./data` from cwd):
`fuser -k 8001/tcp; nohup uv run python -m uvicorn matchlab_server.app:app --port 8001 > /tmp/api.log 2>&1 &`
then verify `curl -s http://127.0.0.1:8001/api/runs | python3 -c "import json,sys; rs=json.load(sys.stdin); print([r['metrics'].get('idsw_persistent_tracklet') for r in rs if r['id'].startswith('tdlpfullsn')])"`
Expected: the refreshed (possibly lower) persistent counts.

- [ ] **Step 3: Report**

No commit (data dir is gitignored). Summarize the before/after table in the final message,
naming which switches moved to `frame_exit` per run.

---

## Self-Review Notes

- Spec coverage: exemption definition (gap-empty GT + border + min absence), fail-safe
  direction, separate reporting, raw-IDsw untouched, headline keys unchanged, docs — all have
  tasks. UI intentionally has none (no component reads `frame_exit` yet — YAGNI; the diff/
  benchmark/dashboard keep reading the now-exit-excluded headline keys).
- Type consistency: `persistent_switch_counts` keyword signature in Task 1 matches Task 2's
  call site; the eval.json shape in Task 2 matches Task 3's `PersistentSwitchLevel`.
- The `frame_exit` sub-dict is keyword-safe with `_persistent_headline` (top-level `t_*`
  keys unchanged) — asserted implicitly by `test_evaluate_run_association_gain`'s headline
  assertions continuing to pass.
