# Switch Scrubbing in the Run Viewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-transition switch records in `eval.json`, and a run-viewer UI that scrubs to
them: classified timeline markers (genuine / frame-exit / raw), a Switches list in the Eval
tab, click = seek + highlight both ids, ‹ › stepping.

**Architecture:** The evaluator records one dict per surviving-run transition at the 1 s
headline threshold, inside the loop that already counts them (`persistent_switch_counts`);
`_is_frame_exit_gap` is refactored to `_locate_frame_exit` returning (verdict, absence) so
the absence span is reported for counted transitions too. The viewer renders those records —
no client-side classification. Spec:
`docs/superpowers/specs/2026-07-24-switch-scrubbing-design.md`.

**Tech Stack:** Python (matchlab_core evaluation + pytest), TypeScript/React (LabRunViewer,
TimelineStrip markers), no new dependencies.

## Global Constraints

- **NO COMMITS.** Leave every change uncommitted; the user commits when satisfied. Do not
  run `git commit` anywhere.
- Counts must not change: `t_*` and `frame_exit` values in every existing test stay
  identical; transitions are recorded only at `_PERSISTENCE_HEADLINE_S` (1.0 s) and only
  when that value is in `thresholds_s`.
- Boolean exemption behavior of the refactored `_locate_frame_exit` must equal the current
  `_is_frame_exit_gap` for every input (two-tier border test, all fail-safe paths).
- Old eval.json artifacts (no `transitions` key) must render exactly today's raw-marker view
  — optional field, graceful fallback, no crashes.
- Times in transition records: seconds rounded to 2 decimals; frame→seconds is
  `frame_idx * seconds_per_frame / stride`.
- Tests: `uv run python -m pytest` (NEVER bare `pytest`/`uv run pytest` — stale venv
  shebangs). Lint: `uv run ruff check packages` (line length 100; note E731 is enabled — no
  lambda assignments). Web typecheck: `cd web && npm run build`.

---

### Task 1: Transition records in the evaluator

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/evaluation.py` (`persistent_switch_counts`,
  `_is_frame_exit_gap` → `_locate_frame_exit`, `evaluate_run` enrichment)
- Test: `packages/matchlab_core/tests/test_gt_eval.py`

**Interfaces:**
- Consumes: existing two-tier exemption logic (commit `177c1d2`).
- Produces (Task 2 relies on this): each level dict returned by `persistent_switch_counts`
  gains `"transitions": list[dict]` — records sorted by `t_to`, shape
  `{gt_track_id, prev_id, new_id, t_from, t_to, prev_run_s, new_run_s, verdict:
  "genuine"|"frame_exit", absence: {"t_from","t_to"}|None}`; `evaluate_run` adds
  `"level"` and `"gt_label"` to each record before writing `eval.json`.

- [ ] **Step 1: Update existing tests to the new return shape and add transition tests**

In `packages/matchlab_core/tests/test_gt_eval.py`:

(a) The pure-function tests that assert full-dict equality gain a `"transitions"` key.
Update these exact assertions:

```python
# test_persistent_flicker_revert_counts_zero
    assert counts["t_0.5s"] == 0 and counts["t_1s"] == 0 and counts["t_2s"] == 0
    assert counts["frame_exit"] == NO_EXITS
    assert counts["transitions"] == []

# test_persistent_flicker_then_handoff_counts_one
    assert counts["t_0.5s"] == 1 and counts["t_1s"] == 1 and counts["t_2s"] == 1
    assert counts["frame_exit"] == NO_EXITS
    assert [(r["prev_id"], r["new_id"], r["verdict"]) for r in counts["transitions"]] == [
        (1, 3, "genuine")
    ]

# test_persistent_boundary_run_survives
    assert counts["t_0.5s"] == 1 and counts["t_1s"] == 1 and counts["t_2s"] == 0
    assert counts["frame_exit"] == NO_EXITS
```

(Where those tests currently do `assert counts == {...}`, replace with the key-wise
assertions above; `NO_EXITS` already exists.)

(b) `test_persistent_stride_normalized` — transition times differ by sampling granularity
across strides (last matched frame 49 at stride 1 vs 48 at stride 2), so compare counts
without transitions and assert the transition separately:

```python
def test_persistent_stride_normalized():
    from matchlab_core.evaluation import persistent_switch_counts

    stride1 = persistent_switch_counts({7: _seq([1] * 50 + [2] * 50)}, 1 / 25.0)
    stride2 = persistent_switch_counts(
        {7: [(2 * i, hid) for i, hid in enumerate([1] * 25 + [2] * 25)]},
        2 / 25.0,
        stride=2,
    )
    strip = lambda c: {k: v for k, v in c.items() if k != "transitions"}
    assert strip(stride1) == strip(stride2)
    assert stride1["t_1s"] == 1
    assert [r["verdict"] for r in stride1["transitions"]] == ["genuine"]
    assert [r["verdict"] for r in stride2["transitions"]] == ["genuine"]
```

Note E731 (no lambda assignment) — write `strip` as a nested `def`:

```python
    def strip(c):
        return {k: v for k, v in c.items() if k != "transitions"}
```

(c) In `test_evaluate_run_association_gain`, the per-level equality becomes:

```python
    for level in ("tracklet", "entity"):
        assert ps[level]["t_0.5s"] == 0 and ps[level]["t_1s"] == 0 and ps[level]["t_2s"] == 0
        assert ps[level]["frame_exit"] == {"t_0.5s": 0, "t_1s": 0, "t_2s": 0}
        assert ps[level]["transitions"] == []
```

(d) Append new tests at the end of the file:

```python
# --- transition records (switch scrubbing; spec:
# docs/superpowers/specs/2026-07-24-switch-scrubbing-design.md) --------------


def test_transition_record_genuine_handoff_times_and_ids():
    from matchlab_core.evaluation import persistent_switch_counts

    counts = persistent_switch_counts({7: _seq([1] * 50 + [2] * 50)}, SPF_25)
    (rec,) = counts["transitions"]
    assert rec["gt_track_id"] == 7
    assert rec["prev_id"] == 1 and rec["new_id"] == 2
    assert rec["t_from"] == 1.96  # frame 49
    assert rec["t_to"] == 2.0  # frame 50
    assert rec["prev_run_s"] == 2.0 and rec["new_run_s"] == 2.0
    assert rec["verdict"] == "genuine"
    assert rec["absence"] is None  # no >=0.2s GT gap in the window


def test_transition_record_frame_exit_carries_absence():
    from matchlab_core.evaluation import persistent_switch_counts

    seqs, boxes = _exit_fixture(BORDER_BOX, INSIDE_BOX)
    counts = persistent_switch_counts(seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME)
    (rec,) = counts["transitions"]
    assert rec["verdict"] == "frame_exit"
    assert rec["absence"] == {"t_from": 1.96, "t_to": 4.0}  # frames 49 -> 100


def test_transition_record_counted_still_reports_absence():
    from matchlab_core.evaluation import persistent_switch_counts

    # Mid-pitch occlusion: counted, but the located absence is still reported
    # so "why was this NOT exempt" is auditable from the artifact.
    seqs, boxes = _exit_fixture(MID_BOX, MID_BOX)
    counts = persistent_switch_counts(seqs, SPF_25, gt_boxes=boxes, frame_size=FRAME)
    (rec,) = counts["transitions"]
    assert rec["verdict"] == "genuine"
    assert rec["absence"] == {"t_from": 1.96, "t_to": 4.0}


def test_transition_records_sorted_and_enriched_via_evaluate_run(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run

    # Reuse the frame-exit integration fixture shape: GT track 1 at the left
    # border, absent 4 s mid-clip, tracker covers it with two ids.
    seq = tmp_path / "SNMOT-003"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-003\nimDir=img1\nframeRate=25\nseqLength=300\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )
    (seq / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-003\nnum_tracklets=1\n"
        "trackletID_1= player team left;10\n"
    )
    rows = []
    for frame in list(range(1, 101)) + list(range(201, 301)):
        rows.append(f"{frame},1,0,500,40,120,1,-1,-1,-1")
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
    (run_dir / "manifest.json").write_text(
        json.dumps({"video": {"fps": 25.0, "frame_count": 300, "sample_stride": 1}})
    )
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps(players))

    result = evaluate_run(run_dir, gt)
    for level in ("tracklet", "entity"):
        recs = result["persistent_switches"][level]["transitions"]
        (rec,) = recs
        assert rec["level"] == level
        assert rec["gt_label"] == "#10 (left)"
        assert rec["verdict"] == "frame_exit"
        assert rec["t_to"] == 8.0  # first matched frame after re-entry
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run python -m pytest packages/matchlab_core/tests/test_gt_eval.py -q`
Expected: the four new tests FAIL with `KeyError: 'transitions'`; updated existing tests
also fail on the missing key. (Counts-only assertions still pass.)

- [ ] **Step 3: Implement**

In `packages/matchlab_core/src/matchlab_core/evaluation.py`:

(a) Add a module-level helper next to `_threshold_key`:

```python
def _frame_to_s(frame_idx: int, seconds_per_frame: float, stride: int) -> float:
    return round(frame_idx * seconds_per_frame / stride, 2)
```

(b) Replace `_is_frame_exit_gap` with `_locate_frame_exit` — same docstring content plus the
return-shape sentence; identical boolean logic, but the located absence is returned even
when the verdict is False (and even when `frame_size` is unknown, since locating the absence
needs only `track_boxes`):

```python
def _locate_frame_exit(
    track_boxes: dict[int, list[float]] | None,
    gap_start: int,
    gap_end: int,
    frame_size: tuple[int, int] | None,
    seconds_per_frame: float,
    stride: int,
) -> tuple[bool, tuple[int, int] | None]:
    """Locate the largest GT-annotation absence inside the transition window
    and apply the two-tier border test (see the module constants above).
    Returns (is_frame_exit, absence) where absence is the (from_frame,
    to_frame) of the largest gap >= _FRAME_EXIT_MIN_ABSENCE_S, or None --
    reported for counted transitions too, so "why was this NOT exempt" is
    auditable from eval.json. Anything unverifiable is (False, ...): the
    switch then counts."""
    if not track_boxes or stride <= 0 or seconds_per_frame <= 0:
        return False, None
    frames = sorted(f for f in track_boxes if gap_start <= f <= gap_end)
    if len(frames) < 2:
        return False, None
    absent_from, absent_to = max(
        zip(frames, frames[1:]), key=lambda pair: pair[1] - pair[0]
    )
    absence_s = (absent_to - absent_from) * seconds_per_frame / stride
    if absence_s < _FRAME_EXIT_MIN_ABSENCE_S:
        return False, None
    absence = (absent_from, absent_to)
    if not frame_size or not frame_size[0] or not frame_size[1]:
        return False, absence
    at_border = [
        _touches_border(track_boxes[absent_from], frame_size),
        _touches_border(track_boxes[absent_to], frame_size),
    ]
    if absence_s >= _FRAME_EXIT_LONG_ABSENCE_S:
        return any(at_border), absence
    return all(at_border), absence
```

(c) In `persistent_switch_counts`, replace the per-threshold classification block with:

```python
    counts = {_threshold_key(t): 0 for t in thresholds_s}
    exits = {_threshold_key(t): 0 for t in thresholds_s}
    transitions: list[dict] = []
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
                is_exit, absence = _locate_frame_exit(
                    track_boxes, a[3], b[2], frame_size, seconds_per_frame, stride
                )
                if is_exit:
                    exits[_threshold_key(t)] += 1
                else:
                    counts[_threshold_key(t)] += 1
                if t == _PERSISTENCE_HEADLINE_S:
                    transitions.append(
                        {
                            "gt_track_id": gt_id,
                            "prev_id": a[0],
                            "new_id": b[0],
                            "t_from": _frame_to_s(a[3], seconds_per_frame, stride),
                            "t_to": _frame_to_s(b[2], seconds_per_frame, stride),
                            "prev_run_s": round(a[1] * seconds_per_frame, 2),
                            "new_run_s": round(b[1] * seconds_per_frame, 2),
                            "verdict": "frame_exit" if is_exit else "genuine",
                            "absence": (
                                {
                                    "t_from": _frame_to_s(absence[0], seconds_per_frame, stride),
                                    "t_to": _frame_to_s(absence[1], seconds_per_frame, stride),
                                }
                                if absence
                                else None
                            ),
                        }
                    )
    transitions.sort(key=lambda r: r["t_to"])
    return {**counts, "frame_exit": exits, "transitions": transitions}
```

Extend the function docstring's exemption paragraph with one sentence: "Transitions at the
headline threshold are additionally returned under `transitions`, sorted by `t_to`, with
the located absence reported for counted transitions too."

(d) In `evaluate_run`, after the per-level loop (right before `result = {`), enrich:

```python
    for level, ps_level in persistent_levels.items():
        for rec in ps_level["transitions"]:
            rec["level"] = level
            rec["gt_label"] = gt_label.get(rec["gt_track_id"], "?")
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run python -m pytest packages/matchlab_core -q && uv run ruff check packages`
Expected: all pass, lint clean. Do NOT commit.

---

### Task 2: Types + run-viewer UI

**Files:**
- Modify: `web/src/lib/types.ts` (`PersistentSwitchLevel`, new `PersistentSwitchTransition`)
- Modify: `web/src/pages/LabRunViewer.tsx` (markers memo + toggle state, stepping buttons,
  EvalTab switches list + `onTransition` prop, click wiring)

**Interfaces:**
- Consumes: Task 1's record shape (with `level` + `gt_label` present in eval.json).
- Produces: user-visible UI; no downstream consumers.

- [ ] **Step 1: Types**

In `web/src/lib/types.ts`, add above `EvalResult` (next to `PersistentSwitchLevel`):

```ts
export interface PersistentSwitchTransition {
  level: "tracklet" | "entity";
  gt_track_id: number;
  gt_label: string;
  prev_id: number;
  new_id: number;
  t_from: number;
  t_to: number;
  prev_run_s: number;
  new_run_s: number;
  verdict: "genuine" | "frame_exit";
  absence: { t_from: number; t_to: number } | null;
}
```

and extend `PersistentSwitchLevel` with `transitions?: PersistentSwitchTransition[];`.

- [ ] **Step 2: Marker classes + stepping in `LabRunViewer`**

(a) State (next to the existing `signal` state):

```ts
const [switchClasses, setSwitchClasses] = useState({
  genuine: true,
  frameExit: false,
  raw: false,
});
```

(b) A shared "go to transition" callback (place after the `seek` definition; mirrors the
existing `onInstance` handler's layer/highlight discipline):

```ts
const goToTransition = (rec: PersistentSwitchTransition) => {
  seek(Math.max(0, rec.t_to - 1));
  setLayers((l) => ({ ...l, gt: true, tracklets: true }));
  setHlGtTrack(rec.gt_track_id);
  if (rec.level === "tracklet") {
    setHlPair([rec.prev_id, rec.new_id]);
    setHlTracklet(null);
    setHlPlayer(null);
  } else {
    setHlPair(null);
    setHlTracklet(null);
    setHlPlayer(rec.new_id < 100000 ? rec.new_id : null);
  }
};
```

(c) Replace the `evalMarkers` memo:

```ts
const switchTransitions = useMemo(() => {
  const ps = artifacts.eval?.persistent_switches;
  if (!ps) return null;
  const recs = [...(ps.tracklet.transitions ?? []), ...(ps.entity.transitions ?? [])];
  return recs.length > 0 || ps.tracklet.transitions ? recs : null;
}, [artifacts.eval]);

const visibleTransitions = useMemo(() => {
  if (!switchTransitions) return [];
  return switchTransitions
    .filter((r) => (r.verdict === "genuine" ? switchClasses.genuine : switchClasses.frameExit))
    .sort((a, b) => a.t_to - b.t_to);
}, [switchTransitions, switchClasses]);

const evalMarkers = useMemo(() => {
  const ev = artifacts.eval;
  if (!ev) return undefined;
  // Pre-transitions artifacts: exactly the old raw view.
  if (!switchTransitions) {
    if (ev.instances.length === 0) return undefined;
    return ev.instances.map((inst) => ({
      t: Math.max(0, inst.t),
      color: inst.level === "entity" ? "#F5C518" : "#8B949E",
      title: `${inst.level} switch @ ${fmtClock(inst.t)} GT ${inst.gt_label}`,
      onClick: () => seek(Math.max(0, inst.t - 1)),
    }));
  }
  const markers = visibleTransitions.map((rec) => ({
    t: Math.max(0, rec.t_to),
    color: rec.verdict === "genuine" ? "#F87171" : "#64748B",
    title: `${rec.verdict === "genuine" ? "switch" : "frame exit"} ${rec.level} @ ${fmtClock(
      rec.t_to,
    )} GT ${rec.gt_label} · ${rec.prev_id} → ${rec.new_id}`,
    onClick: () => goToTransition(rec),
  }));
  if (switchClasses.raw) {
    markers.push(
      ...ev.instances.map((inst) => ({
        t: Math.max(0, inst.t),
        color: "#8B949E",
        title: `raw ${inst.level} switch @ ${fmtClock(inst.t)} GT ${inst.gt_label}`,
        onClick: () => seek(Math.max(0, inst.t - 1)),
      })),
    );
  }
  return markers.length > 0 ? markers : undefined;
}, [artifacts.eval, switchTransitions, visibleTransitions, switchClasses.raw]);
```

(The `goToTransition` capture in a memo is fine — it only closes over stable refs/setters.)

(d) Toggle row + stepping, rendered inside the timeline `Card` between `SignalPicker` and
the "red cells" hint (adjust the existing flex row to fit — keep the hint, right-aligned):

```tsx
{switchTransitions && (
  <div className="flex items-center gap-2 font-mono text-[11px] text-ink-500">
    {(
      [
        ["genuine", "switches", "#F87171"],
        ["frameExit", "frame exits", "#64748B"],
        ["raw", "raw", "#8B949E"],
      ] as const
    ).map(([key, label, color]) => (
      <button
        key={key}
        onClick={() => setSwitchClasses((s) => ({ ...s, [key]: !s[key] }))}
        className={`rounded border px-2 py-0.5 ${
          switchClasses[key]
            ? "border-white/20 text-ink-100"
            : "border-white/8 text-ink-600"
        }`}
      >
        <span style={{ color: switchClasses[key] ? color : undefined }}>●</span> {label}
      </button>
    ))}
    <span className="mx-1 text-ink-700">·</span>
    <button
      className="rounded border border-white/8 px-2 py-0.5 hover:text-ink-100"
      title="previous switch"
      onClick={() => {
        const t = getTime();
        const prev = [...visibleTransitions].reverse().find((r) => r.t_to < t + 0.95);
        if (prev) goToTransition(prev);
      }}
    >
      ‹
    </button>
    <button
      className="rounded border border-white/8 px-2 py-0.5 hover:text-ink-100"
      title="next switch"
      onClick={() => {
        const t = getTime();
        const next = visibleTransitions.find((r) => r.t_to - 1 > t + 0.05);
        if (next) goToTransition(next);
      }}
    >
      ›
    </button>
  </div>
)}
```

(The +0.95 / +0.05 offsets step relative to the landing position `t_to - 1`, so repeated
clicks advance rather than re-selecting the current transition.)

- [ ] **Step 3: Switches list in the Eval tab**

(a) Give `EvalTab` a new prop `onTransition: (rec: PersistentSwitchTransition) => void`,
passed from the page as `onTransition={goToTransition}`.

(b) Inside `EvalTab`, before the raw instance list, compute and render (respecting the
existing `levelFilter` and `gtFilter` state):

```tsx
const transitions = useMemo(() => {
  const ps = ev?.persistent_switches;
  if (!ps) return null;
  let list = [...(ps.tracklet.transitions ?? []), ...(ps.entity.transitions ?? [])];
  if (levelFilter !== "all") list = list.filter((r) => r.level === levelFilter);
  if (gtFilter !== "all") list = list.filter((r) => r.gt_track_id === gtFilter);
  return list.sort((a, b) => a.t_to - b.t_to);
}, [ev, levelFilter, gtFilter]);
```

Render above the instance list (hidden when `transitions === null`):

```tsx
{transitions !== null && (
  <div>
    <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-500">
      Switches (persistent, 1s)
    </div>
    {transitions.length === 0 ? (
      <div className="py-2 text-[12px] text-ink-500">
        No persistent switches at the current filters.
      </div>
    ) : (
      <table className="w-full text-[12px]">
        <tbody>
          {transitions.map((rec, i) => (
            <tr
              key={i}
              className="cursor-pointer border-b border-white/5 last:border-0 hover:bg-turf-900"
              onClick={() => onTransition(rec)}
            >
              <td className="py-1.5">
                <span
                  className={`rounded px-1.5 py-0.5 font-mono text-[10px] uppercase ${
                    rec.verdict === "genuine"
                      ? "bg-team-away/20 text-team-away"
                      : "bg-white/5 text-ink-500"
                  }`}
                >
                  {rec.verdict === "genuine" ? "switch" : "frame exit"}
                </span>
              </td>
              <td className="py-1.5 font-mono text-ink-300">
                {rec.prev_id} → {rec.new_id}
              </td>
              <td className="py-1.5 text-ink-400">{rec.gt_label}</td>
              <td className="py-1.5 font-mono text-ink-400">
                {fmtClock(rec.t_from)}–{fmtClock(rec.t_to)}
              </td>
              <td className="py-1.5 text-right font-mono text-[11px] text-ink-500">
                {rec.prev_run_s.toFixed(1)}s → {rec.new_run_s.toFixed(1)}s
              </td>
              <td className="py-1.5 pl-2 text-[11px] text-ink-600">{rec.level}</td>
            </tr>
          ))}
        </tbody>
      </table>
    )}
  </div>
)}
```

(Check `fmtClock` is imported in `LabRunViewer.tsx` — it is, used by the existing markers.
`team-away` is an existing color token used for "bad" deltas in LabDiff; if the class names
`bg-team-away/20 text-team-away` don't exist in the Tailwind config, use
`bg-red-400/15 text-red-400`.)

- [ ] **Step 4: Typecheck/build**

Run: `cd web && npm run build`
Expected: clean. Do NOT commit.

---

### Task 3: Re-score, restart, docs

**Files:**
- Modify: `docs/implementation-status.md` (persistent-switches bullet gains the transitions
  sentence)
- No repo changes for the re-score (data/ is gitignored).

- [ ] **Step 1: Re-score the imported TDLP runs**

Run: `uv run python /tmp/claude-1000/-home-jeremy-code-MatchDay-lab/8ddff85d-0f98-4833-9883-8a0bbf6958ed/scratchpad/rescore_tdlp.py`
Expected: identical counts to the last re-score (SNMOT-124 oracle: 3 + 16 exits at 1 s) —
transitions are additive. If the scratchpad script is gone, re-evaluate through the running
API is NOT possible for imported runs (manifest 422); recreate the script from the session
transcript.

- [ ] **Step 2: Verify a transition record end-to-end**

Run:
`uv run python -c "import json; ev=json.load(open('data/runs/tdlpfullsne4b9e2/eval.json')); recs=ev['persistent_switches']['tracklet']['transitions']; print(len(recs)); print(recs[0])"`
Expected: 19 records (3 genuine + 16 frame_exit), each with `level`, `gt_label`, `verdict`,
`absence`.

- [ ] **Step 3: Restart the dev API and spot-check the viewer**

From the repo root:
`fuser -k 8001/tcp; nohup uv run python -m uvicorn matchlab_server.app:app --port 8001 > /tmp/api.log 2>&1 &`
Then open `http://127.0.0.1:5176/lab/runs/tdlpfullsne4b9e2` → Eval tab: Switches list shows
3 genuine rows (t≈8–10 s and two at t≈19 s) with frame exits appearing when toggled; timeline
shows 3 red markers by default; ‹ › steps between them; clicking highlights the id pair.

- [ ] **Step 4: Docs**

In `docs/implementation-status.md`, extend the persistent-switches bullet (after the
two-tier sentence) with:

```markdown
  Each level also records per-transition evidence at the 1 s headline (`transitions`:
  prev/new id, window times, run durations, verdict, located absence) — the run viewer's
  switch-scrubbing UI (timeline markers, Eval-tab Switches list, ‹ › stepping) renders these
  directly; pre-feature eval.json artifacts fall back to the raw instance view.
```

- [ ] **Step 5: Final verification — full suites**

Run: `uv run python -m pytest packages -q && uv run ruff check packages && cd web && npm run build`
Expected: all green. Leave everything uncommitted and report: files changed, test counts,
and the SNMOT-124 viewer spot-check outcome.

---

## Self-Review Notes

- Spec coverage: record shape + rounding (Task 1c), absence-on-counted (Task 1 test),
  refactored locator (Task 1b), enrichment + sorting (Task 1d + integration test), 3-class
  markers with genuine-only default + raw fallback (Task 2 Step 2c/d), Switches list with
  level/GT filters + fallback (Task 2 Step 3), click interaction incl. entity-level
  single-player rule and synthetic-id guard (`goToTransition`), stepping (Task 2 Step 2d),
  types (Task 2 Step 1), backfill/docs (Task 3). Out-of-scope items from the spec are not
  implemented anywhere.
- Type consistency: `PersistentSwitchTransition` fields match Task 1's record dict exactly
  (incl. `level`/`gt_label` added by `evaluate_run`); `goToTransition` signature matches the
  `onTransition` prop.
- No-commit constraint stated globally and repeated at every verification step.
