# Switch Layer Attribution (SPO-19) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every ID-switch instance in eval.json carries an evidence-based layer attribution
(`detection` / `online_association` / `refinement` / `offline_association`) or an explicit
`ambiguous` tag, surfaced in the Lab failure browser.

**Architecture:** A pure post-pass module `pitchlab_core/attribution.py` annotates an
already-computed eval result in place; `evaluate_run` always calls it (single-run evidence),
and callers holding an oracle-run eval payload re-invoke it to upgrade ambiguous
tracklet-level switches. Explicit oracle pairing in the benchmark runner
(`oracle_candidate`) and the server evaluate endpoint (`oracle_run_id`). Spec:
`docs/superpowers/specs/2026-07-17-switch-layer-attribution-design.md`.

**Tech Stack:** Python 3.12 (pydantic, motmetrics only in integration tests), FastAPI,
React/TypeScript (hand-mirrored types).

## Global Constraints

- Layers enum exactly: `detection`, `online_association`, `refinement` (reserved, never
  emitted today), `offline_association`, `ambiguous`.
- No attribution path may silently default to a specific layer; insufficient evidence →
  `ambiguous` with an `insufficient_evidence` record.
- All refusals are loud and name what's wrong (house style).
- Matching tolerance default `tol_s = 1.0` (same as `diff_switch_instances`).
- `web/src/lib/types.ts` mirrors the artifact by hand — keep in sync.
- Lint: `uv run ruff check packages` (line-length 100). Frontend gate:
  `cd web && npm run build`.
- Run all commands from the repo root (the worktree root).

---

### Task 1: Core attribution module — matcher, context, single-run rules

**Files:**
- Create: `packages/pitchlab_core/src/pitchlab_core/attribution.py`
- Create: `packages/pitchlab_core/tests/test_attribution.py`

**Interfaces:**
- Produces: `match_instances(a_list: list[dict], b_list: list[dict], tol_s: float) -> list[tuple[int, int]]` (greedy nearest-`t` one-to-one index pairs, closest first);
  `detect_context(manifest: dict) -> dict` (`{"detect_impl": str|None, "oracle_input": bool}`);
  `attribute_switches(result: dict, *, oracle_eval: dict | None = None, oracle_run_id: str | None = None, tol_s: float = 1.0) -> None` (in-place, idempotent);
  `DEFAULT_TOL_S = 1.0`, `LAYERS` tuple.

- [ ] **Step 1: Write the failing tests** (single-run rules only; oracle comparison is Task 2)

```python
"""Layer attribution for ID-switch instances (SPO-19): pure-function tests on
hand-built eval payloads -- no pipeline execution, no motmetrics."""

from __future__ import annotations

import pytest
from pitchlab_core.attribution import attribute_switches, detect_context, match_instances


def _inst(level: str, frame_idx: int, gt: int, prev_id: int, new_id: int, t: float | None = None) -> dict:
    return {
        "level": level,
        "kind": "id_switch",
        "frame_idx": frame_idx,
        "t": round(frame_idx / 25.0, 2) if t is None else t,
        "gt_track_id": gt,
        "gt_label": f"#{gt}",
        "prev_id": prev_id,
        "new_id": new_id,
    }


def _payload(
    instances: list[dict],
    *,
    detect_impl: str | None = "synthetic",
    oracle_input: bool = False,
    sequence: str = "SEQ-1",
    stride: int = 1,
    iou: float = 0.5,
) -> dict:
    return {
        "sequence": sequence,
        "sample_stride": stride,
        "iou_threshold": iou,
        "instances": instances,
        "attribution": {"detect_impl": detect_impl, "oracle_input": oracle_input},
    }


# ---------------------------------------------------------------------------
# match_instances
# ---------------------------------------------------------------------------


def test_match_instances_greedy_one_to_one_closest_first():
    a = [{"t": 1.0}, {"t": 2.0}]
    b = [{"t": 1.9}, {"t": 1.1}]
    # closest pairs first: (a0, b1) dt=0.1, (a1, b0) dt=0.1... then (a1,b0)=0.1
    pairs = match_instances(a, b, tol_s=1.0)
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_match_instances_respects_tolerance():
    a = [{"t": 1.0}]
    b = [{"t": 3.0}]
    assert match_instances(a, b, tol_s=1.0) == []


def test_match_instances_never_double_matches():
    a = [{"t": 1.0}, {"t": 1.05}]
    b = [{"t": 1.02}]
    pairs = match_instances(a, b, tol_s=1.0)
    assert len(pairs) == 1
    assert pairs[0] == (0, 0)  # a0 is closer (0.02 < 0.03)


# ---------------------------------------------------------------------------
# detect_context
# ---------------------------------------------------------------------------


def test_detect_context_no_config_is_unknown_never_oracle():
    ctx = detect_context({"video": {"fps": 25.0}})
    assert ctx == {"detect_impl": None, "oracle_input": False}


def test_detect_context_pristine_oracle():
    manifest = {"config": {"stages": {"detect": {"impl": "oracle", "params": {}}}}}
    assert detect_context(manifest) == {"detect_impl": "oracle", "oracle_input": True}


def test_detect_context_degraded_oracle_is_not_oracle_input():
    manifest = {
        "config": {"stages": {"detect": {"impl": "oracle", "params": {"dropout_rate": 0.1}}}}
    }
    assert detect_context(manifest) == {"detect_impl": "oracle", "oracle_input": False}
    manifest = {
        "config": {"stages": {"detect": {"impl": "oracle", "params": {"jitter_px": 2.0}}}}
    }
    assert detect_context(manifest)["oracle_input"] is False


def test_detect_context_non_oracle_detector():
    manifest = {"config": {"stages": {"detect": {"impl": "rf-detr", "params": {}}}}}
    assert detect_context(manifest) == {"detect_impl": "rf-detr", "oracle_input": False}


# ---------------------------------------------------------------------------
# attribute_switches: single-run evidence
# ---------------------------------------------------------------------------


def test_tracklet_switch_without_oracle_is_ambiguous():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    attribute_switches(payload)
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "ambiguous"
    assert att["evidence"][0]["kind"] == "insufficient_evidence"


def test_tracklet_switch_on_pristine_oracle_run_is_online_association():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)], detect_impl="oracle", oracle_input=True)
    attribute_switches(payload)
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "online_association"
    assert att["evidence"][0]["kind"] == "oracle_input"


def test_entity_switch_with_tracklet_counterpart_inherits():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 5, 1, 1, 100011)])
    attribute_switches(payload)
    by_level = {i["level"]: i for i in payload["instances"]}
    assert by_level["tracklet"]["attribution"]["layer"] == "ambiguous"
    ent = by_level["entity"]["attribution"]
    assert ent["layer"] == "ambiguous"  # inherited
    assert ent["evidence"][0]["kind"] == "tracklet_counterpart"
    assert ent["evidence"][0]["frame_idx"] == 5


def test_entity_switch_counterpart_matches_within_tolerance_not_exact_frame():
    # entity switch one frame later than the tracklet switch (0.04 s at 25 fps)
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 6, 1, 1, 100011)])
    attribute_switches(payload)
    ent = next(i for i in payload["instances"] if i["level"] == "entity")
    assert ent["attribution"]["evidence"][0]["kind"] == "tracklet_counterpart"


def test_entity_only_switch_is_offline_association():
    payload = _payload([_inst("entity", 7, 2, 1, 2)])
    attribute_switches(payload)
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "offline_association"
    assert att["evidence"][0]["kind"] == "entity_only"


def test_tracklet_counterpart_is_consumed_one_to_one():
    # Two entity switches near one tracklet switch: only the closest inherits,
    # the other is association-introduced.
    payload = _payload(
        [
            _inst("tracklet", 5, 1, 10, 11),
            _inst("entity", 5, 1, 1, 100011),
            _inst("entity", 9, 1, 100011, 3),
        ]
    )
    attribute_switches(payload)
    ents = sorted(
        (i for i in payload["instances"] if i["level"] == "entity"), key=lambda i: i["frame_idx"]
    )
    assert ents[0]["attribution"]["evidence"][0]["kind"] == "tracklet_counterpart"
    assert ents[1]["attribution"]["layer"] == "offline_association"


def test_counterpart_matching_is_per_gt_track():
    # Same-time switches on DIFFERENT GT tracks never cross-match.
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 5, 2, 1, 2)])
    attribute_switches(payload)
    ent = next(i for i in payload["instances"] if i["level"] == "entity")
    assert ent["attribution"]["layer"] == "offline_association"


def test_counts_and_context_block():
    payload = _payload(
        [
            _inst("tracklet", 5, 1, 10, 11),
            _inst("entity", 5, 1, 1, 100011),
            _inst("entity", 20, 2, 1, 2),
        ]
    )
    attribute_switches(payload)
    ctx = payload["attribution"]
    assert ctx["tol_s"] == 1.0
    assert ctx["oracle_comparison"] is None
    assert ctx["counts"] == {
        "tracklet": {"ambiguous": 1},
        "entity": {"ambiguous": 1, "offline_association": 1},
    }


def test_attribute_switches_is_idempotent():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 20, 2, 1, 2)])
    attribute_switches(payload)
    first = [dict(i["attribution"]) for i in payload["instances"]]
    attribute_switches(payload)
    assert [i["attribution"] for i in payload["instances"]] == first


def test_missing_context_block_refuses():
    payload = {"sequence": "SEQ-1", "instances": [_inst("tracklet", 5, 1, 10, 11)]}
    with pytest.raises(ValueError, match="attribution context"):
        attribute_switches(payload)


def test_every_instance_gets_an_attribution():
    payload = _payload(
        [_inst("tracklet", 3, 1, 10, 11), _inst("tracklet", 8, 2, 20, 21), _inst("entity", 12, 3, 1, 2)]
    )
    attribute_switches(payload)
    assert all("attribution" in i for i in payload["instances"])
    assert all(i["attribution"]["layer"] for i in payload["instances"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/pitchlab_core/tests/test_attribution.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'pitchlab_core.attribution'`

- [ ] **Step 3: Write the module**

```python
"""Evidence-based layer attribution for ID-switch instances (SPO-19 /
tracklet-modernization Phase 0 exit criteria).

Annotates each `eval.json::instances` record with which pipeline layer broke
identity -- detection, online association, refinement, offline association --
or an EXPLICIT `ambiguous` tag. Ambiguous is a first-class honest outcome:
no rule here ever defaults to a specific layer when evidence is insufficient
(the PRD's "evidence-based attribution or explicit ambiguous, never a guess").

Evidence rules, most-specific first:

Tracklet-level switches (only detection or the online tracker can create one):
  1. The run consumed PRISTINE oracle detections (detect impl "oracle",
     dropout_rate == 0, jitter_px == 0) -> online_association: detection is
     eliminated by construction.
  2. An oracle-run comparison is provided (`oracle_eval`): match this run's
     tracklet-level switches to the oracle run's per GT track by nearest-t
     (greedy, one-to-one, <= tol_s). Disappears under oracle -> detection;
     persists -> online_association. This is categorization support, not
     proof -- the matched oracle instance is recorded so the claim is
     inspectable.
  3. Neither -> ambiguous (insufficient_evidence).

Entity-level switches:
  4. A tracklet-level switch on the same GT track matches within tol_s ->
     the switch pre-exists association; it INHERITS that counterpart's layer
     (motmetrics frame assignment can shift a frame between levels in crowded
     scenes, hence tolerance matching, one-to-one, never exact-frame-only).
  5. No counterpart -> offline_association: association introduced it.

"refinement" is a RESERVED layer value: no refined-tracklet artifact exists
yet (PRD Phase 4). When it lands, the cascade gains a refined level between
tracklet and entity; nothing here emits it today.

`attribute_switches` is idempotent (recomputes from scratch) and operates on
plain eval-result dicts so an already-written eval.json can be re-attributed
(oracle enrichment) without re-scoring the run. The context block seeded by
`detect_context` makes the payload self-describing -- oracle enrichment
REFUSES an oracle payload that does not identify itself as a pristine oracle
run, rather than trusting the caller's word.
"""

from __future__ import annotations

DEFAULT_TOL_S = 1.0

LAYERS = (
    "detection",
    "online_association",
    "refinement",  # reserved: Phase 4 refined-tracklet layer, never emitted today
    "offline_association",
    "ambiguous",
)


def match_instances(
    a_list: list[dict], b_list: list[dict], tol_s: float = DEFAULT_TOL_S
) -> list[tuple[int, int]]:
    """Greedy nearest-`t` one-to-one matching between two lists of switch
    instances (already restricted to one grouping key by the caller), closest
    pairs first, each instance matched at most once, pairs farther apart than
    `tol_s` never match. Returns (index_in_a, index_in_b) pairs in match
    order. Extracted from `pitchlab_server.evaluation.diff_switch_instances`
    (same semantics; stable sort keeps generation order for equal distances)
    so the diff view and attribution share one matcher."""
    candidates = sorted(
        (
            (abs(a["t"] - b["t"]), i, j)
            for i, a in enumerate(a_list)
            for j, b in enumerate(b_list)
            if abs(a["t"] - b["t"]) <= tol_s
        ),
        key=lambda c: c[0],
    )
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _dt, i, j in candidates:
        if i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        pairs.append((i, j))
    return pairs


def detect_context(manifest: dict) -> dict:
    """Derive the attribution context block from a run manifest dict: which
    detect impl produced the detections, and whether they were PRISTINE
    oracle (GT-box) detections. Absent/unknown config -> impl None and
    oracle_input False -- unknown never upgrades to a claim. Non-zero
    dropout/jitter knobs mean deliberately degraded detections, which is NOT
    oracle input (rule 1 must not fire on a sensitivity-analysis run)."""
    detect_cfg = manifest.get("config", {}).get("stages", {}).get("detect", {}) or {}
    impl = detect_cfg.get("impl")
    params = detect_cfg.get("params") or {}
    oracle_input = (
        impl == "oracle"
        and float(params.get("dropout_rate", 0.0) or 0.0) == 0.0
        and float(params.get("jitter_px", 0.0) or 0.0) == 0.0
    )
    return {"detect_impl": impl, "oracle_input": oracle_input}


def attribute_switches(
    result: dict,
    *,
    oracle_eval: dict | None = None,
    oracle_run_id: str | None = None,
    tol_s: float = DEFAULT_TOL_S,
) -> None:
    """(Re)compute every instance's `attribution` in place -- idempotent,
    drops prior attributions first. `result` must carry the context block
    seeded by `detect_context` (evaluate_run does this); a payload without
    one refuses loudly rather than assuming a detector identity."""
    ctx = result.get("attribution")
    if not isinstance(ctx, dict) or "oracle_input" not in ctx:
        raise ValueError(
            "eval payload has no attribution context block "
            "(result['attribution'] with 'oracle_input'); it must be seeded "
            "via detect_context() -- re-evaluate the run"
        )
    if oracle_eval is not None:
        _validate_oracle_comparison(result, oracle_eval, oracle_run_id)

    instances = result.get("instances", [])
    for inst in instances:
        inst.pop("attribution", None)
    tracklet_insts = [i for i in instances if i["level"] == "tracklet"]
    entity_insts = [i for i in instances if i["level"] == "entity"]

    if ctx["oracle_input"]:
        for inst in tracklet_insts:
            inst["attribution"] = {
                "layer": "online_association",
                "evidence": [
                    {
                        "kind": "oracle_input",
                        "detail": (
                            "run consumed pristine ground-truth (oracle) detections; "
                            "the detection layer is eliminated by construction"
                        ),
                    }
                ],
            }
    elif oracle_eval is not None:
        _attribute_tracklet_via_oracle(tracklet_insts, oracle_eval, oracle_run_id, tol_s)
    else:
        for inst in tracklet_insts:
            inst["attribution"] = {
                "layer": "ambiguous",
                "evidence": [
                    {
                        "kind": "insufficient_evidence",
                        "detail": (
                            "no oracle-detection comparison available to separate "
                            "detection from online association"
                        ),
                    }
                ],
            }

    _attribute_entity_via_counterparts(entity_insts, tracklet_insts, tol_s)

    ctx["tol_s"] = tol_s
    ctx["oracle_comparison"] = (
        {"oracle_run": oracle_run_id} if oracle_eval is not None else None
    )
    counts: dict[str, dict[str, int]] = {"tracklet": {}, "entity": {}}
    for inst in instances:
        bucket = counts[inst["level"]]
        layer = inst["attribution"]["layer"]
        bucket[layer] = bucket.get(layer, 0) + 1
    ctx["counts"] = counts


def _group_by_gt(instances: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for inst in instances:
        groups.setdefault(inst["gt_track_id"], []).append(inst)
    return groups


def _attribute_tracklet_via_oracle(
    tracklet_insts: list[dict], oracle_eval: dict, oracle_run_id: str | None, tol_s: float
) -> None:
    oracle_tracklet = [i for i in oracle_eval["instances"] if i["level"] == "tracklet"]
    oracle_by_gt = _group_by_gt(oracle_tracklet)
    for gt_id, insts in _group_by_gt(tracklet_insts).items():
        others = oracle_by_gt.get(gt_id, [])
        matched = dict(match_instances(insts, others, tol_s))
        for idx, inst in enumerate(insts):
            if idx in matched:
                o = others[matched[idx]]
                inst["attribution"] = {
                    "layer": "online_association",
                    "evidence": [
                        {
                            "kind": "oracle_comparison",
                            "outcome": "persists",
                            "oracle_run": oracle_run_id,
                            "oracle_frame_idx": o["frame_idx"],
                            "oracle_t": o["t"],
                            "tol_s": tol_s,
                        }
                    ],
                }
            else:
                inst["attribution"] = {
                    "layer": "detection",
                    "evidence": [
                        {
                            "kind": "oracle_comparison",
                            "outcome": "disappears",
                            "oracle_run": oracle_run_id,
                            "tol_s": tol_s,
                            "detail": (
                                "no tracklet-level switch on this GT track within "
                                "tol_s in the oracle-detections run"
                            ),
                        }
                    ],
                }


def _attribute_entity_via_counterparts(
    entity_insts: list[dict], tracklet_insts: list[dict], tol_s: float
) -> None:
    tr_by_gt = _group_by_gt(tracklet_insts)
    for gt_id, insts in _group_by_gt(entity_insts).items():
        counterparts = tr_by_gt.get(gt_id, [])
        matched = dict(match_instances(insts, counterparts, tol_s))
        for idx, inst in enumerate(insts):
            if idx in matched:
                c = counterparts[matched[idx]]
                inst["attribution"] = {
                    "layer": c["attribution"]["layer"],
                    "evidence": [
                        {
                            "kind": "tracklet_counterpart",
                            "frame_idx": c["frame_idx"],
                            "t": c["t"],
                            "detail": (
                                "switch pre-exists at the tracklet layer; offline "
                                "association carried it through, so it inherits that "
                                "layer's attribution"
                            ),
                        }
                    ],
                }
            else:
                inst["attribution"] = {
                    "layer": "offline_association",
                    "evidence": [
                        {
                            "kind": "entity_only",
                            "detail": (
                                "no tracklet-level switch on this GT track within "
                                "tol_s; the switch was introduced by offline association"
                            ),
                        }
                    ],
                }


def _validate_oracle_comparison(
    result: dict, oracle_eval: dict, oracle_run_id: str | None
) -> None:
    """Loud refusals for an unusable oracle comparison -- never a silently
    degraded attribution. The oracle payload must self-describe as a pristine
    oracle-input run via ITS OWN attribution context (an old eval.json
    without one must be re-evaluated first)."""
    if "instances" not in oracle_eval:
        raise ValueError(
            f"oracle eval payload ({oracle_run_id!r}) has no 'instances' -- not a "
            "scored eval.json"
        )
    octx = oracle_eval.get("attribution")
    if not isinstance(octx, dict) or octx.get("oracle_input") is not True:
        raise ValueError(
            f"oracle eval payload ({oracle_run_id!r}) does not identify itself as a "
            "pristine oracle-detections run (attribution.oracle_input != True); "
            "re-evaluate the oracle run so its context block is present, and check "
            "its detect stage is impl 'oracle' with dropout_rate=0 and jitter_px=0"
        )
    if result["attribution"].get("oracle_input"):
        raise ValueError(
            "target run itself consumed pristine oracle detections; its switches are "
            "already conclusively attributed -- comparing oracle to oracle is a "
            "caller error"
        )
    for key in ("sequence", "sample_stride", "iou_threshold"):
        if result.get(key) != oracle_eval.get(key):
            raise ValueError(
                f"oracle comparison mismatch on {key!r}: "
                f"{result.get(key)!r} != {oracle_eval.get(key)!r} -- the runs are "
                "not comparable"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/pitchlab_core/tests/test_attribution.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pitchlab_core/src/pitchlab_core/attribution.py packages/pitchlab_core/tests/test_attribution.py
git commit -m "Add switch layer-attribution module: matcher, context, single-run rules (SPO-19)"
```

---

### Task 2: Oracle-run comparison + refusals

**Files:**
- Modify: `packages/pitchlab_core/tests/test_attribution.py` (append)
- Modify: `packages/pitchlab_core/src/pitchlab_core/attribution.py` (already written in Task 1 — this task only adds tests; fix the module if any fail)

**Interfaces:**
- Consumes: Task 1's `attribute_switches(result, oracle_eval=..., oracle_run_id=..., tol_s=...)`.
- Produces: verified oracle-comparison behavior later tasks rely on.

- [ ] **Step 1: Append the failing/verifying tests**

```python
# ---------------------------------------------------------------------------
# attribute_switches: oracle-run comparison
# ---------------------------------------------------------------------------


def _oracle_payload(instances: list[dict], **kw) -> dict:
    kw.setdefault("detect_impl", "oracle")
    kw.setdefault("oracle_input", True)
    return _payload(instances, **kw)


def test_oracle_comparison_disappears_attributes_detection():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    oracle = _oracle_payload([])  # clean oracle run: switch disappears
    attribute_switches(payload, oracle_eval=oracle, oracle_run_id="oracle-run-1")
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "detection"
    ev = att["evidence"][0]
    assert ev["kind"] == "oracle_comparison"
    assert ev["outcome"] == "disappears"
    assert ev["oracle_run"] == "oracle-run-1"
    assert payload["attribution"]["oracle_comparison"] == {"oracle_run": "oracle-run-1"}


def test_oracle_comparison_persists_attributes_online_association():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    oracle = _oracle_payload([_inst("tracklet", 6, 1, 50, 51)])  # 0.04 s away
    attribute_switches(payload, oracle_eval=oracle, oracle_run_id="oracle-run-1")
    att = payload["instances"][0]["attribution"]
    assert att["layer"] == "online_association"
    ev = att["evidence"][0]
    assert ev["outcome"] == "persists"
    assert ev["oracle_frame_idx"] == 6


def test_oracle_comparison_matches_per_gt_track_only():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    oracle = _oracle_payload([_inst("tracklet", 5, 2, 50, 51)])  # other GT track
    attribute_switches(payload, oracle_eval=oracle, oracle_run_id="o")
    assert payload["instances"][0]["attribution"]["layer"] == "detection"


def test_oracle_comparison_updates_entity_inheritance():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11), _inst("entity", 5, 1, 1, 100011)])
    attribute_switches(payload)  # baseline: both ambiguous
    oracle = _oracle_payload([])
    attribute_switches(payload, oracle_eval=oracle, oracle_run_id="o")
    by_level = {i["level"]: i for i in payload["instances"]}
    assert by_level["tracklet"]["attribution"]["layer"] == "detection"
    assert by_level["entity"]["attribution"]["layer"] == "detection"  # re-inherited


def test_oracle_refusal_payload_not_marked_oracle():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    not_oracle = _payload([])  # oracle_input False
    with pytest.raises(ValueError, match="does not identify itself"):
        attribute_switches(payload, oracle_eval=not_oracle, oracle_run_id="o")


def test_oracle_refusal_missing_instances():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    with pytest.raises(ValueError, match="instances"):
        attribute_switches(
            payload, oracle_eval={"attribution": {"oracle_input": True}}, oracle_run_id="o"
        )


def test_oracle_refusal_target_is_itself_oracle():
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)], detect_impl="oracle", oracle_input=True)
    oracle = _oracle_payload([])
    with pytest.raises(ValueError, match="oracle to oracle"):
        attribute_switches(payload, oracle_eval=oracle, oracle_run_id="o")


@pytest.mark.parametrize(
    ("field", "value"),
    [("sequence", "SEQ-2"), ("sample_stride", 2), ("iou_threshold", 0.4)],
)
def test_oracle_refusal_on_incomparable_payloads(field, value):
    payload = _payload([_inst("tracklet", 5, 1, 10, 11)])
    kw = {"sequence": "SEQ-1", "stride": 1, "iou": 0.5}
    kw[{"sequence": "sequence", "sample_stride": "stride", "iou_threshold": "iou"}[field]] = value
    oracle = _oracle_payload([], **kw)
    with pytest.raises(ValueError, match=field):
        attribute_switches(payload, oracle_eval=oracle, oracle_run_id="o")
```

- [ ] **Step 2: Run the full attribution test file**

Run: `uv run pytest packages/pitchlab_core/tests/test_attribution.py -q`
Expected: all PASS (Task 1's module already implements this; fix it if not)

- [ ] **Step 3: Commit**

```bash
git add packages/pitchlab_core/tests/test_attribution.py
git commit -m "Test oracle-run comparison attribution + refusals (SPO-19)"
```

---

### Task 3: Wire attribution into evaluate_run + integration tests

**Files:**
- Modify: `packages/pitchlab_core/src/pitchlab_core/evaluation.py` (result assembly in `evaluate_run`, around line 176-192, and the module docstring)
- Modify: `packages/pitchlab_core/tests/test_attribution.py` (append integration tests)

**Interfaces:**
- Consumes: `detect_context`, `attribute_switches` from Task 1.
- Produces: every eval result from `evaluate_run` carries `result["attribution"]` (context + counts) and per-instance `attribution` — the contract Tasks 4–6 rely on.

- [ ] **Step 1: Append failing integration tests** (handcrafted run dirs, mirrors `test_gt_eval.py` fixtures)

```python
# ---------------------------------------------------------------------------
# integration through evaluate_run (handcrafted run dirs, known causes)
# ---------------------------------------------------------------------------

import json  # noqa: E402  (top of file if not already imported)
from pathlib import Path  # noqa: E402


def _write_soccernet_seq(root: Path) -> Path:
    seq = root / "SNMOT-001"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nimDir=img1\nframeRate=25\nseqLength=10\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )
    (seq / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nnum_tracklets=2\n"
        "trackletID_1= player team left;10\n"
        "trackletID_2= player team right;7\n"
    )
    rows = []
    for frame in range(1, 11):  # 1-based MOT frames
        rows.append(f"{frame},1,100,100,40,120,1,-1,-1,-1")
        rows.append(f"{frame},2,500,200,40,120,1,-1,-1,-1")
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))
    return seq


def _tracklet(tid: int, frames: list[tuple[int, float, float]]) -> dict:
    return {
        "tracklet_id": tid,
        "cls": "player",
        "frames": [
            {
                "frame_idx": f,
                "box": {"x1": x, "y1": y, "x2": x + 40, "y2": y + 120},
                "confidence": 0.9,
            }
            for f, x, y in frames
        ],
    }


def _write_run_dir(
    root: Path, name: str, tracklets: list[dict], players: list[dict], config: dict | None = None
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    manifest: dict = {"video": {"fps": 25.0, "frame_count": 10, "sample_stride": 1}}
    if config is not None:
        manifest["config"] = config
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps(players))
    return run_dir


def test_evaluate_run_attributes_every_switch(tmp_path):
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run
    from pitchlab_core.gt import load_soccernet_sequence

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    # GT1 fragments into tracklets 10 and 11 (tracklet switch at frame 5);
    # the associator does NOT merge them, so the switch persists at entity
    # level too. GT2 is tracked cleanly.
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 5)]),
        _tracklet(11, [(f, 100, 100) for f in range(5, 10)]),
        _tracklet(12, [(f, 500, 200) for f in range(0, 10)]),
    ]
    players = [{"player_id": 1, "tracklet_ids": [10], "team": "home"},
               {"player_id": 2, "tracklet_ids": [11], "team": "home"},
               {"player_id": 3, "tracklet_ids": [12], "team": "away"}]
    run_dir = _write_run_dir(tmp_path, "run-frag", tracklets, players)

    result = evaluate_run(run_dir, gt)

    assert result["attribution"]["detect_impl"] is None  # no config in manifest
    assert result["attribution"]["oracle_input"] is False
    assert result["instances"], "expected at least one switch"
    for inst in result["instances"]:
        assert inst["attribution"]["layer"] in (
            "detection", "online_association", "refinement", "offline_association", "ambiguous",
        )
        assert inst["attribution"]["evidence"]
    tracklet_switches = [i for i in result["instances"] if i["level"] == "tracklet"]
    assert all(i["attribution"]["layer"] == "ambiguous" for i in tracklet_switches)
    entity_switches = [i for i in result["instances"] if i["level"] == "entity"]
    assert all(
        i["attribution"]["evidence"][0]["kind"] == "tracklet_counterpart"
        for i in entity_switches
    )
    counts = result["attribution"]["counts"]
    assert counts["tracklet"].get("ambiguous", 0) == len(tracklet_switches)


def test_evaluate_run_oracle_manifest_attributes_online_association(tmp_path):
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run
    from pitchlab_core.gt import load_soccernet_sequence

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 5)]),
        _tracklet(11, [(f, 100, 100) for f in range(5, 10)]),
    ]
    config = {"stages": {"detect": {"impl": "oracle", "params": {}}}}
    run_dir = _write_run_dir(tmp_path, "run-oracle", tracklets, [], config=config)

    result = evaluate_run(run_dir, gt)

    assert result["attribution"]["oracle_input"] is True
    tracklet_switches = [i for i in result["instances"] if i["level"] == "tracklet"]
    assert tracklet_switches
    assert all(
        i["attribution"]["layer"] == "online_association" for i in tracklet_switches
    )


def test_evaluate_run_degraded_oracle_manifest_stays_ambiguous(tmp_path):
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run
    from pitchlab_core.gt import load_soccernet_sequence

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    tracklets = [
        _tracklet(10, [(f, 100, 100) for f in range(0, 5)]),
        _tracklet(11, [(f, 100, 100) for f in range(5, 10)]),
    ]
    config = {"stages": {"detect": {"impl": "oracle", "params": {"dropout_rate": 0.2}}}}
    run_dir = _write_run_dir(tmp_path, "run-degraded", tracklets, [], config=config)

    result = evaluate_run(run_dir, gt)
    assert result["attribution"]["oracle_input"] is False
    tracklet_switches = [i for i in result["instances"] if i["level"] == "tracklet"]
    assert all(i["attribution"]["layer"] == "ambiguous" for i in tracklet_switches)


def test_end_to_end_enrichment_flips_ambiguous_to_detection(tmp_path):
    pytest.importorskip("motmetrics")
    from pitchlab_core.evaluation import evaluate_run
    from pitchlab_core.gt import load_soccernet_sequence

    gt = load_soccernet_sequence(_write_soccernet_seq(tmp_path))
    # Baseline run fragments GT1 (switch); oracle run tracks GT1 cleanly ->
    # the switch disappears under oracle detections -> detection-attributed.
    baseline_dir = _write_run_dir(
        tmp_path,
        "run-base",
        [
            _tracklet(10, [(f, 100, 100) for f in range(0, 5)]),
            _tracklet(11, [(f, 100, 100) for f in range(5, 10)]),
        ],
        [],
    )
    oracle_dir = _write_run_dir(
        tmp_path,
        "run-oracle",
        [_tracklet(20, [(f, 100, 100) for f in range(0, 10)])],
        [],
        config={"stages": {"detect": {"impl": "oracle", "params": {}}}},
    )

    baseline_eval = evaluate_run(baseline_dir, gt)
    oracle_eval = evaluate_run(oracle_dir, gt)

    attribute_switches(baseline_eval, oracle_eval=oracle_eval, oracle_run_id="run-oracle")
    tracklet_switches = [i for i in baseline_eval["instances"] if i["level"] == "tracklet"]
    assert tracklet_switches
    assert all(i["attribution"]["layer"] == "detection" for i in tracklet_switches)
```

Note: `import json` / `from pathlib import Path` go at the top of the file with the
existing imports, not mid-file (the `# noqa` markers above are only plan notation).

- [ ] **Step 2: Run to verify the integration tests fail**

Run: `uv run pytest packages/pitchlab_core/tests/test_attribution.py -q`
Expected: the four new tests FAIL with `KeyError: 'attribution'` (evaluate_run doesn't seed it yet)

- [ ] **Step 3: Wire into evaluate_run**

In `packages/pitchlab_core/src/pitchlab_core/evaluation.py`, inside `evaluate_run`, right
after the `result = {...}` dict literal (currently ending with the `"identity":` entry) and
BEFORE the `merge_quality` call, add:

```python
    from pitchlab_core.attribution import attribute_switches, detect_context

    # SPO-19: every switch instance carries a layer attribution or an explicit
    # ambiguous tag. Single-run evidence only here; callers holding an
    # oracle-counterpart eval payload re-invoke attribute_switches to upgrade
    # ambiguous tracklet-level switches (see pitchlab_core.attribution).
    result["attribution"] = detect_context(manifest)
    attribute_switches(result)
```

(Import at use-site keeps the existing lazy-import convention of this module.)

Also append one sentence to the module docstring (after the SPO-9 paragraph):

```
A seventh annotation pass (SPO-19) attributes every per-instance ID switch
to a pipeline layer -- detection, online association, refinement (reserved),
offline association -- or an explicit `ambiguous` tag, with the evidence
basis recorded per instance; see `pitchlab_core.attribution`.
```

- [ ] **Step 4: Run core tests**

Run: `uv run pytest packages/pitchlab_core/tests/test_attribution.py packages/pitchlab_core/tests/test_gt_eval.py -q`
Expected: all PASS (existing gt_eval assertions are key-specific, not exhaustive — verify none broke)

- [ ] **Step 5: Commit**

```bash
git add packages/pitchlab_core/src/pitchlab_core/evaluation.py packages/pitchlab_core/tests/test_attribution.py
git commit -m "Attribute every ID switch in evaluate_run output (SPO-19)"
```

---

### Task 4: Server — shared matcher + oracle_run_id on the evaluate endpoint

**Files:**
- Modify: `packages/pitchlab_server/src/pitchlab_server/evaluation.py` (`diff_switch_instances` inner matching; `evaluate_run_against_gt` signature)
- Modify: `packages/pitchlab_server/src/pitchlab_server/api/runs.py` (`evaluate_run` endpoint)
- Modify: `packages/pitchlab_server/tests/test_api.py` (append endpoint test)

**Interfaces:**
- Consumes: `match_instances`, `attribute_switches` from `pitchlab_core.attribution`.
- Produces: `evaluate_run_against_gt(run, video, oracle_eval: dict | None = None, oracle_run_id: str | None = None)`;
  `POST /api/runs/{run_id}/evaluate?oracle_run_id=<id>`.

- [ ] **Step 1: Refactor diff_switch_instances onto the shared matcher**

In `packages/pitchlab_server/src/pitchlab_server/evaluation.py`, replace the inner
candidate/matched loop of `diff_switch_instances` (the `candidates = sorted(...)` block
through the two `matched_a`/`matched_b` loops) with:

```python
        pairs = match_instances(a_list, b_list, tol_s)
        matched_a = {i for i, _ in pairs}
        matched_b = {j for _, j in pairs}
        persisted.extend({"a": a_list[i], "b": b_list[j]} for i, j in pairs)
```

with `from pitchlab_core.attribution import match_instances` added to the imports, and a
docstring note that matching is shared with the attribution module. Delete the now-unused
`from collections import defaultdict`? No — `_grouped` still uses it; leave it.

- [ ] **Step 2: Run existing server tests to confirm no behavior change**

Run: `uv run pytest packages/pitchlab_server/tests/test_evaluation.py packages/pitchlab_server/tests/test_api.py -q`
Expected: all PASS

- [ ] **Step 3: Append the failing endpoint test**

Append to `packages/pitchlab_server/tests/test_api.py`:

```python
def test_evaluate_endpoint_with_oracle_run_id(client, video_id):
    """POST /runs/{id}/evaluate?oracle_run_id=... enriches attribution from a
    scored oracle run of the same video; refusals surface as 422."""
    pytest.importorskip("motmetrics")
    from pitchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
    from pitchlab_core.schemas.geometry import Box
    from pitchlab_server.db import session
    from pitchlab_server.models import Run, Video

    # Attach a tiny GT to the shared demo video (120 frames at 20 fps).
    with session() as db:
        video = db.get(Video, video_id)
        frame_count = 120
        gt = GroundTruth(
            source="test",
            sequence="clip",
            fps=20.0,
            width=960,
            height=540,
            seq_length=frame_count,
            tracks=[
                GroundTruthTrack(
                    track_id=1,
                    role="player",
                    frames=[
                        GroundTruthFrame(frame_idx=f, box=Box(x1=100, y1=100, x2=140, y2=220))
                        for f in range(frame_count)
                    ],
                )
            ],
        )
        gt_path = Path(os.environ["PITCHLAB_DATA_DIR"]) / "clip.gt.json"
        gt_path.write_text(gt.model_dump_json())
        video.gt_path = str(gt_path)
        db.commit()

    ids = []
    for _ in range(2):
        resp = client.post(
            "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
        )
        ids.append(resp.json()["id"])
        execute_job(claim_next_job())
    run_id, oracle_id = ids

    # Fabricate run dirs whose tracklets exercise attribution: baseline
    # fragments GT1 at frame 60; the "oracle" run tracks it cleanly.
    def _frames(rng):
        return [
            {
                "frame_idx": f,
                "box": {"x1": 100, "y1": 100, "x2": 140, "y2": 220},
                "confidence": 0.9,
            }
            for f in rng
        ]

    with session() as db:
        base_dir = Path(db.get(Run, run_id).run_dir)
        oracle_dir = Path(db.get(Run, oracle_id).run_dir)
    (base_dir / "tracklets.json").write_text(
        json.dumps(
            [
                {"tracklet_id": 10, "cls": "player", "frames": _frames(range(0, 60))},
                {"tracklet_id": 11, "cls": "player", "frames": _frames(range(60, 120))},
            ]
        )
    )
    (base_dir / "players.json").write_text(json.dumps([]))
    (oracle_dir / "tracklets.json").write_text(
        json.dumps([{"tracklet_id": 20, "cls": "player", "frames": _frames(range(0, 120))}])
    )
    (oracle_dir / "players.json").write_text(json.dumps([]))
    # Mark the oracle run's manifest as a pristine oracle-detections run.
    manifest = json.loads((oracle_dir / "manifest.json").read_text())
    manifest["config"]["stages"]["detect"] = {"impl": "oracle", "params": {}, "enabled": True}
    (oracle_dir / "manifest.json").write_text(json.dumps(manifest))

    # Oracle run must be evaluated first (its eval.json self-describes).
    resp = client.post(f"/api/runs/{run_id}/evaluate", params={"oracle_run_id": oracle_id})
    assert resp.status_code == 422
    assert "no eval.json" in resp.json()["detail"]

    assert client.post(f"/api/runs/{oracle_id}/evaluate").status_code == 200
    oracle_eval = json.loads((oracle_dir / "eval.json").read_text())
    assert oracle_eval["attribution"]["oracle_input"] is True

    resp = client.post(f"/api/runs/{run_id}/evaluate", params={"oracle_run_id": oracle_id})
    assert resp.status_code == 200, resp.text
    enriched = json.loads((base_dir / "eval.json").read_text())
    assert enriched["attribution"]["oracle_comparison"] == {"oracle_run": oracle_id}
    tracklet_switches = [i for i in enriched["instances"] if i["level"] == "tracklet"]
    assert tracklet_switches
    assert all(i["attribution"]["layer"] == "detection" for i in tracklet_switches)

    # Unknown oracle run -> 404; wrong video -> 422.
    assert (
        client.post(f"/api/runs/{run_id}/evaluate", params={"oracle_run_id": "nope"}).status_code
        == 404
    )
```

- [ ] **Step 4: Run to verify it fails**

Run: `uv run pytest packages/pitchlab_server/tests/test_api.py::test_evaluate_endpoint_with_oracle_run_id -q`
Expected: FAIL (endpoint ignores `oracle_run_id`; first assert on 422 fails with 200)

- [ ] **Step 5: Implement server changes**

`packages/pitchlab_server/src/pitchlab_server/evaluation.py` — extend
`evaluate_run_against_gt`:

```python
def evaluate_run_against_gt(
    run: Run,
    video: Video,
    oracle_eval: dict | None = None,
    oracle_run_id: str | None = None,
) -> dict | None:
```

and after `result = evaluate_run(run_dir, gt)` insert:

```python
    if oracle_eval is not None:
        from pitchlab_core.attribution import attribute_switches

        attribute_switches(result, oracle_eval=oracle_eval, oracle_run_id=oracle_run_id)
```

(the existing `eval.json` write below then persists the enriched payload). Document the
params in the docstring: "oracle_eval/oracle_run_id: an already-scored eval.json payload
from a pristine oracle-detections run of the same video, used to upgrade ambiguous
tracklet-level switch attributions; refusals (pitchlab_core.attribution) raise ValueError."

`packages/pitchlab_server/src/pitchlab_server/api/runs.py` — replace the endpoint body:

```python
@router.post("/{run_id}/evaluate", response_model=RunDetailOut)
def evaluate_run(run_id: str, oracle_run_id: str | None = None, db: Session = Depends(get_db)):
    """(Re-)score a completed run against its video's ground truth. Writes the
    eval.json artifact and folds headline metrics into run.metrics.

    `oracle_run_id` (optional): a scored run of the SAME video that consumed
    pristine oracle detections; its eval.json upgrades this run's ambiguous
    tracklet-level switch attributions via oracle comparison (SPO-19). The
    oracle run must already carry an eval.json -- evaluate it first."""
    from pitchlab_server.evaluation import evaluate_run_against_gt, merged_metrics

    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    video = db.get(Video, run.video_id)
    if video is None or not video.gt_path:
        raise HTTPException(422, "Video has no ground truth to evaluate against")

    oracle_eval = None
    if oracle_run_id is not None:
        oracle = db.get(Run, oracle_run_id)
        if oracle is None:
            raise HTTPException(404, "Oracle run not found")
        if oracle.video_id != run.video_id:
            raise HTTPException(422, "Oracle run must be a run of the same video")
        oracle_eval_path = Path(oracle.run_dir) / ARTIFACT_FILES[ArtifactName.EVAL]
        if not oracle_eval_path.exists():
            raise HTTPException(
                422, f"Oracle run '{oracle_run_id}' has no eval.json -- evaluate it first"
            )
        oracle_eval = json.loads(oracle_eval_path.read_text())

    try:
        result = evaluate_run_against_gt(
            run, video, oracle_eval=oracle_eval, oracle_run_id=oracle_run_id
        )
    except ImportError as exc:
        raise HTTPException(501, "motmetrics not installed (uv sync --group eval)") from exc
    except ValueError as exc:  # attribution refusals (pitchlab_core.attribution)
        raise HTTPException(422, str(exc)) from exc
    if result is None:
        raise HTTPException(422, "Run has no tracklets artifact to score")
    run.metrics = merged_metrics(run, result)
    db.commit()
    return _detail(run, db)
```

(`json`, `Path`, `ARTIFACT_FILES`, `ArtifactName` are already imported in runs.py.)

- [ ] **Step 6: Run server tests**

Run: `uv run pytest packages/pitchlab_server -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add packages/pitchlab_server/src/pitchlab_server/evaluation.py packages/pitchlab_server/src/pitchlab_server/api/runs.py packages/pitchlab_server/tests/test_api.py
git commit -m "Share switch matcher; oracle_run_id enrichment on evaluate endpoint (SPO-19)"
```

---

### Task 5: Benchmark runner — explicit oracle_candidate pairing

**Files:**
- Modify: `packages/pitchlab_train/src/pitchlab_train/experiments/benchmark.py`
- Modify: `packages/pitchlab_train/tests/test_benchmark_runner.py` (append)

**Interfaces:**
- Consumes: `attribute_switches` from `pitchlab_core.attribution`.
- Produces: `PipelineCandidate.oracle_candidate: str | None`;
  `_validate_oracle_candidates(candidates: list) -> None` (called from `_expand_candidates` before returning);
  `_enrich_with_oracle(candidates, rows, workdir: Path) -> None` (called in `run()` after the candidate loop, before `_check_missing_provenance`);
  enriched rows carry `row["attribution_oracle"] = {"status": "enriched", "oracle_run_id": ...}` or `{"status": "unavailable", "reason": ...}`.

- [ ] **Step 1: Append failing validation tests**

```python
# ---------------------------------------------------------------------------
# oracle_candidate (SPO-19)
# ---------------------------------------------------------------------------


def test_oracle_candidate_unknown_name_refuses():
    with pytest.raises(RuntimeError, match="ghost"):
        _expand_candidates(
            [{"name": "base", "config": STUB_CONFIG, "oracle_candidate": "ghost"}], []
        )


def test_oracle_candidate_self_reference_refuses():
    with pytest.raises(RuntimeError, match="itself"):
        _expand_candidates(
            [{"name": "base", "config": ORACLE_CONFIG, "oracle_candidate": "base"}], []
        )


def test_oracle_candidate_must_use_oracle_detect():
    with pytest.raises(RuntimeError, match="oracle"):
        _expand_candidates(
            [
                {"name": "base", "config": STUB_CONFIG, "oracle_candidate": "other"},
                {"name": "other", "config": STUB_CONFIG},
            ],
            [],
        )


def test_oracle_candidate_must_be_pristine():
    with pytest.raises(RuntimeError, match="pristine"):
        _expand_candidates(
            [
                {"name": "base", "config": STUB_CONFIG, "oracle_candidate": "orc"},
                {
                    "name": "orc",
                    "config": ORACLE_CONFIG,
                    "overrides": {"stages.detect.params.dropout_rate": 0.1},
                },
            ],
            [],
        )


def test_oracle_candidate_track_config_must_match():
    with pytest.raises(RuntimeError, match="track"):
        _expand_candidates(
            [
                {
                    "name": "base",
                    "config": STUB_CONFIG,
                    "overrides": {"stages.track.params.max_age_frames": 99},
                    "oracle_candidate": "orc",
                },
                {"name": "orc", "config": ORACLE_CONFIG},
            ],
            [],
        )


def test_oracle_candidate_sweep_on_track_params_refuses():
    # Sweep-derived candidates inherit oracle_candidate; a sweep that mutates
    # track params breaks comparability and must refuse at expansion.
    with pytest.raises(RuntimeError, match="track"):
        _expand_candidates(
            [
                {"name": "base", "config": STUB_CONFIG, "oracle_candidate": "orc"},
                {"name": "orc", "config": ORACLE_CONFIG},
            ],
            [SweepSpec(candidate="base", param="stages.track.params.max_age_frames", values=[30])],
        )


def test_oracle_candidate_valid_pairing_expands():
    candidates = _expand_candidates(
        [
            {"name": "base", "config": STUB_CONFIG, "oracle_candidate": "orc"},
            {"name": "orc", "config": ORACLE_CONFIG},
        ],
        [],
    )
    assert [c.name for c in candidates] == ["base", "orc"]
    assert candidates[0].oracle_candidate == "orc"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest packages/pitchlab_train/tests/test_benchmark_runner.py -q -k oracle_candidate`
Expected: FAIL (`oracle_candidate` is not a PipelineCandidate field → pydantic rejects the extra key... note: pydantic BaseModel ignores or forbids extras depending on config — if it silently ignores, the tests fail on "did not raise", which is equally a failing start)

- [ ] **Step 3: Implement**

In `benchmark.py`:

1. Add to `PipelineCandidate`:

```python
    # SPO-19: names another pipeline candidate that is this candidate's
    # oracle counterpart (same tracker, pristine oracle detections). Explicit
    # only, never inferred; validated in _validate_oracle_candidates.
    oracle_candidate: str | None = None
```

2. Add after `_validate_import_candidate`:

```python
def _validate_oracle_candidates(candidates: list[PipelineCandidate | ImportCandidate]) -> None:
    """Eager validation of every `oracle_candidate` pairing (SPO-19), at
    expansion time: the named candidate must exist, be a pipeline candidate,
    not be the candidate itself, resolve to a PRISTINE oracle detect stage
    (impl 'oracle', dropout_rate == 0, jitter_px == 0), and share an
    identical resolved track stage config and sample_stride with the paired
    candidate -- the oracle counterpart is definitionally "same tracker,
    perfect detections", so anything else refuses loudly here rather than
    producing an incomparable enrichment at scoring time."""
    by_name = {c.name: c for c in candidates}
    for c in candidates:
        if not isinstance(c, PipelineCandidate) or c.oracle_candidate is None:
            continue
        target = by_name.get(c.oracle_candidate)
        if target is None:
            raise RuntimeError(
                f"Candidate '{c.name}': oracle_candidate '{c.oracle_candidate}' is not "
                f"a known candidate (known: {sorted(by_name)})"
            )
        if target.name == c.name:
            raise RuntimeError(
                f"Candidate '{c.name}': oracle_candidate cannot name itself"
            )
        if not isinstance(target, PipelineCandidate):
            raise RuntimeError(
                f"Candidate '{c.name}': oracle_candidate '{target.name}' is an import "
                "candidate -- oracle counterparts must be pipeline candidates"
            )
        oracle_config = _load_pipeline_config(target)
        detect = oracle_config.stages[StageKind.DETECT]
        if detect.impl != "oracle":
            raise RuntimeError(
                f"Candidate '{c.name}': oracle_candidate '{target.name}' resolves to "
                f"detect impl '{detect.impl}', not 'oracle'"
            )
        params = detect.params or {}
        if float(params.get("dropout_rate", 0.0) or 0.0) != 0.0 or (
            float(params.get("jitter_px", 0.0) or 0.0) != 0.0
        ):
            raise RuntimeError(
                f"Candidate '{c.name}': oracle_candidate '{target.name}' is not "
                "pristine (dropout_rate/jitter_px must be 0 for attribution)"
            )
        own_config = _load_pipeline_config(c)
        if own_config.stages[StageKind.TRACK] != oracle_config.stages[StageKind.TRACK]:
            raise RuntimeError(
                f"Candidate '{c.name}' and oracle_candidate '{target.name}' have "
                "different resolved track stage configs -- the oracle counterpart "
                "must run the identical tracker"
            )
        if own_config.video.sample_stride != oracle_config.video.sample_stride:
            raise RuntimeError(
                f"Candidate '{c.name}' and oracle_candidate '{target.name}' have "
                "different sample_stride -- their evals are not comparable"
            )
```

3. Call it at the end of `_expand_candidates`, just before `return expanded`:

```python
    _validate_oracle_candidates(expanded)
    return expanded
```

4. Add the enrichment pass (place near `_row_from_import`):

```python
def _enrich_with_oracle(
    candidates: list[PipelineCandidate | ImportCandidate], rows: list[dict], workdir: Path
) -> None:
    """Oracle-comparison enrichment (SPO-19): for each completed row of a
    candidate with `oracle_candidate` set, re-attribute its eval.json against
    the oracle candidate's completed row on the same sequence and rewrite the
    file in place. A missing/failed oracle row records
    `row["attribution_oracle"] = {"status": "unavailable", ...}` and leaves
    the baseline (ambiguous) attribution -- visible, never silent. Headline
    metrics are untouched; only eval.json instances change."""
    from pitchlab_core.attribution import attribute_switches

    by_name = {c.name: c for c in candidates}
    completed = {
        (r["candidate"], r["sequence"]): r for r in rows if r["status"] == "completed"
    }
    for row in rows:
        cand = by_name.get(row["candidate"])
        if not isinstance(cand, PipelineCandidate) or cand.oracle_candidate is None:
            continue
        if row["status"] != "completed":
            continue
        oracle_row = completed.get((cand.oracle_candidate, row["sequence"]))
        if oracle_row is None:
            row["attribution_oracle"] = {
                "status": "unavailable",
                "reason": (
                    f"no completed row for oracle candidate "
                    f"'{cand.oracle_candidate}' on sequence '{row['sequence']}'"
                ),
            }
            continue
        eval_file = workdir / row["eval_path"]
        result = json.loads(eval_file.read_text())
        oracle_eval = json.loads((workdir / oracle_row["eval_path"]).read_text())
        attribute_switches(result, oracle_eval=oracle_eval, oracle_run_id=oracle_row["run_id"])
        eval_file.write_text(json.dumps(result))
        row["attribution_oracle"] = {"status": "enriched", "oracle_run_id": oracle_row["run_id"]}
```

5. In `BenchmarkExperiment.run()`, insert between the candidate loop and
`_check_missing_provenance(rows)`:

```python
        # Oracle-comparison enrichment (SPO-19) -- after all rows exist so an
        # oracle row is available regardless of candidate order, before any
        # aggregation (attribution never changes headline metrics).
        _enrich_with_oracle(candidates, rows, workdir)
```

- [ ] **Step 4: Run validation tests**

Run: `uv run pytest packages/pitchlab_train/tests/test_benchmark_runner.py -q -k oracle_candidate`
Expected: all PASS

- [ ] **Step 5: Append the failing end-to-end enrichment test**

```python
def test_benchmark_oracle_enrichment_end_to_end(tmp_path):
    pytest.importorskip("motmetrics")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    render_demo_video(videos_dir / "clip-a.mp4", duration_s=2, fps=20, width=960, height=540)
    _write_gt_for_clip(videos_dir / "clip-a.mp4", seq_length=40, fps=20)

    manifest_path = _write_manifest_tree(
        tmp_path,
        [{"name": "clip-a", "video": "videos/clip-a.mp4", "gt": "videos/clip-a.gt.json", "role": "tuning"}],
        write_files=False,
    )

    config = ExperimentConfig(
        name="test-benchmark-oracle",
        task="benchmark",
        params={
            "dataset_manifest": str(manifest_path),
            "roles": ["tuning"],
            "candidates": [
                {"name": "base", "config": STUB_CONFIG, "oracle_candidate": "orc"},
                {"name": "orc", "config": ORACLE_CONFIG},
            ],
            "device": "cpu",
        },
        output_dir=str(tmp_path / "exp"),
    )
    result = build(config.task, config).run()

    by_candidate = {row["candidate"]: row for row in result["rows"]}
    assert by_candidate["base"]["status"] == "completed"
    assert by_candidate["orc"]["status"] == "completed"
    assert by_candidate["base"]["attribution_oracle"] == {
        "status": "enriched",
        "oracle_run_id": "orc-clip-a",
    }
    assert "attribution_oracle" not in by_candidate["orc"]

    workdir = next((tmp_path / "exp").glob("*/result.json")).parent
    base_eval = json.loads((workdir / by_candidate["base"]["eval_path"]).read_text())
    assert base_eval["attribution"]["oracle_comparison"] == {"oracle_run": "orc-clip-a"}
    for inst in base_eval["instances"]:
        assert inst["attribution"]["layer"] in (
            "detection", "online_association", "offline_association", "ambiguous",
        )
        # tracklet-level switches must no longer be ambiguous after enrichment
        if inst["level"] == "tracklet":
            assert inst["attribution"]["layer"] in ("detection", "online_association")
    orc_eval = json.loads((workdir / by_candidate["orc"]["eval_path"]).read_text())
    assert orc_eval["attribution"]["oracle_input"] is True
```

- [ ] **Step 6: Run the end-to-end test**

Run: `uv run pytest packages/pitchlab_train/tests/test_benchmark_runner.py::test_benchmark_oracle_enrichment_end_to_end -q`
Expected: PASS after Step 3's implementation (it was written before this test; if it fails, fix the implementation, not the assertion — with one allowed exception: if the stub run on the demo clip yields zero switches, follow the `test_benchmark_end_to_end` pattern and keep the structural asserts only)

- [ ] **Step 7: Run the full train suite + core/server suites**

Run: `uv run pytest packages -q`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add packages/pitchlab_train/src/pitchlab_train/experiments/benchmark.py packages/pitchlab_train/tests/test_benchmark_runner.py
git commit -m "Benchmark runner: explicit oracle_candidate pairing + eval enrichment (SPO-19)"
```

---

### Task 6: Lab UI — attribution pill + layer filter

**Files:**
- Modify: `web/src/lib/types.ts` (EvalInstance, EvalResult)
- Modify: `web/src/components/EvalBits.tsx` (SwitchInstanceRow)
- Modify: `web/src/pages/LabRunViewer.tsx` (EvalTab layer filter)

**Interfaces:**
- Consumes: eval.json shape from Task 3.
- Produces: `AttributionLayer`, `EvalInstanceAttribution` types; every switch row renders a layer pill.

- [ ] **Step 1: types.ts** — after the `EvalInstance` interface, add and modify:

```ts
// SPO-19: evidence-based layer attribution per ID switch. "refinement" is
// reserved (Phase 4 refined-tracklet layer) and never emitted today;
// "ambiguous" is a first-class honest outcome, not a fallback guess.
export type AttributionLayer =
  | "detection"
  | "online_association"
  | "refinement"
  | "offline_association"
  | "ambiguous";

export interface AttributionEvidence {
  kind: string; // oracle_input | oracle_comparison | tracklet_counterpart | entity_only | insufficient_evidence
  detail?: string;
  outcome?: "persists" | "disappears";
  oracle_run?: string | null;
  oracle_frame_idx?: number;
  oracle_t?: number;
  frame_idx?: number;
  t?: number;
  tol_s?: number;
}

export interface EvalInstanceAttribution {
  layer: AttributionLayer;
  evidence: AttributionEvidence[];
}
```

`EvalInstance` gains (optional — eval.json files written before SPO-19 lack it):

```ts
  attribution?: EvalInstanceAttribution;
```

`EvalResult` gains (after `instances`):

```ts
  // SPO-19 context block: how attribution was derived for this payload.
  attribution?: {
    detect_impl: string | null;
    oracle_input: boolean;
    oracle_comparison: { oracle_run: string | null } | null;
    tol_s: number;
    counts: Record<"tracklet" | "entity", Partial<Record<AttributionLayer, number>>>;
  };
```

- [ ] **Step 2: EvalBits.tsx** — add the pill (export it for reuse) and render it in the row:

```tsx
import type { AttributionLayer, EvalInstance, EvalInstanceAttribution } from "../lib/types";

export const LAYER_LABEL: Record<AttributionLayer, string> = {
  detection: "detection",
  online_association: "online assoc",
  refinement: "refinement",
  offline_association: "offline assoc",
  ambiguous: "ambiguous",
};

const LAYER_CLASS: Record<AttributionLayer, string> = {
  detection: "bg-team-away/15 text-team-away",
  online_association: "bg-volt-400/15 text-volt-300",
  refinement: "bg-white/10 text-ink-300",
  offline_association: "bg-team-ref/15 text-team-ref",
  ambiguous: "bg-turf-800 text-ink-400",
};

export function AttributionPill({ attribution }: { attribution?: EvalInstanceAttribution }) {
  if (!attribution) {
    return (
      <span
        className="rounded-full bg-turf-800 px-2 py-0.5 font-mono text-[10px] text-ink-500"
        title="This eval predates layer attribution — re-evaluate the run to attribute its switches."
      >
        unattributed
      </span>
    );
  }
  const tooltip = attribution.evidence
    .map((e) => e.detail ?? `${e.kind}${e.outcome ? `: ${e.outcome}` : ""}`)
    .join("; ");
  return (
    <span
      className={`rounded-full px-2 py-0.5 font-mono text-[10px] ${LAYER_CLASS[attribution.layer]}`}
      title={tooltip}
    >
      {LAYER_LABEL[attribution.layer]}
    </span>
  );
}
```

In `SwitchInstanceRow`, insert `<AttributionPill attribution={inst.attribution} />`
immediately after the existing level pill `<span>`.

- [ ] **Step 3: LabRunViewer.tsx EvalTab layer filter** — alongside the existing
`levelFilter` state:

```tsx
  const [layerFilter, setLayerFilter] = useState<AttributionLayer | "all">("all");

  const layerOptions = useMemo(() => {
    if (!ev) return [];
    const layers = new Set<AttributionLayer>();
    for (const inst of ev.instances) if (inst.attribution) layers.add(inst.attribution.layer);
    return [...layers].sort();
  }, [ev]);
```

Extend `visibleInstances` filtering (before the sort):

```tsx
    if (layerFilter !== "all")
      list = list.filter((inst) => inst.attribution?.layer === layerFilter);
```

Render the chips right after the existing level-filter `<div className="flex items-center gap-1">` block, only when `layerOptions.length > 0`:

```tsx
        {layerOptions.length > 0 && (
          <div className="flex items-center gap-1">
            {(["all", ...layerOptions] as const).map((layer) => (
              <button
                key={layer}
                onClick={() => setLayerFilter(layer as AttributionLayer | "all")}
                className={`rounded-md px-2.5 py-1 text-[12px] transition-colors ${
                  layerFilter === layer
                    ? "bg-turf-800 text-ink-100"
                    : "text-ink-400 hover:bg-turf-900 hover:text-ink-100"
                }`}
              >
                {layer === "all" ? "all layers" : LAYER_LABEL[layer as AttributionLayer]}
              </button>
            ))}
          </div>
        )}
```

Add the imports: `AttributionLayer` type from `../lib/types`, `LAYER_LABEL` from
`../components/EvalBits`. Reset `layerFilter` is not needed (a stale filter simply shows
the empty-state "No switches match the filters." message that already exists).

- [ ] **Step 4: Frontend typecheck + build**

Run: `cd web && npm run build`
Expected: clean tsc + vite build

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/types.ts web/src/components/EvalBits.tsx web/src/pages/LabRunViewer.tsx
git commit -m "Show switch layer attribution in the Lab failure browser (SPO-19)"
```

---

### Task 7: Docs + full verification

**Files:**
- Modify: `docs/implementation-status.md` (evaluation section — describe the attribution layer, its rules, and the oracle enrichment paths; follow the file's existing verified-against-code style)
- Modify: `CLAUDE.md` (one sentence in "Server, jobs, and evaluation" noting eval.json instances now carry per-switch layer attribution incl. explicit ambiguous, with oracle enrichment via the benchmark runner's `oracle_candidate` or `POST /api/runs/{id}/evaluate?oracle_run_id=`)

**Interfaces:** none (docs + verification only).

- [ ] **Step 1: Update docs** — read the current evaluation sections of both files first; keep claims code-verified and linked (docs governance rules).

- [ ] **Step 2: Full verification**

Run: `uv run pytest packages -q && uv run ruff check packages && cd web && npm run build`
Expected: everything green

- [ ] **Step 3: Commit**

```bash
git add docs/implementation-status.md CLAUDE.md
git commit -m "Document switch layer attribution (SPO-19)"
```

---

## Self-review notes

- Spec coverage: attribution module (T1–2), evaluate_run wiring + handcrafted-cause tests
  (T3), shared matcher + server endpoint (T4), benchmark `oracle_candidate` (T5), UI +
  types sync (T6), docs (T7). "Refinement reserved" is enum/types-only by design.
- The acceptance criterion "no silent default" is enforced structurally: the only
  non-evidence path emits `ambiguous` + `insufficient_evidence`, and oracle payloads must
  self-describe (`oracle_input: True`) or be refused.
- Type consistency: `attribute_switches(result, *, oracle_eval, oracle_run_id, tol_s)` is
  used identically in T3 (core), T4 (server), T5 (train).
