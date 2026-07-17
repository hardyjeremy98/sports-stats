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
    stages = (manifest.get("config") or {}).get("stages") or {}
    detect_cfg = stages.get("detect") or {}
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
    ctx["oracle_comparison"] = {"oracle_run": oracle_run_id} if oracle_eval is not None else None
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


def _validate_oracle_comparison(result: dict, oracle_eval: dict, oracle_run_id: str | None) -> None:
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
