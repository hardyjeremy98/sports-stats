"""Team-classification metrics — the slot the evaluator was blind to.

Detection has AP, tracking has HOTA/purity, association has IDF1 and entity
purity. Team classification had nothing: `evaluation.py` carried the predicted
team only as a display string on switch instances, so no run could tell a good
team classifier from a bad one. That mattered more than the omission suggests,
because under the two-pass merge engine the ONLY hard constraints left are
temporal non-overlap and *same team* — kit colour is the last remaining veto,
and SPO-73 measured it falsely vetoing 19% of true re-entry pairs before the
merge rule ever evaluated them.

Two layers, deliberately separate:

**Assignment quality** — is the label right? GT teams are camera-relative
("left"/"right"); the classifiers emit arbitrary cluster labels (HOME/AWAY).
There is no canonical correspondence between the two, so accuracy is scored
under the better of the two global permutations, clustering-accuracy style, and
the permutation actually used is reported. Scoring a fixed HOME=left mapping
would report ~50% on a perfect classifier that happened to label the clusters
the other way round.

**Gate behaviour** — what does the label *do*? This is the metric that matters
for the merge engine, and it is not derivable from accuracy: a classifier can be
95% accurate and still veto every true re-entry pair if its errors land on the
tracklets that re-enter. Pairs are classified by GT (same player / opponents /
same team but different players) and run through the REAL
`reid.gates.TeamConsistencyGate` — never a reimplementation of it here, so this
measures the shipped gate rather than a copy that could drift from it.

The gate block is computed twice: once at the run's configured
`team_min_confidence` and once at 0.0 (the pre-SPO-75 label-only behaviour), so
the confidence-abstention fix's effect is visible per-run instead of resting on
the single 2026-07-26 triage.

Nothing here is a pass/fail gate; this module reports rates.
"""

from __future__ import annotations

from pathlib import Path

from matchlab_core.gt import GroundTruth

# GT roles that carry a team. Referees have `team=None` in `gt.py` and the ball
# is not a scored role, so both drop out naturally rather than by special case.
_TEAMED_ROLES = ("player", "goalkeeper")

# The two possible correspondences between arbitrary classifier cluster labels
# and GT's camera-relative sides.
_PERMUTATIONS = (
    {"home": "left", "away": "right"},
    {"home": "right", "away": "left"},
)


def evaluate_team(
    run_dir: str | Path,
    gt: GroundTruth,
    manifest: dict,
    tracklets: list[dict],
    purity_records: list[dict],
) -> dict | None:
    """Run-dir adapter called by `evaluation.evaluate_run`.

    `purity_records` are `tracklet_purity(...)["tracklets"]` — reusing the
    per-tracklet GT match that pass already computed, rather than matching a
    second time with subtly different rules. Returns None (caller omits the
    section) when the run has no `teams.json`, e.g. imported external runs.
    """
    import json

    run_dir = Path(run_dir)
    teams_path = run_dir / "teams.json"
    if not teams_path.exists():
        return None
    teams = json.loads(teams_path.read_text())

    gt_team = {t.track_id: t.team for t in gt.tracks if t.role in _TEAMED_ROLES and t.team}
    gt_role = {t.track_id: t.role for t in gt.tracks}
    majority_gt = {
        r["tracklet_id"]: r["majority_gt_track_id"]
        for r in purity_records
        if r["majority_gt_track_id"] is not None
    }
    matched_frames = {r["tracklet_id"]: r["matched_frames"] for r in purity_records}

    assignment = _assignment_quality(teams, gt_team, gt_role, majority_gt, matched_frames)
    gate = _gate_behaviour(teams, gt_team, majority_gt, tracklets, manifest)
    return {"assignment": assignment, "gate": gate}


def _assignment_quality(
    teams: list[dict],
    gt_team: dict[int, str],
    gt_role: dict[int, str],
    majority_gt: dict[int, int],
    matched_frames: dict[int, int],
) -> dict:
    """Per-tracklet team accuracy under the better global label permutation.

    Counted two ways: `by_tracklet` treats every tracklet equally, `by_frame`
    weights each by its matched frames so one long tracklet outweighs many
    short ones — the same frame-weighting `tracklet_purity` uses, reported
    alongside rather than instead of the count so a classifier that fails only
    on short fragments is distinguishable from one that fails on long ones.

    A tracklet whose predicted team is UNKNOWN/REFEREE is an ABSTENTION, not an
    error: it is excluded from the accuracy denominator and reported as
    coverage. Per ADR 003, unusable evidence is neutral — scoring abstention as
    a miss would reward a classifier that always guesses.
    """
    pred_team = {t["tracklet_id"]: t["team"] for t in teams}
    pred_conf = {t["tracklet_id"]: t.get("confidence") for t in teams}

    scorable: list[tuple[int, str, str, int, str]] = []  # tid, pred, gt_side, frames, role
    abstained = 0
    for tid, gid in majority_gt.items():
        side = gt_team.get(gid)
        if side is None:
            continue  # referee / unteamed GT track: nothing to be right about
        pred = pred_team.get(tid, "unknown")
        if pred not in ("home", "away"):
            abstained += 1
            continue
        scorable.append((tid, pred, side, matched_frames.get(tid, 0), gt_role.get(gid, "player")))

    scored: list[dict] = []
    best = None
    for perm in _PERMUTATIONS:
        correct = [perm[p] == side for _, p, side, _, _ in scorable]
        frames = [f for _, _, _, f, _ in scorable]
        n_correct = sum(correct)
        frame_correct = sum(f for c, f in zip(correct, frames) if c)
        total_frames = sum(frames)
        cand = {
            "mapping": dict(perm),
            "by_tracklet": round(n_correct / len(scorable), 4) if scorable else None,
            "by_frame": round(frame_correct / total_frames, 4) if total_frames else None,
        }
        if best is None or (cand["by_tracklet"] or 0) > (best["by_tracklet"] or 0):
            best, scored = cand, correct
    alt = [c for c in _PERMUTATIONS if best is not None and c != best["mapping"]]

    per_role: dict[str, dict] = {}
    for role in _TEAMED_ROLES:
        idx = [i for i, (_, _, _, _, r) in enumerate(scorable) if r == role]
        if idx:
            per_role[role] = {
                "tracklets": len(idx),
                "accuracy": round(sum(scored[i] for i in idx) / len(idx), 4),
            }

    # SPO-75 claimed 0.5 separates goalkeepers from outfielders exactly. Making
    # that a standing per-run metric means the claim is re-tested on every run
    # instead of resting on one 2026-07-26 triage over 12 sequences.
    conf_by_role: dict[str, dict] = {}
    for role in _TEAMED_ROLES:
        vals = [
            pred_conf[tid]
            for tid, gid in majority_gt.items()
            if gt_role.get(gid) == role and pred_conf.get(tid) is not None
        ]
        if vals:
            conf_by_role[role] = {
                "n": len(vals),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(sum(vals) / len(vals), 4),
            }

    total = len(scorable) + abstained
    return {
        "scored_tracklets": len(scorable),
        "abstained_tracklets": abstained,
        "coverage": round(len(scorable) / total, 4) if total else None,
        "accuracy": best["by_tracklet"] if best else None,
        "accuracy_by_frame": best["by_frame"] if best else None,
        "mapping": best["mapping"] if best else None,
        "mapping_note": (
            "Classifier cluster labels (home/away) carry no inherent "
            "correspondence to GT's camera-relative sides (left/right); "
            "accuracy is the better of the two global permutations."
        ),
        "accuracy_alt_mapping": (
            round(1 - best["by_tracklet"], 4)
            if best and best["by_tracklet"] is not None and alt
            else None
        ),
        "by_role": per_role,
        "confidence_by_role": conf_by_role,
    }


def _gate_behaviour(
    teams: list[dict],
    gt_team: dict[int, str],
    majority_gt: dict[int, int],
    tracklets: list[dict],
    manifest: dict,
) -> dict:
    """False-veto rate of the real `TeamConsistencyGate` over GT-true pairs.

    Every ordered tracklet pair that the temporal gate lets through is labelled
    from GT — `same_player` (a true re-entry the engine SHOULD be free to
    merge), `opponents` (the pair the gate exists to kill), `same_team` (a hard
    negative the gate cannot help with) — and run through the shipped gate.

    `false_veto_rate` over `same_player` is the headline: merges the team gate
    removes from the engine's reach before any appearance evidence is scored.
    `true_veto_rate` over `opponents` is what the gate buys in exchange.
    """
    try:
        from matchlab_core.reid.gates import TeamConsistencyGate, TemporalOverlapGate
        from matchlab_core.schemas import Team, Tracklet
    except Exception as exc:  # pragma: no cover - lean env without cv2
        # reid.gates pulls cv2 via reid.motion. Degrade loudly with the reason
        # rather than silently dropping the section: an absent metric that
        # looks like "no problem found" is the failure mode this file exists
        # to prevent.
        return {"unavailable": f"{type(exc).__name__}: {exc}"}

    team_by_tid = {t["tracklet_id"]: Team(t["team"]) for t in teams}
    conf_by_tid = {
        t["tracklet_id"]: t["confidence"] for t in teams if t.get("confidence") is not None
    }

    parsed = [Tracklet.model_validate(t) for t in tracklets if t.get("frames")]
    parsed = [t for t in parsed if t.tracklet_id in majority_gt]
    parsed.sort(key=lambda t: (t.start_frame, t.tracklet_id))

    assoc_params = manifest.get("config", {}).get("stages", {}).get("associate", {}).get(
        "params", {}
    )
    tolerance = assoc_params.get("overlap_tolerance_frames", 2)
    configured = assoc_params.get("team_min_confidence")
    # Absent from the manifest means the run predates the param or used a
    # different associator; fall back to the engine's own default rather than
    # inventing 0.0, which would silently report the pre-SPO-75 behaviour.
    if not isinstance(configured, (int, float)):
        configured = _default_team_min_confidence()

    temporal = TemporalOverlapGate(tolerance_frames=int(tolerance))
    pairs: list[tuple[Tracklet, Tracklet, str]] = []
    for i, first in enumerate(parsed):
        for second in parsed[i + 1 :]:
            if temporal.check(first, second) is not None:
                continue  # the team gate never sees these
            ga, gb = majority_gt[first.tracklet_id], majority_gt[second.tracklet_id]
            if ga == gb:
                kind = "same_player"
            elif gt_team.get(ga) and gt_team.get(gb) and gt_team[ga] != gt_team[gb]:
                kind = "opponents"
            elif gt_team.get(ga) and gt_team.get(gb):
                kind = "same_team"
            else:
                continue  # a referee or unteamed GT track on one side
            pairs.append((first, second, kind))

    arms: dict[str, dict] = {}
    for label, min_conf in (("configured", float(configured)), ("label_only", 0.0)):
        gate = TeamConsistencyGate(team_by_tid, conf_by_tid, min_confidence=min_conf)
        counts = {k: {"pairs": 0, "vetoed": 0} for k in ("same_player", "opponents", "same_team")}
        for first, second, kind in pairs:
            counts[kind]["pairs"] += 1
            if gate.check(first, second) is not None:
                counts[kind]["vetoed"] += 1
        arms[label] = {
            "min_confidence": min_conf,
            **counts,
            "false_veto_rate": _rate(counts["same_player"]),
            "true_veto_rate": _rate(counts["opponents"]),
            "same_team_veto_rate": _rate(counts["same_team"]),
        }

    return {
        "overlap_tolerance_frames": int(tolerance),
        "candidate_pairs": len(pairs),
        "note": (
            "Pairs are those surviving the temporal gate, the only gate "
            "checked before team. false_veto_rate counts true re-entry pairs "
            "the team gate removes from the merge engine's reach before any "
            "appearance evidence is scored. 'label_only' replays the "
            "pre-SPO-75 behaviour (confidence ignored) over the same pairs."
        ),
        **arms,
    }


def _rate(c: dict) -> float | None:
    return round(c["vetoed"] / c["pairs"], 4) if c["pairs"] else None


def _default_team_min_confidence() -> float:
    """The reid-engine's own `team_min_confidence` default, resolved through the
    registry rather than hardcoded here so the two cannot drift apart."""
    try:
        from matchlab_core.registry import build
        from matchlab_core.schemas.run import StageKind

        stage = build(StageKind.ASSOCIATE, "reid-engine", {})
        value = getattr(getattr(stage, "params", None), "team_min_confidence", None)
        return float(value) if isinstance(value, (int, float)) else 0.0
    except Exception:
        return 0.0
