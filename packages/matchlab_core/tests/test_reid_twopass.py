"""Two-pass merging over accumulated threads."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.evidence import LLRCalibrator
from matchlab_core.reid.jersey import N_NUMBERS, uniform_prior
from matchlab_core.reid.twopass import (
    FusionModel,
    TrackletEvidence,
    _pool_jersey,
    members_disjoint,
    merge_threads_two_pass,
)


def _peaked(n: int, mass: float = 0.999) -> np.ndarray:
    v = np.full(N_NUMBERS, (1.0 - mass) / (N_NUMBERS - 1))
    v[n] = mass
    return v

START = np.array([0, 50, 100, 150, 200, 60])
END = np.array([10, 60, 110, 160, 210, 105])


def test_interleaved_threads_are_disjoint_though_envelopes_overlap():
    """x = {0-10, 100-110}, y = {50-60}: y sits inside x's gap."""
    assert members_disjoint([0, 2], [1], START, END)
    envelope_ok = END[[0, 2]].max() < START[[1]].min()
    assert not envelope_ok


def test_overlapping_tracklets_are_rejected():
    assert not members_disjoint([2], [5], START, END)
    assert not members_disjoint([0, 2], [1, 5], START, END)


def _model(**weights) -> FusionModel:
    """A calibrator whose LLR rises with cosine similarity."""
    same = np.random.default_rng(0).normal(0.9, 0.05, 4000)
    diff = np.random.default_rng(1).normal(0.1, 0.05, 4000)
    return FusionModel(
        calibrators={"body": LLRCalibrator.fit(same, diff, max_bins=64)},
        weights={"body": 1.0, **weights},
        fps=25.0,
    )


def _ev(tid, start, end, emb, team=0):
    return TrackletEvidence(
        tracklet_id=tid, start=start, end=end, team=team,
        embedding=np.asarray(emb, dtype=float),
    )


def test_merges_matching_appearance_and_separates_the_rest():
    a, b = [1.0, 0.0], [0.0, 1.0]
    ev = [_ev(1, 0, 10, a), _ev(2, 20, 30, a), _ev(3, 40, 50, b)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=0.0)
    assert [1, 2] in res.groups
    assert [3] in res.groups


def test_team_is_a_hard_constraint():
    """Identical appearance must not cross a team boundary."""
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 10, a, team=0), _ev(2, 20, 30, a, team=1)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=-1e9, pass2_score=-1e9)
    assert sorted(res.groups) == [[1], [2]]


def test_temporal_overlap_is_a_hard_constraint():
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 100, a), _ev(2, 50, 150, a)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=-1e9, pass2_score=-1e9)
    assert sorted(res.groups) == [[1], [2]]


def test_interleaved_same_player_tracklets_can_merge():
    """The case an envelope gate would block permanently."""
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 10, a), _ev(2, 100, 110, a), _ev(3, 50, 60, a)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=0.0)
    assert res.groups == [[1, 2, 3]]


def test_missing_supporting_evidence_is_neutral():
    """ADR 003: an absent occupancy/transition channel costs nothing.

    Neither tracklet here has pitch positions, so only appearance and gap
    speak — and the pair still merges on appearance alone.
    """
    ev = [_ev(1, 0, 10, [1.0, 0.0]), _ev(2, 20, 30, [1.0, 0.0])]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=None)
    assert res.groups == [[1, 2]]


def test_no_identity_evidence_abstains_however_low_the_threshold():
    """Neutrality must not let timing alone assert an identity.

    A missing channel contributes zero, but the threshold is absolute, so
    without a required-channel rule a pair with no appearance at all could be
    merged by a short gap. A silent player swap is worse than an unknown
    player, so the pair abstains at any threshold.
    """
    ev = [
        _ev(1, 0, 10, [1.0, 0.0]),
        TrackletEvidence(tracklet_id=2, start=20, end=30, team=0),
    ]
    res = merge_threads_two_pass(ev, model=_model(), min_score=-1e9, pass2_score=-1e9)
    assert sorted(res.groups) == [[1], [2]]


def test_required_channels_are_configurable():
    model = _model()
    model.required = ()
    ev = [
        _ev(1, 0, 10, [1.0, 0.0]),
        TrackletEvidence(tracklet_id=2, start=20, end=30, team=0),
    ]
    res = merge_threads_two_pass(ev, model=model, min_score=-1e9, pass2_score=None)
    assert res.groups == [[1, 2]]


def test_high_threshold_abstains_rather_than_guessing():
    ev = [_ev(1, 0, 10, [1.0, 0.0]), _ev(2, 20, 30, [0.0, 1.0])]
    res = merge_threads_two_pass(ev, model=_model(), min_score=1e9, pass2_score=1e9)
    assert sorted(res.groups) == [[1], [2]]


def test_every_tracklet_appears_exactly_once():
    rng = np.random.default_rng(3)
    ev = [
        _ev(i, i * 20, i * 20 + 10, rng.normal(size=2), team=i % 2)
        for i in range(12)
    ]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=0.0)
    flat = [t for g in res.groups for t in g]
    assert sorted(flat) == list(range(12))


def test_pass2_merges_what_pass1_could_not():
    """Pass 2 exists because both sides are mature by then.

    Three tracklets of one player where the middle one is unlike the first, so
    pass 1 leaves two threads; pooled, the prototypes align.
    """
    ev = [
        _ev(1, 0, 10, [1.0, 0.0]),
        _ev(2, 20, 30, [0.0, 1.0]),
        _ev(3, 40, 50, [1.0, 0.0]),
    ]
    one = merge_threads_two_pass(ev, model=_model(), min_score=2.0, pass2_score=None)
    two = merge_threads_two_pass(ev, model=_model(), min_score=2.0, pass2_score=-2.0)
    assert len(two.groups) <= len(one.groups)


def test_pair_filter_keeps_pairs_silent():
    """Referee exclusion: filtered pairs never merge and leave no trail row."""
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 10, a), _ev(2, 20, 30, a)]
    res = merge_threads_two_pass(
        ev, model=_model(), min_score=-1e9, pass2_score=-1e9,
        pair_filter=lambda x, y: False,
    )
    assert sorted(res.groups) == [[1], [2]]
    assert res.pairs == []


def test_model_round_trips_through_json():
    m = _model(transition=0.5)
    back = FusionModel.from_dict(m.to_dict())
    assert back.weights == m.weights
    assert back.fps == m.fps
    probe = np.array([0.5])
    assert back.calibrators["body"].llr(probe) == pytest.approx(
        m.calibrators["body"].llr(probe)
    )


def test_empty_input():
    assert merge_threads_two_pass([], model=_model(), min_score=0.0).groups == []


# --- _pool_jersey (jersey-evidence accumulation, fix round 1) -----------------


def test_pool_jersey_uniform_times_peaked_is_peaked():
    """A flat (abstaining) side is neutral: pooling with it must not reshape
    the other side's belief -- same do-no-harm convention as `pair_llr`."""
    peaked = _peaked(7)
    pooled = _pool_jersey(uniform_prior(), peaked)
    assert np.allclose(pooled, peaked, atol=1e-9)
    assert pooled.sum() == pytest.approx(1.0)


def test_pool_jersey_none_handling():
    peaked = _peaked(7)
    assert _pool_jersey(None, peaked) is peaked
    assert _pool_jersey(peaked, None) is peaked
    assert _pool_jersey(None, None) is None


def test_pool_jersey_degenerate_contradiction_keeps_a():
    """Two disjoint one-hot reads have zero product support (the s<=0 branch).

    Documents the actual fallback: `a` (the accumulated side) is kept and
    `b`'s reading is silently dropped rather than raising -- see the
    comment on `_pool_jersey` for why that's acceptable (the contradiction
    is still visible wherever this pooled belief is later compared via
    `pair_llr`, which saturates to a strong negative rather than staying
    silent)."""
    one_hot_a = np.zeros(N_NUMBERS)
    one_hot_a[3] = 1.0
    one_hot_b = np.zeros(N_NUMBERS)
    one_hot_b[9] = 1.0
    assert _pool_jersey(one_hot_a, one_hot_b) is one_hot_a


def test_pool_jersey_agreeing_reads_sharpen_the_pooled_belief():
    """Two independent agreeing reads should concentrate MORE mass on the
    shared number than either read alone -- this is what "accumulation"
    means for jersey evidence across a growing thread."""
    once = _peaked(7)
    twice = _pool_jersey(once, _peaked(7))
    assert twice[7] > once[7]


# --- three-tracklet threads (fix round 1: the untested accumulation path) ----


def test_three_tracklet_thread_accumulates_jersey_evidence_and_merges():
    """Pass 1 pools jersey evidence across EVERY member added to a thread so
    far, not just the newest -- so a third tracklet's decision is scored
    against two prior agreeing reads, not one. Appearance alone is kept
    provably too weak to merge (min_score above the single-channel max of
    LOG_CLAMP * body_weight = 6.0), so any merge below can only be jersey's
    accumulated contribution."""
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 10, a), _ev(2, 20, 30, a), _ev(3, 40, 50, a)]
    model = _model()  # weights={"body": 1.0}, no other calibrators

    body_only = merge_threads_two_pass(ev, model=model, min_score=8.0, pass2_score=None)
    assert sorted(body_only.groups) == [[1], [2], [3]]

    jersey = {1: _peaked(7), 2: _peaked(7), 3: _peaked(7)}
    fused = merge_threads_two_pass(
        ev, model=model, min_score=8.0, pass2_score=None,
        jersey_likelihood_by_tid=jersey, jersey_prior=uniform_prior(), jersey_weight=1.0,
    )
    assert [1, 2, 3] in fused.groups


def test_three_tracklet_uniform_jersey_is_bit_identical_to_disabled():
    """Extends the pairwise bit-identity guarantee to the pooling path: three
    tracklets whose likelihoods are all flat (never-read/abstaining) must
    fuse to EXACTLY the jersey-disabled result, pooling included."""
    a = [1.0, 0.0]
    ev = [_ev(1, 0, 10, a), _ev(2, 20, 30, a), _ev(3, 40, 50, a)]
    model = _model()

    base = merge_threads_two_pass(ev, model=model, min_score=1.0, pass2_score=0.5)
    flat = {1: uniform_prior(), 2: uniform_prior(), 3: uniform_prior()}
    fused = merge_threads_two_pass(
        ev, model=model, min_score=1.0, pass2_score=0.5,
        jersey_likelihood_by_tid=flat, jersey_prior=uniform_prior(), jersey_weight=1.0,
    )

    assert fused.groups == base.groups
    base_by_pair = {(p.a, p.b): p.affinity for p in base.pairs}
    fused_by_pair = {(p.a, p.b): p.affinity for p in fused.pairs}
    assert base_by_pair.keys() == fused_by_pair.keys()
    for key, base_affinity in base_by_pair.items():
        if base_affinity is None:
            assert fused_by_pair[key] is None
        else:
            assert fused_by_pair[key] == pytest.approx(base_affinity, abs=1e-12)


def test_score_channels_sums_to_score_and_names_abstentions():
    """The displayed breakdown must reconcile with the decided number, and an
    abstaining channel must be distinguishable from one that voted zero.

    The Lab shows this breakdown to explain why a merge did or did not happen;
    if the parts did not add up to the whole, or a dead input looked like a
    neutral one, the explanation would be worse than none.
    """
    import numpy as np
    from matchlab_core.reid.twopass import FusionModel, TrackletEvidence

    model = FusionModel.load("configs/reid/fusion-footpass-v1.json")
    emb = np.ones(8, dtype=np.float64)
    emb /= np.linalg.norm(emb)
    # No pitch positions at all -> occupancy/transition have nothing to say.
    a = TrackletEvidence(tracklet_id=1, start=0, end=50, embedding=emb).to_state()
    b = TrackletEvidence(tracklet_id=2, start=100, end=150, embedding=emb).to_state()

    total, channels = model.score_channels(a, a.footprint(), b, b.footprint())
    assert total is not None
    by_name = {c["name"]: c for c in channels}
    assert set(by_name) == {"body", "occupancy", "gap", "transition"}

    # Parts reconcile with the whole.
    assert total == pytest.approx(sum(c["contribution"] for c in channels))

    # Body had an embedding on both sides, so it voted.
    assert by_name["body"]["llr"] is not None
    # Occupancy had no pitch positions, so it abstained -- reported as None and
    # contributing exactly nothing, NOT as a 0.0 vote.
    assert by_name["occupancy"]["llr"] is None
    assert by_name["occupancy"]["contribution"] == 0.0

    # Transition abstains: with no calibrated frames there are no entry/exit
    # positions, and the old (0, 0) substitution scored a fabricated
    # "player who did not move" as the prior's maximum positive evidence.
    # raw is None too, distinguishing "no endpoints" from "gap-gated".
    assert by_name["transition"]["llr"] is None
    assert by_name["transition"]["raw"] is None
    assert by_name["transition"]["contribution"] == 0.0


def test_two_pass_records_channels_for_a_rejected_best_candidate():
    """A merge that did NOT happen is the harder one to explain, so the engine
    records the decisive candidate's channels even when it rejects it."""
    import numpy as np
    from matchlab_core.reid.twopass import FusionModel, TrackletEvidence, merge_threads_two_pass

    model = FusionModel.load("configs/reid/fusion-footpass-v1.json")
    rng = np.random.default_rng(0)

    def emb(_seed):
        v = rng.normal(size=8)
        return v / np.linalg.norm(v)

    ev = [
        TrackletEvidence(tracklet_id=1, start=0, end=50, team=0, embedding=emb(1)),
        TrackletEvidence(tracklet_id=2, start=100, end=150, team=0, embedding=emb(2)),
    ]
    # Unreachably high threshold: the pair is scored, then rejected.
    result = merge_threads_two_pass(ev, model=model, min_score=1e6, pass2_score=1e6)

    assert not result.merge_edges
    assert result.channel_breakdowns, "a scored-then-rejected pair must be explained"
    row = result.channel_breakdowns[0]
    assert row["decision"] == "rejected"
    assert {c["name"] for c in row["channels"]} >= {"body", "gap", "jersey"}
    assert row["total"] == pytest.approx(sum(c["contribution"] for c in row["channels"]))
def test_occupancy_shrinkage_fades_sparse_footprints_toward_neutral():
    """With occupancy_shrink_n0 set, a thin pair's occupancy contribution is
    scaled by n/(n+n0); in the data-rich regime the factor approaches 1 and
    the fitted behaviour is unchanged. 0.0 (default) is exactly the old sum."""
    import numpy as np
    from matchlab_core.reid.evidence import LLRCalibrator
    from matchlab_core.reid.threads import ThreadState
    from matchlab_core.reid.twopass import FusionModel

    cal = LLRCalibrator.fit(
        np.linspace(0.0, 1.0, 200), np.r_[np.ones(100), np.zeros(100)]
    )
    def state(n):
        xs = np.linspace(0.4, 0.6, n)
        return ThreadState.from_fragment(
            xs, xs, embedding=np.array([1.0, 0.0]), start=0, end=n,
            exit_xy=np.array([0.5, 0.5]), entry_xy=np.array([0.5, 0.5]),
        )
    a, b = state(30), state(30)
    b = ThreadState.from_fragment(
        np.linspace(0.4, 0.6, 30), np.linspace(0.4, 0.6, 30),
        embedding=np.array([1.0, 0.0]), start=40, end=70,
        exit_xy=np.array([0.5, 0.5]), entry_xy=np.array([0.5, 0.5]),
    )
    base = FusionModel(calibrators={"body": cal, "occupancy": cal}, weights={})
    shrunk = FusionModel(
        calibrators={"body": cal, "occupancy": cal}, weights={},
        occupancy_shrink_n0=300.0,
    )
    s0 = base.score(a, a.footprint(), b, b.footprint())
    s1 = shrunk.score(a, a.footprint(), b, b.footprint())
    # body identical; occupancy scaled by 30/330
    occ = float(cal.llr(0.0))  # identical footprints -> js ~ 0
    from matchlab_core.reid.occupancy import js_distance
    occ = float(cal.llr(js_distance(b.footprint(), a.footprint())))
    assert abs((s0 - s1) - occ * (1 - 30 / 330)) < 1e-9
    # round-trips through serialisation
    assert FusionModel.from_dict(shrunk.to_dict()).occupancy_shrink_n0 == 300.0


def test_pass2_margin_blocks_contested_merges_only():
    """Three same-looking singleton threads: every pass-2 pairing scores the
    same, so no pairing is distinguishable from its alternative and a margin
    bar blocks them all. With an unambiguous pair (third thread orthogonal),
    the same bar lets the clear winner through. min_score is set high so pass
    1 abstains and pass 2 owns every decision."""
    a, b = [1.0, 0.0], [0.0, 1.0]
    contested = [_ev(1, 0, 10, a), _ev(2, 20, 30, a), _ev(3, 40, 50, a)]
    res = merge_threads_two_pass(
        contested, model=_model(), min_score=99.0, pass2_score=0.0,
        pass2_min_margin=0.5,
    )
    assert res.groups == [[1], [2], [3]]
    # margin 0 is the legacy greedy: everything merges
    res0 = merge_threads_two_pass(
        contested, model=_model(), min_score=99.0, pass2_score=0.0,
        pass2_min_margin=0.0,
    )
    assert res0.groups == [[1, 2, 3]]

    clear = [_ev(1, 0, 10, a), _ev(2, 20, 30, a), _ev(3, 40, 50, b)]
    res1 = merge_threads_two_pass(
        clear, model=_model(), min_score=99.0, pass2_score=0.0,
        pass2_min_margin=0.5,
    )
    assert [1, 2] in res1.groups and [3] in res1.groups


def test_footprint_alpha_zero_is_identical_and_positive_pulls_uniform():
    import numpy as np
    from matchlab_core.reid.threads import ThreadState

    s = ThreadState.from_fragment(
        np.array([0.5]), np.array([0.5]), embedding=None, start=0, end=1,
        exit_xy=np.zeros(2), entry_xy=np.zeros(2),
    )
    base = s.footprint()
    assert np.array_equal(s.footprint(alpha=0.0).grid, base.grid)
    smoothed = s.footprint(alpha=1.0).grid
    uniform = np.full_like(base.grid, 1.0 / base.grid.size)
    # strictly closer to uniform than the raw single-cell footprint
    assert np.abs(smoothed - uniform).sum() < np.abs(base.grid - uniform).sum()
    assert np.isclose(smoothed.sum(), 1.0)


def test_gap_binned_calibrators_select_by_gap_and_round_trip():
    import numpy as np
    from matchlab_core.reid.evidence import LLRCalibrator
    from matchlab_core.reid.twopass import FusionModel

    rng = np.random.default_rng(0)
    hi = LLRCalibrator.fit(rng.normal(0.9, 0.05, 2000), rng.normal(0.1, 0.05, 2000))
    lo = LLRCalibrator.fit(rng.normal(0.6, 0.05, 2000), rng.normal(0.4, 0.05, 2000))
    m = FusionModel(
        calibrators={"body": lo},
        calibrators_by_gap={"body": [(2.0, hi)]},
        weights={"body": 1.0},
    )
    # short gap -> sharp calibrator; long gap falls through to the flat one
    assert m._calibrator_for("body", 1.0) is hi
    assert m._calibrator_for("body", 10.0) is lo
    assert m._calibrator_for("occupancy", 1.0) is None
    m2 = FusionModel.from_dict(m.to_dict())
    assert m2.occupancy_alpha == 0.0
    assert len(m2.calibrators_by_gap["body"]) == 1
    assert m2._calibrator_for("body", 1.0).llr(0.8) == m._calibrator_for(
        "body", 1.0
    ).llr(0.8)


def test_contract_round_trips_and_validate_serving_gates_mismatches():
    m = _model()
    m.contract = {"occupancy_coords": "formation-relative", "embedding_dim": 2}
    m2 = FusionModel.from_dict(m.to_dict())
    assert m2.contract == m.contract
    # Matching serving passes; absent keys are never blocked.
    m2.validate_serving(occupancy_coords="formation-relative", embedding_dim=2)
    m2.validate_serving()
    FusionModel(calibrators={}, weights={}).validate_serving(
        occupancy_coords="absolute", embedding_dim=999
    )
    with pytest.raises(ValueError, match="occupancy_coords"):
        m2.validate_serving(occupancy_coords="absolute")
    with pytest.raises(ValueError, match="dim"):
        m2.validate_serving(embedding_dim=256)


def test_serving_diagnostics_flags_units_broken_endpoints():
    """The 2026-08-01 transition units bug (endpoints in cm, not [0,1]) must
    trip the physical-displacement flag; correctly-normalised endpoints must
    not."""
    m = _model()
    rng = np.random.default_rng(2)

    def ev(units: float):
        out = []
        for i in range(12):
            xy = rng.random(2) * units
            out.append(TrackletEvidence(
                tracklet_id=i, start=i * 100, end=i * 100 + 50, team=0,
                embedding=rng.normal(size=4),
                xs=rng.random(5), ys=rng.random(5),
                entry_xy=xy, exit_xy=xy,
            ))
        return out

    good = m.serving_diagnostics(ev(1.0), max_pairs=200)
    bad = m.serving_diagnostics(ev(10500.0), max_pairs=200)
    assert good["transition"]["flag"] is False
    assert bad["transition"]["flag"] is True
    # Body cosines from a 4-dim gaussian are nowhere near the fitted 0.9/0.1
    # bimodal pool's support edges -- but the report must still carry both
    # sides so a reader can compare them.
    assert good["body"]["served_n"] > 0
    assert "fitted_lo" in good["body"] and "served_median" in good["body"]


def test_weights_by_gap_selects_bin_and_round_trips():
    m = _model()
    m.weights_by_gap = {
        "edges": [5.0],
        "weights": [{"body": 2.0}, {"body": 0.5}],
    }
    m2 = FusionModel.from_dict(m.to_dict())
    assert m2.weights_by_gap == m.weights_by_gap
    assert m2._weight_for("body", 1.0) == 2.0
    assert m2._weight_for("body", 10.0) == 0.5
    # A channel absent from the bin's dict falls through to the flat weights.
    assert m2._weight_for("gap", 1.0) == m2.weights.get("gap", 1.0)
    # Scoring uses the per-bin weight: same pair, short vs long gap.
    e = np.array([1.0, 0.0])
    a = _ev(1, 0, 10, e).to_state()
    short = _ev(2, 20, 30, e).to_state()
    long_ = _ev(3, 400, 410, e).to_state()
    fa, fs, fl = a.footprint(), short.footprint(), long_.footprint()
    llr = m2.calibrators["body"].llr(1.0)
    s_short, _ = m2.score_channels(a, fa, short, fs)
    s_long, _ = m2.score_channels(a, fa, long_, fl)
    assert abs(s_short - 2.0 * llr) < 1e-9
    assert abs(s_long - 0.5 * llr) < 1e-9


def _ev_pos(tid, start, end, emb, entry, exit_, team=0):
    # Enough calibrated positions that the gate's starved-side guard
    # (gate_min_positions) does not abstain -- these tests exercise the
    # speed logic, not the quality gate.
    has_pos = entry is not None or exit_ is not None
    anchor = entry if entry is not None else exit_
    return TrackletEvidence(
        tracklet_id=tid, start=start, end=end, team=team,
        embedding=np.asarray(emb, dtype=float),
        xs=np.full(6, anchor[0]) if has_pos else None,
        ys=np.full(6, anchor[1]) if has_pos else None,
        entry_xy=None if entry is None else np.asarray(entry, dtype=float),
        exit_xy=None if exit_ is None else np.asarray(exit_, dtype=float),
    )


def test_motion_gate_vetoes_physically_impossible_pairs():
    """60 m in 1 s (60 m/s) is not football: vetoed with a recorded reason.
    The same displacement over 8 s (7.5 m/s) is a jog: merged."""
    a = [1.0, 0.0]
    for gap_frames, expect_merge in ((25, False), (200, True)):
        ev = [
            _ev_pos(1, 0, 10, a, (0.2, 0.5), (0.2, 0.5)),
            # 60/105 of the pitch width to the right = 60 m displacement
            _ev_pos(2, 10 + gap_frames, 10 + gap_frames + 10, a,
                    (0.2 + 60 / 105, 0.5), (0.2 + 60 / 105, 0.5)),
        ]
        res = merge_threads_two_pass(ev, model=_model(), min_score=0.0,
                                     pass2_score=None)
        merged = [1, 2] in res.groups
        assert merged == expect_merge, (gap_frames, res.groups)
        if not expect_merge:
            reasons = {p.reason for p in res.pairs if p.decision == "rejected"}
            assert any(r == "motion_infeasible" for r in map(str, reasons)) or any(
                getattr(r, "value", r) == "motion_infeasible" for r in reasons
            )


def test_motion_gate_abstains_without_positions():
    """No calibrated endpoints = no veto (ADR 003: missing evidence is
    neutral). A strong body match must still merge."""
    a = [1.0, 0.0]
    ev = [_ev_pos(1, 0, 10, a, None, None), _ev_pos(2, 20, 30, a, None, None)]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=None)
    assert [1, 2] in res.groups


def test_motion_gate_slack_absorbs_endpoint_noise_at_short_gaps():
    """2.5 m apparent displacement over 0.04 s is 62 m/s raw -- but 2.5 m is
    calibration noise, not motion, and must NOT veto. 20 m over the same gap
    is beyond any noise allowance and must."""
    a = [1.0, 0.0]
    for jump_m, expect_merge in ((2.5, True), (20.0, False)):
        ev = [
            _ev_pos(1, 0, 10, a, (0.2, 0.5), (0.2, 0.5)),
            _ev_pos(2, 11, 21, a, (0.2 + jump_m / 105, 0.5),
                    (0.2 + jump_m / 105, 0.5)),
        ]
        res = merge_threads_two_pass(ev, model=_model(), min_score=0.0,
                                     pass2_score=None)
        assert ([1, 2] in res.groups) == expect_merge, (jump_m, res.groups)


def test_motion_gate_applies_in_pass_2():
    """Two threads whose facing ends imply 60 m/s must not agglomerate; the
    same threads with a feasible gap must."""
    a, b = [1.0, 0.0], [0.0, 1.0]
    for gap_frames, expect in ((25, False), (500, True)):
        ev = [
            _ev_pos(1, 0, 10, a, (0.1, 0.5), (0.1, 0.5)),
            # dissimilar tracklet so pass 1 keeps them as separate threads
            _ev_pos(3, 15, 20, b, None, None),
            _ev_pos(2, 20 + gap_frames, 30 + gap_frames, a,
                    (0.1 + 60 / 105, 0.5), (0.1 + 60 / 105, 0.5)),
        ]
        res = merge_threads_two_pass(ev, model=_model(), min_score=100.0,
                                     pass2_score=0.0)
        assert ([1, 2] in res.groups) == expect, (gap_frames, res.groups)


def test_missing_endpoints_never_score_transition_against_pitch_corner():
    """A tracklet with no calibrated frames used to serve entry/exit as a
    ZERO vector -- the pitch corner -- and the transition prior scored real
    displacement evidence against it. Missing must mean absent."""
    m = _model()
    m.prior = __import__(
        "matchlab_core.reid.transition", fromlist=["TransitionPrior"]
    ).TransitionPrior.from_dict({
        "x": {"sigma_inf": 20.0, "tau": 30.0},
        "y": {"sigma_inf": 10.0, "tau": 20.0},
        "impostor_x": 30.0, "impostor_y": 20.0,
    })
    m.weights["transition"] = 1.0
    with_pos = _ev_pos(1, 0, 10, [1.0, 0.0], (0.9, 0.9), (0.9, 0.9)).to_state()
    no_pos = _ev_pos(2, 20, 30, [1.0, 0.0], None, None).to_state()
    _, chans = m.score_channels(
        with_pos, with_pos.footprint(), no_pos, no_pos.footprint()
    )
    t = next(c for c in chans if c["name"] == "transition")
    assert t["llr"] is None and t["contribution"] == 0.0


def test_motion_gate_uses_observation_time_not_span_gap():
    """Sparse calibration: the exit position was OBSERVED 5 s before the
    tracklet ended. 45 m over the 1 s span gap looks like 45 m/s, but over
    the true 6 s since the observation it is 7.5 m/s -- a jog. The gate must
    divide by observation time, or ordinary running on sparsely-calibrated
    runs gets vetoed."""
    a = [1.0, 0.0]
    ev = [
        TrackletEvidence(
            tracklet_id=1, start=0, end=150, team=0,
            embedding=np.asarray(a, dtype=float),
            xs=np.full(6, 0.2), ys=np.full(6, 0.5),
            exit_xy=np.array([0.2, 0.5]), entry_xy=np.array([0.2, 0.5]),
            exit_frame=25,  # observed 125 frames (5 s) before the span end
            entry_frame=0,
        ),
        TrackletEvidence(
            tracklet_id=2, start=175, end=200, team=0,
            embedding=np.asarray(a, dtype=float),
            xs=np.full(6, 0.2 + 45 / 105), ys=np.full(6, 0.5),
            entry_xy=np.array([0.2 + 45 / 105, 0.5]),
            exit_xy=np.array([0.2 + 45 / 105, 0.5]),
            entry_frame=175, exit_frame=200,
        ),
    ]
    res = merge_threads_two_pass(ev, model=_model(), min_score=0.0, pass2_score=None)
    assert [1, 2] in res.groups, res.groups
def test_pass2_merges_are_recorded_as_decisions():
    """Every merge must have a decision row behind it, whichever pass made it.

    Regression for the Lab showing "Correct merges · 1" and "No merged
    decisions" side by side (2026-08-02): only pass 1 was instrumented, so a
    thread-to-thread merge landed in the association trail with no decision and
    no channel working to explain it.
    """
    import numpy as np
    from matchlab_core.reid.twopass import merge_threads_two_pass

    a = np.array([1.0, 0.0])
    # Three fragments of one appearance. Pass 1 is causal and can only grow one
    # thread at a time, so the later pair is left for pass 2 to join.
    ev = [
        _ev(1, 0, 10, a),
        _ev(2, 100, 110, a),
        _ev(3, 200, 210, a),
    ]
    res = merge_threads_two_pass(
        ev, model=_model(), min_score=1e6, pass2_score=-1e6
    )

    merged_pairs = [p for p in res.pairs if p.decision == "merged"]
    assert merged_pairs, "expected pass 2 to merge something"

    merged_decisions = [d for d in res.decisions if d["decision"] == "merged"]
    assert len(merged_decisions) == len(merged_pairs), (
        "every merge needs a decision row -- otherwise the Lab reports a merge "
        "it cannot explain"
    )
    assert all(d["pass_no"] == 2 for d in merged_decisions)
    # And the working is there, reconciling with the recorded total.
    for d in merged_decisions:
        chosen = [c for c in d["candidates"] if c["partner"] == d["chosen"]]
        assert chosen, "the chosen partner must appear among the candidates"
        assert chosen[0]["total"] == pytest.approx(d["total"])
    assert any(b["pass_no"] == 2 for b in res.channel_breakdowns)


def _prior():
    from matchlab_core.reid.transition import DiffusionScale, TransitionPrior

    return TransitionPrior(
        x=DiffusionScale(sigma_inf=29.0, tau=34.0),
        y=DiffusionScale(sigma_inf=15.0, tau=20.0),
        impostor_x=32.0,
        impostor_y=21.0,
    )


def _scored_transition(model, a_ev, b_ev):
    sa, sb = a_ev.to_state(), b_ev.to_state()
    total, channels = model.score_channels(
        sa, sa.footprint(), sb, sb.footprint()
    )
    return total, {c["name"]: c for c in channels}


def test_transition_abstains_without_endpoints_instead_of_fabricating_them():
    """Two tracklets with no calibrated frames used to score displacement 0
    from substituted (0, 0) endpoints -- the prior's MAXIMUM positive evidence,
    from data that never existed."""
    model = _model(transition=1.0)
    model.prior = _prior()
    emb = np.array([1.0, 0.0])
    _, by_name = _scored_transition(
        model, _ev(1, 0, 10, emb), _ev(2, 20, 30, emb)
    )
    assert by_name["transition"]["llr"] is None
    assert by_name["transition"]["raw"] is None
    assert by_name["transition"]["contribution"] == 0.0


def test_transition_abstains_when_only_one_endpoint_is_missing():
    model = _model(transition=1.0)
    model.prior = _prior()
    emb = np.array([1.0, 0.0])
    a = TrackletEvidence(
        tracklet_id=1, start=0, end=10, team=0, embedding=emb,
        exit_xy=np.array([0.5, 0.5]),
    )
    b = _ev(2, 20, 30, emb)  # no entry_xy
    _, by_name = _scored_transition(model, a, b)
    assert by_name["transition"]["llr"] is None
    # And the mirror case: a has no exit, b has an entry.
    a2 = _ev(1, 0, 10, emb)
    b2 = TrackletEvidence(
        tracklet_id=2, start=20, end=30, team=0, embedding=emb,
        entry_xy=np.array([0.5, 0.5]),
    )
    _, by_name = _scored_transition(model, a2, b2)
    assert by_name["transition"]["llr"] is None


def test_transition_votes_with_real_endpoints_and_reports_the_true_gap():
    model = _model(transition=1.0)
    model.prior = _prior()
    emb = np.array([1.0, 0.0])
    a = TrackletEvidence(
        tracklet_id=1, start=0, end=10, team=0, embedding=emb,
        exit_xy=np.array([0.5, 0.5]),
    )
    b = TrackletEvidence(
        tracklet_id=2, start=20, end=30, team=0, embedding=emb,
        entry_xy=np.array([0.52, 0.5]),
    )
    _, by_name = _scored_transition(model, a, b)
    assert by_name["transition"]["llr"] is not None
    assert by_name["transition"]["llr"] > 0.0  # 2.1 m in 0.4 s: plausible
    assert by_name["transition"]["raw"] == pytest.approx((20 - 10) / 25.0)


def test_transition_abstains_on_non_positive_gap_even_with_endpoints():
    """Interleaved pass-2 threads reach scoring with a negative envelope gap;
    dt <= 0 floors the diffusion sigma at millimetres and the sign of the
    saturated output is an artifact of the floor, not evidence."""
    from matchlab_core.reid.threads import ThreadState

    model = _model(transition=1.0)
    model.prior = _prior()
    emb = np.array([1.0, 0.0])
    # Thread a spans [0, 100]; thread b's first member starts at 50 (inside
    # a's envelope) -- members disjoint, envelope gap negative.
    a = ThreadState.from_fragment(
        np.array([0.5]), np.array([0.5]), embedding=emb, start=0, end=100,
        exit_xy=np.array([0.5, 0.5]), entry_xy=np.array([0.5, 0.5]),
    )
    b = ThreadState.from_fragment(
        np.array([0.5]), np.array([0.5]), embedding=emb, start=50, end=60,
        exit_xy=np.array([0.5, 0.5]), entry_xy=np.array([0.5, 0.5]),
    )
    total, channels = model.score_channels(a, a.footprint(), b, b.footprint())
    by_name = {c["name"]: c for c in channels}
    assert by_name["transition"]["llr"] is None
    assert by_name["transition"]["raw"] == pytest.approx(-2.0)  # true gap kept
    assert by_name["gap"]["raw"] == 0.0  # gap channel serves the fitted clamp


def test_merged_thread_endpoint_goes_missing_with_an_uncalibrated_tail():
    """A thread whose temporally-last fragment had no calibrated frames has no
    honest exit point; propagating the earlier fragment's stale exit would be
    wrong evidence, so the merge propagates None."""
    emb = np.array([1.0, 0.0])
    with_xy = _ev(1, 0, 10, emb)
    with_xy = TrackletEvidence(
        tracklet_id=1, start=0, end=10, team=0, embedding=emb,
        exit_xy=np.array([0.3, 0.3]), entry_xy=np.array([0.3, 0.3]),
    )
    without = _ev(2, 20, 30, emb)
    merged = with_xy.to_state().merged_with(without.to_state())
    assert merged.exit_xy is None  # later fragment's (missing) exit wins
    assert merged.entry_xy is not None  # earlier fragment's real entry kept


def test_clip_transition_default_matches_saturate_and_neg_clamp_bounds():
    from matchlab_core.reid.evidence import clip_transition, saturate

    x = np.array([-5000.0, -3.0, -0.5, 0.0, 0.5, 3.0, 5000.0])
    assert clip_transition(x) == pytest.approx(saturate(x))
    clipped = clip_transition(x, 1.0)
    assert clipped.min() >= -1.0
    assert clipped[x >= 0] == pytest.approx(saturate(x)[x >= 0])
    assert np.all(np.diff(clipped) >= 0)  # monotone: ordering survives
    zeroed = clip_transition(x, 0.0)
    assert np.all(zeroed[x < 0] == 0.0)
    assert zeroed[x >= 0] == pytest.approx(saturate(x)[x >= 0])
