"""Episode construction: a query fragment and the field it competes in.

The unit the whole multi-input design operates on. Every cue scores candidates
WITHIN an episode, so the episode's membership rules are load-bearing: a
candidate that should not be in the field inflates every downstream number.
"""

from __future__ import annotations

from matchlab_core.reid.episodes import build_episodes


class Frag:
    """Minimal stand-in; build_episodes only reads these four attributes."""

    def __init__(self, idx, player_id, start, end, team):
        self.idx = idx
        self.player_id = player_id
        self.start = start
        self.end = end
        self.team = team


def _frags(spec):
    return [Frag(i, p, s, e, t) for i, (p, s, e, t) in enumerate(spec)]


def test_candidates_start_strictly_after_the_query_ends():
    frags = _frags([(1, 0, 100, 0), (2, 50, 150, 0), (3, 101, 200, 0)])
    eps = {e.query: e for e in build_episodes(frags)}
    assert eps[0].candidates == [2], "an overlapping fragment cannot be a continuation"


def test_candidates_are_same_team_only():
    frags = _frags([(1, 0, 100, 0), (2, 101, 200, 1), (3, 101, 200, 0)])
    eps = {e.query: e for e in build_episodes(frags)}
    assert eps[0].candidates == [2]


def test_query_with_no_candidates_is_dropped():
    frags = _frags([(1, 0, 100, 0), (2, 0, 100, 0)])
    assert build_episodes(frags) == []


def test_episode_reports_which_candidates_are_the_same_player():
    frags = _frags([(1, 0, 100, 0), (1, 101, 200, 0), (2, 101, 200, 0)])
    ep = build_episodes(frags)[0]
    assert ep.query == 0
    assert ep.candidates == [1, 2]
    assert ep.correct == [1]


def test_episode_with_no_correct_candidate_is_kept_and_marked():
    # 1.2% of real queries have no prior/continuing fragment for their player.
    # These MUST survive as abstain targets -- dropping them would train the
    # model that every episode has an answer.
    frags = _frags([(1, 0, 100, 0), (2, 101, 200, 0)])
    ep = build_episodes(frags)[0]
    assert ep.correct == []


def test_candidates_are_ascending_regardless_of_temporal_order_in_the_list():
    # Fragment 1 is LATER in time than fragment 2, so a naive implementation
    # could emit candidates in encounter order. They must come out ascending
    # by index so feature rows line up across arms.
    frags = _frags([(1, 0, 100, 0), (3, 300, 400, 0), (2, 101, 200, 0)])
    cands = build_episodes(frags)[0].candidates
    assert cands == [1, 2] == sorted(cands)
