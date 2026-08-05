"""FOOTPASS ground truth -> `matchlab_core.stats` MatchEvent stream.

Ground-truth only: this adapter reads labels, never pipeline output. It is the
substrate the Tier 1 stats are validated on.

Two files, both required
------------------------
* tactical HDF5 -- `CLS` on the acting player's row gives the event class, and
  the same row gives that player's pitch position; the other 21 rows at that
  frame give the off-ball context.
* `playbyplay_{split}.json` -- the ONLY place the `replay` flag exists. 10.1% of
  val events (614/6070) are broadcast replays that duplicate live play. Reading
  the h5 alone silently doubles those events and fabricates a possession change
  at each replay boundary (21.5% of raw boundaries). The two streams are
  otherwise identical -- verified exact on the full (frame, team, shirt, class)
  tuple set for all 6 val halves -- which is why the obvious cross-check between
  them proves nothing.

Identity
--------
`PLAYER_ID` is the stable player key and `PLAYER_ID // 100` the stable club key.
`TEAM` is **pitch side**, not club: the club<->TEAM binding rebinds at half-time
(verified 18/18, 21/21, 21/21 players in the three val matches), and shirts
collide across teams within a half. `(TEAM, SHIRT)` is used only to join the
play-by-play row to the h5 row.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matchlab_core.pitch import FIFA_PITCH, PitchSpec
from matchlab_core.stats.schema import ActorKey, MatchEvent, PitchPoint, StatEventType
from matchlab_core.stats.zones import is_plausible, normalised_to_cm

from matchlab_train.datasets.footpass import COL, FootpassHalf, load_half

#: FOOTPASS CLS index -> canonical stats vocabulary. Settled against the data,
#: not against `docs/reference/footpass-setup.md`, whose order is wrong: class 4
#: is 86.6% within 5% of a touchline (throw-in) and class 5 is 0% touchline and
#: 64% near a goal end (shot).
CLS_TO_TYPE: dict[int, StatEventType] = {
    1: StatEventType.CARRY,  # "drive"
    2: StatEventType.PASS,
    3: StatEventType.CROSS,
    4: StatEventType.THROW_IN,
    5: StatEventType.SHOT,
    6: StatEventType.HEADER,
    7: StatEventType.TACKLE,
    8: StatEventType.BLOCK,
}

FOOTPASS_FPS = 25.0

#: TEAM is pitch side: 0 defends the left goal and therefore attacks +x.
_ATTACKS_POSITIVE_X = {0: True, 1: False}


def club_of(player_id: int) -> int:
    """Stable club key. Constant across halves, unlike TEAM."""
    return int(player_id) // 100


def load_replay_flags(playbyplay_path: str | Path) -> dict[str, dict[tuple[int, int, int, int], bool]]:
    """`{half_key: {(frame, team, shirt, cls): is_replay}}`."""
    raw = json.loads(Path(playbyplay_path).read_text())
    out: dict[str, dict[tuple[int, int, int, int], bool]] = {}
    for key, rows in raw["events"].items():
        out[key] = {
            (int(r[0]), int(r[1]), int(r[2]), int(r[3])): bool(r[5]) for r in rows
        }
    return out


def _positions_at_frame(
    rows: np.ndarray, pitch: PitchSpec, attacks_positive_x: bool
) -> tuple[list[PitchPoint], list[int], int]:
    """Attack-normalised positions for every player row at one frame."""
    pts: list[PitchPoint] = []
    clubs: list[int] = []
    rejected = 0
    for r in rows:
        p = normalised_to_cm(
            float(r[COL.X]), float(r[COL.Y]), pitch, attacks_positive_x=attacks_positive_x
        )
        if not is_plausible(p, pitch):
            rejected += 1
            continue
        pts.append(p)
        clubs.append(club_of(int(r[COL.PLAYER_ID])))
    return pts, clubs, rejected


def build_events(
    half: FootpassHalf,
    replay_flags: dict[tuple[int, int, int, int], bool] | None = None,
    *,
    pitch: PitchSpec = FIFA_PITCH,
    with_offball: bool = True,
) -> tuple[list[MatchEvent], int]:
    """Build the MatchEvent stream for one half. Returns (events, n_rejected).

    `replay_flags` comes from `load_replay_flags`. Passing None means the replay
    flag is unavailable, and every event is marked `replay=False` -- that is a
    materially different (and worse) stream, so callers must opt into it
    knowingly rather than by forgetting the argument.
    """
    rows = half.rows
    match_id = half.game_id
    by_frame: dict[int, np.ndarray] = {}
    order = np.argsort(rows[:, COL.FRAME], kind="stable")
    rows = rows[order]
    frames = rows[:, COL.FRAME].astype(int)
    # Contiguous runs of equal frame, since rows are now frame-sorted.
    bounds = np.flatnonzero(np.diff(frames)) + 1
    for chunk in np.split(np.arange(len(frames)), bounds):
        by_frame[int(frames[chunk[0]])] = rows[chunk]

    events: list[MatchEvent] = []
    rejected = 0
    event_rows = rows[rows[:, COL.CLS] > 0]
    event_rows = event_rows[np.argsort(event_rows[:, COL.FRAME], kind="stable")]

    for i, r in enumerate(event_rows):
        cls = int(r[COL.CLS])
        etype = CLS_TO_TYPE.get(cls)
        if etype is None:
            continue
        frame = int(r[COL.FRAME])
        team = int(r[COL.TEAM])
        shirt = int(r[COL.SHIRT])
        player_id = int(r[COL.PLAYER_ID])
        attacks_pos = _ATTACKS_POSITIVE_X[team]
        start = normalised_to_cm(
            float(r[COL.X]), float(r[COL.Y]), pitch, attacks_positive_x=attacks_pos
        )
        if not is_plausible(start, pitch):
            rejected += 1
            continue

        club = club_of(player_id)
        actor = ActorKey(
            player_id=player_id, club_id=club, shirt=shirt, role=int(r[COL.ROLE]) or None
        )

        teammates: list[PitchPoint] = []
        opponents: list[PitchPoint] = []
        if with_offball:
            pts, clubs, rej = _positions_at_frame(by_frame[frame], pitch, attacks_pos)
            rejected += rej
            for p, c in zip(pts, clubs, strict=True):
                (teammates if c == club else opponents).append(p)

        replay = bool(replay_flags.get((frame, team, shirt, cls), False)) if replay_flags else False
        events.append(
            MatchEvent(
                attrs={"team_side": team},
                event_id=i,
                match_id=match_id,
                half=half.half,
                frame_idx=frame,
                t=frame / FOOTPASS_FPS,
                type=etype,
                actor=actor,
                club_id=club,
                start=start,
                replay=replay,
                teammates=teammates,
                opponents=opponents,
            )
        )

    _fill_end_points(events, by_frame, pitch)
    return events, rejected


def _fill_end_points(
    events: list[MatchEvent], by_frame: dict[int, np.ndarray], pitch: PitchSpec
) -> None:
    """End location of a ball-moving event.

    FOOTPASS labels one point per event -- where the acting player was when they
    acted -- so the end point is reconstructed, and which position is read
    differs by type:

    * pass / cross / throw-in: the NEXT actor's position at the next event's
      frame. That is where the ball arrived, whoever received it.
    * carry ("drive"): the SAME player's position at the next event's frame,
      i.e. how far they took it before the next thing happened.

    Both are reconstructions, not labels. A carry whose next event is far in the
    future gets no end point rather than an invented one.

    The successor must have the SAME replay status as the event. 3.8% of live
    ball-moving events on the val split are immediately followed in the raw
    stream by a broadcast replay; reading the "next actor's position" from a
    replay row takes a position from re-broadcast old footage, not from where
    the ball arrived -- and it disagrees with the outcome/receiver, which the
    chain layer resolves against the post-replay live event. So a live event's
    end point comes from the next live event, and a replay's from the next
    replay row (those are filtered out downstream anyway).
    """
    from matchlab_core.stats.chains import DEFAULT_MAX_GAP_S

    for i, ev in enumerate(events):
        nxt = None
        for j in range(i + 1, len(events)):
            if events[j].replay == ev.replay:
                nxt = events[j]
                break
        if nxt is None:
            continue
        if nxt.half != ev.half or (nxt.t - ev.t) > DEFAULT_MAX_GAP_S:
            continue
        if ev.type is StatEventType.SHOT:
            continue
        # The end point must be expressed in the ACTING club's frame of
        # reference, so it is normalised with the acting event's direction --
        # the raw pitch side stamped on the event at build time.
        attacks_pos = _ATTACKS_POSITIVE_X[int(ev.attrs["team_side"])]
        target_player = (
            ev.actor.player_id
            if ev.type is StatEventType.CARRY and ev.actor is not None
            else (nxt.actor.player_id if nxt.actor is not None else None)
        )
        if target_player is None:
            continue
        rows = by_frame.get(nxt.frame_idx)
        if rows is None:
            continue
        match = rows[rows[:, COL.PLAYER_ID] == target_player]
        if not len(match):
            continue
        end = normalised_to_cm(
            float(match[0, COL.X]),
            float(match[0, COL.Y]),
            pitch,
            attacks_positive_x=attacks_pos,
        )
        if is_plausible(end, pitch):
            ev.end = end


def load_half_events(
    tactical_path: str | Path,
    key: str,
    playbyplay_path: str | Path | None = None,
    *,
    pitch: PitchSpec = FIFA_PITCH,
    with_offball: bool = True,
) -> tuple[list[MatchEvent], int]:
    """Load one half's ground-truth event stream, replay flags attached."""
    half = load_half(tactical_path, key)
    flags = load_replay_flags(playbyplay_path)[key] if playbyplay_path else None
    return build_events(half, flags, pitch=pitch, with_offball=with_offball)


def coverage_frames(half: FootpassHalf) -> dict[int, tuple[int, int]]:
    """`{player_id: (observed_frames, total_frames)}` from the in-frame flag.

    `ROI_X` is NaN when the player is off camera, which is the coverage
    denominator the source doc requires on every stat line.
    """
    rows = half.rows
    total = len({int(f) for f in rows[:, COL.FRAME]})
    seen: dict[int, int] = defaultdict(int)
    visible = rows[~np.isnan(rows[:, COL.ROI_X])]
    for pid in visible[:, COL.PLAYER_ID].astype(int):
        seen[int(pid)] += 1
    return {pid: (n, total) for pid, n in seen.items()}
