"""SportsMOT + SoccerTrack ground-truth parsers (SPO-11 part 1).

Mirrors the fixture style of test_gt_eval.py::_write_soccernet_seq: hand-write
a tiny sequence on disk, parse it, assert hand-computed values.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pitchlab_core.gt import GroundTruth, load_soccertrack_sequence, load_sportsmot_sequence


def _write_sportsmot_seq(root: Path) -> Path:
    seq = root / "SPT-001"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SPT-001\nimDir=img1\nframeRate=30\nseqLength=4\n"
        "imWidth=1280\nimHeight=720\nimExt=.jpg\n"
    )
    rows = []
    for frame in range(1, 5):  # 1-based MOT frames
        # Track 1: frame 2 has conf=0 (an ignore region) and must be skipped.
        conf1 = 0 if frame == 2 else 1
        rows.append(f"{frame},1,50,60,30,90,{conf1},-1,-1,-1")
        rows.append(f"{frame},2,200,150,30,90,1,-1,-1,-1")
        rows.append(f"{frame},3,400,300,30,90,1,-1,-1,-1")
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))
    return seq


def test_load_sportsmot_sequence(tmp_path):
    gt = load_sportsmot_sequence(_write_sportsmot_seq(tmp_path))

    assert gt.source == "sportsmot"
    assert gt.sequence == "SPT-001"
    assert gt.fps == 30
    assert gt.width == 1280
    assert gt.height == 720
    assert gt.seq_length == 4

    by_id = {t.track_id: t for t in gt.tracks}
    assert set(by_id) == {1, 2, 3}
    for track in gt.tracks:
        assert track.role == "player"
        assert track.team is None
        assert track.jersey is None

    # Track 1: 4 rows written but frame 2 has conf=0 -> only 3 frames survive.
    t1 = by_id[1]
    assert len(t1.frames) == 3
    assert [f.frame_idx for f in t1.frames] == [0, 2, 3]  # frame 2 (1-based) skipped
    f0 = t1.frames[0]
    assert (f0.box.x1, f0.box.y1, f0.box.x2, f0.box.y2) == (50, 60, 80, 150)

    # Track 2: all 4 frames present, 0-based frame_idx.
    t2 = by_id[2]
    assert [f.frame_idx for f in t2.frames] == [0, 1, 2, 3]
    f0 = t2.frames[0]
    assert (f0.box.x1, f0.box.y1, f0.box.x2, f0.box.y2) == (200, 150, 230, 240)

    # Round-trips through JSON unchanged.
    assert GroundTruth.model_validate_json(gt.model_dump_json()) == gt


def test_load_sportsmot_sequence_missing_gt_txt(tmp_path):
    seq = tmp_path / "SPT-002"
    seq.mkdir()
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SPT-002\nimDir=img1\nframeRate=30\nseqLength=4\n"
        "imWidth=1280\nimHeight=720\nimExt=.jpg\n"
    )
    # gt/gt.txt deliberately absent.
    with pytest.raises(FileNotFoundError) as exc_info:
        load_sportsmot_sequence(seq)
    msg = str(exc_info.value)
    assert "gt.txt" in msg
    assert str(seq / "gt" / "gt.txt") in msg


def test_load_sportsmot_sequence_missing_seqinfo(tmp_path):
    seq = tmp_path / "SPT-003"
    (seq / "gt").mkdir(parents=True)
    (seq / "gt" / "gt.txt").write_text("1,1,0,0,10,10,1,-1,-1,-1")
    # seqinfo.ini deliberately absent.
    with pytest.raises(FileNotFoundError) as exc_info:
        load_sportsmot_sequence(seq)
    msg = str(exc_info.value)
    assert "seqinfo.ini" in msg
    assert str(seq / "seqinfo.ini") in msg


def test_load_sportsmot_sequence_malformed_row(tmp_path):
    """A corrupt gt.txt row (non-numeric field) must fail loudly, naming the
    gt.txt path and the offending row -- not a bare, contextless ValueError."""
    seq = tmp_path / "SPT-004"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SPT-004\nimDir=img1\nframeRate=30\nseqLength=1\n"
        "imWidth=1280\nimHeight=720\nimExt=.jpg\n"
    )
    (seq / "gt" / "gt.txt").write_text("1,1,notanumber,60,30,90,1,-1,-1,-1")
    with pytest.raises(ValueError) as exc_info:
        load_sportsmot_sequence(seq)
    msg = str(exc_info.value)
    assert str(seq / "gt" / "gt.txt") in msg
    assert "notanumber" in msg


# --- SoccerTrack ---------------------------------------------------------
#
# 3-row header: TeamID / PlayerID / attribute name, each column-group of 4
# covering bb_left, bb_top, bb_width, bb_height for one (team, player).
# Groups here: team0/player0, team0/player1, team1/player0, team1/player1,
# team3(ball)/player0. 3 frames, one empty cell (team0/player1 @ frame 1).

_ST_GROUPS = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 0)]


def _write_soccertrack_csv(path: Path) -> Path:
    header_team = [""] + [str(team) for team, _ in _ST_GROUPS for _ in range(4)]
    header_player = [""] + [str(player) for _, player in _ST_GROUPS for _ in range(4)]
    header_attr = [""] + ["bb_left", "bb_top", "bb_width", "bb_height"] * len(_ST_GROUPS)

    # (x, y, w, h) per group per frame; None means "cell empty / player absent".
    data = {
        0: [(10, 10, 20, 30), (100, 100, 40, 120), (300, 100, 40, 120), (400, 200, 40, 120), (500, 300, 10, 10)],
        1: [(12, 10, 20, 30), None, (302, 100, 40, 120), (402, 200, 40, 120), (505, 300, 10, 10)],
        2: [(14, 10, 20, 30), (104, 100, 40, 120), (304, 100, 40, 120), (404, 200, 40, 120), (510, 300, 10, 10)],
    }

    rows = [header_team, header_player, header_attr]
    for frame in sorted(data):
        row = [str(frame)]
        for cell in data[frame]:
            row.extend(["", "", "", ""] if cell is None else [str(v) for v in cell])
        rows.append(row)

    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def test_load_soccertrack_sequence(tmp_path):
    csv_path = _write_soccertrack_csv(tmp_path / "SNMOT-clip.csv")
    gt = load_soccertrack_sequence(csv_path, fps=25.0, width=1920, height=1080)

    assert gt.source == "soccertrack"
    assert gt.fps == 25.0
    assert gt.width == 1920
    assert gt.height == 1080

    by_id = {t.track_id: t for t in gt.tracks}
    # team_id*1000 + player_id; ball fixed at 9999.
    assert set(by_id) == {0, 1, 1000, 1001, 9999}

    t_00 = by_id[0]  # team 0, player 0
    assert t_00.role == "player" and t_00.team == "left"
    assert [f.frame_idx for f in t_00.frames] == [0, 1, 2]
    f0 = t_00.frames[0]
    assert (f0.box.x1, f0.box.y1, f0.box.x2, f0.box.y2) == (10, 10, 30, 40)

    t_01 = by_id[1]  # team 0, player 1 -- empty cell at frame 1
    assert t_01.role == "player" and t_01.team == "left"
    assert [f.frame_idx for f in t_01.frames] == [0, 2]  # frame 1 skipped
    f0 = t_01.frames[0]
    assert (f0.box.x1, f0.box.y1, f0.box.x2, f0.box.y2) == (100, 100, 140, 220)

    t_10 = by_id[1000]  # team 1, player 0
    assert t_10.role == "player" and t_10.team == "right"
    assert len(t_10.frames) == 3

    ball = by_id[9999]
    assert ball.role == "ball"
    assert ball.team is None
    assert len(ball.frames) == 3
    fb0 = ball.frames[0]
    assert (fb0.box.x1, fb0.box.y1, fb0.box.x2, fb0.box.y2) == (500, 300, 510, 310)

    # Round-trips through JSON unchanged.
    assert GroundTruth.model_validate_json(gt.model_dump_json()) == gt


def test_load_soccertrack_sequence_malformed_header(tmp_path):
    csv_path = tmp_path / "bad.csv"
    # attr row has a different column count than the team/player rows above it.
    csv_path.write_text("0,0,0,0\n0,0,0,0\n0,10,10,20,30\n")
    with pytest.raises(ValueError) as exc_info:
        load_soccertrack_sequence(csv_path, fps=25.0, width=1920, height=1080)
    assert str(csv_path) in str(exc_info.value)


def test_load_soccertrack_sequence_bad_attribute_names(tmp_path):
    csv_path = tmp_path / "bad_attrs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "0", "0", "0", "0"])
        w.writerow(["", "0", "0", "0", "0"])
        # Wrong attribute names / order -- should not silently be accepted.
        w.writerow(["", "bb_left", "bb_top", "bb_width", "nonsense"])
        w.writerow(["0", "10", "10", "20", "30"])
    with pytest.raises(ValueError) as exc_info:
        load_soccertrack_sequence(csv_path, fps=25.0, width=1920, height=1080)
    assert str(csv_path) in str(exc_info.value)


def test_load_soccertrack_sequence_missing_file(tmp_path):
    csv_path = tmp_path / "does-not-exist.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_soccertrack_sequence(csv_path, fps=25.0, width=1920, height=1080)
    assert str(csv_path) in str(exc_info.value)


def test_load_soccertrack_sequence_non_numeric_frame_cell(tmp_path):
    """A corrupt data row (non-numeric frame number) must fail loudly, naming
    the CSV path and the offending row."""
    csv_path = tmp_path / "bad_frame.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "0", "0", "0", "0"])
        w.writerow(["", "0", "0", "0", "0"])
        w.writerow(["", "bb_left", "bb_top", "bb_width", "bb_height"])
        w.writerow(["notaframe", "10", "10", "20", "30"])
    with pytest.raises(ValueError) as exc_info:
        load_soccertrack_sequence(csv_path, fps=25.0, width=1920, height=1080)
    msg = str(exc_info.value)
    assert str(csv_path) in msg
    assert "notaframe" in msg


def test_load_soccertrack_sequence_nan_cell_and_short_row(tmp_path):
    """Exercises two edge cases together: (1) a cell group whose 4 values are
    literal "nan" -- the NaN-safe branch must treat it as absent, same as a
    blank cell; (2) a data row shorter than the header declares (trailing
    group entirely missing) -- the row-length guard must treat the missing
    group as absent rather than raising an IndexError."""
    csv_path = tmp_path / "nan_short.csv"
    header_team = [""] + [str(team) for team, _ in _ST_GROUPS for _ in range(4)]
    header_player = [""] + [str(player) for _, player in _ST_GROUPS for _ in range(4)]
    header_attr = [""] + ["bb_left", "bb_top", "bb_width", "bb_height"] * len(_ST_GROUPS)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header_team)
        w.writerow(header_player)
        w.writerow(header_attr)
        # Frame 0: group1 (team0/player1) is all-NaN -> must be treated as absent.
        w.writerow(
            [
                "0",
                "10", "10", "20", "30",  # team0/player0
                "nan", "nan", "nan", "nan",  # team0/player1 -- NaN
                "300", "100", "40", "120",  # team1/player0
                "400", "200", "40", "120",  # team1/player1
                "500", "300", "10", "10",  # ball
            ]
        )
        # Frame 1: row is short -- the trailing ball group's 4 columns are
        # entirely absent from the row (fewer columns than the header).
        w.writerow(
            [
                "1",
                "12", "10", "20", "30",  # team0/player0
                "104", "100", "40", "120",  # team0/player1
                "302", "100", "40", "120",  # team1/player0
                "402", "200", "40", "120",  # team1/player1
                # ball group omitted entirely
            ]
        )

    gt = load_soccertrack_sequence(csv_path, fps=25.0, width=1920, height=1080)
    by_id = {t.track_id: t for t in gt.tracks}

    # team0/player1 (track_id 1): frame 0 was all-NaN -> no frame there.
    t_01 = by_id[1]
    assert [f.frame_idx for f in t_01.frames] == [1]

    # team0/player0 (track_id 0) has valid data on both frames.
    t_00 = by_id[0]
    assert [f.frame_idx for f in t_00.frames] == [0, 1]

    # Ball (9999): present at frame 0; frame 1's row was too short to include it.
    ball = by_id[9999]
    assert [f.frame_idx for f in ball.frames] == [0]


def test_load_soccertrack_sequence_duplicate_group_raises(tmp_path):
    """Two header groups naming the same (team_id, player_id) would silently
    merge their frames into one track -- must fail loudly, naming the path
    and the duplicate id pair, rather than merging."""
    csv_path = tmp_path / "dup_group.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        # Two groups, both (team=0, player=0).
        w.writerow(["", "0", "0", "0", "0", "0", "0", "0", "0"])
        w.writerow(["", "0", "0", "0", "0", "0", "0", "0", "0"])
        w.writerow(
            [
                "",
                "bb_left", "bb_top", "bb_width", "bb_height",
                "bb_left", "bb_top", "bb_width", "bb_height",
            ]
        )
        w.writerow(["0", "10", "10", "20", "30", "50", "50", "20", "30"])
    with pytest.raises(ValueError) as exc_info:
        load_soccertrack_sequence(csv_path, fps=25.0, width=1920, height=1080)
    msg = str(exc_info.value)
    assert str(csv_path) in msg
    assert "(0, 0)" in msg


def test_load_soccertrack_sequence_player_id_out_of_range(tmp_path):
    """PlayerID 1000 for team_id=1 would collide with team_id=0, player_id=1000
    (or with team_id=1, player_id=0's own id space) under the deterministic
    team_id*1000+player_id scheme -- must fail loudly rather than silently
    assigning a colliding track_id."""
    csv_path = tmp_path / "bad_player_id.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "0", "0", "0", "0"])
        w.writerow(["", "1000", "1000", "1000", "1000"])
        w.writerow(["", "bb_left", "bb_top", "bb_width", "bb_height"])
        w.writerow(["0", "10", "10", "20", "30"])
    with pytest.raises(ValueError) as exc_info:
        load_soccertrack_sequence(csv_path, fps=25.0, width=1920, height=1080)
    msg = str(exc_info.value)
    assert str(csv_path) in msg
    assert "1000" in msg
