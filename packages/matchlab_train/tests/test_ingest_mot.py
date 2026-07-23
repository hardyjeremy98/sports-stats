"""Tests for the SportsMOT/SoccerTrack ingest commands, the shared
split-manifest writer, and end-to-end scoreability of an ingested fixture
sequence (SPO-11 part 2)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from matchlab_core.gt import GroundTruth
from matchlab_train.datasets.manifest import update_tier_manifest
from matchlab_train.datasets.soccertrack import ingest_soccertrack
from matchlab_train.datasets.sportsmot import ingest_sportsmot

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point matchlab_server settings (data dir, DB, configs dir) at a
    private tmp tree, the same MATCHLAB_* env-var override existing tests
    use (see test_reid_labels.py's `db` fixture), so these tests can never
    touch the real repo's data/matchlab.db or configs/datasets/*.json."""
    import matchlab_server.db as db_module
    from matchlab_server.settings import get_settings

    data_dir = tmp_path / "data"
    config_dir = tmp_path / "configs"
    monkeypatch.setenv("MATCHLAB_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MATCHLAB_DATABASE_URL", f"sqlite:///{data_dir}/test.db")
    monkeypatch.setenv("MATCHLAB_CONFIG_DIR", str(config_dir))
    get_settings.cache_clear()
    db_module._engine = None
    db_module._session_factory = None

    db_module.init_db()
    yield get_settings()

    db_module._engine = None
    db_module._session_factory = None
    get_settings.cache_clear()


def _write_sportsmot_seq(split_dir: Path, name: str = "SPORTSMOT-001", n_frames: int = 8) -> Path:
    """A tiny hand-written SportsMOT sequence: standard MOT17-style
    seqinfo.ini + gt/gt.txt, plus real (tiny, solid-color) img1/ jpgs so
    ffmpeg has something to stitch."""
    seq = split_dir / name
    (seq / "gt").mkdir(parents=True)
    (seq / "img1").mkdir()
    seq_info = (
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=5\nseqLength={n_frames}\n"
        f"imWidth=64\nimHeight=64\nimExt=.jpg\n"
    )
    (seq / "seqinfo.ini").write_text(seq_info)

    rows = []
    for frame in range(1, n_frames + 1):  # 1-based MOT frames
        rows.append(f"{frame},1,10,10,20,20,1,-1,-1,-1")
        rows.append(f"{frame},2,30,30,20,20,1,-1,-1,-1")
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))

    for i in range(1, n_frames + 1):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:] = (i * 20) % 255
        cv2.imwrite(str(seq / "img1" / f"{i:06d}.jpg"), img)
    return seq


def _write_soccertrack_pair(
    root: Path, name: str = "SOCCERTRACK-001", n_frames: int = 5, fps: float = 5.0, size: int = 64
) -> tuple[Path, Path]:
    """A tiny hand-written SoccerTrack (video, csv) pair: a real (solid-color)
    mp4 written directly with cv2 (no ffmpeg needed -- this ingest path
    doesn't stitch frames) plus a matching bbox CSV for one team-0 player."""
    root.mkdir(parents=True, exist_ok=True)
    mp4 = root / f"{name}.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(n_frames):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (i * 20) % 255
        writer.write(img)
    writer.release()

    lines = [
        ",0,0,0,0",
        ",0,0,0,0",
        ",bb_left,bb_top,bb_width,bb_height",
    ]
    for f in range(n_frames):
        lines.append(f"{f},15,15,20,20")
    csv_path = root / f"{name}.csv"
    csv_path.write_text("\n".join(lines))
    return mp4, csv_path


# --- ingest-sportsmot --------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not on PATH; required to stitch SportsMOT frames")
def test_ingest_sportsmot_registers_video_and_manifest(env, tmp_path):
    root = tmp_path / "sportsmot_root"
    _write_sportsmot_seq(root / "val", name="SPORTSMOT-001")

    registered = ingest_sportsmot(root, split="val")
    assert len(registered) == 1
    video_id, name = registered[0]
    assert name == "SPORTSMOT-001"

    from matchlab_server.db import session
    from matchlab_server.models import Video

    with session() as db:
        video = db.get(Video, video_id)
        assert video is not None
        assert video.gt_path
        gt = GroundTruth.model_validate_json(Path(video.gt_path).read_text())

    assert gt.source == "sportsmot"
    assert len(gt.tracks) == 2
    assert all(t.role == "player" for t in gt.tracks)

    manifest_path = env.config_dir / "datasets" / "sportsmot.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["tier"] == "sportsmot"
    assert manifest["source_split"] == "val"
    seq = next(s for s in manifest["sequences"] if s["name"] == "SPORTSMOT-001")
    assert seq["role"] == "tuning"
    # Paths are recorded relative to the dir containing configs/ (matching
    # the existing hand-maintained soccernet.json), not as absolute paths.
    assert not Path(seq["video"]).is_absolute()
    assert not Path(seq["gt"]).is_absolute()
    assert (env.config_dir.parent / seq["video"]).exists()
    assert (env.config_dir.parent / seq["gt"]).exists()


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not on PATH; required to stitch SportsMOT frames")
def test_ingest_sportsmot_missing_split_dir_raises(env, tmp_path):
    root = tmp_path / "sportsmot_root"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="val"):
        ingest_sportsmot(root, split="val")


# --- ingest-soccertrack -------------------------------------------------------


def test_ingest_soccertrack_registers_video_and_manifest(env, tmp_path):
    root = tmp_path / "soccertrack_root"
    _write_soccertrack_pair(root, name="SOCCERTRACK-001")

    registered = ingest_soccertrack(root)
    assert len(registered) == 1
    video_id, name = registered[0]
    assert name == "SOCCERTRACK-001"

    from matchlab_server.db import session
    from matchlab_server.models import Video

    with session() as db:
        video = db.get(Video, video_id)
        assert video is not None
        assert video.gt_path
        gt = GroundTruth.model_validate_json(Path(video.gt_path).read_text())

    assert gt.source == "soccertrack"
    assert len(gt.tracks) == 1
    assert gt.tracks[0].role == "player" and gt.tracks[0].team == "left"

    manifest_path = env.config_dir / "datasets" / "soccertrack.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["tier"] == "soccertrack"
    seq = next(s for s in manifest["sequences"] if s["name"] == "SOCCERTRACK-001")
    assert seq["role"] == "tuning"
    assert not Path(seq["video"]).is_absolute()
    assert not Path(seq["gt"]).is_absolute()
    assert (env.config_dir.parent / seq["video"]).exists()
    assert (env.config_dir.parent / seq["gt"]).exists()


def test_ingest_soccertrack_no_pairs_raises(env, tmp_path):
    root = tmp_path / "soccertrack_root"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match=str(root)):
        ingest_soccertrack(root)


def test_ingest_soccertrack_respects_role(env, tmp_path):
    root = tmp_path / "soccertrack_root"
    _write_soccertrack_pair(root, name="SOCCERTRACK-HELD-OUT")

    ingest_soccertrack(root, role="held_out")

    manifest_path = env.config_dir / "datasets" / "soccertrack.json"
    manifest = json.loads(manifest_path.read_text())
    seq = next(s for s in manifest["sequences"] if s["name"] == "SOCCERTRACK-HELD-OUT")
    assert seq["role"] == "held_out"


# --- split-manifest writer: merge semantics -----------------------------------


def _touch_pair(tmp_path: Path, stem: str) -> tuple[Path, Path]:
    video = tmp_path / f"{stem}.mp4"
    gt = tmp_path / f"{stem}.gt.json"
    video.write_bytes(b"fake-video-bytes")
    gt.write_text("{}")
    return video, gt


def test_manifest_missing_path_raises_before_writing(env, tmp_path):
    manifest_path = env.config_dir / "datasets" / "sportsmot.json"
    with pytest.raises(FileNotFoundError, match="SEQ-MISSING"):
        update_tier_manifest(
            tier="sportsmot",
            dataset="sportsmot",
            source_split="val",
            entries=[
                {
                    "name": "SEQ-MISSING",
                    "video": str(tmp_path / "nope.mp4"),
                    "gt": str(tmp_path / "nope.gt.json"),
                    "role": "tuning",
                }
            ],
        )
    assert not manifest_path.exists()


def test_manifest_tuning_is_permanent_raises_on_flip_to_held_out(env, tmp_path):
    video, gt = _touch_pair(tmp_path, "SEQ-A")
    update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[{"name": "SEQ-A", "video": str(video), "gt": str(gt), "role": "tuning"}],
    )

    with pytest.raises(RuntimeError, match="SEQ-A"):
        update_tier_manifest(
            tier="sportsmot",
            dataset="sportsmot",
            source_split="val",
            entries=[{"name": "SEQ-A", "video": str(video), "gt": str(gt), "role": "held_out"}],
        )

    # The failed flip must not have mutated the on-disk manifest.
    manifest_path = env.config_dir / "datasets" / "sportsmot.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sequences"][0]["role"] == "tuning"


def test_manifest_new_sequence_recorded_with_requested_role(env, tmp_path):
    video_a, gt_a = _touch_pair(tmp_path, "SEQ-A")
    video_b, gt_b = _touch_pair(tmp_path, "SEQ-B")

    update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[{"name": "SEQ-A", "video": str(video_a), "gt": str(gt_a), "role": "tuning"}],
    )
    manifest_path = update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[{"name": "SEQ-B", "video": str(video_b), "gt": str(gt_b), "role": "held_out"}],
    )

    manifest = json.loads(manifest_path.read_text())
    by_name = {s["name"]: s for s in manifest["sequences"]}
    # SEQ-A (not part of this call's entries) is preserved by the merge.
    assert by_name["SEQ-A"]["role"] == "tuning"
    assert by_name["SEQ-B"]["role"] == "held_out"


def test_manifest_held_out_promoted_to_tuning_is_allowed(env, tmp_path):
    """The README's Roles section only forbids tuning -> held_out; promotion
    the other way (held_out -> tuning) must succeed, not raise."""
    video, gt = _touch_pair(tmp_path, "SEQ-A")
    update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[{"name": "SEQ-A", "video": str(video), "gt": str(gt), "role": "held_out"}],
    )

    manifest_path = update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[{"name": "SEQ-A", "video": str(video), "gt": str(gt), "role": "tuning"}],
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["sequences"][0]["role"] == "tuning"


def test_manifest_notes_merge_and_dedup(env, tmp_path):
    video, gt = _touch_pair(tmp_path, "SEQ-A")
    update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[{"name": "SEQ-A", "video": str(video), "gt": str(gt), "role": "tuning"}],
        notes=["first note"],
    )

    # Re-running with an overlapping notes list must append the new note
    # without repeating the one already recorded.
    manifest_path = update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[{"name": "SEQ-A", "video": str(video), "gt": str(gt), "role": "tuning"}],
        notes=["first note", "second note"],
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["notes"] == ["first note", "second note"]


def test_manifest_sequences_grouped_tuning_before_held_out(env, tmp_path):
    """Ordering is role-grouped (all tuning, then all held_out), each group
    ascending by name -- NOT a flat alphabetical sort across both roles. A
    tuning sequence named ZZZ must sort before a held_out sequence named
    AAA, per configs/datasets/README.md's Determinism section."""
    video_aaa, gt_aaa = _touch_pair(tmp_path, "AAA")
    video_zzz, gt_zzz = _touch_pair(tmp_path, "ZZZ")

    manifest_path = update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[
            {"name": "AAA", "video": str(video_aaa), "gt": str(gt_aaa), "role": "held_out"},
            {"name": "ZZZ", "video": str(video_zzz), "gt": str(gt_zzz), "role": "tuning"},
        ],
    )

    manifest = json.loads(manifest_path.read_text())
    assert [s["name"] for s in manifest["sequences"]] == ["ZZZ", "AAA"]


def test_manifest_path_outside_root_raises_naming_path_and_root(env, tmp_path, tmp_path_factory):
    """A video/gt path that isn't under the configs-dir's parent can't be
    made repo-relative -- must raise loudly (naming both the offending path
    and the root it was checked against), never silently fall back to an
    absolute, unportable path."""
    outside = tmp_path_factory.mktemp("outside-root")
    video = outside / "a.mp4"
    video.write_bytes(b"fake-video-bytes")
    gt = outside / "a.gt.json"
    gt.write_text("{}")

    with pytest.raises(RuntimeError) as exc_info:
        update_tier_manifest(
            tier="sportsmot",
            dataset="sportsmot",
            source_split="val",
            entries=[
                {"name": "SEQ-OUTSIDE", "video": str(video), "gt": str(gt), "role": "tuning"}
            ],
        )

    message = str(exc_info.value)
    assert str(video.resolve()) in message
    assert str(tmp_path) in message  # the root (env.config_dir.parent) it was checked against
    manifest_path = env.config_dir / "datasets" / "sportsmot.json"
    assert not manifest_path.exists()


def test_manifest_output_is_byte_identical_across_identical_runs(env, tmp_path):
    video, gt = _touch_pair(tmp_path, "SEQ-A")
    entries = [{"name": "SEQ-A", "video": str(video), "gt": str(gt), "role": "tuning"}]

    path1 = update_tier_manifest(
        tier="sportsmot", dataset="sportsmot", source_split="val", entries=entries
    )
    content1 = path1.read_bytes()
    path2 = update_tier_manifest(
        tier="sportsmot", dataset="sportsmot", source_split="val", entries=entries
    )
    content2 = path2.read_bytes()

    assert path1 == path2
    assert content1 == content2


def test_manifest_created_preserved_across_days_not_restamped(env, tmp_path):
    """`created` means first-creation date, not last-write date: a no-op
    (or content-changing) re-ingest on a later day must not restamp it, or
    hash_dataset_manifest would produce different bytes for identical
    content and check_evaluation_set would refuse a legitimate comparison."""
    video_a, gt_a = _touch_pair(tmp_path, "SEQ-A")
    entries = [{"name": "SEQ-A", "video": str(video_a), "gt": str(gt_a), "role": "tuning"}]

    manifest_path = update_tier_manifest(
        tier="sportsmot", dataset="sportsmot", source_split="val", entries=entries
    )
    original_bytes = manifest_path.read_bytes()

    # Simulate the manifest having been created on an earlier day: rewrite
    # its "created" field on disk to a fixed past date.
    on_disk = json.loads(manifest_path.read_text())
    on_disk["created"] = "2020-01-01"
    manifest_path.write_text(json.dumps(on_disk, sort_keys=True, indent=2) + "\n")
    backdated_bytes = manifest_path.read_bytes()

    # Re-running with identical content must reproduce the byte-identical
    # (backdated) file -- "created" must not be restamped to today.
    replayed_path = update_tier_manifest(
        tier="sportsmot", dataset="sportsmot", source_split="val", entries=entries
    )
    assert replayed_path.read_bytes() == backdated_bytes

    # Re-running with new content must update the sequences but still leave
    # "created" at the backdated value.
    video_b, gt_b = _touch_pair(tmp_path, "SEQ-B")
    updated_path = update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split="val",
        entries=[{"name": "SEQ-B", "video": str(video_b), "gt": str(gt_b), "role": "tuning"}],
    )
    updated = json.loads(updated_path.read_text())
    assert updated["created"] == "2020-01-01"
    assert {s["name"] for s in updated["sequences"]} == {"SEQ-A", "SEQ-B"}
    assert original_bytes != backdated_bytes  # sanity: the backdate actually changed bytes


# --- end-to-end scoreability (SPO-11 acceptance criterion) -------------------


def test_end_to_end_scoreability_of_ingested_sequence(env, tmp_path):
    """An ingested fixture's .gt.json must be usable by the real evaluation
    path, not just present on disk: build a run dir whose tracklets exactly
    echo the ingested ground truth and confirm evaluate_run scores it near
    perfectly (idf1/hota ~1.0, zero id switches)."""
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    root = tmp_path / "soccertrack_root"
    _write_soccertrack_pair(root, name="SOCCERTRACK-E2E", n_frames=10)
    registered = ingest_soccertrack(root)
    video_id, _name = registered[0]

    from matchlab_server.db import session
    from matchlab_server.models import Video

    with session() as db:
        video = db.get(Video, video_id)
        gt_path = video.gt_path

    gt = GroundTruth.model_validate_json(Path(gt_path).read_text())
    assert gt.tracks, "fixture must have produced at least one GT track"

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {"video": {"fps": gt.fps, "frame_count": gt.seq_length, "sample_stride": 1}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    tracklets = []
    players = []
    for track in gt.tracks:
        tracklets.append(
            {
                "tracklet_id": track.track_id,
                "cls": track.role,
                "frames": [
                    {"frame_idx": f.frame_idx, "box": f.box.model_dump(), "confidence": 0.9}
                    for f in track.frames
                ],
            }
        )
        players.append(
            {"player_id": track.track_id, "tracklet_ids": [track.track_id], "team": "home"}
        )
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps(players))

    result = evaluate_run(run_dir, gt)
    heads = headline_metrics(result)

    assert heads["idf1_tracklet"] > 0.99
    assert heads["idf1_entity"] > 0.99
    assert heads["idsw_tracklet"] == 0
    assert heads["idsw_entity"] == 0
    assert heads["hota_tracklet"] > 0.9
    assert heads["hota_entity"] > 0.9
