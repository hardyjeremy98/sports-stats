"""Export identity-QA pair verdicts (IdentityLabel rows in the server DB) to a
re-ID training dataset: pairs.jsonl + copied evidence crops. Unsure verdicts are
abstention and must never become training labels."""

from __future__ import annotations

import json
import os
import tempfile

os.environ["MATCHLAB_DATA_DIR"] = tempfile.mkdtemp(prefix="matchlab-train-test-")
os.environ["MATCHLAB_DATABASE_URL"] = f"sqlite:///{os.environ['MATCHLAB_DATA_DIR']}/test.db"

import pytest  # noqa: E402
from matchlab_train.datasets.reid_labels import export_reid_labels  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the server DB at a private, per-test sqlite file so this test's
    IdentityLabel rows can never be polluted by other test modules sharing the
    process-wide settings/engine cache."""
    import matchlab_server.db as db_module
    from matchlab_server.settings import get_settings

    db_path = tmp_path / "server.db"
    monkeypatch.setenv("MATCHLAB_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MATCHLAB_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db_module._engine = None
    db_module._session_factory = None

    db_module.init_db()
    yield db_module

    db_module._engine = None
    db_module._session_factory = None
    get_settings.cache_clear()


def _pair_payload(**overrides):
    payload = {
        "tracklet_a": 1,
        "tracklet_b": 2,
        "verdict": "same",
        "crop_a": None,
        "crop_b": None,
        "frame_a": 10,
        "frame_b": 20,
        "source": "manual",
    }
    payload.update(overrides)
    return payload


def _seed(db, tmp_path):
    from matchlab_server.models import IdentityLabel, IdentityLabelKind, Run, Video

    with db.session() as s:
        video = Video(filename="clip.mp4", path=str(tmp_path / "clip.mp4"))
        s.add(video)
        s.commit()
        s.refresh(video)

        run_a_dir = tmp_path / "runs" / "run-a"
        (run_a_dir / "crops").mkdir(parents=True)
        (run_a_dir / "crops" / "face_p3_f120.jpg").write_bytes(b"crop-a-bytes")
        (run_a_dir / "crops" / "face_p4_f125.jpg").write_bytes(b"crop-b-bytes")
        # Add a crop with same basename as run-b's crop (distinct content)
        (run_a_dir / "crops" / "face_p1_f10.jpg").write_bytes(b"run-a-face-p1-f10")
        run_a = Run(
            id="run-a",
            video_id=video.id,
            config_name="stub",
            config_yaml="",
            run_dir=str(run_a_dir),
        )
        s.add(run_a)

        run_b_dir = tmp_path / "runs" / "run-b"
        (run_b_dir / "crops").mkdir(parents=True)
        # Add a crop with same basename as run-a's crop (distinct content)
        (run_b_dir / "crops" / "face_p1_f10.jpg").write_bytes(b"run-b-face-p1-f10")
        run_b = Run(
            id="run-b",
            video_id=video.id,
            config_name="stub",
            config_yaml="",
            run_dir=str(run_b_dir),
        )
        s.add(run_b)
        s.commit()

        same_label = IdentityLabel(
            run_id=run_a.id,
            video_id=video.id,
            kind=IdentityLabelKind.PAIR,
            payload=_pair_payload(
                tracklet_a=3,
                tracklet_b=4,
                verdict="same",
                crop_a="crops/face_p3_f120.jpg",
                crop_b="crops/face_p4_f125.jpg",
                frame_a=120,
                frame_b=125,
                source="manual",
            ),
            note="looks same",
        )
        # run-a also references the same-named crop to test collision handling
        run_a_collision_label = IdentityLabel(
            run_id=run_a.id,
            video_id=video.id,
            kind=IdentityLabelKind.PAIR,
            payload=_pair_payload(
                tracklet_a=11,
                tracklet_b=12,
                verdict="same",
                crop_a="crops/face_p1_f10.jpg",
                crop_b=None,
                frame_a=10,
                frame_b=None,
                source="manual",
            ),
            note="run-a collision test",
        )
        different_label = IdentityLabel(
            run_id=run_b.id,
            video_id=video.id,
            kind=IdentityLabelKind.PAIR,
            payload=_pair_payload(
                tracklet_a=5,
                tracklet_b=6,
                verdict="different",
                crop_a="crops/missing.jpg",
                crop_b=None,
                frame_a=50,
                frame_b=None,
                source="eval_switch",
            ),
            note=None,
        )
        # Add a same-basename collision test: run-b references the same-named crop
        collision_label = IdentityLabel(
            run_id=run_b.id,
            video_id=video.id,
            kind=IdentityLabelKind.PAIR,
            payload=_pair_payload(
                tracklet_a=9,
                tracklet_b=10,
                verdict="same",
                crop_a="crops/face_p1_f10.jpg",
                crop_b=None,
                frame_a=10,
                frame_b=None,
                source="manual",
            ),
            note="run-b collision test",
        )
        unsure_label = IdentityLabel(
            run_id=run_a.id,
            video_id=video.id,
            kind=IdentityLabelKind.PAIR,
            payload=_pair_payload(
                tracklet_a=7,
                tracklet_b=8,
                verdict="unsure",
                source="assoc_candidate",
            ),
            note=None,
        )
        merge_label = IdentityLabel(
            run_id=run_a.id,
            video_id=video.id,
            kind=IdentityLabelKind.MERGE,
            payload={"player_ids": [1, 2]},
            note=None,
        )
        s.add_all([same_label, run_a_collision_label, different_label, collision_label, unsure_label, merge_label])
        s.commit()
        return {
            "video_id": video.id,
            "same_id": same_label.id,
            "run_a_collision_id": run_a_collision_label.id,
            "different_id": different_label.id,
            "collision_id": collision_label.id,
        }


def test_export_reid_labels(db, tmp_path):
    ids = _seed(db, tmp_path)
    workdir = tmp_path / "workdir"

    out_dir = export_reid_labels(workdir)

    assert out_dir == workdir / "datasets" / "reid-labels"
    pairs_path = out_dir / "pairs.jsonl"
    assert pairs_path.exists()

    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line]
    assert len(rows) == 4  # unsure and merge excluded; same, run_a_collision, different, collision

    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {ids["same_id"], ids["run_a_collision_id"], ids["different_id"], ids["collision_id"]}

    same_row = by_id[ids["same_id"]]
    assert same_row["video_id"] == ids["video_id"]
    assert same_row["run_id"] == "run-a"
    assert same_row["tracklet_a"] == 3
    assert same_row["tracklet_b"] == 4
    assert same_row["frame_a"] == 120
    assert same_row["frame_b"] == 125
    assert same_row["label"] == "same"
    assert same_row["source"] == "manual"
    assert same_row["note"] == "looks same"
    assert same_row["crop_a"] == "crops/run-a_face_p3_f120.jpg"
    assert same_row["crop_b"] == "crops/run-a_face_p4_f125.jpg"
    assert isinstance(same_row["created_at"], str)

    copied_a = out_dir / "crops" / "run-a_face_p3_f120.jpg"
    copied_b = out_dir / "crops" / "run-a_face_p4_f125.jpg"
    assert copied_a.read_bytes() == b"crop-a-bytes"
    assert copied_b.read_bytes() == b"crop-b-bytes"

    run_a_collision_row = by_id[ids["run_a_collision_id"]]
    assert run_a_collision_row["run_id"] == "run-a"
    assert run_a_collision_row["tracklet_a"] == 11
    assert run_a_collision_row["tracklet_b"] == 12
    assert run_a_collision_row["frame_a"] == 10
    assert run_a_collision_row["frame_b"] is None
    assert run_a_collision_row["label"] == "same"
    assert run_a_collision_row["source"] == "manual"
    assert run_a_collision_row["note"] == "run-a collision test"
    assert run_a_collision_row["crop_a"] == "crops/run-a_face_p1_f10.jpg"
    assert run_a_collision_row["crop_b"] is None

    diff_row = by_id[ids["different_id"]]
    assert diff_row["run_id"] == "run-b"
    assert diff_row["tracklet_a"] == 5
    assert diff_row["tracklet_b"] == 6
    assert diff_row["frame_a"] == 50
    assert diff_row["frame_b"] is None
    assert diff_row["label"] == "different"
    assert diff_row["source"] == "eval_switch"
    assert diff_row["note"] is None
    assert diff_row["crop_a"] is None  # missing on disk -> nulled
    assert diff_row["crop_b"] is None
    assert not (out_dir / "crops" / "run-b_missing.jpg").exists()

    # Test collision-safe naming: both run-a and run-b reference crops named
    # "face_p1_f10.jpg" with distinct contents; export prefixes with run_id
    collision_row = by_id[ids["collision_id"]]
    assert collision_row["run_id"] == "run-b"
    assert collision_row["tracklet_a"] == 9
    assert collision_row["tracklet_b"] == 10
    assert collision_row["frame_a"] == 10
    assert collision_row["frame_b"] is None
    assert collision_row["label"] == "same"
    assert collision_row["source"] == "manual"
    assert collision_row["note"] == "run-b collision test"
    assert collision_row["crop_a"] == "crops/run-b_face_p1_f10.jpg"
    assert collision_row["crop_b"] is None

    # Verify both run-a and run-b crops exist with correct run_id prefixes
    # and distinct contents (no overwrite)
    run_a_collision_crop = out_dir / "crops" / "run-a_face_p1_f10.jpg"
    run_b_collision_crop = out_dir / "crops" / "run-b_face_p1_f10.jpg"
    assert run_a_collision_crop.exists()
    assert run_b_collision_crop.exists()
    assert run_a_collision_crop.read_bytes() == b"run-a-face-p1-f10"
    assert run_b_collision_crop.read_bytes() == b"run-b-face-p1-f10"


def test_export_reid_labels_empty_db_returns_out_dir(db, tmp_path):
    workdir = tmp_path / "workdir"
    out_dir = export_reid_labels(workdir)
    assert out_dir == workdir / "datasets" / "reid-labels"
    assert (out_dir / "pairs.jsonl").read_text() == ""
