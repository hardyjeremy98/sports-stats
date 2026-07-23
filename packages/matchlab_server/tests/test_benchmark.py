"""GET /api/benchmark — config x GT-video matrix aggregating repeat runs.

Per ADR 004 batch experiments aggregate GT metrics (not artifact counts), so this
endpoint groups completed runs by (config_name, normalized-config-yaml hash) and
computes mean/range per video cell. Rows are seeded directly via the db session
(bypassing the pipeline) so we can control status/metrics/config_yaml precisely.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

os.environ.setdefault("MATCHLAB_DATA_DIR", tempfile.mkdtemp(prefix="matchlab-test-"))
os.environ.setdefault(
    "MATCHLAB_DATABASE_URL", f"sqlite:///{os.environ['MATCHLAB_DATA_DIR']}/test.db"
)
os.environ.setdefault("MATCHLAB_CONFIG_DIR", str(Path(__file__).parents[3] / "configs"))

import pytest
import yaml
from fastapi.testclient import TestClient
from matchlab_server.app import app
from matchlab_server.db import session
from matchlab_server.models import Run, RunStatus, Video


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _expected_hash(config_yaml: str) -> str:
    cfg = yaml.safe_load(config_yaml)
    normalized = yaml.safe_dump(cfg, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _make_video(db, *, gt: bool, tag: str) -> int:
    video = Video(
        filename=f"{tag}-{uuid.uuid4().hex[:8]}.mp4",
        path="/dev/null",
        gt_path=(f"/tmp/{tag}.gt.json" if gt else None),
    )
    db.add(video)
    db.flush()
    return video.id


def _make_run(
    db,
    *,
    video_id: int,
    config_name: str,
    config_yaml: str,
    status: RunStatus,
    metrics: dict | None,
    created_at: datetime,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    run = Run(
        id=run_id,
        video_id=video_id,
        config_name=config_name,
        config_yaml=config_yaml,
        status=status,
        metrics=metrics,
        run_dir=f"/tmp/{run_id}",
        created_at=created_at,
    )
    db.add(run)
    return run_id


CFG_NAME = "cfg-bench-x"
# Same dict content ({"a": 1, "b": 2}), different formatting/key order -> same hash.
CFG_YAML_1A = "b: 2\na: 1\n"
CFG_YAML_1B = "a: 1\nb: 2\n"
# Different content -> different hash, even with the same config_name.
CFG_YAML_2 = "a: 1\nb: 3\n"

FULL_METRICS_1 = {
    "idf1_tracklet": 0.5,
    "idf1_entity": 0.6,
    "mota_entity": 0.4,
    "idsw_tracklet": 3,
    "idsw_entity": 2,
    "assoc_idf1_gain": 0.1,
    "identity_coverage": 0.8,
    "cluster_purity": 0.9,
}
# Same keys except cluster_purity is missing -> tests "missing key in one repeat".
PARTIAL_METRICS_1 = {
    "idf1_tracklet": 0.7,
    "idf1_entity": 0.8,
    "mota_entity": 0.6,
    "idsw_tracklet": 5,
    "idsw_entity": 4,
    "assoc_idf1_gain": 0.3,
    "identity_coverage": 0.6,
}
SINGLE_METRICS = {
    "idf1_tracklet": 0.9,
    "idf1_entity": 0.9,
    "mota_entity": 0.9,
    "idsw_tracklet": 1,
    "idsw_entity": 1,
    "assoc_idf1_gain": 0.0,
    "identity_coverage": 0.95,
    "cluster_purity": 0.95,
}
OTHER_GROUP_METRICS = {"idf1_tracklet": 0.3}


@pytest.fixture(scope="module")
def seeded(client):
    """Seed one shared fixture set covering all the scenarios in the brief."""
    t0 = datetime.now(UTC)
    with session() as db:
        video_a = _make_video(db, gt=True, tag="video-a")
        video_b = _make_video(db, gt=True, tag="video-b")
        video_c = _make_video(db, gt=False, tag="video-c-no-gt")

        # Repeats in one cell (video_a): run1 (older) + run2 (newer, missing
        # cluster_purity, different config_yaml formatting -> same hash as run1).
        run1 = _make_run(
            db,
            video_id=video_a,
            config_name=CFG_NAME,
            config_yaml=CFG_YAML_1A,
            status=RunStatus.COMPLETED,
            metrics=FULL_METRICS_1,
            created_at=t0,
        )
        run2 = _make_run(
            db,
            video_id=video_a,
            config_name=CFG_NAME,
            config_yaml=CFG_YAML_1B,
            status=RunStatus.COMPLETED,
            metrics=PARTIAL_METRICS_1,
            created_at=t0 + timedelta(seconds=1),
        )
        # Single-run cell (video_b), same group (hash matches run1/run2).
        run3 = _make_run(
            db,
            video_id=video_b,
            config_name=CFG_NAME,
            config_yaml=CFG_YAML_1A,
            status=RunStatus.COMPLETED,
            metrics=SINGLE_METRICS,
            created_at=t0 + timedelta(seconds=2),
        )
        # Different config_yaml content, same config_name -> separate group.
        run4 = _make_run(
            db,
            video_id=video_a,
            config_name=CFG_NAME,
            config_yaml=CFG_YAML_2,
            status=RunStatus.COMPLETED,
            metrics=OTHER_GROUP_METRICS,
            created_at=t0 + timedelta(seconds=3),
        )
        # Excluded: not completed.
        _make_run(
            db,
            video_id=video_a,
            config_name=CFG_NAME,
            config_yaml=CFG_YAML_1A,
            status=RunStatus.RUNNING,
            metrics=FULL_METRICS_1,
            created_at=t0 + timedelta(seconds=4),
        )
        # Excluded: video has no ground truth.
        _make_run(
            db,
            video_id=video_c,
            config_name=CFG_NAME,
            config_yaml=CFG_YAML_1A,
            status=RunStatus.COMPLETED,
            metrics=FULL_METRICS_1,
            created_at=t0 + timedelta(seconds=5),
        )
        # Excluded: completed + GT video, but metrics has no benchmark keys.
        _make_run(
            db,
            video_id=video_a,
            config_name=CFG_NAME,
            config_yaml=CFG_YAML_1A,
            status=RunStatus.COMPLETED,
            metrics={"n_tracklets": 42},
            created_at=t0 + timedelta(seconds=6),
        )
        db.commit()

    return {
        "video_a": video_a,
        "video_b": video_b,
        "video_c": video_c,
        "run1": run1,
        "run2": run2,
        "run3": run3,
        "run4": run4,
        "hash_1": _expected_hash(CFG_YAML_1A),
        "hash_2": _expected_hash(CFG_YAML_2),
    }


def _group(body, config_hash):
    matches = [g for g in body["groups"] if g["config_hash"] == config_hash]
    assert len(matches) == 1, f"expected exactly one group for hash {config_hash}, got {matches}"
    return matches[0]


def test_hash_normalization_merges_differently_formatted_yaml(client, seeded):
    """CFG_YAML_1A and CFG_YAML_1B differ only in key order/whitespace but encode
    the same dict, so runs 1 and 2 must land in the same group/hash."""
    body = client.get("/api/benchmark").json()
    group = _group(body, seeded["hash_1"])
    assert group["config_name"] == CFG_NAME
    cell_a = group["cells"][str(seeded["video_a"])]
    assert sorted(cell_a["run_ids"]) == sorted([seeded["run1"], seeded["run2"]])


def test_different_yaml_content_is_a_different_group_even_same_config_name(client, seeded):
    body = client.get("/api/benchmark").json()
    assert seeded["hash_1"] != seeded["hash_2"]
    group2 = _group(body, seeded["hash_2"])
    assert group2["config_name"] == CFG_NAME
    assert group2["n_runs"] == 1
    cell = group2["cells"][str(seeded["video_a"])]
    assert cell["n_runs"] == 1
    assert cell["run_ids"] == [seeded["run4"]]
    assert cell["metrics_range"] == {}


def test_repeat_cell_mean_and_range(client, seeded):
    body = client.get("/api/benchmark").json()
    group = _group(body, seeded["hash_1"])
    cell_a = group["cells"][str(seeded["video_a"])]
    assert cell_a["n_runs"] == 2
    # run_ids newest first: run2 was created after run1.
    assert cell_a["run_ids"] == [seeded["run2"], seeded["run1"]]
    assert cell_a["metrics_mean"]["idf1_tracklet"] == pytest.approx((0.5 + 0.7) / 2)
    assert cell_a["metrics_range"]["idf1_tracklet"] == [0.5, 0.7]
    # cluster_purity only present on run1 -> mean over the one present value,
    # no range entry even though the cell has 2 runs.
    assert cell_a["metrics_mean"]["cluster_purity"] == pytest.approx(0.9)
    assert "cluster_purity" not in cell_a["metrics_range"]


def test_single_run_cell_has_no_range(client, seeded):
    body = client.get("/api/benchmark").json()
    group = _group(body, seeded["hash_1"])
    cell_b = group["cells"][str(seeded["video_b"])]
    assert cell_b["n_runs"] == 1
    assert cell_b["run_ids"] == [seeded["run3"]]
    assert cell_b["metrics_range"] == {}
    assert cell_b["metrics_mean"]["idf1_tracklet"] == pytest.approx(0.9)


def test_group_n_runs_is_total_across_cells(client, seeded):
    body = client.get("/api/benchmark").json()
    group = _group(body, seeded["hash_1"])
    # run1 + run2 (video_a) + run3 (video_b) = 3; excluded runs must not count.
    assert group["n_runs"] == 3


def test_excluded_runs_do_not_appear(client, seeded):
    body = client.get("/api/benchmark").json()
    group = _group(body, seeded["hash_1"])
    all_run_ids = {rid for cell in group["cells"].values() for rid in cell["run_ids"]}
    for cell in _group(body, seeded["hash_2"])["cells"].values():
        all_run_ids |= set(cell["run_ids"])
    assert seeded["run1"] in all_run_ids
    assert seeded["run2"] in all_run_ids
    assert seeded["run3"] in all_run_ids
    assert seeded["run4"] in all_run_ids
    # The non-completed / no-GT / no-benchmark-metrics runs are excluded, so
    # only the four runs above show up anywhere in these two groups.
    assert len(all_run_ids) == 4


def test_video_without_gt_excluded_from_videos_list(client, seeded):
    body = client.get("/api/benchmark").json()
    video_ids = {v["video_id"] for v in body["videos"]}
    assert seeded["video_a"] in video_ids
    assert seeded["video_b"] in video_ids
    assert seeded["video_c"] not in video_ids


def test_video_entries_have_null_sequence_in_v1(client, seeded):
    body = client.get("/api/benchmark").json()
    by_id = {v["video_id"]: v for v in body["videos"]}
    assert by_id[seeded["video_a"]]["sequence"] is None
    assert by_id[seeded["video_b"]]["sequence"] is None
    assert "filename" in by_id[seeded["video_a"]]


def test_groups_sorted_by_config_name_then_hash(client, seeded):
    body = client.get("/api/benchmark").json()
    keys = [(g["config_name"], g["config_hash"]) for g in body["groups"]]
    assert keys == sorted(keys)
