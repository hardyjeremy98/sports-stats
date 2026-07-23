"""Identity QA label storage: same/different tracklet-pair verdicts, entity merge/split
flags, and roster labels. Labels are annotations only — they never mutate run artifacts."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MATCHLAB_DATA_DIR", tempfile.mkdtemp(prefix="matchlab-test-"))
os.environ.setdefault(
    "MATCHLAB_DATABASE_URL", f"sqlite:///{os.environ['MATCHLAB_DATA_DIR']}/test.db"
)
os.environ.setdefault("MATCHLAB_CONFIG_DIR", str(Path(__file__).parents[3] / "configs"))

import pytest
from fastapi.testclient import TestClient
from matchlab_core.demo import render_demo_video
from matchlab_server.app import app
from matchlab_server.worker import claim_next_job, execute_job


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def video_id(client):
    path = Path(os.environ["MATCHLAB_DATA_DIR"]) / "identity_qa_clip.mp4"
    render_demo_video(path, duration_s=6, fps=20, width=960, height=540)
    with open(path, "rb") as f:
        resp = client.post(
            "/api/videos", files={"file": ("identity_qa_clip.mp4", f, "video/mp4")}
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


@pytest.fixture(scope="module")
def run_id(client, video_id):
    resp = client.post(
        "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
    )
    assert resp.status_code == 200, resp.text
    rid = resp.json()["id"]
    execute_job(claim_next_job())
    return rid


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


def test_post_pair_label(client, run_id, video_id):
    resp = client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "pair",
            "payload": _pair_payload(),
            "note": "looks same",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "pair"
    assert body["run_id"] == run_id
    assert body["video_id"] == video_id
    assert body["payload"] == _pair_payload()
    assert body["note"] == "looks same"


def test_post_merge_label(client, run_id, video_id):
    resp = client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "merge",
            "payload": {"player_ids": [3, 4, 5]},
            "note": None,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "merge"
    assert body["video_id"] == video_id
    assert body["payload"] == {"player_ids": [3, 4, 5]}


def test_post_split_label(client, run_id, video_id):
    resp = client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "split",
            "payload": {"player_id": 7, "tracklet_ids_out": [10, 11]},
            "note": None,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "split"
    assert body["video_id"] == video_id
    assert body["payload"] == {"player_id": 7, "tracklet_ids_out": [10, 11]}


def test_post_roster_label(client, run_id, video_id):
    resp = client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "roster",
            "payload": {"player_id": 9, "roster_label": "Home #7"},
            "note": None,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "roster"
    assert body["video_id"] == video_id
    assert body["payload"] == {"player_id": 9, "roster_label": "Home #7"}


def test_post_unknown_kind_422(client, run_id):
    resp = client.post(
        "/api/identity_qa",
        json={"run_id": run_id, "kind": "bogus", "payload": {}, "note": None},
    )
    assert resp.status_code == 422


def test_post_pair_missing_verdict_422(client, run_id):
    bad = _pair_payload()
    del bad["verdict"]
    resp = client.post(
        "/api/identity_qa",
        json={"run_id": run_id, "kind": "pair", "payload": bad, "note": None},
    )
    assert resp.status_code == 422


def test_post_missing_run_404(client):
    resp = client.post(
        "/api/identity_qa",
        json={
            "run_id": "does-not-exist",
            "kind": "pair",
            "payload": _pair_payload(),
            "note": None,
        },
    )
    assert resp.status_code == 404


def test_list_filters_by_run_id_and_kind_newest_first(client, run_id, video_id):
    other_run_resp = client.post(
        "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
    )
    other_run_id = other_run_resp.json()["id"]
    execute_job(claim_next_job())

    first = client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "roster",
            "payload": {"player_id": 101, "roster_label": "a"},
            "note": None,
        },
    ).json()
    second = client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "roster",
            "payload": {"player_id": 102, "roster_label": "b"},
            "note": None,
        },
    ).json()
    client.post(
        "/api/identity_qa",
        json={
            "run_id": other_run_id,
            "kind": "roster",
            "payload": {"player_id": 103, "roster_label": "c"},
            "note": None,
        },
    )

    by_run = client.get(
        "/api/identity_qa", params={"run_id": run_id, "kind": "roster"}
    ).json()
    ids = [item["id"] for item in by_run]
    assert ids[:2] == [second["id"], first["id"]]  # newest first
    assert all(item["run_id"] == run_id and item["kind"] == "roster" for item in by_run)


def test_get_with_unknown_kind_422(client, run_id):
    resp = client.get("/api/identity_qa", params={"run_id": run_id, "kind": "bogus"})
    assert resp.status_code == 422


def test_get_with_valid_kind_filter(client, run_id):
    # Ensure at least one pair label exists for this run
    client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "pair",
            "payload": _pair_payload(),
            "note": None,
        },
    )
    # Filter by valid kind and verify it returns 200 and items match kind
    resp = client.get("/api/identity_qa", params={"run_id": run_id, "kind": "pair"})
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["kind"] == "pair" for item in body)


def test_post_merge_payload_with_pair_kind_422(client, run_id):
    # POST a merge payload (well-formed for merge) under kind="pair" should fail 422
    # because pair expects tracklet_a, tracklet_b, verdict, etc., not player_ids
    resp = client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "pair",
            "payload": {"player_ids": [1, 2]},
            "note": None,
        },
    )
    assert resp.status_code == 422


def test_delete_label_then_404_on_second_delete(client, run_id):
    resp = client.post(
        "/api/identity_qa",
        json={
            "run_id": run_id,
            "kind": "merge",
            "payload": {"player_ids": [1, 2]},
            "note": None,
        },
    )
    label_id = resp.json()["id"]

    del_resp = client.delete(f"/api/identity_qa/{label_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"ok": True}

    second = client.delete(f"/api/identity_qa/{label_id}")
    assert second.status_code == 404
