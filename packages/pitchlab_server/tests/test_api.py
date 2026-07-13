"""API + worker integration test on a temp data dir with the stub pipeline."""

import os
import tempfile
from pathlib import Path

os.environ["PITCHLAB_DATA_DIR"] = tempfile.mkdtemp(prefix="pitchlab-test-")
os.environ["PITCHLAB_DATABASE_URL"] = (
    f"sqlite:///{os.environ['PITCHLAB_DATA_DIR']}/test.db"
)
os.environ["PITCHLAB_CONFIG_DIR"] = str(Path(__file__).parents[3] / "configs")

import pytest
from fastapi.testclient import TestClient
from pitchlab_core.demo import render_demo_video
from pitchlab_server.app import app
from pitchlab_server.worker import claim_next_job, execute_job


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def video_id(client):
    path = Path(os.environ["PITCHLAB_DATA_DIR"]) / "clip.mp4"
    render_demo_video(path, duration_s=6, fps=20, width=960, height=540)
    with open(path, "rb") as f:
        resp = client.post("/api/videos", files={"file": ("clip.mp4", f, "video/mp4")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["duration_s"] > 4
    return body["id"]


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_configs_and_registry(client):
    configs = client.get("/api/configs").json()
    names = {c["name"] for c in configs}
    assert {"v1-default", "stub-synthetic"} <= names
    registry = client.get("/api/registry").json()["stages"]
    assert "botsort" in registry["track"]
    assert "learned-motr" in registry["track"]  # the reserved slot is visible


def test_full_run_lifecycle(client, video_id):
    resp = client.post(
        "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["id"]

    job_id = claim_next_job()
    assert job_id is not None
    execute_job(job_id)

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["status"] == "completed", detail["error"]
    assert detail["manifest"]["metrics"]["n_tracklets"] > 10

    stats = client.get(f"/api/runs/{run_id}/artifacts/stats").json()
    assert stats["players"]

    # QA items were imported into the queue; accept one -> a labeled example.
    qa = client.get("/api/qa", params={"run_id": run_id}).json()
    if qa:
        rec = qa[0]
        accepted = client.post(f"/api/qa/{rec['id']}/accept").json()
        assert accepted["status"] == "accepted"
        # double decision is a conflict
        assert client.post(f"/api/qa/{rec['id']}/accept").status_code == 409


def test_run_diff(client, video_id):
    ids = []
    for cfg in ("stub-synthetic", "stub-synthetic"):
        resp = client.post("/api/runs", json={"video_id": video_id, "config_name": cfg})
        ids.append(resp.json()["id"])
        execute_job(claim_next_job())
    diff = client.get(f"/api/runs/{ids[0]}/diff/{ids[1]}").json()
    assert diff["run_a"]["status"] == "completed"
    assert diff["config_changes"] == []  # identical configs
    assert "n_tracklets" in diff["metric_deltas"]


def test_path_traversal_blocked(client, video_id):
    resp = client.post(
        "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
    )
    run_id = resp.json()["id"]
    assert client.get(f"/api/runs/{run_id}/files/../../secrets").status_code == 404
