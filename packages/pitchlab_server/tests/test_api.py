"""API + worker integration test on a temp data dir with the stub pipeline."""

import json
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


def test_run_diff_eval_payloads_and_switch_diff(client, video_id):
    ids = []
    for cfg in ("stub-synthetic", "stub-synthetic"):
        resp = client.post("/api/runs", json={"video_id": video_id, "config_name": cfg})
        ids.append(resp.json()["id"])
        execute_job(claim_next_job())

    eval_a = {
        "levels": {},
        "association": {},
        "instances": [
            {
                "level": "tracklet",
                "kind": "id_switch",
                "frame_idx": 100,
                "t": 5.0,
                "gt_track_id": 1,
                "gt_label": "home_7",
                "prev_id": 10,
                "new_id": 11,
            }
        ],
    }
    run_dir_a = Path(os.environ["PITCHLAB_DATA_DIR"]) / "runs" / ids[0]
    (run_dir_a / "eval.json").write_text(json.dumps(eval_a))

    # Run B has no eval.json -> switch_diff must be None even though eval_a exists.
    diff = client.get(f"/api/runs/{ids[0]}/diff/{ids[1]}").json()
    assert diff["eval_a"] == eval_a
    assert diff["eval_b"] is None
    assert diff["switch_diff"] is None

    # Once run B also has an eval.json, switch_diff is populated.
    eval_b = {
        "levels": {},
        "association": {},
        "instances": [
            {
                "level": "tracklet",
                "kind": "id_switch",
                "frame_idx": 102,
                "t": 5.2,
                "gt_track_id": 1,
                "gt_label": "home_7",
                "prev_id": 10,
                "new_id": 11,
            }
        ],
    }
    run_dir_b = Path(os.environ["PITCHLAB_DATA_DIR"]) / "runs" / ids[1]
    (run_dir_b / "eval.json").write_text(json.dumps(eval_b))

    diff2 = client.get(f"/api/runs/{ids[0]}/diff/{ids[1]}").json()
    assert diff2["eval_b"] == eval_b
    assert diff2["switch_diff"]["counts"] == {"fixed": 0, "introduced": 0, "persisted": 1}
    assert diff2["switch_diff"]["persisted"] == [{"a": eval_a["instances"][0], "b": eval_b["instances"][0]}]


def test_path_traversal_blocked(client, video_id):
    resp = client.post(
        "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
    )
    run_id = resp.json()["id"]
    assert client.get(f"/api/runs/{run_id}/files/../../secrets").status_code == 404


def test_run_diff_same_video_guard(client, video_id):
    """Verify that diffing two runs on different videos returns 422."""
    # Create a second video
    path_b = Path(os.environ["PITCHLAB_DATA_DIR"]) / "clip_b.mp4"
    render_demo_video(path_b, duration_s=6, fps=20, width=960, height=540)
    with open(path_b, "rb") as f:
        resp = client.post("/api/videos", files={"file": ("clip_b.mp4", f, "video/mp4")})
    assert resp.status_code == 200, resp.text
    video_id_b = resp.json()["id"]

    # Create runs on different videos
    resp_a = client.post(
        "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
    )
    run_a = resp_a.json()["id"]
    execute_job(claim_next_job())

    resp_b = client.post(
        "/api/runs", json={"video_id": video_id_b, "config_name": "stub-synthetic"}
    )
    run_b = resp_b.json()["id"]
    execute_job(claim_next_job())

    # Diffing runs on different videos should return 422
    diff_resp = client.get(f"/api/runs/{run_a}/diff/{run_b}")
    assert diff_resp.status_code == 422, diff_resp.text
    assert diff_resp.json()["detail"] == "Run diff requires two runs on the same video"

    # Diffing runs on the same video should still succeed
    resp_c = client.post(
        "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
    )
    run_c = resp_c.json()["id"]
    execute_job(claim_next_job())

    diff_resp_same = client.get(f"/api/runs/{run_a}/diff/{run_c}")
    assert diff_resp_same.status_code == 200, diff_resp_same.text


def test_evaluate_endpoint_with_oracle_run_id(client, video_id):
    """POST /runs/{id}/evaluate?oracle_run_id=... enriches attribution from a
    scored oracle run of the same video; refusals surface as 422."""
    pytest.importorskip("motmetrics")
    from pitchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
    from pitchlab_core.schemas.geometry import Box
    from pitchlab_server.db import session
    from pitchlab_server.models import Run, Video

    # Attach a tiny GT to the shared demo video (120 frames at 20 fps).
    with session() as db:
        video = db.get(Video, video_id)
        frame_count = 120
        gt = GroundTruth(
            source="test",
            sequence="clip",
            fps=20.0,
            width=960,
            height=540,
            seq_length=frame_count,
            tracks=[
                GroundTruthTrack(
                    track_id=1,
                    role="player",
                    frames=[
                        GroundTruthFrame(frame_idx=f, box=Box(x1=100, y1=100, x2=140, y2=220))
                        for f in range(frame_count)
                    ],
                )
            ],
        )
        gt_path = Path(os.environ["PITCHLAB_DATA_DIR"]) / "clip.gt.json"
        gt_path.write_text(gt.model_dump_json())
        video.gt_path = str(gt_path)
        db.commit()

    # Earlier tests in this module may leave queued jobs behind; drain them so
    # claim_next_job below picks up THIS test's runs.
    while (stale := claim_next_job()) is not None:
        execute_job(stale)

    ids = []
    for _ in range(2):
        resp = client.post(
            "/api/runs", json={"video_id": video_id, "config_name": "stub-synthetic"}
        )
        ids.append(resp.json()["id"])
        execute_job(claim_next_job())
    run_id, oracle_id = ids

    # Fabricate run dirs whose tracklets exercise attribution: baseline
    # fragments GT1 at frame 60; the "oracle" run tracks it cleanly.
    def _frames(rng):
        return [
            {
                "frame_idx": f,
                "box": {"x1": 100, "y1": 100, "x2": 140, "y2": 220},
                "confidence": 0.9,
            }
            for f in rng
        ]

    with session() as db:
        base_dir = Path(db.get(Run, run_id).run_dir)
        oracle_dir = Path(db.get(Run, oracle_id).run_dir)
    (base_dir / "tracklets.json").write_text(
        json.dumps(
            [
                {"tracklet_id": 10, "cls": "player", "frames": _frames(range(0, 60))},
                {"tracklet_id": 11, "cls": "player", "frames": _frames(range(60, 120))},
            ]
        )
    )
    (base_dir / "players.json").write_text(json.dumps([]))
    (oracle_dir / "tracklets.json").write_text(
        json.dumps([{"tracklet_id": 20, "cls": "player", "frames": _frames(range(0, 120))}])
    )
    (oracle_dir / "players.json").write_text(json.dumps([]))
    # Mark the oracle run's manifest as a pristine oracle-detections run.
    manifest = json.loads((oracle_dir / "manifest.json").read_text())
    manifest["config"]["stages"]["detect"] = {"impl": "oracle", "params": {}, "enabled": True}
    (oracle_dir / "manifest.json").write_text(json.dumps(manifest))
    # The worker auto-scored both runs at completion (the video already had
    # GT); drop the oracle run's stale eval.json so the "evaluate it first"
    # refusal is exercised against the fabricated tracklets above.
    (oracle_dir / "eval.json").unlink()

    # Oracle run must be evaluated first (its eval.json self-describes).
    resp = client.post(f"/api/runs/{run_id}/evaluate", params={"oracle_run_id": oracle_id})
    assert resp.status_code == 422
    assert "no eval.json" in resp.json()["detail"]

    assert client.post(f"/api/runs/{oracle_id}/evaluate").status_code == 200
    oracle_eval = json.loads((oracle_dir / "eval.json").read_text())
    assert oracle_eval["attribution"]["oracle_input"] is True

    resp = client.post(f"/api/runs/{run_id}/evaluate", params={"oracle_run_id": oracle_id})
    assert resp.status_code == 200, resp.text
    enriched = json.loads((base_dir / "eval.json").read_text())
    assert enriched["attribution"]["oracle_comparison"] == {"oracle_run": oracle_id}
    tracklet_switches = [i for i in enriched["instances"] if i["level"] == "tracklet"]
    assert tracklet_switches
    assert all(i["attribution"]["layer"] == "detection" for i in tracklet_switches)

    # Unknown oracle run -> 404.
    assert (
        client.post(f"/api/runs/{run_id}/evaluate", params={"oracle_run_id": "nope"}).status_code
        == 404
    )
