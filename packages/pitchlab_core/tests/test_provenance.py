"""Provenance recorder tests (SPO-10 part 1): hashing utilities, the stage
contribution hook, and end-to-end manifest wiring."""

from __future__ import annotations

import json

import pytest
from pitchlab_core.provenance import (
    DEFAULT_PACKAGE_NAMES,
    check_evaluation_set,
    collect_package_versions,
    hash_dataset_manifest,
    hash_evaluation_set,
    sha256_file,
)

# --- sha256_file --------------------------------------------------------


def test_sha256_file_known_digest(tmp_path):
    f = tmp_path / "weights.bin"
    f.write_bytes(b"hello world")
    # Known-answer test: sha256("hello world").
    assert sha256_file(f) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


# --- hash_evaluation_set / hash_dataset_manifest: stable canonical hashing --


def test_hash_evaluation_set_ignores_key_order_and_whitespace():
    a = '{"b": 2, "a": 1}'
    b = '{\n  "a": 1,\n  "b": 2\n}'
    assert hash_evaluation_set(a) == hash_evaluation_set(b)


def test_hash_evaluation_set_changes_on_semantic_change():
    a = json.dumps({"tracks": [{"track_id": 1, "frames": [{"box": {"x1": 1.0}}]}]})
    b = json.dumps({"tracks": [{"track_id": 1, "frames": [{"box": {"x1": 2.0}}]}]})
    assert hash_evaluation_set(a) != hash_evaluation_set(b)


def test_hash_dataset_manifest_ignores_key_order_and_whitespace(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text('{"tier": "soccernet", "sequences": []}')
    f2.write_text('{\n  "sequences": [],\n  "tier": "soccernet"\n}')
    assert hash_dataset_manifest(f1) == hash_dataset_manifest(f2)


def test_hash_dataset_manifest_changes_on_semantic_change(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps({"sequences": [{"name": "SNMOT-116"}]}))
    f2.write_text(json.dumps({"sequences": [{"name": "SNMOT-117"}]}))
    assert hash_dataset_manifest(f1) != hash_dataset_manifest(f2)


# --- check_evaluation_set: the refusal primitive ------------------------


def test_check_evaluation_set_equal_returns_none():
    assert check_evaluation_set("abc123", "abc123", "some-context") is None


def test_check_evaluation_set_mismatch_raises_naming_both_hashes_and_context():
    with pytest.raises(RuntimeError) as exc_info:
        check_evaluation_set("abc123", "def456", "benchmark run xyz")
    msg = str(exc_info.value)
    assert "abc123" in msg
    assert "def456" in msg
    assert "benchmark run xyz" in msg


# --- collect_package_versions --------------------------------------------


def test_collect_package_versions_absent_package_is_unknown():
    versions = collect_package_versions(["pitchlab-core", "definitely-not-a-real-package-xyz"])
    assert versions["definitely-not-a-real-package-xyz"] == "unknown"
    assert versions["pitchlab-core"] != "unknown"  # workspace package is installed


def test_default_package_names_include_expected_set():
    assert set(DEFAULT_PACKAGE_NAMES) == {
        "pitchlab-core", "torch", "trackers", "supervision", "inference",
        "ultralytics", "transformers", "numpy", "opencv-python", "motmetrics",
    }


# --- Stage hook: yolo-local detector --------------------------------------


def test_yolo_local_provenance_reports_path_and_correct_sha256(tmp_path):
    from pitchlab_core.stages.detect.yolo_local import YoloLocalDetector

    weights = tmp_path / "football-player-detection.pt"
    weights.write_bytes(b"fake weights content")

    stage = YoloLocalDetector(weights=str(weights))
    models = stage.provenance()

    assert len(models) == 1
    m = models[0]
    assert m.weights_path == str(weights)
    assert m.weights_sha256 == sha256_file(weights)
    assert m.architecture == "yolo"
    assert m.license.code == "AGPL-3.0 (ultralytics, local-eval only, non-shippable)"


def test_yolo_local_provenance_missing_weights_null_hash(tmp_path):
    from pitchlab_core.stages.detect.yolo_local import YoloLocalDetector

    missing = tmp_path / "nope.pt"
    stage = YoloLocalDetector(weights=str(missing))
    models = stage.provenance()

    assert models[0].weights_sha256 is None
    assert models[0].weights_path == str(missing)


# --- Stage hook: roboflow detector (param-level, no network/prepare) -----


def test_roboflow_detector_provenance_uses_model_id_as_revision_no_weights_hash():
    from pitchlab_core.stages.detect.roboflow import RoboflowDetector

    stage = RoboflowDetector(player_model_id="football-players-detection-3zvbc/11")
    models = stage.provenance()

    assert len(models) == 1
    m = models[0]
    assert m.revision == "football-players-detection-3zvbc/11"
    assert m.weights_path is None
    assert m.weights_sha256 is None
    assert m.lineage == "hosted (unpinned)"
    assert m.license.code == "proprietary hosted API (Roboflow)"


def test_roboflow_detector_provenance_includes_ball_model_when_enabled():
    from pitchlab_core.stages.detect.roboflow import RoboflowDetector

    stage = RoboflowDetector(
        player_model_id="football-players-detection-3zvbc/11",
        use_ball_model=True,
        ball_model_id="football-ball-detection-rejhg/2",
    )
    models = stage.provenance()

    assert len(models) == 2
    revisions = {m.revision for m in models}
    assert revisions == {
        "football-players-detection-3zvbc/11",
        "football-ball-detection-rejhg/2",
    }


def test_roboflow_detector_provenance_no_ball_model_when_disabled():
    from pitchlab_core.stages.detect.roboflow import RoboflowDetector

    stage = RoboflowDetector(player_model_id="football-players-detection-3zvbc/11")
    assert len(stage.provenance()) == 1


# --- Default stage hook: no models -----------------------------------------


def test_default_stage_provenance_is_empty_list():
    from pitchlab_core.stages.identity.none import NoIdentityResolver

    stage = NoIdentityResolver()
    assert stage.provenance() == []


# --- Stage hook: global-reid associator (osnet embedder) -------------------


def test_global_reid_provenance_hashes_explicit_local_weights(tmp_path):
    from pitchlab_core.stages.associate.global_reid import GlobalReidAssociator

    weights = tmp_path / "osnet.pth"
    weights.write_bytes(b"fake osnet checkpoint")

    stage = GlobalReidAssociator(embedder="osnet", embedder_params={"weights": str(weights)})
    models = stage.provenance()

    assert len(models) == 1
    m = models[0]
    assert m.architecture == "osnet"
    assert m.weights_path == str(weights)
    assert m.weights_sha256 == sha256_file(weights)
    assert m.license.code == "MIT (vendored deep-person-reid arch)"
    assert m.lineage == "pretrained (MSMT17), no fine-tuning"


def test_global_reid_provenance_no_local_weights_before_prepare():
    from pitchlab_core.stages.associate.global_reid import GlobalReidAssociator

    # No explicit weights path -> resolved only inside prepare() (HF
    # download), which this test deliberately never calls (no network).
    stage = GlobalReidAssociator(embedder="osnet")
    models = stage.provenance()

    assert models[0].weights_path is None
    assert models[0].weights_sha256 is None


# --- Stage hook: face identity resolver -------------------------------------


def test_face_identity_provenance_reports_pack_without_a_local_path():
    from pitchlab_core.stages.identity.face import FaceIdentityResolver

    stage = FaceIdentityResolver()
    models = stage.provenance()

    assert len(models) == 1
    m = models[0]
    assert m.architecture == "insightface-buffalo_l"
    assert m.revision == "buffalo_l"
    assert m.weights_path is None
    assert m.weights_sha256 is None
    assert "research-only" in m.license.weights


# --- Stage hook: siglip team classifier (param-level, no network) ----------


def test_siglip_provenance_default_model_reports_verified_apache_license():
    from pitchlab_core.stages.team.siglip import SiglipTeamClassifier

    stage = SiglipTeamClassifier()  # default model_name
    models = stage.provenance()

    assert len(models) == 1
    m = models[0]
    assert m.architecture == "siglip"
    assert m.revision == "google/siglip-base-patch16-224"
    assert m.weights_path is None
    assert m.weights_sha256 is None
    assert m.lineage == "pretrained (HuggingFace hub)"
    assert m.license.code == "Apache-2.0 (transformers library)"
    assert "Apache-2.0" in m.license.weights


def test_siglip_provenance_nondefault_model_weights_license_unknown():
    from pitchlab_core.stages.team.siglip import SiglipTeamClassifier

    # A caller-configured, unverified checkpoint: the code license (the
    # transformers runtime) is still knowable, but this specific checkpoint's
    # license was never checked, so it must not inherit the default's claim.
    stage = SiglipTeamClassifier(model_name="someone/other-siglip-finetune")
    m = stage.provenance()[0]

    assert m.revision == "someone/other-siglip-finetune"
    assert m.license.code == "Apache-2.0 (transformers library)"
    assert m.license.weights == "unknown"


# --- Stage hook: roboflow-keypoints calibrator (param-level, no network) ---


def test_roboflow_keypoints_provenance_uses_model_id_as_revision():
    from pitchlab_core.stages.calibrate.roboflow_keypoints import RoboflowKeypointCalibrator

    stage = RoboflowKeypointCalibrator(model_id="football-field-detection-f07vi/14")
    models = stage.provenance()

    assert len(models) == 1
    m = models[0]
    assert m.revision == "football-field-detection-f07vi/14"
    assert m.weights_path is None
    assert m.weights_sha256 is None
    assert m.lineage == "hosted (unpinned)"
    assert m.license.code == "proprietary hosted API (Roboflow)"


# --- Pipeline-level: manifest carries a full provenance block --------------


def test_pipeline_manifest_has_provenance_block(tmp_path):
    from pitchlab_core.config import PipelineConfig
    from pitchlab_core.demo import render_demo_video
    from pitchlab_core.runner import PipelineRunner
    from pitchlab_core.schemas.run import StageStatus

    config_path = (
        __file__.rsplit("/tests/", 1)[0] + "/../../configs/pipeline.stub.yaml"
    )
    video = render_demo_video(
        tmp_path / "clip.mp4", duration_s=4, fps=15, width=640, height=360
    )
    config = PipelineConfig.from_yaml(config_path)
    runner = PipelineRunner(
        run_id="prov-test", video_path=video, config=config, run_dir=tmp_path / "run"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error

    on_disk = json.loads((tmp_path / "run" / "manifest.json").read_text())
    prov = on_disk["provenance"]

    assert prov["git_revision"] != ""  # "unknown" or a real short sha, never blank
    for name in DEFAULT_PACKAGE_NAMES:
        assert name in prov["package_versions"]

    completed_kinds = {
        s["kind"] for s in on_disk["stages"] if s["status"] == "completed"
    }
    assert completed_kinds  # sanity: the stub run actually executed stages
    assert set(prov["stages"]) == completed_kinds

    for stage_prov in prov["stages"].values():
        assert "impl" in stage_prov
        assert "params" in stage_prov
        assert "models" in stage_prov

    assert prov["evaluation_set_hash"] == "unknown"
    assert prov["evaluation_set_source"] is None
