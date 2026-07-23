"""TDLP-full subprocess-bridge pure functions (the parts that run without the
external venvs): MOT-layout fabrication, det-file writing, frame-index
round-tripping, and subprocess failure handling. Hand-built fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.schemas.detections import Detection, DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import TrackletFrame
from matchlab_core.stages.track.tdlp_full import bridge
from matchlab_core.video import Frame


def _det(x1, y1, x2, y2, conf, cls="player") -> Detection:
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), confidence=conf, cls=DetectionClass(cls))


def _fd(frame_idx, dets) -> FrameDetections:
    return FrameDetections(frame_idx=frame_idx, t=frame_idx / 25.0, detections=dets)


# --- write_det_file -------------------------------------------------------


def test_write_det_file_maps_source_to_local_and_filters_classes(tmp_path: Path):
    # source frames 0, 5, 10 -> local 0, 1, 2 (stride-5 sampling)
    source_to_local = {0: 0, 5: 1, 10: 2}
    detections = [
        _fd(0, [_det(100, 100, 140, 220, 0.9, "player"),
                _det(500, 500, 510, 510, 0.4, "ball")]),   # ball excluded
        _fd(5, [_det(10, 20, 30, 60, 0.5, "referee")]),
        _fd(10, [_det(1, 2, 3, 4, 0.7, "goalkeeper")]),
    ]
    out = tmp_path / "det.txt"
    n = bridge.write_det_file(detections, source_to_local, out)

    assert n == 3  # ball dropped
    lines = out.read_text().splitlines()
    # local frame = source_to_local+1 (1-based MOT); xywh top-left; %.2f/%.6f
    assert lines[0] == "1,-1,100.00,100.00,40.00,120.00,0.900000,-1,-1,-1"
    assert lines[1] == "2,-1,10.00,20.00,20.00,40.00,0.500000,-1,-1,-1"
    assert lines[2] == "3,-1,1.00,2.00,2.00,2.00,0.700000,-1,-1,-1"


def test_write_det_file_skips_frames_outside_sampling(tmp_path: Path):
    out = tmp_path / "det.txt"
    # frame_idx 7 is not in the decoded set -> its detections are dropped
    n = bridge.write_det_file([_fd(7, [_det(1, 1, 2, 2, 0.9)])], {0: 0}, out)
    assert n == 0
    assert out.read_text() == ""


# --- write_seqinfo --------------------------------------------------------


def test_write_seqinfo(tmp_path: Path):
    out = tmp_path / "seqinfo.ini"
    bridge.write_seqinfo(out, name="clip", seq_length=42, width=1920, height=1080, fps=25.0)
    text = out.read_text()
    assert "seqLength=42" in text
    assert "imWidth=1920" in text
    assert "imHeight=1080" in text
    assert "frameRate=25" in text
    assert "imExt=.jpg" in text


# --- remap_tracklets_to_source -------------------------------------------


def _tracklet(tid, frame_indices) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(frame_idx=fi, box=Box(x1=0, y1=0, x2=1, y2=1), confidence=1.0)
            for fi in frame_indices
        ],
    )


def test_remap_tracklets_to_source():
    local_to_source = [0, 5, 10]  # stride-5
    remapped = bridge.remap_tracklets_to_source([_tracklet(3, [0, 1, 2])], local_to_source)
    assert [f.frame_idx for f in remapped[0].frames] == [0, 5, 10]
    assert remapped[0].tracklet_id == 3


def test_remap_tracklets_out_of_range_raises():
    with pytest.raises(ValueError, match="outside the fabricated sequence"):
        bridge.remap_tracklets_to_source([_tracklet(1, [0, 3])], [0, 5, 10])


# --- stage_sequence (real files, no subprocess) ---------------------------


def test_stage_sequence_writes_layout_and_mapping(tmp_path: Path):
    frames = [
        Frame(frame_idx=0, t=0.0, image=np.zeros((8, 12, 3), dtype=np.uint8)),
        Frame(frame_idx=5, t=0.2, image=np.zeros((8, 12, 3), dtype=np.uint8)),
    ]
    detections = [_fd(0, [_det(1, 1, 3, 5, 0.9)]), _fd(5, [_det(2, 2, 4, 6, 0.8)])]
    layout = bridge.stage_sequence(frames, detections, tmp_path, seq_name="clip", fps=25.0)

    assert layout.local_to_source == [0, 5]
    assert layout.width == 12 and layout.height == 8
    assert layout.n_detections == 2
    seq = tmp_path / "test" / "clip"
    assert (seq / "img1" / "000001.jpg").exists()
    assert (seq / "img1" / "000002.jpg").exists()
    assert (seq / "det" / "det.txt").exists()
    assert "seqLength=2" in (seq / "seqinfo.ini").read_text()


def test_stage_sequence_empty_clip_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="No frames decoded"):
        bridge.stage_sequence([], [], tmp_path, seq_name="clip", fps=25.0)


# --- parse_tracker_output (round-trip) ------------------------------------


def test_parse_tracker_output_remaps_frames(tmp_path: Path):
    mot = tmp_path / "clip.txt"
    # MOT 1-based local frames 1 and 2, one track -> source 0 and 5
    mot.write_text("1,7,10,20,30,40,0.9,-1,-1,-1\n2,7,11,21,30,40,0.8,-1,-1,-1\n")
    tracklets = bridge.parse_tracker_output(mot, local_to_source=[0, 5])
    assert len(tracklets) == 1
    t = tracklets[0]
    assert t.tracklet_id == 7
    assert [f.frame_idx for f in t.frames] == [0, 5]
    # xywh -> xyxy conversion preserved from _parse_mot_tracklets
    assert t.frames[0].box.x2 == pytest.approx(40.0)  # x=10 + w=30


# --- run_external ---------------------------------------------------------


def test_run_external_success_returns_stdout(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text("print('hello-from-external')\n")
    out = bridge.run_external(
        Path(sys.executable), script, [], cwd=tmp_path, timeout_s=30, label="probe"
    )
    assert "hello-from-external" in out


def test_run_external_nonzero_exit_raises_with_stderr_tail(tmp_path: Path):
    script = tmp_path / "boom.py"
    script.write_text("import sys\nsys.stderr.write('kaboom-detail\\n')\nsys.exit(3)\n")
    with pytest.raises(RuntimeError) as exc:
        bridge.run_external(
            Path(sys.executable), script, [], cwd=tmp_path, timeout_s=30, label="probe"
        )
    msg = str(exc.value)
    assert "exit 3" in msg
    assert "kaboom-detail" in msg
    assert "probe" in msg
