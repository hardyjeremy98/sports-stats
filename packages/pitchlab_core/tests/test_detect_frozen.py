"""Frozen det-replay detect stage (SPO-30): replays exported det.txt so every
in-repo Phase 3 tracker candidate consumes byte-identical detections."""

from pathlib import Path

from pitchlab_core.registry import build
from pitchlab_core.schemas.run import StageKind


def _write_det_txt(p: Path) -> None:
    # MOT: frame(1-based),id,x,y,w,h,conf,-1,-1,-1
    p.write_text(
        "1,-1,10,20,30,40,0.9,-1,-1,-1\n"
        "1,-1,50,60,15,25,0.8,-1,-1,-1\n"
        "3,-1,11,21,30,40,0.7,-1,-1,-1\n"
    )


class _Ctx:
    """Minimal StageContext double: the frozen stage only needs video + frames()."""

    def __init__(self, frame_count, fps, stride, video_path="clip.mp4"):
        self.video = type(
            "V", (), {"frame_count": frame_count, "fps": fps, "path": video_path}
        )()
        self.config = type(
            "C",
            (),
            {"video": type("VC", (), {"sample_stride": stride, "max_frames": None})()},
        )()

    def frames(self):
        frame_cls = type("F", (), {})
        for idx in range(0, self.video.frame_count, self.config.video.sample_stride):
            f = frame_cls()
            f.frame_idx = idx
            f.t = idx / self.video.fps
            f.image = None
            yield f


def test_frozen_replays_det_txt_by_frame(tmp_path):
    det = tmp_path / "det.txt"
    _write_det_txt(det)
    stage = build(StageKind.DETECT, "frozen", {"det_path": str(det)})
    out = stage.detect(_Ctx(frame_count=4, fps=25.0, stride=1))
    by_idx = {fd.frame_idx: fd for fd in out.frames}
    assert len(by_idx[0].detections) == 2  # two rows at mot frame 1 -> frame_idx 0
    assert len(by_idx[1].detections) == 0  # no rows at mot frame 2
    assert len(by_idx[2].detections) == 1  # one row at mot frame 3 -> frame_idx 2
    d = by_idx[0].detections[0]
    assert (d.box.x1, d.box.y1, d.box.x2, d.box.y2) == (10, 20, 40, 60)  # xywh->xyxy
    assert abs(d.confidence - 0.9) < 1e-9


def test_frozen_resolves_det_from_exchange_dir_by_video_stem(tmp_path):
    # <exchange_dir>/<video-stem>/det.txt, like the oracle resolves sibling GT.
    seq_dir = tmp_path / "SNMOT-124"
    seq_dir.mkdir()
    _write_det_txt(seq_dir / "det.txt")
    stage = build(StageKind.DETECT, "frozen", {"exchange_dir": str(tmp_path)})
    out = stage.detect(
        _Ctx(frame_count=4, fps=25.0, stride=1, video_path="data/videos/x/SNMOT-124.mp4")
    )
    by_idx = {fd.frame_idx: fd for fd in out.frames}
    assert len(by_idx[0].detections) == 2  # resolved the right seq's det.txt


def test_frozen_requires_det_path_or_exchange_dir():
    import pytest

    with pytest.raises(ValueError):
        build(StageKind.DETECT, "frozen", {})


def test_frozen_provenance_hashes_det_file(tmp_path):
    det = tmp_path / "det.txt"
    _write_det_txt(det)
    stage = build(StageKind.DETECT, "frozen", {"det_path": str(det)})
    prov = stage.provenance()
    assert prov[0].weights_sha256 is not None
    assert prov[0].architecture == "frozen-detections"
