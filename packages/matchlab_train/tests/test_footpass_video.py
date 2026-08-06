from __future__ import annotations

import os

import numpy as np
import pytest
from matchlab_train.datasets.footpass_video import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    MatchVideo,
    match_video_path,
)
from matchlab_train.datasets.paths import data_root

cv2 = pytest.importorskip("cv2")


@pytest.mark.parametrize(
    "key,expected",
    [
        ("game_18_H1", "game_18.mp4"),
        ("game_18_H2", "game_18.mp4"),  # BOTH halves share one file
        ("game_47_H2", "game_47.mp4"),
    ],
)
def test_both_halves_resolve_to_one_match_video(key, expected):
    assert match_video_path("/vids", key).name == expected


@pytest.fixture
def synthetic_video(tmp_path):
    """20 frames whose top-left pixel encodes the frame index, so a seek that lands
    on the wrong frame is detectable rather than merely plausible."""
    path = tmp_path / "game_1.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (EXPECTED_WIDTH, EXPECTED_HEIGHT)
    )
    for i in range(20):
        frame = np.zeros((EXPECTED_HEIGHT, EXPECTED_WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (i * 10, i * 10, i * 10)
        writer.write(frame)
    writer.release()
    return path


def test_reads_the_requested_frames(synthetic_video):
    with MatchVideo(synthetic_video) as v:
        clip = v.read_clip(5, 4)
    assert clip.shape == (4, EXPECTED_HEIGHT, EXPECTED_WIDTH, 3)
    # Frame content increases with index; lossy encoding means approximate.
    means = [float(f.mean()) for f in clip]
    assert means == sorted(means)
    assert means[0] == pytest.approx(50, abs=12)


def test_seek_lands_on_the_requested_frame(synthetic_video):
    with MatchVideo(synthetic_video) as v:
        first = v.read_clip(12, 1)[0].mean()
        again = v.read_clip(12, 1)[0].mean()
        earlier = v.read_clip(3, 1)[0].mean()
    assert first == pytest.approx(again)  # repeatable, not stateful
    assert earlier < first  # seeking backwards works


def test_clip_running_past_the_end_raises(synthetic_video):
    with MatchVideo(synthetic_video) as v, pytest.raises(RuntimeError, match="read failed"):
        v.read_clip(18, 10)


def test_frames_are_rgb_not_bgr(tmp_path):
    """OpenCV decodes BGR. Feeding BGR to an ImageNet/Kinetics-pretrained backbone
    is a silent accuracy loss, not an error."""
    path = tmp_path / "game_2.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (EXPECTED_WIDTH, EXPECTED_HEIGHT)
    )
    for _ in range(5):
        frame = np.zeros((EXPECTED_HEIGHT, EXPECTED_WIDTH, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # pure RED in BGR memory order
        writer.write(frame)
    writer.release()
    with MatchVideo(path) as v:
        clip = v.read_clip(1, 1)
    r, g, b = (float(clip[0, :, :, c].mean()) for c in range(3))
    assert r > 200 and g < 60 and b < 60


def test_wrong_resolution_is_rejected(tmp_path):
    path = tmp_path / "game_3.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 176))
    for _ in range(3):
        writer.write(np.zeros((176, 320, 3), dtype=np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="expected 640x352"):
        MatchVideo(path)


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MatchVideo(tmp_path / "nope.mp4")


# --- against the real extracted footage -------------------------------------------

_root = data_root()
_val_video = (
    _root / "footpass" / "videos_352x640" / "game_18.mp4" if _root else None
)
requires_val_video = pytest.mark.skipif(
    _val_video is None or not _val_video.is_file(),
    reason="FOOTPASS VAL video not extracted",
)


@requires_val_video
def test_real_val_video_properties():
    with MatchVideo(_val_video) as v:
        assert (v.width, v.height) == (640, 352)
        assert v.fps == 25.0
        # Must cover the maximum tactical frame index for game_18_H2 (149,181).
        assert v.frame_count >= 149_181


@requires_val_video
@pytest.mark.skipif(
    not os.environ.get("MATCHLAB_SLOW_TESTS"),
    reason="decodes real footage; set MATCHLAB_SLOW_TESTS=1 to run",
)
def test_reads_a_clip_spanning_the_half_boundary():
    """Frames are continuous across halves, so a clip straddling the H1/H2 boundary
    is readable from the same file. If halves were separate videos this would fail."""
    with MatchVideo(_val_video) as v:
        clip = v.read_clip(75_300, 50)
    assert clip.shape == (50, 352, 640, 3)
    assert clip.std() > 1.0  # real footage, not black frames
