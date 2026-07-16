"""Oracle detector: emits the video's ground-truth boxes as detections instead
of running a real detector. This isolates the tracker: feed it perfect input
and see what tracking/association failures remain once detection quality is no
longer a confound (the "tracker ceiling" experiment). Optional dropout/jitter
knobs (both off by default) support sensitivity analysis around that ceiling.

GT resolution order: explicit `gt_path` param if set, else the sibling
`<video>.gt.json` convention (same file `pitchlab_train.gt_clips` writes for
ingested SoccerNet clips) next to the video file. Missing GT is a loud error,
never silent empty output -- an oracle run without GT is meaningless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from pitchlab_core.gt import GroundTruth
from pitchlab_core.interfaces import Detector, DetectOutput, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import Detection, DetectionClass, FrameDetections
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.detect.ball_utils import resolve_ball_track

# GT roles that become detections. "other" (non-participants: staff,
# photographers) has no detector-class equivalent and is skipped.
_ROLE_TO_CLASS: dict[str, DetectionClass] = {
    "player": DetectionClass.PLAYER,
    "goalkeeper": DetectionClass.GOALKEEPER,
    "referee": DetectionClass.REFEREE,
    "ball": DetectionClass.BALL,
}


class Params(BaseModel):
    gt_path: str | None = None
    dropout_rate: float = Field(default=0.0, ge=0.0, le=1.0)  # per-detection Bernoulli drop
    jitter_px: float = Field(default=0.0, ge=0.0)  # uniform +/- offset per box corner
    seed: int = 0


@register(StageKind.DETECT, "oracle")
class OracleDetector(Detector):
    def __init__(self, **params):
        self.params = Params(**params)

    def detect(self, ctx: StageContext) -> DetectOutput:
        meta = ctx.video
        gt = _load_gt(self.params.gt_path, meta.path)
        by_frame = _index_by_frame(gt)
        rng = np.random.default_rng(self.params.seed)
        p = self.params

        frames_out: list[FrameDetections] = []
        stride = ctx.config.video.sample_stride
        max_frames = ctx.config.video.max_frames
        count = 0
        for frame_idx in range(0, meta.frame_count, stride):
            if max_frames is not None and count >= max_frames:
                break
            count += 1
            dets: list[Detection] = []
            for cls, box in by_frame.get(frame_idx, []):
                if p.dropout_rate > 0.0 and rng.random() < p.dropout_rate:
                    continue
                dets.append(
                    Detection(box=_jitter(box, p.jitter_px, rng), confidence=1.0, cls=cls)
                )
            t = frame_idx / meta.fps
            frames_out.append(FrameDetections(frame_idx=frame_idx, t=t, detections=dets))

        ball = resolve_ball_track(frames_out, fps=meta.fps)
        return DetectOutput(frames=frames_out, ball=ball)


def _sibling_gt_path(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.parent / f"{p.stem}.gt.json"


def _load_gt(gt_path_param: str | None, video_path: str) -> GroundTruth:
    sibling = _sibling_gt_path(video_path)
    if gt_path_param is not None:
        explicit = Path(gt_path_param)
        if explicit.exists():
            return GroundTruth.model_validate_json(explicit.read_text())
        raise FileNotFoundError(
            f"Oracle detector: explicit gt_path={explicit} does not exist "
            f"(sibling convention path would be {sibling})."
        )
    if sibling.exists():
        return GroundTruth.model_validate_json(sibling.read_text())
    raise FileNotFoundError(
        f"Oracle detector: no ground truth found for video {video_path!r}. Tried "
        f"explicit gt_path=<not set> and sibling convention {sibling}. Set "
        "params.gt_path to a GT json path, or place one at the sibling path."
    )


def _index_by_frame(gt: GroundTruth) -> dict[int, list[tuple[DetectionClass, Box]]]:
    by_frame: dict[int, list[tuple[DetectionClass, Box]]] = {}
    for track in gt.tracks:
        cls = _ROLE_TO_CLASS.get(track.role)
        if cls is None:  # role == "other" (or unrecognized): not a detection
            continue
        for f in track.frames:
            by_frame.setdefault(f.frame_idx, []).append((cls, f.box))
    return by_frame


def _jitter(box: Box, jitter_px: float, rng: np.random.Generator) -> Box:
    if jitter_px <= 0.0:
        return box
    dx1, dy1, dx2, dy2 = rng.uniform(-jitter_px, jitter_px, size=4)
    return Box(x1=box.x1 + dx1, y1=box.y1 + dy1, x2=box.x2 + dx2, y2=box.y2 + dy2)
