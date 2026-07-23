"""RTMPose keypoint front-end for the multi-cue tracker (SPO-37).

RTMPose (MMPose/OpenMMLab) is the shippable pose cue: the `rtmlib` runtime and
MMPose code are Apache-2.0, and the ONNX weight *files* are redistributed under
Apache-2.0. It runs top-down (image + person boxes -> per-box keypoints) via
onnxruntime with **no mmcv dependency**, so it is far lighter and less fragile
than the full MMPose stack. Like the other heavy detectors it is not a declared
dependency; supply it per invocation with `uv run --with rtmlib`.

LICENSING — the decisive nuance (verified in the SPO-36/37/38 shipping audit):
the code and weight-files are Apache, but every stock RTMPose *body* checkpoint
is trained on **body7**, which includes **AI Challenger (non-commercial)** plus
research-only mixes (MPII/Halpe/CrowdPose/...). So the training-data axis of the
stock weights is **non-shippable**, and the SPO-41 certification gate must
refuse them. The shippable path keeps this Apache architecture + runtime and
**retrains the head on COCO (CC-BY) + synthetic keypoints** — a cheap retrain,
tracked as a follow-up. This component records that honestly rather than
pretending the Apache weight-file license clears the axis.

There is no `pose` StageKind slot: RTMPose is a component the assembled
multi-cue tracker (SPO-42) calls to attach keypoints to detections, not a
pipeline stage on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from matchlab_core.provenance import LicenseAxes, ModelProvenance
from matchlab_core.schemas.geometry import Box

# Default: RTMPose-m, COCO 17-keypoint, body7-trained (Apache files, NC data).
_DEFAULT_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)


@dataclass(frozen=True)
class DetectionPose:
    """Keypoints for one detection: (x, y, score) per body keypoint, in
    source-image coordinates — the format the multi-cue feature assembly
    consumes."""

    keypoints: list[tuple[float, float, float]]


def _to_poses(kpts, scores) -> list[DetectionPose]:
    """Map rtmlib's parallel (N, K, 2) keypoints and (N, K) scores arrays to a
    list of `DetectionPose`. Pure -> unit-testable without the model."""
    poses: list[DetectionPose] = []
    for det_kpts, det_scores in zip(kpts, scores):
        keypoints = [
            (float(xy[0]), float(xy[1]), float(s)) for xy, s in zip(det_kpts, det_scores)
        ]
        poses.append(DetectionPose(keypoints=keypoints))
    return poses


class RTMPoseEstimator:
    def __init__(
        self,
        onnx_model: str = _DEFAULT_ONNX,
        model_input_size: tuple[int, int] = (192, 256),
        architecture: str = "rtmpose-m",
    ):
        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.architecture = architecture
        self._model = None

    def prepare(self, device: str = "cpu") -> None:
        try:
            from rtmlib import RTMPose
        except ImportError as exc:
            raise RuntimeError(
                "The RTMPose front-end needs the rtmlib package (Apache-2.0, "
                "not a declared dependency): supply it per invocation with "
                "`uv run --with rtmlib ...`."
            ) from exc
        backend_device = "cuda" if device == "cuda" else "cpu"
        self._model = RTMPose(
            self.onnx_model,
            model_input_size=self.model_input_size,
            backend="onnxruntime",
            device=backend_device,
        )

    def estimate(self, image, boxes: Sequence[Box]) -> list[DetectionPose]:
        """Per-detection keypoints for `boxes` in `image`. Loud if not
        prepared (the model must be loaded first)."""
        if self._model is None:
            raise RuntimeError("RTMPoseEstimator.estimate called before prepare()")
        if not boxes:
            return []
        bboxes = [[b.x1, b.y1, b.x2, b.y2] for b in boxes]
        kpts, scores = self._model(image, bboxes=bboxes)
        return _to_poses(kpts, scores)

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            architecture=self.architecture,
            revision=self.onnx_model.rsplit("/", 1)[-1],
            lineage="RTMPose (MMPose) top-down keypoints via rtmlib/onnxruntime",
            license=LicenseAxes(
                code="Apache-2.0 (MMPose / rtmlib)",
                weights="Apache-2.0 (OpenMMLab redistributed ONNX)",
                training_data=(
                    "body7 incl. AI Challenger (non-commercial) + research-only "
                    "mixes — stock weights non-shippable; retrain head on "
                    "COCO (CC-BY) + synthetic for shipping"
                ),
            ),
        )
