"""PARSeq jersey-digit reader (Apache-2.0 code, CC BY-NC fine-tuned weights).

The recogniser is the ONLY stage of the reference jersey pipeline this repo
did not already have. Its main-subject filter is replaced by `crops.py`, which
gates on measured per-frame box overlap rather than inferring contaminants
statistically from a bag of images; its ViTPose localiser is replaced by
`pose/rtmpose.py`; and its confidence-weighted vote is replaced by the
calibrated evidence layer in `reid/jersey.py`.

The checkpoint was fine-tuned on hockey and SoccerNet data, so any accuracy
figure measured on SoccerNet-derived footage is train-adjacent and optimistic.
That is recorded in provenance and must be disclosed in every report, not
mitigated.

Dependency isolation, same as `rtmpose.py`: torch and PARSeq are not declared
dependencies -- supply them per invocation.

TOKENIZER MAPPING -- the single worst failure mode in this module is a wrong
digit/EOS column mapping: it produces confident, plausible-looking, wrong
jersey readings, and every downstream measurement is then unattributable
garbage. `prepare()` therefore never assumes a fixed tokenizer layout. It
derives the digit columns by looking each digit character up in the loaded
tokenizer's `_itos` index->character mapping at runtime, reads the EOS column
from the tokenizer's own `eos_id`, and refuses loudly (`RuntimeError`) if the
mapping does not look as expected -- a missing digit, a digit character
appearing more than once, or an EOS column that collides with a digit column.

Confirmed against the real fine-tuned checkpoint by probe (2026-07-31, see
`.superpowers/sdd/2026-07-30-jersey-ocr-merge-channel/parseq-probe-output.md`):
`tok._itos[:14] == ['[E]', '0', '1', ..., '9', 'a', 'b', 'c']`, EOS column 0,
digit columns 1..10, forward output (1, 26, 95). One correction to the
probe's own stated conclusion, made empirically against the real checkpoint
in fix round 1: the checkpoint's `state_dict` keys are BARE (`pos_queries`),
while the hub-loaded `PARSeq` LightningModule's own state_dict keys are
`model.*` (it wraps the net as `self.model`) -- so the checkpoint's keys
must GAIN a `model.` prefix before `load_state_dict`, not lose one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matchlab_core.crops import ScoredCrop
from matchlab_core.pose.rtmpose import DetectionPose, RTMPoseEstimator
from matchlab_core.provenance import LicenseAxes, ModelProvenance
from matchlab_core.reid.jersey import DIGITS, EOS
from matchlab_core.schemas.geometry import Box

# COCO-17 keypoints bounding the region a number is printed on.
_TORSO_KEYPOINTS = (5, 6, 11, 12)  # L/R shoulder, L/R hip
_MIN_TORSO_PX = 4.0                # below this the region cannot hold a digit
_DEFAULT_CKPT = "data/weights/parseq-jersey.ckpt"


@dataclass
class CropRead:
    """One crop's contribution to a tracklet's number evidence."""

    char_probs: np.ndarray  # (3, 11): positions x (digits 0-9, EOS)
    legibility: float       # in [0, 1]; 0 contributes nothing
    frame_idx: int


def torso_box(
    pose: DetectionPose,
    crop_shape: tuple[int, int],
    *,
    min_keypoint_score: float = 0.3,
    pad_frac: float = 0.15,
) -> Box | None:
    """The shoulders-to-hips region, padded and clamped, or None.

    Returning None is the honest outcome when the pose is unreliable: a guessed
    region produces a confident read of the wrong pixels, and this channel's
    whole safety argument rests on unreadable meaning neutral rather than wrong.
    """
    pts = [pose.keypoints[i] for i in _TORSO_KEYPOINTS]
    good = [(x, y) for x, y, s in pts if s >= min_keypoint_score]
    if len(good) < 3:
        return None
    xs = [p[0] for p in good]
    ys = [p[1] for p in good]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w < _MIN_TORSO_PX or h < _MIN_TORSO_PX:
        return None
    px, py = pad_frac * w, pad_frac * h
    height, width = crop_shape
    return Box(
        x1=max(0.0, min(xs) - px),
        y1=max(0.0, min(ys) - py),
        x2=min(float(width), max(xs) + px),
        y2=min(float(height), max(ys) + py),
    )


def _resolve_digit_and_eos_columns(itos: list[str], eos_col: int) -> tuple[list[int], int]:
    """Derive digit columns 0..9 from a tokenizer's index->character mapping
    (`_itos`), failing loudly rather than assuming a layout. `eos_col` is
    read from the tokenizer itself (`eos_id`), not inferred.

    Raises `RuntimeError` naming what was found vs. what was expected when:
    a digit character is missing from the mapping, a digit character appears
    more than once in the mapping (so "the digit column" is ambiguous), or
    the given EOS column collides with a resolved digit column.
    """
    missing = [str(d) for d in range(DIGITS) if str(d) not in itos]
    if missing:
        raise RuntimeError(
            f"PARSeq tokenizer mapping is missing digit character(s) {missing} "
            f"-- expected '0'..'9' each to appear in itos, got: {itos!r}."
        )
    duplicated = [str(d) for d in range(DIGITS) if itos.count(str(d)) > 1]
    if duplicated:
        raise RuntimeError(
            f"PARSeq tokenizer mapping has duplicate digit character(s) "
            f"{duplicated} in itos -- a digit's column is ambiguous: "
            f"itos={itos!r}."
        )
    digit_cols = [itos.index(str(d)) for d in range(DIGITS)]

    if eos_col in digit_cols:
        raise RuntimeError(
            f"PARSeq tokenizer's EOS column ({eos_col}) collides with a "
            f"digit column: resolved digit columns {digit_cols} from "
            f"itos={itos!r}. Refusing to guess an EOS index."
        )
    return digit_cols, eos_col


def _add_lightning_prefix(state_dict: dict) -> dict:
    """Add one leading `model.` to every key that doesn't already have it.

    Named for the fix-round-1 probe's framing (a Lightning-wrapper prefix
    mismatch), but the direction is the opposite of that probe's stated
    conclusion. Empirically (verified against the real checkpoint, see the
    fix-round-1 report): `torch.hub.load("baudm/parseq", "parseq", ...)`
    returns `strhub.models.parseq.system.PARSeq`, a LightningModule whose
    *own* state_dict keys are `model.*` (it wraps the actual net as
    `self.model`), while `data/weights/parseq-jersey.ckpt`'s `state_dict`
    keys are bare (`pos_queries`, not `model.pos_queries`) -- i.e. the
    checkpoint holds `LightningModule.model.state_dict()`, not
    `LightningModule.state_dict()`. So the checkpoint's keys must gain the
    prefix to match what `model.load_state_dict` expects, not lose one.
    """
    return {
        (key if key.startswith("model.") else f"model.{key}"): value
        for key, value in state_dict.items()
    }


class JerseyReader:
    """Quality-gated crops in, per-crop character distributions out."""

    def __init__(self, checkpoint: str = _DEFAULT_CKPT, min_keypoint_score: float = 0.3):
        self.checkpoint = checkpoint
        self.min_keypoint_score = min_keypoint_score
        self._model = None
        self._digit_cols: list[int] | None = None
        self._eos_col: int | None = None
        self._pose = RTMPoseEstimator()

    def prepare(self, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The PARSeq jersey reader needs torch and the parseq hub repo "
                "(Apache-2.0 code, CC BY-NC weights; none are declared "
                "dependencies): supply them per invocation with `uv run "
                "--with torch --with timm --with pytorch_lightning --with nltk ...` "
                "-- baudm/parseq's hubconf refuses to load without "
                "pytorch_lightning and nltk present."
            ) from exc
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True)
        # weights_only=False: this is a full Lightning training checkpoint
        # (optimizer states, callbacks, ...), not a bare weights-only file --
        # torch>=2.6 defaults to weights_only=True and refuses to unpickle it.
        # Trusted because the checkpoint is a locally-provisioned research
        # artifact (data/weights/), never a network-fetched file.
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        raw_state_dict = state.get("state_dict", state)
        state_dict = _add_lightning_prefix(raw_state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        tok = model.tokenizer
        itos = list(tok._itos)
        self._digit_cols, self._eos_col = _resolve_digit_and_eos_columns(itos, tok.eos_id)
        self._model = model
        self._device = device
        self._pose.prepare(device=device)

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            architecture="parseq",
            revision="jersey-finetune",
            weights_path=self.checkpoint,
            lineage=(
                "PARSeq scene-text recogniser fine-tuned on hockey + SoccerNet "
                "jersey digits (Koshkina & Elder, CVPRW 2024). TRAIN-ADJACENT to "
                "any SoccerNet-derived evaluation."
            ),
            license=LicenseAxes(
                code="Apache-2.0",
                weights="CC-BY-NonCommercial-3.0",
                training_data="SoccerNet (research) + McGill Hockey (research)",
            ),
        )

    def read(self, crops: list[ScoredCrop]) -> list[CropRead]:
        """One `CropRead` per crop whose torso was locatable. Crops without a
        usable torso are ABSENT from the result, not present with zero
        confidence -- missing evidence, not evidence of absence."""
        if self._model is None:
            raise RuntimeError("JerseyReader.read called before prepare()")
        out: list[CropRead] = []
        for crop in crops:
            h, w = crop.image.shape[:2]
            poses = self._pose.estimate(
                crop.image, [Box(x1=0.0, y1=0.0, x2=float(w), y2=float(h))]
            )
            if not poses:
                continue
            box = torso_box(poses[0], (h, w), min_keypoint_score=self.min_keypoint_score)
            if box is None:
                continue
            patch = crop.image[int(box.y1):int(box.y2), int(box.x1):int(box.x2)]
            if patch.size == 0:
                continue
            probs = self._char_probs(patch)
            out.append(
                CropRead(
                    char_probs=probs,
                    legibility=float(crop.quality),
                    frame_idx=crop.frame_idx,
                )
            )
        return out

    def _char_probs(self, patch: np.ndarray) -> np.ndarray:
        """(3, 11) digit+EOS distribution for one torso patch."""
        import cv2
        import torch

        rgb = cv2.cvtColor(cv2.resize(patch, (128, 32)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        x = x.sub_(0.5).div_(0.5).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(x)
        full = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        cols = list(self._digit_cols) + [self._eos_col]
        rows = min(3, full.shape[0])
        out = np.zeros((3, DIGITS + 1), dtype=np.float64)
        out[:, EOS] = 1.0  # unread positions terminate the string
        for i in range(rows):
            sub = full[i, cols]
            total = float(sub.sum())
            out[i] = sub / total if total > 0 else out[i]
        return out
