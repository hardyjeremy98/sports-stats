"""PARSeq jersey-digit reader (Apache-2.0 code, CC BY-NC fine-tuned weights).

The recogniser is the ONLY stage of the reference jersey pipeline this repo
did not already have. Its main-subject filter is replaced by `crops.py`, which
gates on measured per-frame box overlap rather than inferring contaminants
statistically from a bag of images; its ViTPose localiser is replaced by a
fixed vertical band (`number_band`, below); and its confidence-weighted vote
is replaced by the calibrated evidence layer in `reid/jersey.py`.

LOCALISATION -- `number_band` replaced an RTMPose torso crop. Measured on a
300-tracklet vertical-band sweep (task-4b jersey-ocr merge-channel report):
the whole player crop resized to 128x32 destroyed the digits (0.0800
tracklet accuracy); a fixed band at y in [0.12, 0.50] of the crop height
scored best (0.2533); and the RTMPose torso crop scored WORSE than every band
in the sweep (0.137) because RTMPose keypoints are unreliable on the ~120x51
px player crops this pipeline actually has. Pose is not a fallback option
here -- it is measured actively harmful, so it has been deleted rather than
left as a dead code path that invites its own return.

LEGIBILITY -- crops are scored by `ocr/legibility.py`'s `LegibilityClassifier`
before being read, not by pose reliability. Measured at 450 tracklets (band +
vote): reading every crop gave 73 correct / 377 wrong; HARD-gating at
legibility >= 0.9 gave 235 correct / 29 wrong -- a 13x fall in wrong reads.
That hard gate has since been REPLACED (offline 868-fragment sweep,
2026-07-31 jersey-ocr merge-channel rule sweep) by a soft per-crop weight
(`legibility ** a * confidence ** b`, a1 b1) feeding a Sigma-w-normalised
tracklet posterior with a top1-vs-top2 log-odds margin abstention
(tau=2.0, in `matchlab_core.reid.jersey.tracklet_likelihood`): the sweep's
soft-weight family dominated the hard gate on held-out pairs (AUC 0.800 vs
0.707, veto precision 0.999). `min_legibility` now only floors out crops
whose band score is so low the read is attractor-biased garbage (see
`JerseyReader.read`), not a legibility gate in its own right.

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
from matchlab_core.ocr.legibility import LegibilityClassifier
from matchlab_core.provenance import LicenseAxes, ModelProvenance
from matchlab_core.reid.jersey import DIGITS, EOS

_DEFAULT_CKPT = "data/weights/parseq-jersey.ckpt"

# Vertical band a jersey number lives in, as a fraction of crop height.
# Measured best on a 300-tracklet sweep -- see module docstring.
NUMBER_BAND_LO = 0.12
NUMBER_BAND_HI = 0.50


@dataclass
class CropRead:
    """One crop's contribution to a tracklet's number evidence."""

    char_probs: np.ndarray  # (3, 11): positions x (digits 0-9, EOS)
    legibility: float       # band-legibility score in [0, 1]; 0 contributes nothing
    frame_idx: int
    confidence: float = 1.0  # decode-confidence: P(argmax number | char_probs)


def number_band(
    image: np.ndarray, lo: float = NUMBER_BAND_LO, hi: float = NUMBER_BAND_HI
) -> np.ndarray:
    """The fixed vertical band of `image` a jersey number lives in.

    Replaces an RTMPose torso localisation, measured worse than this fixed
    band (0.137 vs. 0.2533 tracklet accuracy on a 300-tracklet sweep) because
    RTMPose keypoints are unreliable on the small (~120x51 px) player crops
    this pipeline has -- see the module docstring. A degenerate (zero-height)
    input returns the original image unchanged rather than an empty slice, so
    a malformed crop upstream fails downstream (empty read) instead of
    silently vanishing here.
    """
    h = image.shape[0]
    if h <= 0:
        return image
    return image[int(lo * h):int(hi * h), :]


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

    def __init__(self, checkpoint: str = _DEFAULT_CKPT, min_legibility: float = 0.1):
        self.checkpoint = checkpoint
        self.min_legibility = min_legibility
        self._model = None
        self._digit_cols: list[int] | None = None
        self._eos_col: int | None = None
        self._legibility = LegibilityClassifier()

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
        self._legibility.prepare(device=device)

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
        """One `CropRead` for every crop with a locatable band, carrying its
        band-legibility score in `.legibility` and its decode-confidence in
        `.confidence`.

        NO HARD LEGIBILITY GATE (offline 868-fragment sweep, 2026-07-31
        jersey-ocr merge-channel rule sweep): gating crops out at a fixed
        legibility threshold discards graded information a soft per-crop
        weight (`legibility ** a * confidence ** b`, applied downstream in
        `tracklet_likelihood`'s weights) uses instead, and the sweep's winning
        rule family dominated the shipped hard-gate rule on held-out pairs
        (AUC 0.800 vs 0.707) using exactly this soft weighting. `min_legibility`
        remains only as a floor to skip crops whose band score is so low the
        read is attractor-biased garbage rather than borderline evidence:
        measured over 16,477 sub-gate crops, low-legibility reads spike on a
        handful of attractor numbers (10, 3, 1, 8, 11) rather than spreading
        neutrally, so they are actively misleading, not merely low-weight --
        default 0.1 is a floor, not the old 0.9 gate.

        If every crop's band is degenerate (zero-size), this returns an empty
        list -- the abstention path, not an error."""
        if self._model is None:
            raise RuntimeError("JerseyReader.read called before prepare()")
        if not crops:
            return []
        bands = [number_band(crop.image) for crop in crops]
        scores = self._legibility.score(bands)
        out: list[CropRead] = []
        for crop, band, score in zip(crops, bands, scores):
            if score < self.min_legibility:
                continue
            if band.size == 0:
                continue
            probs, confidence = self._char_probs(band)
            out.append(
                CropRead(
                    char_probs=probs,
                    legibility=float(score),
                    frame_idx=crop.frame_idx,
                    confidence=confidence,
                )
            )
        return out

    def _char_probs(self, patch: np.ndarray) -> tuple[np.ndarray, float]:
        """(3, 11) digit+EOS distribution for one torso patch, plus the
        decode-confidence P(argmax number | char_probs) used as the `b`
        (confidence) factor of the soft per-crop weight."""
        import cv2
        import torch

        from matchlab_core.reid.jersey import crop_number_logprobs

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

        logprobs = crop_number_logprobs(out)
        finite = logprobs[np.isfinite(logprobs)]
        confidence = float(np.exp(finite.max())) if finite.size else 0.0
        return out, confidence
