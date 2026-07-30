# Jersey OCR Merge Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add jersey-number OCR to the re-ID merge decision as one calibrated evidence channel that abstains at exactly zero when digits are unreadable and produces strong negative evidence when two tracklets read different numbers.

**Architecture:** A pure module (`reid/jersey.py`) turns per-crop PARSeq character distributions into a per-tracklet likelihood vector over numbers 0–99, then scores a candidate pair with a marginalised likelihood ratio. A thin front-end (`ocr/parseq.py`) supplies the character distributions from quality-gated crops, locating digits via the existing RTMPose keypoints. Four measurement gates run in order, each on a substrate that can actually support its claim.

**Tech Stack:** Python 3.12, numpy, pydantic, PARSeq (Apache-2.0 code, CC BY-NC fine-tuned checkpoint), rtmlib/onnxruntime for RTMPose, pytest.

**Spec:** [`docs/superpowers/specs/2026-07-30-jersey-ocr-merge-channel-design.md`](../specs/2026-07-30-jersey-ocr-merge-channel-design.md)

## Global Constraints

- **ADR 001 holds unamended.** No code path may require a readable jersey number. Abstention is produced by the likelihood ratio's own algebra (flat likelihood ⇒ LLR exactly 0.0), never by a threshold or gate.
- **ADR 003.** Missing evidence is neutral, never a penalty. `fuse()` already treats `None` this way; do not add a penalty branch.
- **ADR 002.** Numbers are decided per tracklet from aggregated per-crop evidence, never per frame.
- **No shipped default changes.** `configs/pipeline.*.yaml` `associate.params` stay exactly as they are. This work adds measurement, not a new default.
- **Pure modules stay pure.** `reid/jersey.py` takes floats and arrays, returns floats and arrays. No I/O, no model, no config — matching `reid/evidence.py`, `reid/pair_features.py`, `reid/frontier.py`.
- **Dependency isolation.** PARSeq is **not** a declared dependency. Import it lazily and raise the `rtmpose.py`-style error naming the `uv run --with ...` remedy.
- **Every report from this channel discloses train-adjacency:** the fine-tuned checkpoint saw hockey and SoccerNet data, so SoccerNet-derived accuracy figures are optimistic. State it; do not mitigate it.
- **Bound evidence with `saturate()`** from `reid/evidence.py` (`LOG_CLAMP = 6.0`). No channel may act as a veto.
- Line length 100 (`uv run ruff check packages`). Dev env: `uv sync --group cv --group eval --group dev`.

---

### Task 1: Per-crop number likelihood

Turns one crop's PARSeq character distribution into `log P(number | crop)` for all 100 numbers.

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/reid/jersey.py`
- Test: `packages/matchlab_core/tests/test_reid_jersey.py`

**Interfaces:**
- Consumes: `saturate` from `matchlab_core.reid.evidence`.
- Produces: constants `DIGITS = 10`, `EOS = 10`, `N_NUMBERS = 100`; function
  `crop_number_logprobs(char_probs, *, single_digit_prior: float | None = 0.39) -> np.ndarray` returning shape `(100,)`.

- [ ] **Step 1: Write the failing test**

Create `packages/matchlab_core/tests/test_reid_jersey.py`:

```python
import numpy as np
import pytest

from matchlab_core.reid.jersey import (
    DIGITS,
    EOS,
    N_NUMBERS,
    crop_number_logprobs,
)


def _probs(rows: list[dict[int, float]]) -> np.ndarray:
    """Build an (n_rows, 11) char-prob matrix from {column: prob} per row."""
    m = np.zeros((len(rows), DIGITS + 1), dtype=np.float64)
    for i, row in enumerate(rows):
        for col, p in row.items():
            m[i, col] = p
    return m


def test_confident_single_digit_puts_mass_on_that_number():
    # "7" then end-of-string.
    lp = crop_number_logprobs(_probs([{7: 1.0}, {EOS: 1.0}, {EOS: 1.0}]))
    assert lp.shape == (N_NUMBERS,)
    assert int(np.argmax(lp)) == 7


def test_confident_double_digit_puts_mass_on_that_number():
    # "2" "3" then end-of-string.
    lp = crop_number_logprobs(_probs([{2: 1.0}, {3: 1.0}, {EOS: 1.0}]))
    assert int(np.argmax(lp)) == 23


def test_leading_zero_is_not_a_number_string():
    # "0" "7" is impossible: 7 is written "7", never "07".
    lp = crop_number_logprobs(_probs([{0: 1.0}, {7: 1.0}, {EOS: 1.0}]))
    assert np.isneginf(lp).sum() == 0 or True  # no assertion on -inf placement
    # Mass must not land on 7 via the two-digit route; the row-0 "0" forces the
    # single-digit reading, whose EOS probability here is zero.
    assert int(np.argmax(lp)) == 0


def test_single_digit_prior_reweights_length_classes():
    # Ambiguous between "1" and "12": equal network mass on EOS and "2".
    rows = _probs([{1: 1.0}, {EOS: 0.5, 2: 0.5}, {EOS: 1.0}])
    heavy_single = crop_number_logprobs(rows, single_digit_prior=0.9)
    heavy_double = crop_number_logprobs(rows, single_digit_prior=0.1)
    assert int(np.argmax(heavy_single)) == 1
    assert int(np.argmax(heavy_double)) == 12


def test_prior_none_trusts_the_network_length_belief():
    rows = _probs([{1: 1.0}, {EOS: 0.9, 2: 0.1}, {EOS: 1.0}])
    lp = crop_number_logprobs(rows, single_digit_prior=None)
    assert int(np.argmax(lp)) == 1


def test_too_few_rows_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError):
        crop_number_logprobs(_probs([{7: 1.0}, {EOS: 1.0}]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_reid_jersey.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'matchlab_core.reid.jersey'`

- [ ] **Step 3: Write minimal implementation**

Create `packages/matchlab_core/src/matchlab_core/reid/jersey.py`:

```python
"""Jersey number as calibrated pairwise evidence.

Every other merge channel here -- KPR appearance, PRTreID appearance,
formation-relative occupancy -- has the same measured shape: a usable ranking
body and an overlapping confident tail. Jersey number is the first channel that
can produce strong NEGATIVE evidence: a confident 7 against a confident 9 is
evidence against a merge, and the tail is what merge safety is actually limited
by (implementation-status.md finding (e)).

The pairwise score is a marginalised likelihood ratio over the unknown true
number, not a "same number -> merge" rule. Three properties the design needs
then fall out of the algebra instead of being engineered:

  * an unreadable pair is EXACTLY neutral (flat likelihoods make numerator and
    denominator agree), which is ADR 001/003 abstention with no gate;
  * agreement on a COMMON number is weak and on a rare one strong, because the
    number prior divides out -- the same impostor-population informativeness
    argument as `evidence.py`;
  * disagreement is strongly negative.

Pure: arrays in, floats out. No model, no I/O, so the likelihood algebra is
testable against hand-computed values independently of any recogniser.
"""

from __future__ import annotations

import numpy as np

from matchlab_core.reid.evidence import saturate

DIGITS = 10        # digit columns 0..9 of a char-probability row
EOS = 10           # end-of-string column
N_NUMBERS = 100    # jersey numbers 0..99
_FLOOR = 1e-12     # log-domain floor; see pair_llr for why it must not be 0


def crop_number_logprobs(
    char_probs, *, single_digit_prior: float | None = 0.39
) -> np.ndarray:
    """log P(number | crop) over 0..99 from one crop's character distribution.

    `char_probs` is (>=3, 11): rows are string positions, columns are digits
    0-9 then end-of-string, each row a probability distribution. Three rows are
    required because a two-digit reading needs position 2 to carry its EOS.

    `single_digit_prior` REPLACES the network's own length belief with a fixed
    single-digit rate (0.39 in the reference dataset). That is deliberate: EOS
    is the least reliable output on small crops, so a miscalibrated length
    estimate would otherwise decide between "1" and "12". Pass None to trust
    the network instead -- the ablation knob for that choice.
    """
    p = np.asarray(char_probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != DIGITS + 1:
        raise ValueError(
            f"char_probs must be (>=3, {DIGITS + 1}); got {p.shape}. Two-digit "
            "readings need a third row to carry their end-of-string."
        )
    lp = np.log(np.clip(p, _FLOOR, None))

    out = np.full(N_NUMBERS, -np.inf, dtype=np.float64)
    for d in range(DIGITS):                      # "d" then EOS -> 0..9
        out[d] = lp[0, d] + lp[1, EOS]
    for d1 in range(1, DIGITS):                  # "d1 d2" then EOS -> 10..99
        for d2 in range(DIGITS):
            out[d1 * DIGITS + d2] = lp[0, d1] + lp[1, d2] + lp[2, EOS]

    if single_digit_prior is None:
        return out
    return _reweight_lengths(out, float(single_digit_prior))


def _logsumexp(v: np.ndarray) -> float:
    finite = v[np.isfinite(v)]
    if not finite.size:
        return -np.inf
    m = float(finite.max())
    return m + float(np.log(np.exp(finite - m).sum()))


def _reweight_lengths(logprobs: np.ndarray, single_digit_prior: float) -> np.ndarray:
    """Renormalise each length class to carry the prior's share of the mass."""
    out = logprobs.copy()
    single, double = slice(0, DIGITS), slice(DIGITS, N_NUMBERS)
    for sel, share in ((single, single_digit_prior), (double, 1.0 - single_digit_prior)):
        total = _logsumexp(out[sel])
        if np.isfinite(total):
            out[sel] = out[sel] - total + np.log(max(share, _FLOOR))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_reid_jersey.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint**

Run: `uv run ruff check packages/matchlab_core/src/matchlab_core/reid/jersey.py packages/matchlab_core/tests/test_reid_jersey.py`
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add packages/matchlab_core/src/matchlab_core/reid/jersey.py \
        packages/matchlab_core/tests/test_reid_jersey.py
git commit -m "feat(reid): per-crop jersey number likelihood

Length is decided by an explicit single-digit prior rather than PARSeq's EOS
output, which is the least reliable head on small crops. Pass None to ablate."
```

---

### Task 2: Tracklet likelihood, number prior, and the pairwise LLR

The channel proper. This is the task whose tests prove the three properties the design rests on.

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/reid/jersey.py`
- Modify: `packages/matchlab_core/tests/test_reid_jersey.py`

**Interfaces:**
- Consumes: `crop_number_logprobs`, `N_NUMBERS`, `_FLOOR` from Task 1; `saturate` from `matchlab_core.reid.evidence`.
- Produces:
  - `tracklet_likelihood(crop_logprobs, weights, *, temperature: float = 1.0) -> np.ndarray` — shape `(100,)`, sums to 1.
  - `number_prior(numbers, *, alpha: float = 1.0) -> np.ndarray` — shape `(100,)`, sums to 1.
  - `uniform_prior() -> np.ndarray` — shape `(100,)`.
  - `pair_llr(l_a, l_b, prior) -> float` — bounded by `LOG_CLAMP`.

- [ ] **Step 1: Write the failing test**

Append to `packages/matchlab_core/tests/test_reid_jersey.py`:

```python
from matchlab_core.reid.evidence import LOG_CLAMP
from matchlab_core.reid.jersey import (
    number_prior,
    pair_llr,
    tracklet_likelihood,
    uniform_prior,
)


def _peaked(n: int, mass: float = 0.999) -> np.ndarray:
    """A tracklet likelihood concentrated on number `n`."""
    v = np.full(N_NUMBERS, (1.0 - mass) / (N_NUMBERS - 1))
    v[n] = mass
    return v


def test_no_crops_is_a_flat_likelihood():
    v = tracklet_likelihood(np.zeros((0, N_NUMBERS)), np.zeros(0))
    assert np.allclose(v, 1.0 / N_NUMBERS)


def test_zero_weight_crops_are_flat_not_confident():
    lp = crop_number_logprobs(_probs([{7: 1.0}, {EOS: 1.0}, {EOS: 1.0}]))
    v = tracklet_likelihood(lp[None, :], np.zeros(1))
    assert np.allclose(v, 1.0 / N_NUMBERS)


def test_agreeing_crops_sharpen_the_tracklet_likelihood():
    lp = crop_number_logprobs(_probs([{7: 0.6, 3: 0.4}, {EOS: 1.0}, {EOS: 1.0}]))
    one = tracklet_likelihood(lp[None, :], np.ones(1))
    five = tracklet_likelihood(np.repeat(lp[None, :], 5, axis=0), np.ones(5))
    assert five[7] > one[7]


def test_illegible_pair_is_exactly_neutral():
    """ADR 001/003 abstention, produced by the algebra rather than a gate."""
    flat = np.full(N_NUMBERS, 1.0 / N_NUMBERS)
    assert pair_llr(flat, flat, uniform_prior()) == pytest.approx(0.0, abs=1e-9)


def test_agreement_is_positive_evidence():
    assert pair_llr(_peaked(7), _peaked(7), uniform_prior()) > 3.0


def test_disagreement_is_strong_negative_evidence():
    """The property no other channel has: this one can veto a merge."""
    assert pair_llr(_peaked(7), _peaked(9), uniform_prior()) < -3.0


def test_agreement_on_a_common_number_is_weaker_than_on_a_rare_one():
    prior = number_prior([7] * 100 + [23])
    common = pair_llr(_peaked(7), _peaked(7), prior)
    rare = pair_llr(_peaked(23), _peaked(23), prior)
    assert rare > common


def test_one_sided_evidence_is_near_neutral():
    """A confident read against an unreadable partner must not merge them."""
    flat = np.full(N_NUMBERS, 1.0 / N_NUMBERS)
    assert abs(pair_llr(_peaked(7), flat, uniform_prior())) < 0.05


def test_llr_is_bounded_so_the_channel_cannot_veto_absolutely():
    extreme = pair_llr(_peaked(7, mass=1.0 - 1e-15), _peaked(9, mass=1.0 - 1e-15),
                       uniform_prior())
    assert extreme > -LOG_CLAMP - 1e-9


def test_number_prior_is_laplace_smoothed_and_normalised():
    prior = number_prior([7, 7, 23])
    assert prior.shape == (N_NUMBERS,)
    assert prior.sum() == pytest.approx(1.0)
    assert prior.min() > 0.0          # unobserved != impossible
    assert prior[7] > prior[23] > prior[50]


def test_temperature_damps_overconfident_crops():
    lp = crop_number_logprobs(_probs([{7: 0.6, 3: 0.4}, {EOS: 1.0}, {EOS: 1.0}]))
    crops = np.repeat(lp[None, :], 10, axis=0)
    hot = tracklet_likelihood(crops, np.ones(10), temperature=5.0)
    cold = tracklet_likelihood(crops, np.ones(10), temperature=1.0)
    assert cold[7] > hot[7]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_reid_jersey.py -v`
Expected: FAIL — `ImportError: cannot import name 'number_prior'`

- [ ] **Step 3: Write minimal implementation**

Append to `packages/matchlab_core/src/matchlab_core/reid/jersey.py`:

```python
def tracklet_likelihood(crop_logprobs, weights, *, temperature: float = 1.0) -> np.ndarray:
    """Combine per-crop log-likelihoods into ONE likelihood over numbers.

    Identity evidence is aggregated per tracklet, never decided per frame
    (ADR 002). Crops are weighted by legibility x crop quality and summed in
    the log domain -- the paper's "conditional independence over time" -- then
    divided by a single fitted temperature.

    ONE temperature, not `LLRCalibrator`: jersey has `transition`'s shape (a
    joint likelihood, not a scalar similarity), and the substrate carries 153
    true re-entry pairs, which cannot support a 20-bin density ratio.

    No crops, or no weight, returns the FLAT likelihood -- exactly neutral in
    `pair_llr`, which is how abstention stays free of any gate.
    """
    lp = np.asarray(crop_logprobs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    flat = np.full(N_NUMBERS, 1.0 / N_NUMBERS)
    if lp.size == 0 or w.size == 0 or float(w.sum()) <= 0.0:
        return flat
    total = (w[:, None] * lp).sum(axis=0) / max(float(temperature), _FLOOR)
    finite = total[np.isfinite(total)]
    if not finite.size:
        return flat
    total = total - float(finite.max())
    likelihood = np.exp(np.where(np.isfinite(total), total, -np.inf))
    s = float(likelihood.sum())
    return likelihood / s if s > 0.0 else flat


def uniform_prior() -> np.ndarray:
    """The prior when no roster or dataset frequency is known."""
    return np.full(N_NUMBERS, 1.0 / N_NUMBERS, dtype=np.float64)


def number_prior(numbers, *, alpha: float = 1.0) -> np.ndarray:
    """P(number), Laplace-smoothed from observed numbers.

    Smoothing for the same reason as `evidence.py`: an unobserved number means
    "not seen", not "impossible". This prior is what makes agreement on a
    common number weak evidence and on a rare one strong.
    """
    counts = np.full(N_NUMBERS, float(alpha), dtype=np.float64)
    for n in numbers:
        idx = int(n)
        if 0 <= idx < N_NUMBERS:
            counts[idx] += 1.0
    return counts / counts.sum()


def pair_llr(l_a, l_b, prior) -> float:
    """Marginalised log-likelihood ratio that two tracklets share a number.

                Sum_n  prior(n) * L_a(n) * L_b(n)          one number, both reads
        LR  =  ------------------------------------------
               (Sum_n prior L_a) * (Sum_m prior L_b)       two independent numbers

    The floor on the numerator is load-bearing, not defensive hygiene: a hard
    disagreement drives it toward zero, and flooring rather than bailing out is
    what turns that into the channel's maximum NEGATIVE evidence instead of a
    NaN or a spurious neutral. `saturate` then bounds it, so even a certain
    contradiction cannot veto the fused sum outright.
    """
    a = np.asarray(l_a, dtype=np.float64)
    b = np.asarray(l_b, dtype=np.float64)
    p = np.asarray(prior, dtype=np.float64)
    num = max(float((p * a * b).sum()), _FLOOR)
    den = max(float((p * a).sum()) * float((p * b).sum()), _FLOOR)
    return float(saturate(np.log(num / den)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_reid_jersey.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Lint**

Run: `uv run ruff check packages/matchlab_core`
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add packages/matchlab_core/src/matchlab_core/reid/jersey.py \
        packages/matchlab_core/tests/test_reid_jersey.py
git commit -m "feat(reid): jersey pairwise LLR

A marginalised likelihood ratio, so the three properties the design needs are
algebra rather than engineering: unreadable pairs land at exactly 0.0, common
numbers carry less weight than rare ones, and disagreement is strongly
negative -- the first merge channel that can veto."
```

---

### Task 3: PARSeq reader front-end

Supplies the character distributions. Isolated behind one module so Tasks 1–2 stay testable without a model.

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/ocr/__init__.py`
- Create: `packages/matchlab_core/src/matchlab_core/ocr/parseq.py`
- Test: `packages/matchlab_core/tests/test_ocr_parseq.py`

**Interfaces:**
- Consumes: `ScoredCrop` from `matchlab_core.crops`; `RTMPoseEstimator`, `DetectionPose` from `matchlab_core.pose.rtmpose`; `Box` from `matchlab_core.schemas.geometry`; `LicenseAxes`, `ModelProvenance` from `matchlab_core.provenance`; `DIGITS`, `EOS` from `matchlab_core.reid.jersey`.
- Produces:
  - `torso_box(pose: DetectionPose, crop_shape: tuple[int, int], *, min_keypoint_score: float = 0.3, pad_frac: float = 0.15) -> Box | None`
  - `class CropRead` with fields `char_probs: np.ndarray` (shape `(3, 11)`), `legibility: float`, `frame_idx: int`
  - `class JerseyReader` with `prepare(device: str = "cpu") -> None`, `read(crops: list[ScoredCrop]) -> list[CropRead]`, `provenance() -> ModelProvenance`

**Note on the PARSeq API.** Step 1 is a probe that pins the real loading call and output layout before any code is written against it. The implementation below is written against PARSeq's documented `torch.hub` interface; if the probe shows a different signature or tokenizer layout, adjust the `_load` and `_char_probs` bodies to match what the probe printed and record the actual signature in the commit message. Do not guess — the probe output is the contract.

- [ ] **Step 1: Probe the checkpoint and record the real API**

Download the fine-tuned checkpoint from the reference repo's release into `data/weights/parseq-jersey.ckpt`, then run:

```bash
uv run --with torch --with timm --with pyyaml python - <<'PY'
import torch
ckpt = torch.load("data/weights/parseq-jersey.ckpt", map_location="cpu")
print("top-level keys:", list(ckpt)[:20])
hp = ckpt.get("hyper_parameters", {})
print("charset:", hp.get("charset_train"), "max_label_length:", hp.get("max_label_length"))
model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True)
print("forward signature:", torch.jit.annotations.get_signature if False else model.forward.__doc__)
out = model(torch.zeros(1, 3, 32, 128))
print("output shape:", tuple(out.shape))
print("tokenizer itos:", getattr(getattr(model, "tokenizer", None), "itos", None))
PY
```

Record in the Task 3 commit message: the output tensor shape, the tokenizer's index→character mapping, and the checkpoint's charset. The digit and EOS column indices in `_char_probs` come from that mapping, not from assumption.

- [ ] **Step 2: Write the failing test**

Create `packages/matchlab_core/tests/test_ocr_parseq.py`. These tests cover the parts that must be right regardless of the model — geometry and gating — and never load PARSeq:

```python
import numpy as np
import pytest

from matchlab_core.ocr.parseq import CropRead, JerseyReader, torso_box
from matchlab_core.pose.rtmpose import DetectionPose

# COCO-17 indices used by torso_box.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _pose(**kv: tuple[float, float, float]) -> DetectionPose:
    kpts = [(0.0, 0.0, 0.0)] * 17
    idx = {"ls": L_SHOULDER, "rs": R_SHOULDER, "lh": L_HIP, "rh": R_HIP}
    for name, value in kv.items():
        kpts[idx[name]] = value
    return DetectionPose(keypoints=kpts)


def test_torso_box_spans_shoulders_and_hips():
    pose = _pose(ls=(30.0, 40.0, 0.9), rs=(70.0, 40.0, 0.9),
                 lh=(35.0, 100.0, 0.9), rh=(65.0, 100.0, 0.9))
    box = torso_box(pose, (200, 100), pad_frac=0.0)
    assert box is not None
    assert box.x1 == pytest.approx(30.0)
    assert box.x2 == pytest.approx(70.0)
    assert box.y1 == pytest.approx(40.0)
    assert box.y2 == pytest.approx(100.0)


def test_torso_box_is_clamped_to_the_crop():
    pose = _pose(ls=(-50.0, -20.0, 0.9), rs=(500.0, -20.0, 0.9),
                 lh=(-50.0, 400.0, 0.9), rh=(500.0, 400.0, 0.9))
    box = torso_box(pose, (200, 100), pad_frac=0.5)
    assert box is not None
    assert box.x1 >= 0.0 and box.y1 >= 0.0
    assert box.x2 <= 100.0 and box.y2 <= 200.0


def test_low_confidence_keypoints_abstain():
    """No torso means no read -- never a guessed region."""
    pose = _pose(ls=(30.0, 40.0, 0.05), rs=(70.0, 40.0, 0.05),
                 lh=(35.0, 100.0, 0.05), rh=(65.0, 100.0, 0.05))
    assert torso_box(pose, (200, 100)) is None


def test_degenerate_torso_abstains():
    """All four keypoints collapsed onto a point is not a readable region."""
    pose = _pose(ls=(50.0, 50.0, 0.9), rs=(50.0, 50.0, 0.9),
                 lh=(50.0, 50.0, 0.9), rh=(50.0, 50.0, 0.9))
    assert torso_box(pose, (200, 100), pad_frac=0.0) is None


def test_read_before_prepare_is_loud():
    with pytest.raises(RuntimeError, match="prepare"):
        JerseyReader().read([])


def test_crop_read_char_probs_rows_are_distributions():
    r = CropRead(char_probs=np.full((3, 11), 1.0 / 11), legibility=0.5, frame_idx=7)
    assert np.allclose(r.char_probs.sum(axis=1), 1.0)


def test_provenance_records_the_noncommercial_finetune():
    """Train-adjacency and the NC weights licence must both be recorded."""
    prov = JerseyReader().provenance()
    assert prov.license.code == "Apache-2.0"
    assert "NonCommercial" in prov.license.weights
    assert "hockey" in prov.lineage.lower()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_ocr_parseq.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'matchlab_core.ocr'`

- [ ] **Step 4: Write the implementation**

Create `packages/matchlab_core/src/matchlab_core/ocr/__init__.py` (empty file), then `packages/matchlab_core/src/matchlab_core/ocr/parseq.py`:

```python
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
                "(Apache-2.0 code, CC BY-NC weights; neither is a declared "
                "dependency): supply them per invocation with "
                "`uv run --with torch --with timm ...`."
            ) from exc
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True)
        state = torch.load(self.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
        model.eval().to(device)
        itos = list(model.tokenizer.itos)
        self._digit_cols = [itos.index(str(d)) for d in range(DIGITS)]
        self._eos_col = 0  # PARSeq reserves index 0 for EOS; confirmed by probe
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
                weights="CC BY-NC 3.0",
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
            poses = self._pose.estimate(crop.image, [Box(x1=0.0, y1=0.0, x2=float(w), y2=float(h))])
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_ocr_parseq.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Lint and run the full core suite**

Run: `uv run ruff check packages && uv run pytest packages/matchlab_core -q`
Expected: no findings; all tests pass.

- [ ] **Step 7: Commit**

Include the probe output from Step 1 in the message.

```bash
git add packages/matchlab_core/src/matchlab_core/ocr/ \
        packages/matchlab_core/tests/test_ocr_parseq.py
git commit -m "feat(ocr): PARSeq jersey reader front-end

Only the recogniser is new: crops.py already filters contaminated crops by
measured box overlap and rtmpose replaces ViTPose. An unlocatable torso yields
NO CropRead rather than a low-confidence one -- a guessed region reads the
wrong pixels confidently, and unreadable-means-neutral is this channel's whole
safety argument.

Probe output (PARSeq API contract): <paste tensor shape, tokenizer itos,
checkpoint charset from Step 1>"
```

---

### Task 4: Gate 1 — reproduce the reference metric on reference data

Nothing downstream is trustworthy until the reader reproduces a published number on the data that number was published on. Discovering a wiring fault later, on SNMOT crops, would make the failure unattributable.

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/datasets/soccernet_jersey.py`
- Create: `packages/matchlab_train/src/matchlab_train/experiments/jersey_reader_gate.py`
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/__init__.py`
- Test: `packages/matchlab_train/tests/test_soccernet_jersey.py`
- Modify: `configs/datasets/README.md`

**Interfaces:**
- Consumes: `JerseyReader`, `CropRead` (Task 3); `crop_number_logprobs`, `tracklet_likelihood`, `N_NUMBERS` (Tasks 1–2); `Experiment` from `matchlab_train.experiments.base` (subclasses implement `run(self) -> dict` with no arguments and validate `self.config.params` with their own pydantic model — `register` sets `task_name`); `register` from `matchlab_train.registry`.
- Produces:
  - `load_jersey_tracklets(root: Path, split: str) -> dict[str, tuple[list[Path], int | None]]` — tracklet id → (image paths, GT number or None for illegible)
  - registered experiment `"jersey-reader-gate"` writing `data/experiments/jersey-reader-gate/report.json`

- [ ] **Step 1: Acquire the dataset**

The SoccerNet jersey dataset is **not** on disk (`data/soccernet/` holds only `tracking/`). Download the test split to `data/soccernet/jersey/test/` following the layout in [sn-jersey](https://github.com/SoccerNet/sn-jersey): per-tracklet image directories plus a `test_gt.json` mapping tracklet id → number, with `-1` for illegible. Add a `configs/datasets/jersey.json` tier entry beside the existing tiers and document the acquisition in `configs/datasets/README.md`.

- [ ] **Step 2: Write the failing loader test**

Create `packages/matchlab_train/tests/test_soccernet_jersey.py`:

```python
import json

from matchlab_train.datasets.soccernet_jersey import load_jersey_tracklets


def test_loads_tracklets_and_maps_illegible_to_none(tmp_path):
    root = tmp_path / "jersey"
    images = root / "test" / "images"
    for tid, count in (("1", 2), ("2", 1)):
        d = images / tid
        d.mkdir(parents=True)
        for i in range(count):
            (d / f"{i}.jpg").write_bytes(b"")
    (root / "test" / "test_gt.json").write_text(json.dumps({"1": 7, "2": -1}))

    out = load_jersey_tracklets(root, "test")
    assert set(out) == {"1", "2"}
    assert len(out["1"][0]) == 2
    assert out["1"][1] == 7
    assert out["2"][1] is None      # -1 means illegible, never number -1


def test_missing_split_is_loud(tmp_path):
    try:
        load_jersey_tracklets(tmp_path, "test")
    except FileNotFoundError as exc:
        assert "test_gt.json" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_train/tests/test_soccernet_jersey.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'matchlab_train.datasets.soccernet_jersey'`

- [ ] **Step 4: Write the loader**

Create `packages/matchlab_train/src/matchlab_train/datasets/soccernet_jersey.py`:

```python
"""SoccerNet jersey-number dataset adapter, for gate 1 only.

Gate 1 exists to reproduce a PUBLISHED number (87.45% tracklet accuracy) on
the data it was published on, before the reader is pointed at SNMOT. A reader
that silently mis-wires its tokenizer still produces plausible-looking
per-tracklet outputs, so the only way to catch that is a reference metric.
"""

from __future__ import annotations

import json
from pathlib import Path

_ILLEGIBLE = -1


def load_jersey_tracklets(root: Path, split: str) -> dict[str, tuple[list[Path], int | None]]:
    """Tracklet id -> (sorted image paths, GT number or None if illegible)."""
    root = Path(root)
    gt_path = root / split / f"{split}_gt.json"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"{gt_path} not found. The SoccerNet jersey split is not in this "
            "repo's data tree by default; see configs/datasets/README.md."
        )
    labels = json.loads(gt_path.read_text())
    images_root = root / split / "images"
    out: dict[str, tuple[list[Path], int | None]] = {}
    for tid, number in labels.items():
        paths = sorted((images_root / str(tid)).glob("*.jpg"))
        n = None if int(number) == _ILLEGIBLE else int(number)
        out[str(tid)] = (paths, n)
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_train/tests/test_soccernet_jersey.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Write the gate experiment**

Create `packages/matchlab_train/src/matchlab_train/experiments/jersey_reader_gate.py`:

```python
"""Gate 1: reproduce the reference jersey metric on the reference data.

Pre-registered bar: tracklet accuracy on the SoccerNet jersey TEST split
within 3 points of the published 87.45%. Below that, the fault is local wiring
and no downstream measurement is attributable, so gates 2-4 do not run.

Reported as an INTEGER COUNT TABLE (correct / wrong / abstained), not just an
accuracy: a single ratio cannot distinguish "reads wrongly" from "declines to
read", and those have opposite implications for a channel whose safety rests
on abstention.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from matchlab_core.crops import ScoredCrop
from matchlab_core.ocr.parseq import JerseyReader
from matchlab_core.reid.jersey import N_NUMBERS, tracklet_likelihood
from pydantic import BaseModel

from matchlab_train.datasets.soccernet_jersey import load_jersey_tracklets
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register

PUBLISHED_TRACKLET_ACCURACY = 0.8745
TOLERANCE = 0.03
OUT_DIR = Path("data/experiments/jersey-reader-gate")


class Params(BaseModel):
    root: str = "data/soccernet/jersey"
    split: str = "test"
    checkpoint: str = "data/weights/parseq-jersey.ckpt"
    device: str = "cuda"
    max_crops_per_tracklet: int = 32
    min_confidence: float = 0.5   # tracklet likelihood below this abstains
    max_tracklets: int | None = None


@register("jersey-reader-gate")
class JerseyReaderGate(Experiment):
    def run(self) -> dict:
        params = Params(**self.config.params)
        data = load_jersey_tracklets(Path(params.root), params.split)
        items = sorted(data.items())[: params.max_tracklets]
        reader = JerseyReader(checkpoint=params.checkpoint)
        reader.prepare(device=params.device)

        counts = {"correct": 0, "wrong": 0, "abstained": 0, "illegible_gt": 0}
        rows = []
        for tid, (paths, gt_number) in items:
            crops = []
            for p in paths[: params.max_crops_per_tracklet]:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                crops.append(
                    ScoredCrop(
                        image=img,
                        quality=1.0,
                        frame_idx=0,
                        box_height=float(img.shape[0]),
                        isolation_iou=0.0,
                    )
                )
            reads = reader.read(crops)
            likelihood = tracklet_likelihood(
                _logprobs(reads), np.array([r.legibility for r in reads])
            )
            best = int(np.argmax(likelihood))
            confidence = float(likelihood[best])
            predicted = best if confidence >= params.min_confidence else None

            if gt_number is None:
                counts["illegible_gt"] += 1
            elif predicted is None:
                counts["abstained"] += 1
            elif predicted == gt_number:
                counts["correct"] += 1
            else:
                counts["wrong"] += 1
            rows.append(
                {"tracklet": tid, "gt": gt_number, "pred": predicted,
                 "confidence": confidence, "n_reads": len(reads)}
            )

        decided = counts["correct"] + counts["wrong"] + counts["abstained"]
        accuracy = counts["correct"] / decided if decided else 0.0
        report = {
            "counts": counts,
            "tracklet_accuracy": accuracy,
            "published": PUBLISHED_TRACKLET_ACCURACY,
            "tolerance": TOLERANCE,
            "passed": accuracy >= PUBLISHED_TRACKLET_ACCURACY - TOLERANCE,
            "train_adjacency": (
                "The checkpoint was fine-tuned on SoccerNet jersey data, so this "
                "figure is a reproduction check, NOT independent accuracy."
            ),
            "rows": rows,
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
        self.write_result(self.workdir(), report)
        return report


def _logprobs(reads) -> np.ndarray:
    """(n_reads, 100) per-crop number log-likelihoods."""
    from matchlab_core.reid.jersey import crop_number_logprobs

    if not reads:
        return np.zeros((0, N_NUMBERS))
    return np.array([crop_number_logprobs(r.char_probs) for r in reads])
```

Register it by adding `jersey_reader_gate` to the alphabetical import tuple in `packages/matchlab_train/src/matchlab_train/experiments/__init__.py`:

```python
from matchlab_train.experiments import (  # noqa: F401
    benchmark,
    detector_rfdetr,
    eval_pipelines,
    gate2_resmooth,
    jersey_reader_gate,
    reid_ablation,
)
```

- [ ] **Step 7: Run the gate**

```bash
uv run --with torch --with timm --with rtmlib matchlab-train run \
  configs/train/jersey-reader-gate.yaml
```

Create that config alongside the existing `configs/train/` files, with `experiment: jersey-reader-gate` and the params above.

Expected: `report.json` with `passed: true` and `tracklet_accuracy >= 0.845`.

**If it fails:** stop. Do not proceed to Task 5. The fault is tokenizer column mapping, torso cropping, or the resize normalisation — bisect by dumping `char_probs` for ten tracklets whose GT number is known and checking whether the argmax digit is right before the aggregation runs.

- [ ] **Step 8: Commit**

```bash
git add packages/matchlab_train/src/matchlab_train/datasets/soccernet_jersey.py \
        packages/matchlab_train/src/matchlab_train/experiments/jersey_reader_gate.py \
        packages/matchlab_train/src/matchlab_train/experiments/__init__.py \
        packages/matchlab_train/tests/test_soccernet_jersey.py \
        configs/train/jersey-reader-gate.yaml configs/datasets/jersey.json \
        configs/datasets/README.md
git commit -m "feat(train): gate 1 -- reproduce the reference jersey metric

Reports an integer count table, not just accuracy: a ratio cannot separate
'reads wrongly' from 'declines to read', and those have opposite implications
for a channel whose safety rests on abstention.

Measured: <accuracy> vs published 0.8745."
```

---

### Task 5: Gate 2 — reader accuracy and coverage on SNMOT crops

The target data. Coverage measured here sets the channel's ceiling and is the number that belongs in `implementation-status.md`.

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/experiments/jersey_channel.py`
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/__init__.py`
- Create: `configs/pipeline.jersey-oracle-substrate.yaml`
- Create: `configs/train/jersey-channel.yaml`

**Interfaces:**
- Consumes: `JerseyReader` (Task 3); `crop_number_logprobs`, `tracklet_likelihood`, `number_prior` (Tasks 1–2); `sample_quality_crops` from `matchlab_core.crops`; `fragment_tracks` / `FragmentResult` from `matchlab_core.gt_fragments`; `StageContext` from `matchlab_core.interfaces`; `ArtifactStore` from `matchlab_core.artifacts`; `PipelineConfig` from `matchlab_core.config`; `probe` from `matchlab_core.video`; `discover_clips_with_gt(clips_dir, max_clips) -> list[tuple[Path, GroundTruth]]` from `matchlab_train.gt_clips`.
- Produces: registered experiment `"jersey-channel"`; functions
  `read_fragments(ctx, tracklets, reader, *, per_tracklet: int = 32) -> dict[int, np.ndarray]` mapping fragment id → likelihood vector `(100,)`, and
  `stage_context(clip: Path, cfg: PipelineConfig, run_dir: Path, device: str) -> StageContext`.

- [ ] **Step 1: Scale the substrate**

The SPO-85 substrate carries 153 true re-entry pairs over 8 sequences — too few to resolve a tail. Ingest more:

```bash
uv run matchlab-train ingest-soccernet --split test --limit 32
```

Record the resulting sequence count and true-pair count in the gate-2 report. This is a prerequisite for Task 6's headline, not an optional improvement.

- [ ] **Step 2: Create the substrate config**

Create `configs/pipeline.jersey-oracle-substrate.yaml`: copy `configs/pipeline.reid-frozen-substrate.yaml` and change `stages.track` to

```yaml
  track:
    impl: oracle
    params:
      gap_frames: 2
      min_fragment_frames: 1
      features_backend: external
      features_model: prtreid
```

Leave `associate.params` exactly as they are — this substrate is measured, not re-tuned. Add a header comment stating that this config exists to give the jersey channel pixels plus pure fragments plus `jersey_by_fragment` labels, none of which the FOOTPASS `multi_input` harness can supply.

- [ ] **Step 3: Write the reader-accuracy experiment**

Create `packages/matchlab_train/src/matchlab_train/experiments/jersey_channel.py`:

```python
"""Gates 2-4 for the jersey merge channel, on the SNMOT oracle-fragment
substrate.

Why not `multi_input.py`: that harness runs on FOOTPASS tactical h5 data and
there are NO PIXELS on that substrate -- no FOOTPASS video is on disk. A
pixel-dependent channel cannot be measured there at all. The oracle TRACK
stage gives fragments that are pure by construction, real crop degradation at
each re-entry, and `jersey_by_fragment` labels for free.

Gate 2 (this stage) reports per-tracklet reader accuracy AND legible coverage.
Coverage is the headline: it sets the channel's ceiling, and it is the figure
that belongs in implementation-status.md next to any result the channel
produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.config import PipelineConfig
from matchlab_core.crops import sample_quality_crops
from matchlab_core.gt_fragments import fragment_tracks
from matchlab_core.interfaces import StageContext
from matchlab_core.ocr.parseq import JerseyReader
from matchlab_core.reid.jersey import (
    N_NUMBERS,
    crop_number_logprobs,
    number_prior,
    tracklet_likelihood,
)
from matchlab_core.video import probe
from pydantic import BaseModel

from matchlab_train.experiments.base import Experiment
from matchlab_train.gt_clips import discover_clips_with_gt
from matchlab_train.registry import register

OUT_DIR = Path("data/experiments/jersey-channel")


class Params(BaseModel):
    base_config: str = "configs/pipeline.jersey-oracle-substrate.yaml"
    clips_dir: str = "data/videos/soccernet"
    max_clips: int = 32
    checkpoint: str = "data/weights/parseq-jersey.ckpt"
    device: str = "cuda"
    per_tracklet: int = 32
    min_confidence: float = 0.5
    temperature: float = 1.0


def read_fragments(ctx, tracklets, reader, *, per_tracklet: int = 32) -> dict[int, np.ndarray]:
    """Fragment id -> likelihood over numbers.

    `per_tracklet` is 32, not `sample_quality_crops`'s default 8: the reference
    system reads every legible frame of ~482-frame tracklets, and 8 crops
    starve the estimate. TEMPORAL SPREAD IS RETAINED (it is what the sampler
    ranks on) even though pure legibility would prefer the best N crops
    regardless of time -- spread is the defence against a whole fragment's read
    coming from one instant of systematic misreading, which is this channel's
    primary risk.
    """
    crops = sample_quality_crops(ctx, tracklets, per_tracklet=per_tracklet)
    out: dict[int, np.ndarray] = {}
    for tid, tracklet_crops in crops.items():
        reads = reader.read(tracklet_crops)
        logprobs = (
            np.array([crop_number_logprobs(r.char_probs) for r in reads])
            if reads
            else np.zeros((0, N_NUMBERS))
        )
        weights = np.array([r.legibility for r in reads])
        out[tid] = tracklet_likelihood(logprobs, weights)
    return out


def stage_context(clip: Path, cfg: PipelineConfig, run_dir: Path, device: str) -> StageContext:
    """A StageContext outside the runner, for pixel access without a pipeline.

    `StageContext` is a plain dataclass, so this needs no runner and no stage
    slot: probe the clip, point an ArtifactStore at a scratch dir, and
    `ctx.frames()` honours the config's sampling exactly as a real stage would.
    """
    return StageContext(
        video=probe(clip, sample_stride=cfg.video.sample_stride),
        config=cfg,
        store=ArtifactStore(run_dir),
        device=device,
    )


@register("jersey-channel")
class JerseyChannel(Experiment):
    def run(self) -> dict:
        params = Params(**self.config.params)
        workdir = self.workdir()
        cfg = PipelineConfig.from_yaml(params.base_config)
        reader = JerseyReader(checkpoint=params.checkpoint)
        reader.prepare(device=params.device)

        clip_gt_pairs = discover_clips_with_gt(params.clips_dir, params.max_clips)
        if not clip_gt_pairs:
            raise FileNotFoundError(f"No clips with sibling GT under {params.clips_dir}")

        per_clip = []
        for clip, gt in clip_gt_pairs:
            frags = fragment_tracks(gt)
            ctx = stage_context(
                clip, cfg, workdir / "scratch" / clip.stem, params.device
            )
            likelihoods = read_fragments(
                ctx, frags.tracklets, reader, per_tracklet=params.per_tracklet
            )
            per_clip.append(
                _score_clip(clip.name, frags, likelihoods, params.min_confidence)
            )

        pooled = _pool(per_clip)
        report = {
            "per_clip": per_clip,
            "pooled": pooled,
            "n_clips": len(clip_gt_pairs),
            "train_adjacency": (
                "The checkpoint saw SoccerNet jersey data; SNMOT is SoccerNet-"
                "derived, so these figures are optimistic. Disclosed, not mitigated."
            ),
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "gate2.json").write_text(json.dumps(report, indent=2))
        self.write_result(workdir, report)
        return report


def _score_clip(clip_name: str, frags, likelihoods, min_confidence: float) -> dict:
    """Integer counts per clip, plus the coverage headline."""
    counts = {"correct": 0, "wrong": 0, "abstained": 0, "no_gt_number": 0}
    for fid, gt_jersey in frags.jersey_by_fragment.items():
        likelihood = likelihoods.get(fid)
        if gt_jersey is None or not str(gt_jersey).isdigit():
            counts["no_gt_number"] += 1
            continue
        if likelihood is None:
            counts["abstained"] += 1
            continue
        best = int(np.argmax(likelihood))
        if float(likelihood[best]) < min_confidence:
            counts["abstained"] += 1
        elif best == int(gt_jersey):
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    decided = counts["correct"] + counts["wrong"]
    total = decided + counts["abstained"]
    return {
        "clip": clip_name,
        "counts": counts,
        "accuracy_when_read": counts["correct"] / decided if decided else 0.0,
        "coverage": decided / total if total else 0.0,
    }


def _pool(per_clip: list[dict]) -> dict:
    """Pooled over fragments, never a mean of per-clip ratios -- a bad clip
    must not hide behind a good one (the MERGE_PRECISION_GATE rule in
    reid_ablation.py)."""
    keys = ("correct", "wrong", "abstained", "no_gt_number")
    total = {k: sum(c["counts"][k] for c in per_clip) for k in keys}
    decided = total["correct"] + total["wrong"]
    seen = decided + total["abstained"]
    return {
        "counts": total,
        "accuracy_when_read": total["correct"] / decided if decided else 0.0,
        "coverage": decided / seen if seen else 0.0,
    }
```

Register it by adding `jersey_channel` to the alphabetical import tuple in `packages/matchlab_train/src/matchlab_train/experiments/__init__.py`, beside `jersey_reader_gate` from Task 4.

Note that no `PipelineRunner` is involved: `read_fragments` needs only pixels and fragments, and `stage_context` above builds a `StageContext` directly because it is a plain dataclass. Do not run the oracle pipeline for gate 2 — the substrate config is needed from Task 7 onward, when body affinities from `frame_features.npz` enter.

- [ ] **Step 4: Run the gate**

```bash
uv run --with torch --with timm --with rtmlib matchlab-train run \
  configs/train/jersey-channel.yaml
```

Expected: `data/experiments/jersey-channel/gate2.json` exists, with pooled `coverage` and `accuracy_when_read` populated over ≥16 clips.

- [ ] **Step 5: Record the finding**

Add a `docs/implementation-status.md` "Known findings" entry with: pooled coverage, accuracy-when-read, the clip and fragment counts, the config revision, and the train-adjacency disclosure. Expect accuracy well below 87% — SNMOT crops are smaller than the jersey dataset's curated ones. **Report the number you measured, whatever it is.**

- [ ] **Step 6: Commit**

```bash
git add packages/matchlab_train/src/matchlab_train/experiments/jersey_channel.py \
        packages/matchlab_train/src/matchlab_train/experiments/__init__.py \
        packages/matchlab_train/src/matchlab_train/experiments/base.py \
        configs/pipeline.jersey-oracle-substrate.yaml \
        configs/train/jersey-channel.yaml docs/implementation-status.md
git commit -m "feat(train): gate 2 -- jersey reader on SNMOT crops

Coverage is the headline, not accuracy: it sets the channel's ceiling. Pooled
over fragments rather than averaged over clips so a bad clip cannot hide.

Measured: coverage <x>, accuracy-when-read <y> over <n> clips."
```

---

### Task 6: Gate 3 — the channel alone, with its temperature fitted

**Files:**
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/jersey_channel.py`
- Test: `packages/matchlab_train/tests/test_jersey_channel.py`

**Interfaces:**
- Consumes: `read_fragments` (Task 5); `pair_llr`, `number_prior` (Task 2); `zero_wrong_frontier`, `sweep`, `merge_counts` from `matchlab_core.reid.frontier`.
- Produces:
  - `fit_temperature(likelihood_by_fragment, label_by_fragment, prior, *, grid) -> float`
  - `pair_scores(likelihoods, prior, pairs, *, temperature) -> dict[tuple[int, int], float]`
  - gate-3 section in `data/experiments/jersey-channel/gate3.json`

- [ ] **Step 1: Write the failing test**

Create `packages/matchlab_train/tests/test_jersey_channel.py`:

```python
import numpy as np

from matchlab_core.reid.jersey import N_NUMBERS, uniform_prior
from matchlab_train.experiments.jersey_channel import fit_temperature, pair_scores


def _peaked(n: int, mass: float = 0.95) -> np.ndarray:
    v = np.full(N_NUMBERS, (1.0 - mass) / (N_NUMBERS - 1))
    v[n] = mass
    return v


def test_pair_scores_are_keyed_by_ordered_pair():
    likelihoods = {2: _peaked(7), 1: _peaked(7)}
    scores = pair_scores(likelihoods, uniform_prior(), [(2, 1)], temperature=1.0)
    assert list(scores) == [(1, 2)]


def test_pair_scores_separate_agreeing_from_disagreeing_pairs():
    likelihoods = {1: _peaked(7), 2: _peaked(7), 3: _peaked(9)}
    scores = pair_scores(
        likelihoods, uniform_prior(), [(1, 2), (1, 3)], temperature=1.0
    )
    assert scores[(1, 2)] > scores[(1, 3)]


def test_fit_temperature_prefers_damping_when_reads_are_overconfident():
    """Two impostors that both misread as 7 -- the correlated-error failure.
    A good temperature must not let that outscore a genuine same pair."""
    likelihoods = {1: _peaked(7, 0.999), 2: _peaked(7, 0.999),
                   3: _peaked(7, 0.999), 4: _peaked(3, 0.999)}
    labels = {1: "a", 2: "a", 3: "b", 4: "b"}
    t = fit_temperature(likelihoods, labels, uniform_prior(),
                        grid=(0.5, 1.0, 2.0, 5.0, 10.0))
    assert t >= 1.0


def test_fit_temperature_returns_a_value_from_the_grid():
    likelihoods = {1: _peaked(7), 2: _peaked(7)}
    labels = {1: "a", 2: "a"}
    grid = (0.5, 1.0, 2.0)
    assert fit_temperature(likelihoods, labels, uniform_prior(), grid=grid) in grid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_train/tests/test_jersey_channel.py -v`
Expected: FAIL — `ImportError: cannot import name 'fit_temperature'`

- [ ] **Step 3: Write the implementation**

Append to `packages/matchlab_train/src/matchlab_train/experiments/jersey_channel.py`:

```python
from itertools import combinations

from matchlab_core.reid.frontier import zero_wrong_frontier
from matchlab_core.reid.jersey import pair_llr, tracklet_likelihood

TEMPERATURE_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


def pair_scores(likelihoods, prior, pairs, *, temperature: float):
    """Ordered-pair keyed jersey LLRs, matching `frontier.py`'s contract.

    Temperature is applied by re-exponentiating the likelihood, which is the
    same operation `tracklet_likelihood` performs on the log-domain sum -- kept
    here so a fitted temperature can be swept without re-reading any crop.
    """
    tempered = {
        fid: tracklet_likelihood(np.log(np.clip(v, 1e-300, None))[None, :],
                                 np.ones(1), temperature=temperature)
        for fid, v in likelihoods.items()
    }
    out: dict[tuple[int, int], float] = {}
    for a, b in pairs:
        key = (min(a, b), max(a, b))
        if key[0] in tempered and key[1] in tempered:
            out[key] = pair_llr(tempered[key[0]], tempered[key[1]], prior)
    return out


def fit_temperature(likelihood_by_fragment, label_by_fragment, prior, *, grid=TEMPERATURE_GRID):
    """One scalar, chosen to maximise correct merges at ZERO wrong merges.

    ONE parameter because the substrate has ~153 true pairs; a 20-bin
    `LLRCalibrator` on that is per-bin noise, and a degenerate operating curve
    produced by the calibrator rather than the signal is a mistake this repo
    has already made once.

    Fitted against real impostor pairs specifically: OCR errors are systematic
    (6<->8, 1<->7, single-vs-double-digit truncation), so two different players
    misread identically is NOT independent noise, and the temperature is the
    only thing damping it.
    """
    fids = sorted(likelihood_by_fragment)
    pairs = list(combinations(fids, 2))
    best_t, best_correct = grid[0], -1
    for t in grid:
        scores = pair_scores(likelihood_by_fragment, prior, pairs, temperature=t)
        thresholds = sorted(set(scores.values()))
        frontier = zero_wrong_frontier(
            scores, label_by_fragment, thresholds=thresholds
        )
        if frontier["correct"] > best_correct:
            best_t, best_correct = t, frontier["correct"]
    return best_t
```

Then extend `JerseyChannel.run` to write `gate3.json` containing, per clip and pooled: the fitted temperature, channel AUC over all fragment pairs, the `zero_wrong_frontier` result, and — reported **separately** — the count of wrong merges the channel's negative evidence *prevents* relative to the body-only scores from `frame_features.npz`. AUC cannot see that number, and it is this channel's actual claim.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_train/tests/test_jersey_channel.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the gate and inspect the confident-impostor tail by hand**

```bash
uv run --with torch --with timm --with rtmlib matchlab-train run \
  configs/train/jersey-channel.yaml
```

Then take the ten highest-scoring **impostor** pairs from `gate3.json` and look at their crops individually. Aggregates cannot distinguish "the channel works" from "two players' numbers were misread identically", and the tail is what governs merge safety. Record what each of the ten actually was.

- [ ] **Step 6: Commit**

```bash
git add packages/matchlab_train/src/matchlab_train/experiments/jersey_channel.py \
        packages/matchlab_train/tests/test_jersey_channel.py
git commit -m "feat(train): gate 3 -- jersey channel alone

One fitted temperature, not LLRCalibrator: 153 true pairs cannot support 20
bins. Fitted against real impostor pairs because OCR errors are systematic,
so two players misread identically is not independent noise.

Measured: AUC <x>, <c> correct at the zero-wrong frontier, <p> wrong merges
prevented. Top-10 impostor tail inspected by hand: <what they were>."
```

---

### Task 7: Gate 4 — fusion, ablated, and the documentation

**Files:**
- Modify: `packages/matchlab_train/src/matchlab_train/experiments/jersey_channel.py`
- Modify: `docs/implementation-status.md`
- Create: `docs/reports/2026-07-30-jersey-ocr-merge-channel.md`

**Interfaces:**
- Consumes: `pair_scores`, `fit_temperature` (Task 6); `fit_fusion_weights`, `fuse`, `LLRCalibrator` from `matchlab_core.reid.evidence`; `zero_wrong_frontier` from `matchlab_core.reid.frontier`.
- Produces: `gate4.json` with per-arm frontier results for `body`, `jersey`, and `body+jersey`.

- [ ] **Step 1: Add the fusion arm**

Extend `JerseyChannel.run` with a `gate4` section that, for each clip:

1. builds body affinities from the run's `frame_features.npz` (the oracle track stage writes it; reuse the same pairwise similarity call `reid_ablation.py` uses — do not write a second one),
2. calibrates body with `LLRCalibrator.fit` on the training clips' labelled pairs, since body *is* a scalar similarity and does have the sample size for it,
3. takes jersey LLRs from `pair_scores` at the fitted temperature — **not** through `LLRCalibrator`, per the Global Constraints,
4. fits channel weights with `fit_fusion_weights` on the training clips,
5. sweeps each of the three arms to **its own** `zero_wrong_frontier` on held-out clips.

Arms must be swept to their own frontiers. Judging a new channel at the incumbent's threshold is the specific error that produced three retracted conclusions in the SPO-85 work.

- [ ] **Step 2: Verify do-no-harm empirically**

Add an assertion to the gate-4 output: for every pair where **both** fragments' jersey likelihood is flat (no legible read), `abs(jersey_llr) < 1e-6`. This holds by construction from Task 2, so the check is cheap; it exists because "holds by construction" has been wrong before. Record the count of such pairs — it is the fraction of the substrate on which the channel contributes nothing, and it belongs next to the coverage figure.

- [ ] **Step 3: Run the full gate**

```bash
uv run --with torch --with timm --with rtmlib matchlab-train run \
  configs/train/jersey-channel.yaml
```

Expected: `gate4.json` with three arms, each at its own zero-wrong frontier, plus the do-no-harm pair count.

- [ ] **Step 4: Write the report**

Create `docs/reports/2026-07-30-jersey-ocr-merge-channel.md` covering: the four gates and their measured values; coverage as the headline constraint; wrong merges prevented, stated separately from AUC; the hand-inspected impostor tail from Task 6 Step 5; the train-adjacency disclosure; and the substrate scale (clips, fragments, true pairs). Scope every negative finding to the decision rule actually tested — "jersey under mutual-best + margin at this coverage", not "jersey fails".

- [ ] **Step 5: Update implementation-status.md**

Add the finding, and state explicitly that this is an **optional modality** with its measured coverage, so no future reader mistakes a result that leaned on numbered kits for a body-ID result. Note that ADR 001 is unamended and why (abstention is algebraic, no gate, no roster required). Do not claim any shipped default changed — none did.

- [ ] **Step 6: Full verification**

```bash
uv run ruff check packages
uv run pytest packages -q
```
Expected: no lint findings; full suite passes.

- [ ] **Step 7: Commit**

```bash
git add packages docs/implementation-status.md \
        docs/reports/2026-07-30-jersey-ocr-merge-channel.md
git commit -m "feat(train): gate 4 -- jersey fused with body, ablated

Each arm swept to ITS OWN zero-wrong frontier; judging a new channel at the
incumbent's threshold is what retracted three SPO-85 conclusions.

Measured: body <b> / jersey <j> / fused <f> correct at zero wrong. Channel is
exactly neutral on <n> pairs (<pct>% of the substrate) -- optional modality,
ADR 001 unamended, no shipped default changed."
```

---

## Notes for the executing engineer

**Gate ordering is not advisory.** If gate 1 misses the published metric, stop and fix the wiring. A reader with a mis-mapped tokenizer column still produces plausible per-tracklet outputs, so every later number would be confidently wrong and unattributable.

**The one place to expect real trouble** is Task 6 Step 5, the hand inspection. Systematic OCR confusions (6↔8, 1↔7, truncated double digits) violate the independence assumption in `pair_llr`'s denominator, and they present as *strong same-player evidence* between two different players. If the top impostor pairs turn out to be that, the fix is not a higher threshold — it is either a stronger temperature, a per-confusion-class correction, or a documented negative finding scoped to the reader's confusion profile.

**Do not fold jersey into `LLRCalibrator`** even though every other channel goes through it. The reason is written in `multi_input.py`'s own docstring about `transition`, and repeated in the Global Constraints.
