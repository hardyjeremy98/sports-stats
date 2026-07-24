# PRD: Pitch Calibration Rebuild (PnLCalib) + Smooth Game-State Reconstruction

**Status:** Draft for decomposition (2026-07-24).
**Tracking:** SPO-60.
**Owner:** Jeremy
**Precedence:** Planning document. Sits below the accepted ADRs and
`../../../docs/player-identity-vision.md`; where this PRD and an ADR disagree, the ADR wins.
**Depends on:** nothing hard. Runs orthogonally to the re-ID program (SPO-51..59); shares
only the common stage/registry/artifact contracts.
**Supersedes:** the reverted PnLCalib integration attempt (commits `aae572e`, `9f3bd4e`,
`b47f50f`, all reverted 2026-07-24). That attempt failed on **implementation quality, not
model choice** — this PRD re-does it with the integration bugs designed out and hard gates
in front of every claim.
**Related:** ADR 004 (measure before you believe — the same discipline applied here to a
geometry task), the external-spotters setup and exchange-contract reference docs (the
isolation pattern this PRD reuses), `docs/implementation-status.md`.

---

## Problem Statement

The Lab's pitch keypoint prediction is effectively non-functional, and the "Game state"
view built on top of it does not work well:

1. **Per-frame keypoint prediction is fragile.** The current default calibrator (a
   YOLOv8-pose 32-keypoint model + per-frame RANSAC homography) jitters frame to frame,
   fails outright on camera pans past featureless stretches of pitch, and is known-fragile
   on non-broadcast footage. Failure handling is "carry the last good homography with decay
   for up to ~4 seconds, then give up" — so the projected world silently drifts, then
   vanishes.

2. **The Game state view flickers and jumps.** Player dots on the 2D pitch blink out
   whenever calibration confidence dips below a hard gate, jump when a carried homography
   snaps back to a fresh estimate, and jitter because two independent, uncoordinated
   exponential smoothers (one on the homography, one on per-player positions) fight each
   other.

3. **The in-repo pitch template is not a real pitch.** It is a 120×70 m template with
   non-physical geometry (scaled penalty boxes but a true-size centre circle). It is not a
   projective image of any real pitch, so mapping *any* geometrically-correct calibration
   onto it warps everything by ~10–14% — overlay keypoints land visibly off the lines, and
   downstream pitch-space quantities (minimap positions, metric speeds) are distorted. This
   is the trap that sank the previous integration attempt.

4. **There is no calibration metric anywhere in the repo.** Evaluation covers detection,
   tracking, identity, and action spotting — but nothing scores homography accuracy,
   reprojection error, or pitch-space positional quality. Calibration is judged entirely by
   eye via two Lab overlays, which is how a broken integration previously survived until
   qualitative review.

## Solution

Rebuild pitch calibration on **PnLCalib** (Gutiérrez-Pérez & Agudo, CVIU 2025) — the
published state of the art for soccer field registration on SoccerNet-Calibration,
WorldCup 2014, and TS-WorldCup, and the successor to *No Bells, Just Whistles* — run in an
isolated sibling environment behind a subprocess exchange contract, exactly like the
existing external-spotters (T-DEED) pattern. Then build a proper **offline global
smoother** that turns per-frame homography estimates into one coherent, gap-free camera
trajectory for the whole clip, and reconstruct the Game state view from that.

Three phases, two hard gates:

- **Phase 0 — physically-correct pitch geometry.** Introduce a real FIFA 105×68 m pitch
  specification alongside the legacy template, selected per config, and make every
  consumer (calibration, fusion, evaluation, the web pitch renderer) draw its geometry
  from the selected spec instead of hardcoded constants. This lands first, as its own
  reviewable unit, so the geometry refactor never again entangles with model integration.

- **Phase 1 — external calibrator, gated standalone (Gate 1, hard).** Stand up PnLCalib in
  an `external-calibrators/` sibling environment behind a validated JSON exchange
  contract, with an in-repo permissive reference CLI for smoke/CI. Before any pipeline
  wiring is trusted, the external CLI must reproduce PnLCalib's published accuracy on the
  official SoccerNet-Calibration test-split evaluation protocol, and produce stable,
  visually-correct per-frame predictions across a panel of ingested clips spanning
  different pans and perspectives. Integration bugs must surface at the exchange boundary,
  where they are attributable — not downstream in a warped minimap.

- **Phase 2 — smooth game-state reconstruction (Gate 2, hard).** Replace the naive
  EMA-and-carry logic with an offline global smoother: collect all per-frame homography
  estimates for the clip, reject outliers, smooth in a stable camera parameterization
  (not naive matrix blending), and interpolate gaps up to a cap using both past and future
  frames — the Lab is offline and the whole clip is available, per the same philosophy as
  offline identity association. Fuse player positions through the smoothed trajectory with
  one coordinated smoothing story. The gate: project **ground-truth player tracks**
  through the smoothed homographies on GT-tracked sequences and pass quantitative
  pitch-space thresholds for jitter, coverage, and teleport-freedom. These metrics are
  folded into the standard run evaluation so game-state quality is tracked permanently,
  not just for this PRD.

Both gates are **hard**: human sign-off on the measured numbers before the phase is called
done.

> **Scope change (2026-07-25, owner):** the amateur-footage qualitative check originally
> planned here (SPO-71) is **removed from this PRD entirely**, consistent with the
> current development scope (CLAUDE.md): development and evaluation run against
> broadcast-style benchmark data, the team owns no phone/amateur footage, and
> phone-footage validation is an explicit later phase. Everything amateur-related below
> is retained only where struck through or marked removed, for record.

## User Stories

1. As a Lab researcher, I want per-frame pitch keypoints that land on the actual pitch
   lines across different clips, pans, and perspectives, so that I can trust the
   homography every downstream stage consumes.
2. As a Lab researcher, I want the imported model validated against the official
   SoccerNet-Calibration evaluation protocol before it enters the pipeline, so that
   integration bugs cannot masquerade as model failure (or vice versa).
3. As a Lab user, I want the Game state view to show every player as a smoothly moving dot
   for the whole clip, so that I can read the game at a glance without flicker or blinking
   gaps.
4. As a Lab user, I want player dots not to teleport when the camera pans or when
   calibration recovers from a weak stretch, so that I never mistake a projection artifact
   for a real movement.
5. As a Lab user, I want the pitch drawn with physically-correct proportions, so that
   positions, distances, and the overlay keypoints correspond to the real world.
6. As a Lab researcher, I want pitch-space positions computed against a real FIFA-geometry
   pitch model, so that metric quantities derived from them (distances, speeds, zones) are
   physically meaningful.
7. As a Lab researcher, I want calibration quality expressed as numbers in the run
   evaluation (jitter, coverage, teleports, reprojection quality), so that regressions are
   caught by the benchmark matrix instead of by eye.
8. As a Lab researcher, I want the game-state gate computed by projecting ground-truth
   player tracks (not our tracker's output) through the smoothed homographies, so that the
   gate isolates calibration + smoothing quality from tracker errors.
9. As a Lab researcher, I want gaps in calibration bridged by interpolation from both past
   and future frames rather than by carrying a decaying stale estimate, so that short
   featureless pans don't produce drift or blackouts.
10. As a Lab researcher, I want outlier per-frame homographies rejected before smoothing,
    so that one bad frame cannot yank the whole projected world.
11. As a Lab researcher, I want a single coordinated smoothing story between calibration
    and player-position fusion, so that two filters no longer fight and every smoothing
    decision has one owner.
12. As a pipeline operator, I want the heavyweight calibrator isolated in a sibling
    environment behind a subprocess contract, so that its torch/CUDA stack and GPL code
    never leak into the repo's dependency tree (dependency hygiene, per the research
    posture — not a shipping boundary).
13. As a pipeline operator, I want an in-repo permissive reference CLI that exercises the
    full exchange contract without a GPU or real weights, so that CI and smoke configs can
    test the seam cheaply.
14. As a pipeline operator, I want contract violations from the external calibrator to
    raise loudly rather than degrade silently, so that a broken external env never produces
    a quietly-empty calibration artifact.
15. As a pipeline operator, I want a config that pairs the oracle detector with the real
    calibrator, so that I can evaluate calibration and game-state quality with tracking
    held at ground truth.
16. As a Lab user, I want the pitch-keypoints overlay to distinguish fresh, smoothed, and
    interpolated calibration per frame, so that I can see where estimates are direct
    versus inferred.
17. As a Lab researcher, I want the smoother usable as a pure library function on recorded
    homography estimates, so that I can iterate on smoothing parameters offline without
    re-running the GPU model.
18. ~~As a Lab researcher, I want a qualitative amateur-footage review recorded as an
    explicit known finding.~~ *(Removed 2026-07-25 — see Scope change note; number kept so
    later story references stay stable.)*
19. As a future developer, I want the pitch geometry (dimensions, keypoint vertices, line
    segments) defined in exactly one spec consumed by core, evaluation, and the web
    renderer, so that a geometry change can never half-apply.
20. As a future developer, I want the exchange contract documented well enough that a
    different calibration model can be swapped in behind the same seam, so that model
    upgrades are config changes plus one adapter, not rewrites.
21. As a benchmark reviewer, I want both gates to end in explicit human sign-off over
    measured numbers on named datasets and splits, so that "calibration works now" is a
    verifiable claim tied to evidence.

## Implementation Decisions

**Model choice.** PnLCalib: HRNetV2-w48 heatmap models predicting pitch keypoints *and*
field lines, followed by a points-and-lines nonlinear refinement. Published SOTA on
SoccerNet-Calibration, WorldCup 2014, and TS-WorldCup; GPL-2.0 code with
SoccerNet-trained weights. The previous attempt's failure is attributed (by the owner) to
integration bugs, not the model; the model survives on merit. The 2023 SoccerNet
challenge-winner (Sportlight) is the named fallback if Gate 1 fails for model-side
reasons.

**Isolation pattern.** Same as external-spotters: the GPL model lives in a sibling
`external-calibrators/` directory with its own pinned environment, never a dependency of
any in-repo package, reached only via a subprocess CLI. Licensing terms are recorded as
provenance facts; they gate nothing (research posture).

**Exchange contract.** The pipeline freezes sampled frames to disk and writes a job
manifest; the external CLI returns one homography record per frame (frame index,
optional 3×3 homography, confidence, point count) in a schema-validated JSON exchange.
The external side returns *raw per-frame estimates only* — all temporal smoothing lives
on the pipeline side, so smoothing behavior is identical for every calibrator. Any
contract violation raises a typed bridge error; the bridge never yields silently-empty
results. A permissive in-repo reference CLI implements the same contract for smoke/CI.

**Coordinate convention (the hard-won lesson).** The external model calibrates against
real pitch geometry; the pipeline-side template it is reconciled with must therefore also
be a real pitch. Phase 0 introduces a physically-correct FIFA 105×68 m pitch spec —
dimensions, canonical keypoint vertices, and line segments — selected via a per-config
`pitch:` seam, with the legacy 120×70 template retained for old configs and synthetic
runs. Every consumer (calibrate stages, minimap fusion, evaluation, the web pitch
renderer) reads geometry from the selected spec; the web renderer receives the pitch
dimensions through run artifacts/manifest rather than hardcoding them. Template vertices
behind the camera's horizon line are culled before any fitting or reprojection (second
lesson from the reverted attempt).

**Modules.** Ten, in three phases:

*Phase 0:* (1) the pitch-spec geometry module (deep: all pitch geometry behind one
rarely-changing interface); (2) pitch-spec-aware web rendering.

*Phase 1:* (3) the calibration bridge (subprocess seam, schema validation, loud
failures); (4) the in-repo permissive reference CLI; (5) the external-calibrators sibling
env + PnLCalib adapter CLI, with a setup doc and exchange-contract doc mirroring the
spotting ones; (6) the Gate 1 harness — runs the external CLI on the
SoccerNet-Calibration test split and computes the official challenge metrics, runnable
before and independently of any pipeline wiring.

*Phase 2:* (7) the offline global smoother (deepest module: a pure function from raw
per-frame homography estimates — with gaps and outliers — to a smoothed, gap-interpolated
homography trajectory; smooths in a decomposed camera parameterization, not naive matrix
blending; no I/O); (8) the PnLCalib calibrate stage (thin orchestration: freeze frames →
bridge → smoother → calibration artifact); (9) minimap-fusion cleanup (one coordinated
smoothing story; remove the second fighting EMA and the hard blank-out gate, replacing
them with policy driven by the smoother's per-frame status); (10) the Gate 2 harness —
projects GT player tracks through the smoothed homographies and computes pitch-space
jitter, coverage, and teleport metrics.

**Calibration artifact.** Keeps the existing per-frame record shape (homography,
keypoints, confidence) extended with an explicit per-frame provenance status
(fresh / smoothed / interpolated / absent) replacing the current boolean, so the overlay
and fusion can distinguish direct estimates from inferred ones. The frontend type mirror
is updated in the same change.

**Evaluation integration.** The Gate 2 metrics become a permanent calibration/game-state
section of the standard run evaluation for GT-tracked videos, with headline numbers
folded into run metrics so the dashboard, diff view, and benchmark matrix track them.
The gate itself is thresholds over those metrics plus sign-off.

**Gate definitions.**

- *Gate 1 (hard):* the external CLI reproduces PnLCalib's published
  SoccerNet-Calibration test-split results under the official evaluation protocol within
  a small stated tolerance, and a clip panel spanning distinct pans/perspectives (drawn
  from ingested sequences) shows stable per-frame predictions — quantified by inlier and
  reprojection statistics, confirmed by human review of the keypoints overlay. Pass is
  recorded with dataset, split, weights version, and numbers before Phase 2 conclusions
  are drawn.
- *Gate 2 (hard):* on GT-tracked sequences (tuning/held-out roles per the dataset-tier
  declarations), GT tracks projected through the smoothed homographies must meet
  provisional thresholds of: valid-homography coverage ≥ 99% of sampled frames per clip;
  no projected GT track exhibiting implausible frame-to-frame speed (provisionally
  > 12 m/s) in more than 1% of its frames; zero teleports (provisionally > 2 m
  displacement at sampling stride) attributable to homography refresh events; and
  projected positions staying within pitch bounds + margin. Thresholds are provisional
  and finalized (with justification) during implementation; the gate closes with a
  before/after review of the Game state view on the same clips.

**~~Non-blocking amateur check~~ — removed.** *(2026-07-25 scope change: no amateur
footage exists and phone-footage validation is a later phase; SPO-71 cancelled. See the
Scope change note at the top.)*

**Operational posture.** Real-model runs (GPU, external env) are human-initiated, mirroring
the external-spotters convention — smoke configs against the reference CLI are the only
unattended path.

## Testing Decisions

A good test exercises **external behavior through a stable interface** — never internal
implementation details. The three module groups chosen for tests:

- **Offline global smoother** (highest value): unit tests with synthetic camera
  trajectories — a smooth pan with injected per-frame noise must come out with bounded
  jitter; injected outlier frames must be rejected, not blended; gaps shorter than the cap
  must be interpolated through both endpoints; gaps longer than the cap must yield
  explicit absence, never a decayed stale estimate; identity/degenerate inputs must not
  crash. Assertions are on projected-point behavior (where known pitch points land), not
  on internal matrix entries.
- **Bridge + reference CLI**: contract tests — manifest round-trip against the reference
  CLI; malformed output, missing frames, and non-invertible homographies raise typed
  errors; the reference CLI's output validates against the exchange schema. No GPU in CI.
  Prior art: the spotting bridge/exchange tests and the (reverted) calibrate-stage test
  suite, which is a legitimate starting point to resurrect.
- **Pitch-spec geometry**: invariants — physically-consistent dimensions and landmark
  positions for the FIFA spec; keypoint/line derivation self-consistency; the legacy
  template still selectable and unchanged; spec selection actually reaches consumers.

The Gate 1 and Gate 2 harnesses are validated by their gate runs (measured numbers on
named datasets) rather than dedicated unit suites, per the owner's test selection; their
metric math lives in evaluation code where existing evaluation-test prior art applies if
coverage is later wanted.

## Out of Scope

- **Amateur footage — entirely.** (Broadened 2026-07-25: originally only the *hard gate*
  was out of scope with a non-blocking check retained; the check itself is now removed
  too. Phone-footage validation is a later development phase per the repo scope note.)
- **Training or fine-tuning any model.** This PRD imports pretrained weights only.
- **Ball tracking / ball game-state improvements.** The ball dot renders as today;
  improving ball detection or smoothing is separate work.
- **Online/streaming calibration.** The smoother is deliberately offline whole-clip;
  a causal variant is out of scope.
- **Standing up the Sportlight alternative.** It is a named fallback only; no work
  happens on it unless Gate 1 fails for model-side reasons.
- **Broader camera-model outputs** (lens distortion fields, full intrinsics artifacts)
  beyond what is needed to produce the per-frame homography record.
- **Changes to detection, tracking, association, or identity stages** beyond the
  minimap-fusion smoothing cleanup.

## Further Notes

- **History:** this is the second attempt. The first (three commits, same day, all
  reverted) produced results the owner describes as bad *due to implementation*, and its
  two intermediate fixes — the non-physical pitch template and behind-horizon keypoint
  pollution — are baked into this PRD as Phase 0 and a contract-level requirement rather
  than discovered mid-integration again. The reverted code is a legitimate reference for
  resurrection where it was sound (bridge, reference CLI, shared smoothing seam,
  stage skeleton), but nothing is trusted until it passes the gates that didn't exist
  last time.
- **Why gates before wiring:** the previous attempt was verified only by eyeballing one
  broadcast clip after full pipeline integration, which made model quality and
  integration bugs indistinguishable. Gate 1's standalone official-protocol evaluation
  exists precisely to separate those two failure classes.
- **Licensing provenance (facts, not gates):** PnLCalib code is GPL-2.0 with
  SoccerNet-trained weights; SoccerNet data is research-licensed. Recorded for provenance
  honesty; the sibling-env isolation is dependency hygiene only.
- **Datasets:** SoccerNet-Calibration (Gate 1) will need downloading under the existing
  SoccerNet data conventions; Gate 2 uses already-ingestable GT-tracked sequences
  (SoccerNet tracking / SportsMOT / SoccerTrack) respecting the checked-in tuning vs
  held-out tier roles.
