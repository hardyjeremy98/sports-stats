"""External feature backend for the `oracle` tracker (SPO-85): the part-based
re-ID models that live in the isolated CAMELTrack venv (KPR, PRTreID), reached
through the same subprocess bridge `tdlp-full` already uses.

`gen_features.py` takes `--img-dir` + `--det-file` and embeds whatever boxes it
is handed, so GT fragment boxes need no new external machinery -- and because
`join_features_to_tracklets` matches by IoU, a fragment box claims its own
det.txt row at IoU 1.0.

`--no-pose` is passed: pose is a TDLP tracker feature and nothing under
`matchlab_core/reid/` reads keypoints, so skipping it removes most of the
runtime from a retrieval measurement.

A named benchmark arm must never silently run a different model, so a missing
checkpoint raises and names the acquisition step rather than falling back.
"""

from __future__ import annotations

from pathlib import Path

from matchlab_core.frame_features import FrameFeatures
from matchlab_core.interfaces import StageContext
from matchlab_core.schemas import Detection, FrameDetections, Tracklet
from matchlab_core.stages.track.tdlp_full import bridge

# Checkpoint per model, relative to the external-trackers checkout root.
_WEIGHTS: dict[str, str] = {
    "kpr": (
        "CAMELTrack/pretrained_models/reid/"
        "kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"
    ),
    "prtreid": "CAMELTrack/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar",
}

_ACQUIRE: dict[str, str] = {
    "kpr": "acquire it from the CAMELTrack pretrained_models release",
    "prtreid": (
        "download it from the SoccerNet Zenodo release "
        "(https://zenodo.org/records/10653453) together with the hrnet32 backbone "
        "(https://zenodo.org/records/10604211), and install the package into the "
        "CAMELTrack venv with `uv pip install --no-deps "
        "'prtreid @ git+https://github.com/VlSomers/prtreid'` -- --no-deps matters: "
        "prtreid's own pin would replace the KPR fork of torchreid and break the kpr arm"
    ),
}


def _resolve_weights(model: str, external_root: str) -> Path:
    if model not in _WEIGHTS:
        raise ValueError(
            f"Unknown external re-ID model {model!r}. Known: {sorted(_WEIGHTS)}."
        )
    path = Path(external_root) / _WEIGHTS[model]
    if not path.exists():
        raise RuntimeError(
            f"External re-ID weights for {model!r} not found at {path} -- "
            f"{_ACQUIRE[model]}. Refusing to run this arm with a different model."
        )
    return path


def tracklets_to_detections(tracklets: list[Tracklet], fps: float) -> list[FrameDetections]:
    """Fragment boxes as per-frame detections, so the existing sequence-staging
    path can write them to det.txt unchanged."""
    by_frame: dict[int, list[Detection]] = {}
    for t in tracklets:
        for tf in t.frames:
            by_frame.setdefault(tf.frame_idx, []).append(
                Detection(box=tf.box, confidence=tf.confidence, cls=t.cls)
            )
    return [
        FrameDetections(frame_idx=idx, t=idx / fps, detections=dets)
        for idx, dets in sorted(by_frame.items())
    ]


def embed_external(
    ctx: StageContext,
    tracklets: list[Tracklet],
    *,
    model: str,
    external_root: str = "../external-trackers",
    camel_python: str = "CAMELTrack/.venv/bin/python",
    gen_features_script: str = "bridge/gen_features.py",
    batch: int = 16,
    timeout_s: float | None = 7200.0,
    keep_work: bool = False,
) -> FrameFeatures:
    """Stage the run's frames + fragment boxes as a MOT sequence, run the
    external feature generator over them, and join the result back."""
    root = Path(external_root)
    weights = _resolve_weights(model, str(root))
    python = root / camel_python
    script = root / gen_features_script
    for path, what in ((python, "CAMELTrack venv python"), (script, "gen_features.py")):
        if not path.exists():
            raise RuntimeError(
                f"Oracle tracker (external backend): {what} not found at {path}. "
                "Set params.features_backend to 'in-repo', or fix the external-trackers "
                "checkout."
            )

    work = ctx.store.run_dir / "_oracle_work"
    work.mkdir(parents=True, exist_ok=True)
    layout = bridge.stage_sequence(
        ctx.frames(),
        tracklets_to_detections(tracklets, ctx.video.fps),
        work,
        seq_name=Path(ctx.video.path).stem,
        fps=ctx.video.fps,
    )
    feat_dir = work / "feat"
    feat_dir.mkdir(parents=True, exist_ok=True)

    bridge.run_external(
        python,
        script,
        # Absolute paths: run_external runs with cwd=external_root, so a
        # run-dir-relative path would resolve against the wrong tree.
        [
            "--img-dir",
            str(layout.img_dir.resolve()),
            "--det-file",
            str(layout.det_file.resolve()),
            "--out-dir",
            str(feat_dir.resolve()),
            "--weights",
            str(weights.resolve()),
            "--reid-model",
            model,
            "--device",
            ctx.device,
            "--kpr-batch",
            str(batch),
            # Flags last: keeps the arg list parseable pairwise.
            "--no-pose",
        ],
        cwd=root,
        timeout_s=timeout_s,
        label=f"oracle feature-gen ({model})",
    )

    feats = bridge.join_features_to_tracklets(
        tracklets,
        feat_dir,
        layout.local_to_source,
        width=layout.width,
        height=layout.height,
    )
    feats.meta.update({"backend": "external", "model": model, "stage": "oracle"})
    if not keep_work:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    return feats
