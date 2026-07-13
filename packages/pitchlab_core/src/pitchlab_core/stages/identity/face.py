"""Face-based identity at tracklet level (locked decision: identity method is
experimental — face-ID, AI upscaling, and jersey OCR all compete behind the
IdentityResolver interface; this is the face candidate).

Per entity: pick the K highest-resolution candidate frames across its
tracklets (biggest boxes ≈ closest to camera), crop the head region, optionally
upscale, embed with insightface/ArcFace. Entity embedding = quality-weighted
mean. Entities whose embeddings are near-identical get the same label — that
both names players consistently and surfaces association misses for QA.

LICENSING: insightface *code* is MIT, but its distributed model packs
(buffalo_l) are research-only. Commercial deployment needs a licensed pack —
gate on that before enabling this stage in production.

Evidence crops are written to the run's crops/ dir so the Lab UI can show WHY
an identity was assigned.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from pitchlab_core.interfaces import IdentityResolver, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import (
    IdentityEvidence,
    IdentityKind,
    PlayerEntity,
    PlayerIdentity,
    Tracklet,
)
from pitchlab_core.schemas.run import StageKind


class Params(BaseModel):
    candidates_per_entity: int = 6
    min_box_height_px: int = 60       # below this a face crop is hopeless
    min_face_score: float = 0.5
    # 'none' | 'realesrgan' — upscaling low-res candidates before embedding.
    upscaler: str = "none"
    upscale_below_px: int = 120
    cluster_cosine_threshold: float = 0.45
    save_crops: bool = True


@register(StageKind.IDENTITY, "face")
class FaceIdentityResolver(IdentityResolver):
    def __init__(self, **params):
        self.params = Params(**params)
        self._app = None
        self._upscaler = None

    def prepare(self, ctx: StageContext) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "The 'face' identity resolver needs insightface "
                "(pip install 'pitchlab-core[face]'). Use identity impl 'none' "
                "to run without it."
            ) from exc
        self._app = FaceAnalysis(name="buffalo_l")
        self._app.prepare(ctx_id=0 if ctx.device == "cuda" else -1, det_size=(320, 320))
        if self.params.upscaler == "realesrgan":
            self._upscaler = _load_realesrgan(ctx.device)

    def resolve(
        self, ctx: StageContext, players: list[PlayerEntity], tracklets: list[Tracklet]
    ) -> list[PlayerEntity]:
        p = self.params
        tr_by_id = {tr.tracklet_id: tr for tr in tracklets}

        # Candidate frames per entity: tallest boxes across all its tracklets.
        wanted: dict[int, list[tuple[int, int]]] = {}  # frame_idx -> [(player_id, tracklet_id)]
        cand_boxes: dict[tuple[int, int], tuple] = {}
        for ent in players:
            frames = [
                (tf.box.height, tf, tid)
                for tid in ent.tracklet_ids
                if (tr := tr_by_id.get(tid))
                for tf in tr.frames
            ]
            frames.sort(key=lambda x: -x[0])
            for h, tf, tid in frames[: p.candidates_per_entity]:
                if h < p.min_box_height_px:
                    continue
                wanted.setdefault(tf.frame_idx, []).append((ent.player_id, tid))
                cand_boxes[(ent.player_id, tf.frame_idx)] = tf.box

        # One pass over the video collecting embeddings.
        embeddings: dict[int, list[tuple[float, np.ndarray]]] = {}
        evidence: dict[int, list[IdentityEvidence]] = {}
        crops_dir = ctx.store.crops_dir() if p.save_crops else None
        for frame in ctx.frames():
            for player_id, tid in wanted.get(frame.frame_idx, []):
                box = cand_boxes.get((player_id, frame.frame_idx))
                if box is None:
                    continue
                crop, upscaled = self._head_crop(frame.image, box)
                if crop is None:
                    continue
                faces = self._app.get(crop)
                if not faces:
                    continue
                best = max(faces, key=lambda f: f.det_score)
                if best.det_score < p.min_face_score:
                    continue
                emb = best.normed_embedding
                embeddings.setdefault(player_id, []).append((float(best.det_score), emb))
                crop_rel = None
                if crops_dir is not None:
                    crop_rel = f"crops/face_p{player_id}_f{frame.frame_idx}.jpg"
                    cv2.imwrite(str(ctx.store.run_dir / crop_rel), crop)
                evidence.setdefault(player_id, []).append(
                    IdentityEvidence(
                        tracklet_id=tid,
                        frame_idx=frame.frame_idx,
                        score=round(float(best.det_score), 4),
                        crop_artifact=crop_rel,
                        upscaled=upscaled,
                    )
                )

        # Quality-weighted mean embedding per entity, then greedy clustering.
        entity_emb: dict[int, np.ndarray] = {}
        for pid, scored in embeddings.items():
            ws = np.array([s for s, _ in scored])
            es = np.stack([e for _, e in scored])
            mean = (es * ws[:, None]).sum(axis=0) / ws.sum()
            entity_emb[pid] = mean / np.linalg.norm(mean)

        labels = _cluster_labels(entity_emb, p.cluster_cosine_threshold)

        out = []
        for ent in players:
            pid = ent.player_id
            if pid in entity_emb:
                scores = [s for s, _ in embeddings[pid]]
                ent = ent.model_copy(
                    update={
                        "identity": PlayerIdentity(
                            kind=IdentityKind.FACE,
                            label=labels[pid],
                            confidence=round(float(np.mean(scores)), 4),
                            evidence=evidence.get(pid, []),
                        )
                    }
                )
            out.append(ent)
        return out

    def _head_crop(self, image: np.ndarray, box) -> tuple[np.ndarray | None, bool]:
        """Upper ~40% of the player box, padded. Optionally super-resolved."""
        h, w = image.shape[:2]
        bw = box.width
        x1 = max(0, int(box.x1 - 0.2 * bw))
        x2 = min(w, int(box.x2 + 0.2 * bw))
        y1 = max(0, int(box.y1 - 0.1 * box.height))
        y2 = min(h, int(box.y1 + 0.4 * box.height))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None, False
        if self._upscaler is not None and crop.shape[0] < self.params.upscale_below_px:
            try:
                crop, _ = self._upscaler.enhance(crop, outscale=2)
                return crop, True
            except Exception:
                return crop, False
        return crop, False


def _cluster_labels(entity_emb: dict[int, np.ndarray], threshold: float) -> dict[int, str]:
    """Greedy cosine clustering; entities in one cluster share a label."""
    labels: dict[int, str] = {}
    reps: list[tuple[str, np.ndarray]] = []
    for pid, emb in sorted(entity_emb.items()):
        assigned = None
        for label, rep in reps:
            if 1.0 - float(np.dot(emb, rep)) < threshold:
                assigned = label
                break
        if assigned is None:
            assigned = f"P{len(reps) + 1}"
            reps.append((assigned, emb))
        labels[pid] = assigned
    return labels


def _load_realesrgan(device: str):
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise RuntimeError(
            "upscaler='realesrgan' needs the realesrgan package "
            "(pip install realesrgan). Set upscaler='none' to skip."
        ) from exc
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    return RealESRGANer(
        scale=2,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        model=model,
        half=device == "cuda",
    )
