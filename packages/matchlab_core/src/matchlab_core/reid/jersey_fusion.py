"""Task 7 (jersey-OCR merge-channel fusion ablation): pure functions comparing
body-only, jersey-only, and fused (body + jersey) merge decisions on the same
868-fragment SNMOT substrate the jersey evidence cache was built from.

Kept separate from `jersey_channel.py` (which drives the reader model itself)
because everything here is post-hoc scoring over two already-computed score
dicts -- no model, no I/O beyond what the caller passes in. Pure: dicts and
arrays in, dicts out, so alignment/fusion-arithmetic properties are testable
without GPU or the cache file.
"""

from __future__ import annotations

import numpy as np

FragKey = tuple[str, int]  # (clip_stem, fragment_id) -- unique across all 32 clips


def verify_fragment_alignment(
    cache_rows: list[dict], npz_meta_by_clip: dict[str, dict]
) -> None:
    """Fail loudly if the evidence cache's (clip, frag) -> gt_track mapping
    disagrees with the oracle stage's own `gt_track_by_fragment` for that
    fragment id. A prior index-keyed misalignment silently scrambled 42% of a
    channel -- this check exists so that failure mode cannot recur silently.
    """
    mismatches: list[str] = []
    for row in cache_rows:
        clip, frag, gt_track = row["clip"], row["frag"], row["gt_track"]
        stem = clip.removesuffix(".mp4")
        meta = npz_meta_by_clip.get(stem)
        if meta is None:
            mismatches.append(f"{clip}: no frame_features for this clip")
            continue
        mapping = meta.get("gt_track_by_fragment", {})
        npz_gt_track = mapping.get(str(frag))
        if npz_gt_track is None:
            mismatches.append(f"{clip} frag {frag}: not present in npz gt_track_by_fragment")
        elif int(npz_gt_track) != int(gt_track):
            mismatches.append(
                f"{clip} frag {frag}: cache gt_track={gt_track} != npz gt_track={npz_gt_track}"
            )
    if mismatches:
        preview = "\n".join(mismatches[:20])
        raise ValueError(
            f"Fragment alignment check FAILED on {len(mismatches)}/{len(cache_rows)} rows "
            f"(cache vs oracle-stage gt_track_by_fragment). First 20:\n{preview}"
        )


def pooled_pairs(
    keys: list[FragKey], labels: dict[FragKey, object]
) -> list[tuple[FragKey, FragKey]]:
    """Every unordered within-clip pair, ordered (min, max) by the same key
    convention `frontier.merge_counts` reads against -- pairs never cross
    clips (identity is only comparable within one clip's roster)."""
    by_clip: dict[str, list[FragKey]] = {}
    for k in keys:
        by_clip.setdefault(k[0], []).append(k)
    out: list[tuple[FragKey, FragKey]] = []
    for ks in by_clip.values():
        ks = sorted(ks, key=lambda k: k[1])
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                out.append((ks[i], ks[j]))
    return out


def fuse_sum(body_llr: dict, jersey_llr: dict) -> dict:
    """Unweighted-sum fusion arm: fused = body + jersey (missing key = 0)."""
    keys = set(body_llr) | set(jersey_llr)
    return {k: body_llr.get(k, 0.0) + jersey_llr.get(k, 0.0) for k in keys}


def fuse_weighted(body_llr: dict, jersey_llr: dict, weights: np.ndarray) -> dict:
    """Fitted-weight fusion arm: fused = w_body * body + w_jersey * jersey."""
    wb, wj = float(weights[0]), float(weights[1] if len(weights) > 1 else 1.0)
    keys = set(body_llr) | set(jersey_llr)
    return {k: wb * body_llr.get(k, 0.0) + wj * jersey_llr.get(k, 0.0) for k in keys}


def assert_do_no_harm(
    body_llr: dict,
    jersey_llr: dict,
    fused_llr: dict,
    *,
    body_scale: float = 1.0,
    tol: float = 1e-9,
) -> int:
    """On every pair where jersey evidence is exactly 0 (abstention on at
    least one side), fused ranking must equal body ranking. For the
    unweighted-sum arm (body_scale=1.0) that means fused == body exactly; for
    the fitted-weight arm it means fused == w_body * body -- a positive scale
    of body preserves body's ranking, which is the property the spec asks
    for, not bit-identical LLR values. Returns the count checked; raises on
    any violation."""
    checked = 0
    for k, j in jersey_llr.items():
        if abs(j) <= tol:
            b = body_llr.get(k, 0.0)
            expected = body_scale * b
            f = fused_llr.get(k, expected)
            if abs(f - expected) > tol:
                raise AssertionError(
                    f"do-no-harm violated at {k}: jersey={j}, body={b}, "
                    f"expected={expected}, fused={f}"
                )
            checked += 1
    return checked


def veto_impact(
    body_llr: dict, fused_llr: dict, labels: dict, *, body_threshold: float
) -> dict:
    """Pairs body-alone would merge at its own best threshold, split into
    those fused still merges (agrees) vs. refuses (a jersey veto) -- and
    whether that veto was correct (the pair was actually wrong) or a false
    veto (the pair was actually correct, and fusion lost it)."""
    would_merge = {k for k, s in body_llr.items() if s >= body_threshold}
    correct_veto = wrong_veto = still_merges = 0
    for k in would_merge:
        same = labels.get(k[0]) == labels.get(k[1])
        fused_would_merge = fused_llr.get(k, body_llr[k]) >= body_threshold
        if fused_would_merge:
            still_merges += 1
        elif same:
            wrong_veto += 1  # false veto: lost a correct merge
        else:
            correct_veto += 1  # true veto: refused a wrong merge
    return {
        "body_merges_at_threshold": len(would_merge),
        "still_merges_when_fused": still_merges,
        "correct_vetoes": correct_veto,
        "false_vetoes": wrong_veto,
    }
