"""The B2 re-ID engine's pure modules (PRD: sports-stats#1).

Deep modules behind small interfaces, composed by the `reid-engine` associate
stage: tracklet representation (`representation`), constraint gates (`gates`),
the merge engine (`merge`), and closed-roster naming (`naming`). Everything
here is pure logic over schemas + numpy — no I/O, no stage context — so each
seam is independently testable and later slices swap internals, not shapes.
"""
