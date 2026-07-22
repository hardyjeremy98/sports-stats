"""TDLP-full: the SOTA offline link-prediction tracker (MixSort-YOLOX-class
detections → RTMPose keypoints → KPR 6-part appearance → released
`tdlp_sportsmot` head) wired into pitchlab as a native TRACK stage.

The heavy, separately-licensed stack (tracklab/KPR, motrack, released TDLP
weights) is NOT vendored here and never enters the lab dependency tree. It
lives in the isolated `external-trackers/` sibling checkout, in two dedicated
venvs. `bridge.py` is the ONLY seam to it: it fabricates the MOT-sequence
layout those tools already understand from arbitrary pitchlab frames +
detections, shells out to each venv, and parses the MOT result back into
native `Tracklet`s. The in-repo code imports nothing from TDLP/CAMELTrack.
"""

from pitchlab_core.stages.track.tdlp_full import stage as stage  # noqa: F401
