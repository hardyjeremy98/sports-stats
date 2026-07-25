"""Camera-calibration subprocess exchange: the pipeline-side seam for external
pitch-registration models run in their own isolated environment, reached only
as a subprocess exchanging JSON. Mirrors matchlab_core.spotting. The permissive
in-repo reference calibrator (`reference_cli`) satisfies the same contract with
no model or GPU, so the bridge and the `pnlcalib` stage run anywhere."""
