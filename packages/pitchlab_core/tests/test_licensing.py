"""Per-axis licensing certification gate tests (SPO-41).

The gate answers one question for the shippable-tracker program: is every
component on the shipping path permissive on *all three* provenance axes
(code, weights, training data)? A single non-permissive axis -- or an
unverifiable ("unknown") one -- must refuse certification loudly, naming the
offending stage/model/axis/value. The existing evaluation-set and
embedder-provenance gates are the precedent (fail-closed, name the mismatch).
"""

from __future__ import annotations

import pytest
from pitchlab_core.licensing import (
    AxisVerdict,
    LicenseCertificationError,
    assert_stack_shippable,
    certify_license_axes,
    certify_stack,
    classify_axis,
)
from pitchlab_core.provenance import (
    LicenseAxes,
    ModelProvenance,
    RunProvenance,
    StageProvenance,
)

# --- classify_axis: permissive tokens -----------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "Apache-2.0",
        "apache 2.0 (YOLOX code)",
        "MIT",
        "MIT License (TDLP)",
        "BSD-3-Clause",
        "ISC",
        "CC0-1.0",
        "public domain",
        "The Unlicense",
        "CC BY 4.0",
        "synthetic (RandPerson) -- no real-person data",
        "owned footage",
        "permissive",
    ],
)
def test_classify_axis_permissive(value):
    assert classify_axis(value) == AxisVerdict.PERMISSIVE


# --- classify_axis: non-permissive markers win --------------------------


@pytest.mark.parametrize(
    "value",
    [
        "AGPL-3.0 (ultralytics, local-eval only, non-shippable)",
        "GPL-3.0",
        "LGPL-2.1",
        "CC BY-NC 4.0 (SportsMOT) -- selection-only, non-shippable",
        "non-commercial",
        "noncommercial research use",
        "research-only (Market-1501)",
        "proprietary",
        # A permissive token is present but a non-permissive marker must win:
        "Apache-2.0 code, weights trained on non-commercial SportsMOT",
        "MIT code but non-shippable checkpoint",
    ],
)
def test_classify_axis_non_permissive_marker_wins(value):
    assert classify_axis(value) == AxisVerdict.NON_PERMISSIVE


# --- classify_axis: unverifiable is fail-closed -------------------------


@pytest.mark.parametrize("value", ["unknown", "", "   ", "TBD", "see upstream repo"])
def test_classify_axis_unknown_is_unknown(value):
    assert classify_axis(value) == AxisVerdict.UNKNOWN


# --- certify_license_axes: all three axes must pass ---------------------


def test_certify_license_axes_all_permissive_passes():
    axes = LicenseAxes(
        code="Apache-2.0", weights="Apache-2.0", training_data="synthetic (RandPerson)"
    )
    cert = certify_license_axes(axes)
    assert cert.passed is True
    assert {a.axis for a in cert.axes} == {"code", "weights", "training_data"}
    assert all(a.verdict == AxisVerdict.PERMISSIVE for a in cert.axes)


def test_certify_license_axes_one_non_permissive_axis_fails():
    axes = LicenseAxes(
        code="Apache-2.0",
        weights="Apache-2.0",
        training_data="CC BY-NC 4.0 (SportsMOT)",
    )
    cert = certify_license_axes(axes)
    assert cert.passed is False
    bad = [a for a in cert.axes if a.verdict != AxisVerdict.PERMISSIVE]
    assert len(bad) == 1
    assert bad[0].axis == "training_data"


def test_certify_license_axes_unknown_axis_fails_closed():
    axes = LicenseAxes(code="Apache-2.0", weights="unknown", training_data="Apache-2.0")
    cert = certify_license_axes(axes)
    assert cert.passed is False
    assert any(a.axis == "weights" and a.verdict == AxisVerdict.UNKNOWN for a in cert.axes)


# --- certify_stack: walk every stage/model on the shipping path ---------


def _clean_shipping_prov() -> RunProvenance:
    """A stack where every declared model is permissive on all axes."""
    return RunProvenance(
        git_revision="abc123",
        stages={
            "detect": StageProvenance(
                impl="rfdetr-local",
                models=[
                    ModelProvenance(
                        architecture="rf-detr-base",
                        license=LicenseAxes(
                            code="Apache-2.0",
                            weights="Apache-2.0",
                            training_data="COCO (permissive commercial use)",
                        ),
                    )
                ],
            ),
            "track": StageProvenance(
                impl="multi-cue-tdlp",
                models=[
                    ModelProvenance(
                        architecture="tdlp-head",
                        license=LicenseAxes(
                            code="MIT",
                            weights="MIT (retrained)",
                            training_data="synthetic (MOTSynth-permissive)",
                        ),
                    )
                ],
            ),
        },
    )


def test_certify_stack_clean_passes():
    cert = certify_stack(_clean_shipping_prov())
    assert cert.passed is True
    assert cert.findings == []


def test_certify_stack_flags_offending_model_axis():
    prov = _clean_shipping_prov()
    # Poison one axis of one model.
    prov.stages["track"].models[0].license.weights = "CC BY-NC 4.0 (SportsMOT)"
    cert = certify_stack(prov)
    assert cert.passed is False
    assert len(cert.findings) == 1
    f = cert.findings[0]
    assert f.stage == "track"
    assert f.axis == "weights"
    assert f.verdict == AxisVerdict.NON_PERMISSIVE
    assert "tdlp-head" in f.model


def test_certify_stack_undeclared_license_fails_closed():
    """A model that never declared its license (all axes default "unknown")
    cannot be certified -- recording the license per axis is mandatory."""
    prov = RunProvenance(
        stages={
            "detect": StageProvenance(
                impl="mystery", models=[ModelProvenance(architecture="mystery-net")]
            )
        }
    )
    cert = certify_stack(prov)
    assert cert.passed is False
    assert {f.axis for f in cert.findings} == {"code", "weights", "training_data"}


def test_certify_stack_stage_without_models_is_noop():
    """A model-free stage (e.g. a pure-motion tracker) contributes nothing to
    certify -- it has no weights/training-data to vet."""
    prov = RunProvenance(
        stages={"track": StageProvenance(impl="iou", models=[])}
    )
    cert = certify_stack(prov)
    assert cert.passed is True
    assert cert.findings == []


# --- assert_stack_shippable: the refusal primitive ----------------------


def test_assert_stack_shippable_passes_silently_on_clean_stack():
    # Must not raise.
    assert_stack_shippable(_clean_shipping_prov(), context="bar-a:in-house")


def test_assert_stack_shippable_refuses_and_names_the_offender():
    prov = _clean_shipping_prov()
    prov.stages["detect"].models[0].license.training_data = "research-only (Objects365?)"
    with pytest.raises(LicenseCertificationError) as exc:
        assert_stack_shippable(prov, context="bar-a:in-house")
    msg = str(exc.value)
    # Names the context, stage, axis, and the offending value for diagnosis.
    assert "bar-a:in-house" in msg
    assert "detect" in msg
    assert "training_data" in msg
    assert "research-only" in msg
