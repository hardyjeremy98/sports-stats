"""Per-axis licensing certification gate (SPO-41).

The shippable-tracker program's non-negotiable: every component on the
shipping path must be permissive on *all three* provenance axes -- code,
weights, and training data (`provenance.LicenseAxes`). This module turns the
free-text license strings the stages already record into a machine verdict and
provides the refusal primitive the Bar A acceptance step uses to certify (or
loudly refuse) an assembled stack.

Design rules, chosen to fail *closed*:

- A **non-permissive marker wins.** If a value contains any known
  non-permissive token (AGPL/GPL, non-commercial / CC BY-NC, research-only,
  "non-shippable", "selection-only", proprietary), the axis is NON_PERMISSIVE
  even if a permissive token is also present -- this is exactly how the repo
  already annotates e.g. "Apache-2.0 (ultralytics, local-eval only,
  non-shippable)".
- Otherwise a recognised **permissive token** (Apache/MIT/BSD/ISC/CC0/public
  domain/Unlicense/CC BY (attribution-only)/synthetic/owned/permissive) →
  PERMISSIVE.
- Anything else -- including the default "unknown", blank, or an unrecognised
  string -- is UNKNOWN, which never certifies. Recording the license per axis
  is therefore mandatory, not optional.

The gate mirrors `provenance.check_evaluation_set`: pure functions, and a
refusal that names the offending stage / model / axis / value so a failure is
diagnosable at a glance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from matchlab_core.provenance import LicenseAxes, RunProvenance

AXES: tuple[str, str, str] = ("code", "weights", "training_data")


class AxisVerdict(str, Enum):
    PERMISSIVE = "permissive"
    NON_PERMISSIVE = "non_permissive"
    UNKNOWN = "unknown"


# Non-permissive markers are matched as substrings against the lowercased,
# whitespace-normalised value. Ordering vs. the permissive set does not matter
# because non-permissive is always checked first (it wins).
_NON_PERMISSIVE_MARKERS: tuple[str, ...] = (
    "agpl",
    "gpl",  # also catches lgpl; copyleft is treated as non-shippable here
    "non-commercial",
    "noncommercial",
    "non commercial",
    "by-nc",  # CC BY-NC / BY-NC-SA / BY-NC-ND
    "cc-by-nc",
    "research-only",
    "research only",
    "non-shippable",
    "selection-only",
    "proprietary",
)

# Permissive tokens matched as word-ish substrings. CC BY (attribution-only)
# is commercial-OK; the BY-NC marker above is checked first so CC BY-NC never
# reaches here.
_PERMISSIVE_TOKENS: tuple[str, ...] = (
    "apache",
    "mit",
    "bsd",
    "isc",
    "cc0",
    "public domain",
    "public-domain",
    "unlicense",
    "cc by",
    "cc-by",
    "synthetic",
    "owned",
    "self-recorded",
    "permissive",
)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def classify_axis(value: str) -> AxisVerdict:
    """Classify one free-text license axis value. Fail-closed: unrecognised or
    "unknown" values are UNKNOWN, and any non-permissive marker wins."""
    text = _normalise(value)
    if not text or text == "unknown":
        return AxisVerdict.UNKNOWN
    if any(marker in text for marker in _NON_PERMISSIVE_MARKERS):
        return AxisVerdict.NON_PERMISSIVE
    if any(token in text for token in _PERMISSIVE_TOKENS):
        return AxisVerdict.PERMISSIVE
    return AxisVerdict.UNKNOWN


@dataclass(frozen=True)
class AxisResult:
    axis: str
    value: str
    verdict: AxisVerdict


@dataclass(frozen=True)
class ComponentCertification:
    passed: bool
    axes: list[AxisResult]


def certify_license_axes(axes: LicenseAxes) -> ComponentCertification:
    """Certify a single component's three axes. Passes iff all three are
    PERMISSIVE."""
    results = [
        AxisResult(axis=axis, value=getattr(axes, axis), verdict=classify_axis(getattr(axes, axis)))
        for axis in AXES
    ]
    passed = all(r.verdict == AxisVerdict.PERMISSIVE for r in results)
    return ComponentCertification(passed=passed, axes=results)


@dataclass(frozen=True)
class LicenseFinding:
    stage: str
    model: str
    axis: str
    value: str
    verdict: AxisVerdict


@dataclass(frozen=True)
class StackCertification:
    passed: bool
    findings: list[LicenseFinding] = field(default_factory=list)


def _model_label(model) -> str:
    """A human identifier for a model in a finding: architecture, and the
    revision when it adds signal."""
    arch = model.architecture or "unknown"
    if model.revision and model.revision != "unknown":
        return f"{arch}@{model.revision}"
    return arch


def certify_stack(prov: RunProvenance) -> StackCertification:
    """Walk every model declared by every stage and collect a finding for each
    axis that is not PERMISSIVE. Passes iff there are no findings. A stage with
    no models contributes nothing (a pure-motion tracker has no weights or
    training data to vet)."""
    findings: list[LicenseFinding] = []
    for stage_name, stage in prov.stages.items():
        for model in stage.models:
            cert = certify_license_axes(model.license)
            for axis_result in cert.axes:
                if axis_result.verdict != AxisVerdict.PERMISSIVE:
                    findings.append(
                        LicenseFinding(
                            stage=stage_name,
                            model=_model_label(model),
                            axis=axis_result.axis,
                            value=axis_result.value,
                            verdict=axis_result.verdict,
                        )
                    )
    return StackCertification(passed=not findings, findings=findings)


class LicenseCertificationError(RuntimeError):
    """Raised by the certification gate when a stack carries any
    non-permissive or unverifiable axis on the shipping path."""


def _format_findings(findings: list[LicenseFinding]) -> str:
    return "; ".join(
        f"{f.stage}/{f.model} {f.axis}={f.value!r} ({f.verdict.value})" for f in findings
    )


def assert_stack_shippable(prov: RunProvenance, context: str) -> None:
    """Refusal primitive: raise `LicenseCertificationError` naming the context
    and every offending stage/model/axis/value if the stack is not fully
    permissive on the shipping path. No-op on a clean stack. Mirrors
    `provenance.check_evaluation_set`."""
    cert = certify_stack(prov)
    if not cert.passed:
        raise LicenseCertificationError(
            f"License certification failed in {context}: "
            f"{_format_findings(cert.findings)}"
        )
