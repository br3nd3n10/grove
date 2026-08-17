"""The protocol summary must describe the guardrails that actually ship.

A summary table is the first thing a reader trusts and the last thing anyone
updates. When it still named `heldout_forced_pass`, `unchecked_keys` and UUID
`experts[*].id` pairing, every one of those had been replaced in code months
of commits earlier. A stale summary is not a cosmetic problem: it tells a
reader the system checks something it does not.

These tests read the shipped docs and specs, so they fail when the two drift.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_PROTOCOL = _ROOT / "docs" / "EXPERIMENT_PROTOCOL.md"
_SPEC_DIR = _ROOT / "experiments"


def _audit_table() -> str:
    """The 'What the audit said, and what changed' table, and only that."""
    text = _PROTOCOL.read_text()
    start = text.index("## What the audit said, and what changed")
    end = text.index("## How a predeclared experiment works")
    return text[start:end]


def _audit_module():
    """Load the audit script so the doc can be checked against its allowlist."""
    script = _ROOT / "scripts" / "audit_evaluation_report.py"
    spec = importlib.util.spec_from_file_location("audit_evaluation_report", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "current",
    [
        "heldout_forced_pass_oracle_free",
        "unchecked_prose_keys",
        "experts[*].pairing_key",
        "required_resolved_identity",
        "derived from the decision rules",
        "unverified",
    ],
)
def test_protocol_summary_names_current_guardrails(current):
    assert current in _audit_table(), current


@pytest.mark.parametrize(
    "stale",
    [
        "`route_positive_source: heldout_forced_pass`",
        "as `unchecked_keys`",
        "paired on `experts[*].id`",
        "`EXP-003` declares `requires_control_report`, so the checker refuses",
    ],
)
def test_protocol_summary_drops_superseded_wording(stale):
    assert stale not in _audit_table(), stale


def test_protocol_summary_matches_each_spec_gap_permission():
    """Gap permissions are per spec, so the summary may not state one number."""
    table = _audit_table()
    permissions = {
        path.name: json.loads(path.read_text())["permitted_provenance_gaps"]
        for path in sorted(_SPEC_DIR.glob("*.json"))
    }
    exp002 = next(value for name, value in permissions.items() if "EXP-002" in name)
    exp003 = next(value for name, value in permissions.items() if "EXP-003" in name)

    assert len(exp002) == 2
    assert exp003 == ["models.base.aggregate_sha256"]
    assert "EXP-002 is single-arm and permits two gaps" in table
    assert "EXP-003 is paired and permits **one**" in table


def test_protocol_names_the_worker_manifest_as_required_for_exp003():
    exp003 = json.loads(
        (_SPEC_DIR / "EXP-003-correction-source-ab.json").read_text()
    )

    assert exp003["required_resolved_identity"] == [
        "provenance.worker.model_manifest_sha256"
    ]
    assert "provenance.worker.model_manifest_sha256" in _audit_table()


def test_protocol_exit_two_list_covers_the_current_refusals():
    text = _PROTOCOL.read_text()
    start = text.index("### Exactly what exits 2")
    section = text[start : text.index("###", start + 10)]

    for clause in (
        "need a control arm and none was supplied",
        "control_required_setup",
        "provenance carries no digest",
        "adapter digest",
    ):
        assert clause in section, clause


def test_the_annotated_artifact_carries_an_adjacent_status_record():
    """The historical file is preserved; the correction lives beside it."""
    artifact = _ROOT / "docs" / "data" / "final-real-cycle-annotated-2026-08-07.json"
    status = artifact.with_suffix(".STATUS.md")

    assert artifact.exists()
    assert status.exists()
    body = status.read_text()
    assert "Historical reconciliation only" in body
    assert "unverified" in body
    # The historical file itself is untouched, which is why the note is needed.
    annotated = json.loads(artifact.read_text())
    assert "authoritative for rollback metrics" in (
        annotated["rollback_audit_correction"]["note"]
    )
    assert "no longer\ndefensible" in body or "no longer defensible" in body


def test_verification_record_retracts_the_authoritative_wording():
    text = (_ROOT / "docs" / "VERIFICATION_2026-08-01.md").read_text()

    assert "that annotation is not verified evidence" in text
    assert "integrity.status: unverified" in text


def test_protocol_exit_two_section_states_the_new_refusals():
    """Every refusal these rounds added must be findable where a reader looks."""
    text = _PROTOCOL.read_text()
    start = text.index("### Exactly what exits 2")
    section = text[start : text.index("###", start + 10)]

    for clause in (
        "64 lowercase hexadecimal characters",
        "not a silent fallback to the",
        "absent, null/unmeasured or non-numeric",
        "not a rollback evaluation",
        "closed vocabulary",
        "`not_rollback`",
        "`rollback_disabled`",
        "`rollback_not_run`",
        "unevaluable or refused",
        "`falsified_hypotheses` is",
        "`passed: false`",
        "non-finite",
        "`NaN`, `Infinity` or `-Infinity`",
        "rather than a traceback",
    ):
        assert clause in section, clause


def test_protocol_names_the_whole_rollback_vocabulary_the_audit_accepts():
    """The doc's accepted list and the script's allowlist must not drift."""
    section = _PROTOCOL.read_text()
    start = section.index("### Exactly what exits 2")
    section = section[start : section.index("###", start + 10)]

    for label in sorted(_audit_module().ROLLBACK_LABELS):
        assert f"`{label}`" in section, label


def test_protocol_says_an_unusable_run_publishes_no_hypothesis_outcome():
    text = " ".join(_PROTOCOL.read_text().split())

    assert (
        "A run the checker says it cannot judge may not also report a "
        "falsified hypothesis." in text
    )
    # Including a rule that had already been graded and satisfied.
    assert "has its `passed` withdrawn too" in text


@pytest.mark.parametrize(
    "limit",
    [
        "Grove's core thesis is not proven",
        "EXP-002",
        "EXP-003",
        # The two blockers the objective names explicitly.
        "50-task",
        "external trust anchors",
    ],
)
def test_readme_keeps_its_stated_limits(limit):
    """The new guardrails refuse bad evidence; they do not prove the thesis."""
    assert limit in (_ROOT / "README.md").read_text(), limit
