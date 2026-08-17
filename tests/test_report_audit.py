from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from pathlib import Path

import pytest

from grove.store import GroveStore

_SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_evaluation_report.py"
_SPEC = importlib.util.spec_from_file_location("audit_evaluation_report", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def _database(tmp_path, metrics: dict[str, object]) -> str:
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="rollback-run",
            label="rollback_drill",
            cohort="all",
            metrics=metrics,
        )
    return str(database)


def _report(
    tmp_path, rollback: dict[str, object], *, bound: bool = True
) -> str:
    report = tmp_path / "report.json"
    payload: dict[str, object] = {"rollback": rollback, "other": "preserved"}
    if bound:
        payload["rollback_audit"] = {"run_id": "rollback-run"}
    report.write_text(json.dumps(payload))
    return str(report)


def test_audit_detects_every_stale_rollback_field(tmp_path, capsys):
    metrics = {
        "active_experts": 0,
        "added_parameters": 0,
        "capability": 0.5,
        "tasks": 2,
    }
    database = _database(tmp_path, metrics)
    report = _report(
        tmp_path,
        {**metrics, "active_experts": 1, "added_parameters": 123},
    )

    assert main(["--db", database, "--report", report]) == 1

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] == "stale"
    assert verdict["authoritative_db_record_label"] == "rollback_drill"
    assert [item["field"] for item in verdict["disagreements"]] == [
        "active_experts",
        "added_parameters",
    ]


def test_audit_accepts_matching_rollback_report(tmp_path, capsys):
    metrics = {
        "active_experts": 0,
        "added_parameters": 0,
        "capability": 0.5,
        "tasks": 2,
    }
    database = _database(tmp_path, metrics)
    report = _report(tmp_path, metrics)

    assert main(["--db", database, "--report", report]) == 0

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] == "clean"
    assert verdict["disagreements"] == []


def test_annotation_preserves_input_and_records_correction(tmp_path, capsys):
    metrics = {
        "active_experts": 0,
        "added_parameters": 0,
        "capability": 0.5,
        "tasks": 2,
    }
    database = _database(tmp_path, metrics)
    report = _report(
        tmp_path,
        {**metrics, "active_experts": 1, "added_parameters": 123},
    )
    original = Path(report).read_text()
    annotated = tmp_path / "annotated.json"

    assert (
        main(
            [
                "--db",
                database,
                "--report",
                report,
                "--annotate",
                str(annotated),
            ]
        )
        == 1
    )
    capsys.readouterr()

    assert Path(report).read_text() == original
    content = json.loads(annotated.read_text())
    correction = content["rollback_audit_correction"]
    assert content["rollback"]["active_experts"] == 1
    assert correction["original_stale_values"]["active_experts"] == 1
    assert correction["corrected_values"]["active_experts"] == 0
    assert correction["authoritative_db_evaluation"]["label"] == "rollback_drill"
    assert correction["generated_by"] == "scripts/audit_evaluation_report.py"


def test_report_run_id_binds_audit_to_matching_record_not_older_corrected_one(
    tmp_path, capsys
):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="older-corrected",
            label="rollback_drill_corrected",
            cohort="all",
            metrics={"active_experts": 0},
        )
        store.record_evaluation(
            run_id="newer-report-run",
            label="rollback_drill",
            cohort="all",
            metrics={"active_experts": 1},
        )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "rollback_audit": {"run_id": "newer-report-run"},
                "rollback": {"active_experts": 1},
            }
        )
    )

    assert main(["--db", str(database), "--report", str(report)]) == 0

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] == "clean"
    assert verdict["authoritative_db_record_run_id"] == "newer-report-run"
    assert verdict["binding"]["mode"] == "run_id"


def test_legacy_label_selection_is_marked_unbound(tmp_path, capsys):
    metrics = {"active_experts": 0}
    database = _database(tmp_path, metrics)
    report = _report(tmp_path, metrics, bound=False)

    assert main(["--db", database, "--report", report]) == 2

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] == "unbound"
    assert verdict["authoritative"] is False
    assert verdict["binding"]["mode"] == "unbound_legacy_label_recency"
    assert verdict["binding"]["unbound_evidence"] is True


def test_cli_run_id_overrides_missing_report_binding(tmp_path, capsys):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="specific-run",
            label="rollback_drill",
            cohort="all",
            metrics={"active_experts": 2},
        )
    report = _report(tmp_path, {"active_experts": 2}, bound=False)

    assert (
        main(
            [
                "--db",
                str(database),
                "--report",
                report,
                "--run-id",
                "specific-run",
            ]
        )
        == 0
    )

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["binding"] == {"mode": "run_id", "value": "specific-run"}


def test_contradictory_evaluation_and_run_selectors_refuse(tmp_path, capsys):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        evaluation_id = store.record_evaluation(
            run_id="first-run",
            label="rollback_drill",
            cohort="all",
            metrics={"active_experts": 1},
        )
        store.record_evaluation(
            run_id="second-run",
            label="rollback_drill",
            cohort="all",
            metrics={"active_experts": 2},
        )
    report = _report(tmp_path, {"active_experts": 1}, bound=False)

    assert main(
        [
            "--db",
            str(database),
            "--report",
            report,
            "--evaluation-id",
            evaluation_id,
            "--run-id",
            "second-run",
        ]
    ) == 2

    assert "different or missing" in capsys.readouterr().err


def test_ambiguous_run_id_refuses_recency_guess(tmp_path, capsys):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="duplicate-run",
            label="rollback_drill",
            cohort="all",
            metrics={"active_experts": 1},
        )
        store.record_evaluation(
            run_id="duplicate-run",
            label="rollback_drill",
            cohort="all",
            metrics={"active_experts": 2},
        )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "rollback": {"active_experts": 1},
                "rollback_audit": {"run_id": "duplicate-run"},
            }
        )
    )

    assert main(["--db", str(database), "--report", str(report)]) == 2
    assert "identifies multiple" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Finding 8: --annotate must never land on an input artifact
# --------------------------------------------------------------------------


def test_annotation_refuses_database_alias(tmp_path, capsys):
    """``--annotate evidence.db`` used to succeed and destroy the database."""
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    before = Path(database).read_bytes()

    assert main(["--db", database, "--report", report, "--annotate", database]) == 2

    error = capsys.readouterr().err
    assert "must differ from --db input" in error
    assert Path(database).read_bytes() == before


def test_annotation_refuses_the_report_itself(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    before = Path(report).read_bytes()

    assert main(["--db", database, "--report", report, "--annotate", report]) == 2

    assert "must differ from --report input" in capsys.readouterr().err
    assert Path(report).read_bytes() == before


def test_annotation_refuses_symlink_to_database(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    alias = tmp_path / "alias.json"
    alias.symlink_to(database)
    before = Path(database).read_bytes()

    assert main(["--db", database, "--report", report, "--annotate", str(alias)]) == 2

    assert "same file" in capsys.readouterr().err
    assert Path(database).read_bytes() == before


def test_annotation_refuses_a_hard_link_to_the_database(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    alias = tmp_path / "hardlink.db"
    os.link(database, alias)
    before = Path(database).read_bytes()

    assert main(["--db", database, "--report", report, "--annotate", str(alias)]) == 2

    assert Path(database).read_bytes() == before


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_annotation_refuses_database_sidecar_name(tmp_path, capsys, suffix):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    sidecar = Path(f"{database}{suffix}")
    sidecar.write_bytes(b"sqlite sidecar")
    before = sidecar.read_bytes()

    assert main(
        ["--db", database, "--report", report, "--annotate", str(sidecar)]
    ) == 2

    assert sidecar.read_bytes() == before
    assert suffix in capsys.readouterr().err


def test_annotation_refuses_symlink_to_database_sidecar(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    sidecar = Path(f"{database}-wal")
    sidecar.write_bytes(b"sqlite sidecar")
    alias = tmp_path / "alias-wal"
    alias.symlink_to(sidecar)

    assert main(
        ["--db", database, "--report", report, "--annotate", str(alias)]
    ) == 2

    assert "same file" in capsys.readouterr().err
    assert sidecar.read_bytes() == b"sqlite sidecar"


def test_annotation_refuses_hard_link_to_database_sidecar(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    sidecar = Path(f"{database}-shm")
    sidecar.write_bytes(b"sqlite sidecar")
    alias = tmp_path / "hardlink-shm"
    os.link(sidecar, alias)
    before = sidecar.read_bytes()

    assert main(
        ["--db", database, "--report", report, "--annotate", str(alias)]
    ) == 2

    assert sidecar.read_bytes() == before
    assert alias.read_bytes() == before


def test_annotation_refuses_resolved_database_sidecar(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    database_alias = tmp_path / "database-alias.db"
    database_alias.symlink_to(database)
    sidecar = Path(f"{database}-journal")
    sidecar.write_bytes(b"sqlite sidecar")

    assert main(
        [
            "--db",
            str(database_alias),
            "--report",
            report,
            "--annotate",
            str(sidecar),
        ]
    ) == 2

    assert "same file" in capsys.readouterr().err


def test_annotation_failure_preserves_database_bytes(tmp_path):
    """After a refused annotation the database is still a readable database."""
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})

    assert main(["--db", database, "--report", report, "--annotate", database]) == 2

    with GroveStore(database) as store:
        assert [record["label"] for record in store.evaluations()] == [
            "rollback_drill"
        ]


def test_annotation_to_a_new_destination_still_works(tmp_path):
    database = _database(tmp_path, {"capability": 0.75})
    report = _report(tmp_path, {"capability": 0.5})
    output = tmp_path / "annotated.json"

    assert main(
        ["--db", database, "--report", report, "--annotate", str(output)]
    ) == 1

    annotated = json.loads(output.read_text())
    assert annotated["rollback"] == {"capability": 0.5}
    assert annotated["rollback_audit_correction"]["corrected_values"] == {
        "capability": 0.75
    }


# --------------------------------------------------------------------------
# Finding 7: a database row is not authoritative just because it is a row
# --------------------------------------------------------------------------


def test_editing_db_metrics_breaks_evaluation_binding(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})

    assert main(["--db", database, "--report", report]) == 0
    capsys.readouterr()

    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "UPDATE evaluations SET metrics_json = ?",
            (json.dumps({"capability": 1.0}, sort_keys=True, separators=(",", ":")),),
        )
    connection.close()

    assert main(["--db", database, "--report", report]) == 2
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] == "tampered"
    assert verdict["integrity"]["intact"] is False
    assert verdict["integrity"]["tampered_evaluation_ids"]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("id", "eval_changed"),
        ("run_id", "changed-run"),
        ("label", "rollback_edited"),
        ("cohort", "changed-cohort"),
        ("created_at", "2099-01-01T00:00:00+00:00"),
    ],
)
def test_audit_digest_binds_evaluation_selector_fields(
    tmp_path, capsys, column: str, value: str
):
    database = _database(tmp_path, {"capability": 0.5})
    connection = sqlite3.connect(database)
    evaluation_id = connection.execute("SELECT id FROM evaluations").fetchone()[0]
    connection.close()
    report = tmp_path / "report.json"
    binding = (
        {"run_id": "rollback-run"}
        if column == "id"
        else {"evaluation_id": evaluation_id}
    )
    report.write_text(
        json.dumps({"rollback": {"capability": 0.5}, "rollback_audit": binding})
    )

    connection = sqlite3.connect(database)
    with connection:
        connection.execute(f"UPDATE evaluations SET {column} = ?", (value,))
    connection.close()

    assert main(["--db", database, "--report", str(report)]) == 2
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] == "tampered"
    assert verdict["integrity"]["tampered_evaluation_ids"]

# --------------------------------------------------------------------------
# Finding 4: an unhashed row is unverified, and unverified is not clean
# --------------------------------------------------------------------------


def _blank_digests(database: str) -> None:
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("UPDATE evaluations SET metrics_sha256 = ''")
    connection.close()


def _drop_digest_column(database: str) -> None:
    """Rebuild the table without the digest column, as a legacy file has it."""
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("ALTER TABLE evaluations DROP COLUMN metrics_sha256")
    connection.close()


def _drop_digest_schema_column(database: str) -> None:
    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "ALTER TABLE evaluations DROP COLUMN evaluation_digest_schema"
        )
    connection.close()


def test_blank_evaluation_digest_is_unverified_and_exit_2(tmp_path, capsys):
    """The real 2026-07-31 database reports exactly this shape.

    Four rows, none hashed. The audit used to print ``status: clean`` with
    ``checked: 0`` and exit 0, which reads as "the database agrees" when in
    fact nothing was checked.
    """
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    _blank_digests(database)

    assert main(["--db", database, "--report", report]) == 2

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] == "unverified"
    assert verdict["authoritative"] is False
    assert verdict["integrity"]["status"] == "unverified"
    assert verdict["integrity"]["intact"] is False
    assert verdict["integrity"]["checked"] == 0
    assert verdict["integrity"]["unhashed_evaluation_ids"]


def test_missing_evaluation_digest_column_is_unverified(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    _drop_digest_column(database)

    assert main(["--db", database, "--report", report]) == 2

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["integrity"]["status"] == "unverified"
    assert verdict["integrity"]["authoritative"] is False


def test_missing_evaluation_digest_schema_is_unverified(tmp_path, capsys):
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})
    _drop_digest_schema_column(database)

    assert main(["--db", database, "--report", report]) == 2

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["integrity"]["status"] == "unverified"
    assert verdict["integrity"]["authoritative"] is False


def test_unverified_evidence_cannot_be_called_clean(tmp_path, capsys):
    """A report that agrees with an unverifiable row is not a clean audit."""
    database = _database(tmp_path, {"capability": 0.5})
    report = _report(tmp_path, {"capability": 0.5})

    assert main(["--db", database, "--report", report]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "clean"

    _blank_digests(database)
    assert main(["--db", database, "--report", report]) == 2
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] != "clean"
    assert verdict["disagreements"] == []


def test_annotation_of_unverified_evidence_is_marked_unverified(tmp_path):
    """The artifact carries its own status, not a note in a lost log line."""
    database = _database(tmp_path, {"capability": 0.75})
    report = _report(tmp_path, {"capability": 0.5})
    _blank_digests(database)
    output = tmp_path / "annotated.json"

    assert main(["--db", database, "--report", report, "--annotate", str(output)]) == 2

    annotated = json.loads(output.read_text())["rollback_audit_correction"]
    assert annotated["evidence_status"] == "unverified"
    assert annotated["db_record_is_authoritative"] is False
    assert "HISTORICAL RECONCILIATION ONLY" in annotated["note"]
    assert "authoritative for rollback metrics" not in annotated["note"]


def test_annotation_of_verified_evidence_keeps_the_authoritative_note(tmp_path):
    database = _database(tmp_path, {"capability": 0.75})
    report = _report(tmp_path, {"capability": 0.5})
    output = tmp_path / "annotated.json"

    assert main(["--db", database, "--report", report, "--annotate", str(output)]) == 1

    annotated = json.loads(output.read_text())["rollback_audit_correction"]
    assert annotated["evidence_status"] == "clean"
    assert annotated["db_record_is_authoritative"] is True
    assert "authoritative for rollback metrics" in annotated["note"]


# --------------------------------------------------------------------------
# Reviewer B blocker 2: an exact selector narrows rollback scope, never widens
# it. A baseline row may be inspected; it may not become rollback evidence.
# --------------------------------------------------------------------------


def _baseline_database(tmp_path, metrics: dict[str, object]) -> tuple[str, str, str]:
    """A database whose only exactly selectable row is a clean baseline."""
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        baseline_id = store.record_evaluation(
            run_id="baseline-run",
            label="baseline",
            cohort="all",
            metrics=metrics,
        )
        store.record_evaluation(
            run_id="rollback-run",
            label="rollback_drill",
            cohort="all",
            metrics={**metrics, "capability": 0.99},
        )
    return str(database), baseline_id, "baseline-run"


@pytest.mark.parametrize("selector", ["--evaluation-id", "--run-id"])
def test_exact_selector_cannot_authorize_nonrollback_record(
    tmp_path, capsys, selector
):
    """A matching baseline row is not the rollback measurement under audit."""
    metrics = {
        "active_experts": 0,
        "added_parameters": 0,
        "capability": 0.5,
        "tasks": 2,
    }
    database, baseline_id, baseline_run = _baseline_database(tmp_path, metrics)
    value = baseline_id if selector == "--evaluation-id" else baseline_run
    # Metrics agree exactly, so nothing but the label's scope can refuse this.
    report = _report(tmp_path, metrics, bound=False)

    assert main(["--db", database, "--report", report, selector, value]) == 2

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["authoritative"] is False
    assert verdict["status"] != "clean"
    assert verdict["status"] == "unbound"
    assert verdict["authoritative_db_record"]["label"] == "baseline"
    assert verdict["binding"]["scope"] == "rollback"
    assert verdict["binding"]["scope_refused"] is True
    assert "not a rollback evaluation" in verdict["binding"]["reason"]
    assert verdict["disagreements"] == []


@pytest.mark.parametrize("selector", ["--evaluation-id", "--run-id"])
def test_exact_nonrollback_selector_refuses_annotation(tmp_path, selector):
    metrics = {"active_experts": 0, "capability": 0.5}
    database, baseline_id, baseline_run = _baseline_database(tmp_path, metrics)
    value = baseline_id if selector == "--evaluation-id" else baseline_run
    report = _report(tmp_path, metrics, bound=False)
    output = tmp_path / "annotated.json"

    assert (
        main(
            [
                "--db",
                database,
                "--report",
                report,
                selector,
                value,
                "--annotate",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_report_bound_to_a_baseline_row_is_refused(tmp_path, capsys):
    """The refusal follows the report's own binding, not only CLI flags."""
    metrics = {"active_experts": 0, "capability": 0.5}
    database, _, baseline_run = _baseline_database(tmp_path, metrics)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "rollback": metrics,
                "rollback_audit": {"run_id": baseline_run},
            }
        )
    )

    assert main(["--db", database, "--report", str(report)]) == 2

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["authoritative"] is False
    assert verdict["binding"]["scope_refused"] is True


@pytest.mark.parametrize("selector", ["--evaluation-id", "--run-id"])
def test_exact_rollback_selector_is_still_authoritative(tmp_path, capsys, selector):
    """The narrowing case still works: a rollback row selected exactly is clean."""
    metrics = {"active_experts": 0, "capability": 0.5}
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        rollback_id = store.record_evaluation(
            run_id="rollback-run",
            label="rollback_drill",
            cohort="all",
            metrics=metrics,
        )
    value = rollback_id if selector == "--evaluation-id" else "rollback-run"
    report = _report(tmp_path, metrics, bound=False)

    assert main(["--db", str(database), "--report", report, selector, value]) == 0

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["status"] == "clean"
    assert verdict["authoritative"] is True
    assert verdict["binding"].get("scope_refused") is None


# --------------------------------------------------------------------------
# Review round 6: a label that merely contains "rollback" is not a rollback
# evaluation. `not_rollback` says so in the label and must be refused.
# --------------------------------------------------------------------------

NON_ROLLBACK_LABELS = [
    # A label that merely contains the word.
    "not_rollback",
    "not-rollback",
    "no_rollback",
    "norollback",
    "pre_rollback",
    "post-rollback",
    "baseline_before_rollback",
    "rollbackless",
    "rollbacks_disabled",
    # A label that starts with the word and then denies the rollback. A prefix
    # rule accepted every one of these.
    "rollback_disabled",
    "rollback_not_run",
    "rollback_cancelled",
    "rollback_skipped",
    "rollback_not_performed",
    "rollback_dry_run",
    "rollback_simulated",
    "rollback_planned",
    "rollback_edited",
    "rollback_",
    "rollback_drill_reverted",
    # Nothing to do with a rollback at all.
    "after_growth",
    "baseline",
    "frozen_baseline",
]

# Non-ASCII near misses. Lowercasing maps some of these onto an accepted label
# and NFKC normalisation maps others, which is why neither is applied.
NON_ASCII_ROLLBACK_LABELS = [
    "rollbac\u212a",        # Kelvin sign, lowercases to "k"
    "rollback\u0130",       # dotted capital I
    "rollback\u0131",       # dotless i
    "rollback\u017f",       # long s
    "\uff52ollback",        # fullwidth r
    "\u0433ollback",        # Cyrillic homoglyph
    "rollback\u200b",       # zero-width space
    "\ufeffrollback",       # byte-order mark
    "rollback\u00a0",       # NBSP padding
    "\u00a0rollback_drill",  # NBSP padding
    "rollback\u0301",       # combining acute
    "\uff52\uff4f\uff4c\uff4c\uff42\uff41\uff43\uff4b",  # fullwidth "rollback"
]

ROLLBACK_LABELS = [
    "rollback",
    "rollback_drill",
    "rollback_drill_corrected",
    "Rollback_Drill",
    " rollback_drill ",
]


def _labelled_database(tmp_path, label: str, metrics: dict[str, object]):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        evaluation_id = store.record_evaluation(
            run_id="probe-run",
            label=label,
            cohort="all",
            metrics=metrics,
        )
    return str(database), evaluation_id


@pytest.mark.parametrize("label", NON_ROLLBACK_LABELS)
@pytest.mark.parametrize("selector", ["--evaluation-id", "--run-id"])
def test_a_label_that_only_contains_rollback_is_not_rollback_evidence(
    tmp_path, capsys, label, selector
):
    """`not_rollback` used to satisfy the substring test and authorise itself."""
    metrics = {"active_experts": 0, "capability": 0.5}
    database, evaluation_id = _labelled_database(tmp_path, label, metrics)
    value = evaluation_id if selector == "--evaluation-id" else "probe-run"
    # Metrics agree exactly, so only the label's scope can refuse this.
    report = _report(tmp_path, metrics, bound=False)

    assert main(["--db", database, "--report", report, selector, value]) == 2

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["authoritative"] is False
    assert verdict["status"] == "unbound"
    assert verdict["binding"]["scope_refused"] is True
    assert verdict["binding"]["scope"] == "rollback"
    assert repr(label) in verdict["binding"]["reason"]
    # It is not listed as a rollback record either.
    assert verdict["rollback_record_count"] == 0
    assert verdict["rollback_records"] == []


@pytest.mark.parametrize("label", ROLLBACK_LABELS)
@pytest.mark.parametrize("selector", ["--evaluation-id", "--run-id"])
def test_a_real_rollback_label_is_still_rollback_evidence(
    tmp_path, capsys, label, selector
):
    metrics = {"active_experts": 0, "capability": 0.5}
    database, evaluation_id = _labelled_database(tmp_path, label, metrics)
    value = evaluation_id if selector == "--evaluation-id" else "probe-run"
    report = _report(tmp_path, metrics, bound=False)

    assert main(["--db", database, "--report", report, selector, value]) == 0

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["authoritative"] is True
    assert verdict["status"] == "clean"
    assert verdict["binding"].get("scope_refused") is None
    assert verdict["rollback_record_count"] == 1


@pytest.mark.parametrize("label", NON_ROLLBACK_LABELS)
def test_a_near_miss_label_refuses_annotation(tmp_path, label):
    metrics = {"active_experts": 0, "capability": 0.5}
    database, evaluation_id = _labelled_database(tmp_path, label, metrics)
    report = _report(tmp_path, metrics, bound=False)
    output = tmp_path / "annotated.json"

    assert (
        main(
            [
                "--db",
                database,
                "--report",
                report,
                "--evaluation-id",
                evaluation_id,
                "--annotate",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


@pytest.mark.parametrize("label", NON_ROLLBACK_LABELS)
def test_a_near_miss_label_is_not_selectable_by_legacy_recency(
    tmp_path, capsys, label
):
    """Unbound legacy selection must not reach a near-miss label either."""
    metrics = {"active_experts": 0, "capability": 0.5}
    database, _ = _labelled_database(tmp_path, label, metrics)
    report = _report(tmp_path, metrics, bound=False)

    assert main(["--db", database, "--report", report]) == 2

    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert "no rollback evaluation records" in error["error"]


def test_the_rollback_label_predicate_is_a_closed_vocabulary():
    for label in ROLLBACK_LABELS:
        assert _MODULE._is_rollback_record({"label": label}) is True, label
    for label in NON_ROLLBACK_LABELS:
        assert _MODULE._is_rollback_record({"label": label}) is False, label
    for label in NON_ASCII_ROLLBACK_LABELS:
        assert _MODULE._is_rollback_record({"label": label}) is False, label
    # Non-string labels are refused rather than crashing the audit, and an
    # object whose __str__ says "rollback_drill" is still not a label.

    class _Pretender:
        def __str__(self) -> str:
            return "rollback_drill"

    for label in (None, 0, 1, False, True, {}, [], _Pretender()):
        assert _MODULE._is_rollback_record({"label": label}) is False, label
    # An absent label key fails closed rather than raising.
    assert _MODULE._is_rollback_record({}) is False


@pytest.mark.parametrize("label", NON_ASCII_ROLLBACK_LABELS)
@pytest.mark.parametrize("selector", ["--evaluation-id", "--run-id"])
def test_a_non_ascii_lookalike_label_is_not_rollback_evidence(
    tmp_path, capsys, label, selector
):
    metrics = {"active_experts": 0, "capability": 0.5}
    database, evaluation_id = _labelled_database(tmp_path, label, metrics)
    value = evaluation_id if selector == "--evaluation-id" else "probe-run"
    report = _report(tmp_path, metrics, bound=False)

    assert main(["--db", database, "--report", report, selector, value]) == 2

    verdict = json.loads(capsys.readouterr().out)
    assert verdict["authoritative"] is False
    assert verdict["binding"]["scope_refused"] is True
    assert verdict["rollback_record_count"] == 0
