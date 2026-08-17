"""Audit rollback metrics in an evaluation report against Grove's SQLite evidence.

The database is authoritative only when the report names the exact rollback
evaluation row by ``rollback_audit.evaluation_id`` or ``rollback_audit.run_id``
(or the matching CLI flags). Legacy reports without an exact binding still use
the old rollback-label/recency selection, but the verdict marks them as unbound
evidence rather than authoritative. This command is read-only with respect to
both inputs; ``--annotate`` writes a separate report copy with an explicit
correction record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

_MISSING = object()


def _has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        row["name"] == column
        for row in connection.execute(f"PRAGMA table_info({table})")
    )

EVALUATION_DIGEST_SCHEMA = "grove-evaluation-v2"


def _evaluation_digest(record: dict[str, Any]) -> str:
    """Match GroveStore's digest over identity, selectors, and metrics."""
    payload = json.dumps(
        {
            "schema": EVALUATION_DIGEST_SCHEMA,
            "id": record["id"],
            "run_id": record["run_id"],
            "label": record["label"],
            "cohort": record["cohort"],
            "metrics_json": record["metrics_json"],
            "created_at": record["created_at"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_evaluations(database: Path) -> list[dict[str, Any]]:
    """Load evaluation rows using only the SQLite schema used by GroveStore."""
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        hashed = _has_column(connection, "evaluations", "metrics_sha256")
        schema_marked = _has_column(
            connection, "evaluations", "evaluation_digest_schema"
        )
        columns = "id, run_id, label, cohort, metrics_json, created_at" + (
            ", metrics_sha256" if hashed else ""
        ) + (", evaluation_digest_schema" if schema_marked else "")
        rows = connection.execute(
            f"SELECT {columns} FROM evaluations ORDER BY created_at, id"
        )
        evaluations: list[dict[str, Any]] = []
        for row in rows:
            evaluations.append(
                {
                    "id": row["id"],
                    "run_id": row["run_id"],
                    "label": row["label"],
                    "cohort": row["cohort"],
                    "metrics": json.loads(row["metrics_json"]),
                    "metrics_json": row["metrics_json"],
                    "metrics_sha256": row["metrics_sha256"] if hashed else "",
                    "evaluation_digest_schema": (
                        row["evaluation_digest_schema"] if schema_marked else ""
                    ),
                    "created_at": row["created_at"],
                }
            )
        return evaluations
    finally:
        connection.close()


def _evaluation_integrity(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute each stored evaluation digest.

    Without this the database was treated as authoritative simply because it is
    a database. An edited ``metrics_json`` no longer matches the digest written
    beside it, and the audit refuses instead of quoting the edit back as ground
    truth.

    Absence is its own state. ``intact = not tampered`` reported a database with
    no digests at all -- which is what every row written before digest recording
    is -- as clean with ``checked: 0``. Nothing was checked. Such a row is
    ``unverified``: readable history, not evidence, and never authoritative.
    """
    tampered: list[str] = []
    unhashed: list[str] = []
    checked = 0
    for record in evaluations:
        if (
            not record["metrics_sha256"]
            or record["evaluation_digest_schema"] != EVALUATION_DIGEST_SCHEMA
        ):
            unhashed.append(record["id"])
            continue
        checked += 1
        if _evaluation_digest(record) != record["metrics_sha256"]:
            tampered.append(record["id"])
    if tampered:
        status, reason = "tampered", (
            f"{len(tampered)} evaluation row(s) disagree with their stored digest"
        )
    elif unhashed:
        status, reason = "unverified", (
            f"{len(unhashed)} evaluation row(s) carry no current-schema digest, "
            "so they cannot be checked and must not be quoted as authoritative"
        )
    else:
        status, reason = "clean", (
            f"{len(evaluations)} evaluation row(s) verified against their digests"
        )
    return {
        "status": status,
        "reason": reason,
        "checked": checked,
        "tampered_evaluation_ids": tampered,
        "unhashed_evaluation_ids": unhashed,
        "intact": status == "clean",
        "authoritative": status == "clean",
    }


# The rollback evidence scope is a closed vocabulary, not a pattern. A prefix
# rule read ``rollback_disabled`` and ``rollback_not_run`` as rollback records:
# labels that say in words that no rollback happened. A substring rule before it
# read ``not_rollback`` the same way. Any pattern wide enough to admit future
# labels is wide enough to admit their negations, so the accepted set is listed
# instead. Adding a genuinely new rollback label is a deliberate edit here.
#
#   ``rollback``                   -- the bare label used by evidence tests
#   ``rollback_drill``             -- what ``run_first_real_cycle`` writes
#   ``rollback_drill_corrected``   -- the corrected row in checked-in history
ROLLBACK_LABELS = frozenset(
    {
        "rollback",
        "rollback_drill",
        "rollback_drill_corrected",
    }
)


def _is_rollback_record(evaluation: dict[str, Any]) -> bool:
    """The one label predicate that defines the rollback evidence scope.

    Fails closed. A non-string label is not a label, and a non-ASCII one is not
    comparable to this vocabulary without a normalisation step that would itself
    widen acceptance: ``lower()`` maps the Kelvin sign onto ``k``, ``casefold()``
    maps the long s onto ``s``, and NFKC folds fullwidth text into ASCII. Only
    ASCII spaces are trimmed, because only they are certain to be padding.
    """
    raw = evaluation.get("label")
    if not isinstance(raw, str) or not raw.isascii():
        return False
    return raw.strip(" ").lower() in ROLLBACK_LABELS


def _rollback_records(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        evaluation for evaluation in evaluations if _is_rollback_record(evaluation)
    ]


def _authoritative_record(
    records: list[dict[str, Any]],
    *,
    evaluation_id: str | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not records:
        raise ValueError("database contains no rollback evaluation records")
    matches = records
    if evaluation_id is not None:
        matches = [record for record in matches if record["id"] == evaluation_id]
    if run_id is not None:
        matches = [record for record in matches if record["run_id"] == run_id]
    if not matches:
        if evaluation_id is not None and run_id is not None:
            raise ValueError(
                f"evaluation id {evaluation_id} and run_id {run_id} identify "
                "different or missing rollback records"
            )
        if evaluation_id is not None:
            raise ValueError(
                f"database contains no rollback evaluation id {evaluation_id}"
            )
        raise ValueError(f"database contains no rollback run_id {run_id}")
    if run_id is not None and evaluation_id is None and len(matches) > 1:
        raise ValueError(
            f"run_id {run_id} identifies multiple evaluation records; provide "
            "an exact evaluation id"
        )
    if evaluation_id is not None and run_id is not None:
        return matches[-1], {
            "mode": "evaluation_id_and_run_id",
            "evaluation_id": evaluation_id,
            "run_id": run_id,
        }
    if evaluation_id is not None:
        return matches[-1], {"mode": "evaluation_id", "value": evaluation_id}
    if run_id is not None:
        return matches[-1], {"mode": "run_id", "value": run_id}
    corrected = [
        record
        for record in records
        if str(record["label"]).lower().endswith("corrected")
    ]
    return (corrected or records)[-1], {
        "mode": "unbound_legacy_label_recency",
        "unbound_evidence": True,
        "authoritative": False,
        "reason": (
            "legacy label/recency selection is not authoritative; provide an "
            "exact evaluation id or run id"
        ),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {prefix: value}
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], child))
        return flattened
    return {prefix: value}


def _same_json(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _disagreements(reported: Any, authoritative: Any) -> list[dict[str, Any]]:
    reported_fields = _flatten(reported)
    authoritative_fields = _flatten(authoritative)
    fields = sorted(set(reported_fields) | set(authoritative_fields))
    disagreements: list[dict[str, Any]] = []
    for field in fields:
        reported_value = reported_fields.get(field, _MISSING)
        authoritative_value = authoritative_fields.get(field, _MISSING)
        if reported_value is not _MISSING and authoritative_value is not _MISSING:
            if _same_json(reported_value, authoritative_value):
                continue
        elif reported_value is _MISSING and authoritative_value is _MISSING:
            continue
        disagreements.append(
            {
                "field": field,
                "reported": (None if reported_value is _MISSING else reported_value),
                "authoritative": (
                    None if authoritative_value is _MISSING else authoritative_value
                ),
                "reported_present": reported_value is not _MISSING,
                "authoritative_present": authoritative_value is not _MISSING,
            }
        )
    return disagreements


def _report_binding(report: dict[str, Any]) -> tuple[str | None, str | None]:
    binding = report.get("rollback_audit", {})
    if not isinstance(binding, dict):
        binding = {}
    nested_evaluation_id = binding.get("evaluation_id")
    top_evaluation_id = report.get("rollback_evaluation_id")
    nested_run_id = binding.get("run_id")
    top_run_id = report.get("rollback_run_id")
    if (
        nested_evaluation_id is not None
        and top_evaluation_id is not None
        and str(nested_evaluation_id) != str(top_evaluation_id)
    ):
        raise ValueError("report contains contradictory rollback evaluation ids")
    if (
        nested_run_id is not None
        and top_run_id is not None
        and str(nested_run_id) != str(top_run_id)
    ):
        raise ValueError("report contains contradictory rollback run ids")
    evaluation_id = (
        nested_evaluation_id
        if nested_evaluation_id is not None
        else top_evaluation_id
    )
    run_id = nested_run_id if nested_run_id is not None else top_run_id
    return (
        str(evaluation_id) if evaluation_id is not None else None,
        str(run_id) if run_id is not None else None,
    )


def _merge_selector(
    name: str, cli_value: str | None, report_value: str | None
) -> str | None:
    if (
        cli_value is not None
        and report_value is not None
        and str(cli_value) != report_value
    ):
        raise ValueError(f"--{name} contradicts the report's rollback {name}")
    return str(cli_value) if cli_value is not None else report_value


def _verdict(
    report: dict[str, Any],
    database: Path,
    *,
    evaluation_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    evaluations = _load_evaluations(database)
    records = _rollback_records(evaluations)
    report_evaluation_id, report_run_id = _report_binding(report)
    selected_evaluation_id = _merge_selector(
        "evaluation-id", evaluation_id, report_evaluation_id
    )
    selected_run_id = _merge_selector("run-id", run_id, report_run_id)
    # Exact selectors remain inspectable even when a tampered label no longer
    # looks like a rollback record. Legacy selection intentionally stays
    # restricted to rollback labels.
    selection_records = (
        evaluations
        if selected_evaluation_id is not None or selected_run_id is not None
        else records
    )
    authoritative, binding = _authoritative_record(
        selection_records,
        evaluation_id=selected_evaluation_id,
        run_id=selected_run_id,
    )
    # Locating a row is not authorising it. An exact evaluation_id or run_id
    # narrows the rollback evidence scope; it does not redefine it. Without
    # this, a report could name a clean ``baseline`` row, match its metrics and
    # be called authoritative rollback evidence -- a different measurement
    # quoted as the one under audit.
    if not _is_rollback_record(authoritative):
        binding = dict(binding)
        binding.update(
            {
                "scope": "rollback",
                "scope_refused": True,
                "authoritative": False,
                "reason": (
                    f"selected evaluation is labelled {authoritative['label']!r}, "
                    "not a rollback evaluation; an exact selector may inspect "
                    "any row but cannot make a non-rollback row rollback "
                    "evidence"
                ),
            }
        )
    disagreements = _disagreements(report.get("rollback", {}), authoritative["metrics"])
    integrity = _evaluation_integrity(evaluations)
    binding_authoritative = binding.get("authoritative", True)
    evidence_authoritative = integrity["authoritative"] and binding_authoritative
    return {
        # A legacy label/recency guess is a separate refusal state. A stale
        # comparison is meaningful only after both the row and its selector
        # binding are authoritative.
        "status": (
            integrity["status"]
            if integrity["status"] != "clean"
            else "unbound"
            if not binding_authoritative
            else "stale"
            if disagreements
            else "clean"
        ),
        "authoritative": evidence_authoritative,
        "integrity": integrity,
        "report_rollback": report.get("rollback", {}),
        "rollback_record_count": len(records),
        "rollback_records": [
            {
                "id": record["id"],
                "run_id": record["run_id"],
                "label": record["label"],
                "created_at": record["created_at"],
            }
            for record in records
        ],
        "authoritative_db_record": {
            "id": authoritative["id"],
            "run_id": authoritative["run_id"],
            "label": authoritative["label"],
            "created_at": authoritative["created_at"],
            "metrics": authoritative["metrics"],
        },
        "authoritative_db_record_label": authoritative["label"],
        "authoritative_db_record_run_id": authoritative["run_id"],
        "disagreements": disagreements,
        "binding": binding,
    }


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _same_file(left: Path, right: Path) -> bool:
    """True when two paths name the same file, through aliases or links."""
    try:
        if left.resolve(strict=False) == right.resolve(strict=False):
            return True
    except OSError:
        pass
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except OSError:
        return False
    return (
        left_stat.st_ino == right_stat.st_ino
        and left_stat.st_dev == right_stat.st_dev
    )


def _annotation_inputs(inputs: dict[str, Path]) -> dict[str, Path]:
    """Include SQLite sidecars under supplied and filesystem aliases of db."""
    artifacts = dict(inputs)
    database = inputs.get("db")
    if database is None:
        return artifacts
    database_paths = [database]
    try:
        database_stat = database.stat()
    except OSError:
        database_stat = None
    try:
        resolved = database.resolve(strict=False)
    except OSError:
        resolved = database
    if resolved not in database_paths:
        database_paths.append(resolved)
    # Hard links have no canonical path to resolve. Discover aliases beside
    # the input so their independently named SQLite sidecars are protected.
    if database_stat is not None:
        try:
            for candidate in database.parent.iterdir():
                try:
                    candidate_stat = candidate.stat()
                except OSError:
                    continue
                if (
                    candidate_stat.st_ino == database_stat.st_ino
                    and candidate_stat.st_dev == database_stat.st_dev
                    and candidate not in database_paths
                ):
                    database_paths.append(candidate)
        except OSError:
            pass
    for index, base in enumerate(database_paths):
        suffix_label = "" if index == 0 else f"[alias-{index}]"
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            artifacts[f"db{suffix}{suffix_label}"] = Path(f"{base}{suffix}")
    return artifacts


def _reject_annotation_alias(output: Path, inputs: dict[str, Path]) -> None:
    """Refuse to write an annotation over any input or SQLite sidecar."""
    for name, candidate in _annotation_inputs(inputs).items():
        if _same_file(output, candidate):
            raise ValueError(
                f"--annotate output must differ from --{name} input; "
                f"{output} and {candidate} are the same file"
            )


def _annotate(
    report: dict[str, Any],
    verdict: dict[str, Any],
    output: Path,
    inputs: dict[str, Path],
) -> None:
    _reject_annotation_alias(output, inputs)
    authoritative = verdict["authoritative_db_record"]
    integrity = verdict["integrity"]
    annotated = dict(report)
    annotated["rollback_audit_correction"] = {
        "generated_by": "scripts/audit_evaluation_report.py",
        "evidence_status": integrity["status"],
        "db_record_is_authoritative": verdict["authoritative"],
        "note": (
            "The SQLite evaluation record is authoritative for rollback metrics; "
            "the original report rollback block is preserved unchanged."
            if verdict["authoritative"]
            else "HISTORICAL RECONCILIATION ONLY. Evaluation integrity is "
            f"{integrity['status']}: {integrity['reason']}. These values are "
            "readable history, not verified evidence, and this file must not be "
            "cited as a clean or tamper-evident record. The original report "
            "rollback block is preserved unchanged."
        ),
        "authoritative_db_evaluation": {
            "id": authoritative["id"],
            "run_id": authoritative["run_id"],
            "label": authoritative["label"],
            "created_at": authoritative["created_at"],
        },
        "original_stale_values": report.get("rollback", {}),
        "corrected_values": authoritative["metrics"],
        "disagreements": verdict["disagreements"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(annotated, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a report rollback block with Grove SQLite evidence."
    )
    parser.add_argument("--db", required=True, type=Path, help="Grove SQLite database")
    parser.add_argument("--report", required=True, type=Path, help="JSON report")
    parser.add_argument(
        "--annotate",
        type=Path,
        metavar="OUTPUT.json",
        help="write a separate report copy with rollback corrections",
    )
    parser.add_argument(
        "--evaluation-id",
        help="bind the report to an exact rollback evaluation row id",
    )
    parser.add_argument("--run-id", help="bind the report to an exact rollback run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Checked before the database is opened and before the report is read,
        # so a colliding path costs nothing and destroys nothing.
        if args.annotate is not None:
            _reject_annotation_alias(
                args.annotate, {"report": args.report, "db": args.db}
            )
        report = json.loads(args.report.read_text())
        if not isinstance(report, dict):
            raise TypeError("report JSON must contain an object")
        verdict = _verdict(
            report, args.db, evaluation_id=args.evaluation_id, run_id=args.run_id
        )
        if args.annotate is not None:
            if verdict["binding"].get("unbound_evidence"):
                raise ValueError(
                    "refusing annotation selected by legacy label/recency; "
                    "provide --evaluation-id or --run-id"
                )
            if verdict["binding"].get("scope_refused"):
                raise ValueError(
                    "refusing annotation from a non-rollback evaluation: "
                    f"{verdict['binding'].get('reason')}"
                )
            _annotate(
                report,
                verdict,
                args.annotate,
                {"report": args.report, "db": args.db},
            )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(verdict, indent=2, sort_keys=True))
    # Both a disagreeing digest and a missing one mean the database cannot be
    # quoted as ground truth, so both exit 2 rather than 0 or 1.
    if verdict.get("authoritative") is not True:
        return 2
    return 1 if verdict["disagreements"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
