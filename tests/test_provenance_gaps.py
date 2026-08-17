"""Report-writer provenance declarations for optional self-repair decoding."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from grove import experiment
from grove.provenance import (
    canonical_hash,
    canonical_json,
    collect_provenance,
    missing_fields,
)


def _provenance(
    tmp_path: Path,
    decoding,
    *,
    correction_source: str = "canonical",
):
    return collect_provenance(
        repo_root=tmp_path,
        base_model="model@abc",
        verifier_suite_version="suite-v1",
        extra={
            "correction_source": correction_source,
            "self_repair_attempts": 3,
            "self_repair_decoding": experiment._self_repair_decoding_record(
                decoding,
                correction_source=correction_source,
            ),
        },
    )


def test_canonical_provenance_declares_self_repair_decoding_not_applicable(
    tmp_path: Path,
):
    record = _provenance(tmp_path, None)

    declared = record["extra"]["self_repair_decoding"]
    assert declared == {
        "applicable": False,
        "reason": "correction source 'canonical'; self-repair decoding not used",
    }
    assert "extra.self_repair_decoding" not in missing_fields(record)


def test_self_repair_without_sampled_decoding_is_declared_truthfully(
    tmp_path: Path,
):
    record = _provenance(tmp_path, None, correction_source="self-repair")

    declared = record["extra"]["self_repair_decoding"]
    assert declared == {
        "applicable": True,
        "declared": False,
        "reason": (
            "self-repair ran under the default greedy decoding; "
            "no sampled decoding was declared"
        ),
    }
    assert "extra.self_repair_decoding" not in missing_fields(record)


def test_empty_decoding_uses_the_falsy_canonical_declaration(tmp_path: Path):
    record = _provenance(tmp_path, {})

    assert record["extra"]["self_repair_decoding"] == {
        "applicable": False,
        "reason": "correction source 'canonical'; self-repair decoding not used",
    }


@pytest.mark.parametrize(
    "decoding",
    [
        {"temperature": 0.8, "base_seed": 20260809},
        MappingProxyType({"temperature": 0.8, "base_seed": 20260809}),
    ],
)
def test_supplied_self_repair_decoding_is_a_dict_copy(tmp_path: Path, decoding):
    record = _provenance(tmp_path, decoding, correction_source="self-repair")

    declared = record["extra"]["self_repair_decoding"]
    assert isinstance(declared, dict)
    assert declared is not decoding
    assert declared == dict(decoding)
    # A non-dict Mapping would fail inside json.dumps before this assertion if
    # the report writer returned it without coercing it to a dict.
    canonical_json(record)
    canonical_hash(record)


def test_declared_decoding_values_add_no_extra_gap_paths(tmp_path: Path):
    record = _provenance(
        tmp_path,
        {"temperature": 0.8, "base_seed": 20260809},
        correction_source="self-repair",
    )

    extra_gaps = [path for path in missing_fields(record) if path.startswith("extra.")]
    assert extra_gaps == []
