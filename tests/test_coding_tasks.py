import copy
import json
from collections import Counter

from grove.coding_tasks import coding_catalog


def test_real_workload_has_disjoint_governed_splits() -> None:
    catalog = coding_catalog()
    counts = Counter(item.role.value for item in catalog)

    assert counts == {
        "regression": 103,
        "train": 20,
        "target": 4,
        "future": 2,
    }
    assert len({item.task.id for item in catalog}) == len(catalog)
    assert len({item.task.prompt for item in catalog}) == len(catalog)


def test_hidden_suites_never_put_expected_values_in_tasks() -> None:
    for item in coding_catalog():
        assert item.task.expected is None
        assert item.suite.task_id == item.task.id
        assert item.suite.cases


def test_regression_cohort_is_blind_to_escaped_path_family() -> None:
    for item in coding_catalog():
        if item.role.value != "regression":
            continue
        assert item.task.metadata["failure_type"] == "core_python"
        assert len(item.suite.cases) >= 3


def test_suite_cases_are_json_serializable() -> None:
    for item in coding_catalog():
        for case in item.suite.cases:
            json.dumps(case.payload)
            json.dumps(case.expected)


def test_every_reference_solution_passes_its_own_suite() -> None:
    catalog = coding_catalog()
    assert len({item.task.id for item in catalog}) == len(catalog)
    for item in catalog:
        namespace: dict[str, object] = {}
        exec(item.reference_solution, namespace)  # noqa: S102
        solve = namespace["solve"]
        for case in item.suite.cases:
            actual = solve(copy.deepcopy(case.payload))
            assert actual == case.expected, (
                f"{item.task.id}: solve({case.payload!r}) returned "
                f"{actual!r}, expected {case.expected!r}"
            )
            assert type(actual) is type(case.expected) or isinstance(
                actual, type(case.expected)
            )
