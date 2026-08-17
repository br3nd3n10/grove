"""Correction provenance: self-generated repairs versus human references.

The audit's "material assumption gap" was that the decisive training targets came
from human canonical solutions. These tests pin the alternative path and the
comparison that makes the two sources measurable against each other.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from grove.corrections import (
    CanonicalReferenceSource,
    SelfRepairSource,
    compare_correction_sources,
    correction_proposals,
    derive_attempt_seed,
    repair_feedback,
    repair_prompt,
    summarize_correction_sources,
)
from grove.demo import DemoMathBackend, demo_live_tasks
from grove.experiment import select_training_proposal
from grove.models import Task, Verification
from grove.verifiers import VerifierRegistry

REGISTRY = VerifierRegistry()


def verify(task: Task, response: str) -> Verification:
    return REGISTRY.verify(task, response)


def subtraction_failures() -> list[tuple[Task, str]]:
    backend = DemoMathBackend()
    return [
        (task, backend.generate(task, None))
        for task in demo_live_tasks()
        if task.metadata["operation"] == "subtract"
    ]


class RepairingBackend:
    """A base that cannot solve the family cold but can repair when told why."""

    def __init__(self, *, succeed_on: int = 1) -> None:
        self.succeed_on = succeed_on
        self.calls: list[str] = []

    def generate(self, task: Task, expert=None) -> str:
        self.calls.append(task.prompt)
        if len(self.calls) < self.succeed_on:
            return "still wrong"
        left, right = task.metadata["operands"]
        return str(left - right)


def test_repair_prompt_carries_feedback_without_leaking_hidden_cases():
    task = Task("t1", "Subtract 3 from 10", expected="7", verifier="numeric")

    prompt = repair_prompt(task, "unsupported", "expected 7, received unsupported")

    assert task.prompt in prompt
    assert "unsupported" in prompt
    assert "expected 7, received unsupported" in prompt
    # The verifier's own hidden case list must never appear in a repair prompt.
    assert "hidden" not in prompt.lower() or "hidden verifier" in prompt.lower()


def test_self_repair_is_rejected_when_the_verifier_rejects_it():
    """A model that cannot fix its failure produces no training target at all."""
    source = SelfRepairSource(DemoMathBackend(), attempts=3)
    task, failed = subtraction_failures()[0]

    proposal = source.propose(task, failed, verify)

    assert proposal.accepted is False
    assert proposal.response is None
    assert proposal.attempts == 3
    assert proposal.provenance["reason"] == "no attempt cleared the verifier"
    assert len(proposal.provenance["rejected_attempts"]) == 3


def test_verified_self_repair_becomes_an_accepted_proposal():
    source = SelfRepairSource(RepairingBackend(), attempts=3)
    task, failed = subtraction_failures()[0]

    proposal = source.propose(task, failed, verify)

    assert proposal.accepted is True
    assert proposal.response == task.expected
    assert proposal.attempts == 1
    assert proposal.source == "self-repair-v1"
    assert proposal.provenance["origin"] == "model self-repair"


def test_self_repair_retries_and_records_every_rejected_attempt():
    source = SelfRepairSource(RepairingBackend(succeed_on=3), attempts=3)
    task, failed = subtraction_failures()[0]

    proposal = source.propose(task, failed, verify)

    assert proposal.accepted is True
    assert proposal.attempts == 3
    assert len(proposal.provenance["rejected_attempts"]) == 2


def test_self_repair_grades_against_the_original_task_not_the_repair_prompt():
    """Rewriting the prompt must not change which verifier suite grades it."""
    seen: list[Task] = []

    class Recorder:
        def generate(self, task, expert=None):
            seen.append(task)
            return "7"

    task, failed = subtraction_failures()[0]
    graded: list[Task] = []

    def recording_verify(graded_task, response):
        graded.append(graded_task)
        return verify(graded_task, response)

    SelfRepairSource(Recorder()).propose(task, failed, recording_verify)

    assert seen[0].prompt != task.prompt
    assert seen[0].id == task.id
    assert graded[0] is task


def test_canonical_source_reports_a_missing_reference_instead_of_inventing_one():
    source = CanonicalReferenceSource({})
    task, failed = subtraction_failures()[0]

    proposal = source.propose(task, failed, verify)

    assert proposal.accepted is False
    assert proposal.response is None
    assert proposal.provenance["reason"] == "no reference solution registered"


def test_comparison_separates_human_yield_from_self_repair_yield():
    failures = subtraction_failures()
    references = {task.id: str(task.expected) for task, _ in failures}

    summary = compare_correction_sources(
        failures,
        [CanonicalReferenceSource(references), SelfRepairSource(DemoMathBackend())],
        verify,
    )

    human = summary["per_source"]["canonical-reference-v1"]
    model = summary["per_source"]["self-repair-v1"]
    assert human["verified_rate"] == 1.0
    # The frozen demo base genuinely cannot self-repair this family.
    assert model["verified_rate"] == 0.0
    assert summary["coverage"]["only_canonical-reference-v1"] == sorted(references)
    assert summary["coverage"]["both"] == []


def test_comparison_reports_agreement_when_both_sources_succeed():
    failures = subtraction_failures()
    references = {task.id: str(task.expected) for task, _ in failures}

    summary = compare_correction_sources(
        failures,
        [CanonicalReferenceSource(references), SelfRepairSource(RepairingBackend())],
        verify,
    )

    assert summary["coverage"]["both"] == sorted(references)
    assert summary["coverage"]["neither"] == []
    assert summary["per_source"]["self-repair-v1"]["mean_attempts"] == 1.0


def test_unsolvable_family_leaves_both_sources_empty():
    failures = [
        (replace(task, expected="impossible"), failed)
        for task, failed in subtraction_failures()
    ]

    summary = compare_correction_sources(
        failures,
        [CanonicalReferenceSource({}), SelfRepairSource(DemoMathBackend())],
        verify,
    )

    assert summary["coverage"]["neither"] == sorted(task.id for task, _ in failures)


def test_comparison_reports_generation_calls_for_each_source():
    failures = subtraction_failures()[:1]
    references = {task.id: str(task.expected) for task, _ in failures}
    backend = RepairingBackend()

    summary = compare_correction_sources(
        failures,
        [CanonicalReferenceSource(references), SelfRepairSource(backend)],
        verify,
    )

    assert summary["per_source"]["canonical-reference-v1"]["generation_calls"] == 0
    assert summary["per_source"]["self-repair-v1"]["generation_calls"] == 1
    assert summary["total_generation_calls"] == 1
    assert summary["proposals"]["self-repair-v1"][0]["generation_calls"] == 1


def test_self_repair_generation_calls_count_exhausted_retries():
    failures = subtraction_failures()[:1]
    backend = DemoMathBackend()

    summary = compare_correction_sources(
        failures,
        [SelfRepairSource(backend, attempts=3)],
        verify,
    )

    model = summary["per_source"]["self-repair-v1"]
    assert model["generation_calls"] == 3
    assert summary["total_generation_calls"] == 3
    assert summary["proposals"]["self-repair-v1"][0]["generation_calls"] == 3


# --------------------------------------------------------------------------
# Finding 12: the comparison must describe the corrections that were trained on
# --------------------------------------------------------------------------


class CountingRepairBackend:
    """A nondeterministic repairer: every call returns a different answer."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, task: Task, expert=None) -> str:
        self.calls += 1
        left, right = task.metadata["operands"]
        # The right answer, padded differently on every call: two calls both
        # verify and still produce different text, exactly like a real
        # nondeterministic repairer.
        return f"{left - right}" + " " * self.calls


def test_compare_proposals_are_the_proposals_used_for_training():
    """Regenerating meant the reported yield described untrained corrections."""
    failures = subtraction_failures()
    backend = CountingRepairBackend()
    source = SelfRepairSource(backend, attempts=3)

    proposals = correction_proposals(failures, [source], verify)
    calls_after_comparison = backend.calls
    summary = summarize_correction_sources(proposals, failures)
    reused = {item.task_id: item for item in proposals[source.name]}

    trained = []
    for task, failed in failures:
        proposal, was_reused = select_training_proposal(
            task, failed, source, verify, reused=reused
        )
        assert was_reused is True
        trained.append(proposal)

    # No second generation pass: the comparison's cost is the training cost.
    assert backend.calls == calls_after_comparison
    assert calls_after_comparison == len(failures)
    reported = summary["proposals"][source.name]
    assert [item["response_sha256"] for item in reported] == [
        hashlib.sha256(item.response.encode()).hexdigest() for item in trained
    ]
    assert summary["per_source"][source.name]["verified"] == sum(
        item.accepted for item in trained
    )


def test_without_reuse_the_training_proposal_is_generated_again():
    """The old path, pinned so the difference is visible rather than assumed."""
    failures = subtraction_failures()[:1]
    backend = CountingRepairBackend()
    source = SelfRepairSource(backend, attempts=3)

    proposals = correction_proposals(failures, [source], verify)
    comparison_response = proposals[source.name][0].response

    task, failed = failures[0]
    regenerated, was_reused = select_training_proposal(task, failed, source, verify)

    assert was_reused is False
    assert regenerated.response != comparison_response
    assert backend.calls == 2


def test_summary_records_a_response_hash_for_every_proposal():
    failures = subtraction_failures()[:1]
    references = {task.id: str(task.expected) for task, _ in failures}

    summary = compare_correction_sources(
        failures,
        [CanonicalReferenceSource(references), SelfRepairSource(DemoMathBackend())],
        verify,
    )

    canonical = summary["proposals"]["canonical-reference-v1"][0]
    assert canonical["response_sha256"] == hashlib.sha256(
        str(failures[0][0].expected).encode()
    ).hexdigest()
    # A source that produced nothing records no hash rather than an empty one.
    assert summary["proposals"]["self-repair-v1"][0]["response_sha256"] is None


# --------------------------------------------------------------------------
# EXP-004: fair regime -- sampled, seeded attempts and honest feedback
# --------------------------------------------------------------------------


class DecodingRecorder:
    """A backend that records the decoding of every call and never repairs."""

    def __init__(self) -> None:
        self.calls: list[dict | None] = []

    def generate(self, task: Task, expert=None, *, decoding=None) -> str:
        self.calls.append(decoding)
        return "still wrong"


def test_sampled_repair_attempts_get_distinct_recorded_seeds():
    """EXP-003 wasted 60 calls on near-identical greedy attempts.

    A sampled regime must give every attempt its own derived seed, record all
    of them, and reproduce the same seed sequence for the same base seed.
    """
    backend = DecodingRecorder()
    source = SelfRepairSource(
        backend, attempts=3, decoding={"temperature": 0.8}, seed=42
    )
    task, failed = subtraction_failures()[0]

    proposal = source.propose(task, failed, verify)

    seeds = [call["seed"] for call in backend.calls]
    assert len(seeds) == 3
    assert len(set(seeds)) == 3
    assert all(call["temperature"] == 0.8 for call in backend.calls)
    assert proposal.provenance["attempt_seeds"] == seeds
    assert proposal.provenance["base_seed"] == 42
    assert proposal.provenance["decoding"] == {"temperature": 0.8}
    assert seeds == [derive_attempt_seed(42, task.id, n) for n in (1, 2, 3)]
    # Task-scoped: another task never shares an attempt seed by accident.
    assert derive_attempt_seed(42, "another_task", 1) != seeds[0]

    rerun = DecodingRecorder()
    SelfRepairSource(
        rerun, attempts=3, decoding={"temperature": 0.8}, seed=42
    ).propose(task, failed, verify)
    assert [call["seed"] for call in rerun.calls] == seeds


def test_unconfigured_self_repair_keeps_the_legacy_backend_call_shape():
    """Without decoding, backends that only accept (task, expert) still work."""
    source = SelfRepairSource(DemoMathBackend(), attempts=2)
    task, failed = subtraction_failures()[0]

    proposal = source.propose(task, failed, verify)

    assert proposal.provenance["decoding"] is None
    assert proposal.provenance["base_seed"] is None
    assert proposal.provenance["attempt_seeds"] == [None, None]


def test_first_repair_prompt_carries_the_verifiers_actual_reason():
    """The failed answer is re-verified so attempt one repairs with real detail."""
    prompts: list[str] = []

    class Recorder:
        def generate(self, task, expert=None):
            prompts.append(task.prompt)
            return "nope"

    task, failed = subtraction_failures()[0]
    initial_reason = verify(task, failed).reason

    SelfRepairSource(Recorder(), attempts=1).propose(task, failed, verify)

    assert initial_reason in prompts[0]
    # The EXP-003 generic sentence must be gone.
    assert "did not satisfy the hidden verifier suite" not in prompts[0]


def test_repair_feedback_uses_only_allowlisted_detail():
    """Reason, counts, exit class and exception class -- nothing else."""
    verification = Verification(
        False,
        0.0,
        "one or more hidden cases failed",
        {
            "expected": "SECRET_EXPECTED_VALUE",
            "hidden_cases": [["SECRET_PAYLOAD", "SECRET_EXPECTED_VALUE"]],
            "cases": 7,
            "cases_passed": 3,
            "exit_code": 1,
            "timed_out": False,
            "output_limited": False,
            "stderr_tail": "Traceback (most recent call last):\n"
            "KeyError: 'SECRET_PAYLOAD_KEY'",
            "case_pass_bitmap": [True, False],
        },
    )

    feedback = repair_feedback(verification)

    assert "SECRET_EXPECTED_VALUE" not in feedback
    assert "SECRET_PAYLOAD" not in feedback
    assert "SECRET_PAYLOAD_KEY" not in feedback
    assert "3 of 7 hidden cases passed" in feedback
    assert "exited with status 1" in feedback
    # The exception *class* is allowed; its message is not.
    assert "KeyError" in feedback


def test_no_hidden_suite_expected_value_appears_in_a_repair_prompt():
    """For every catalog task, an adversarial verification cannot leak.

    Even when the verification details carry the entire hidden suite -- the
    payload/expected pairs, an ``expected`` field, and a stderr full of
    expected values -- the repair prompt built from it may not contain one.
    """
    import json

    from grove.coding_tasks import coding_catalog

    failed_response = "def solve(payload):\n    return None"
    for item in coding_catalog():
        details = {
            "cases": len(item.suite.cases),
            "cases_passed": 0,
            "exit_code": 1,
            "stderr_tail": "".join(
                f"ValueError: expected {case.expected!r} for {case.payload!r}\n"
                for case in item.suite.cases
            ),
            "hidden_cases": [
                [case.payload, case.expected] for case in item.suite.cases
            ],
            "expected": [case.expected for case in item.suite.cases],
        }
        verification = Verification(
            False, 0.0, "one or more hidden cases failed", details
        )
        prompt = repair_prompt(
            item.task, failed_response, repair_feedback(verification)
        )
        for case in item.suite.cases:
            for hidden in (json.dumps(case.expected), json.dumps(case.payload)):
                if len(hidden) >= 3 and hidden not in item.task.prompt:
                    assert hidden not in prompt, (item.task.id, hidden)


def test_summary_records_decoding_and_seeds_per_proposal():
    """The report must carry what decoding produced each proposal."""
    failures = subtraction_failures()[:1]
    source = SelfRepairSource(
        DecodingRecorder(), attempts=2, decoding={"temperature": 0.8}, seed=7
    )

    summary = compare_correction_sources(failures, [source], verify)

    recorded = summary["proposals"]["self-repair-v1"][0]
    assert recorded["decoding"] == {"temperature": 0.8}
    assert len(recorded["attempt_seeds"]) == 2
    assert all(isinstance(seed, int) for seed in recorded["attempt_seeds"])
