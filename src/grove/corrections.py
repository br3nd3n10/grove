"""Where a training target comes from, and how to compare the options.

The 2026-08-06 audit found that failures only selected *which* examples to train
on; the corrections themselves were human-written canonical solutions
(``canonical-reference-v1``). That weakens "the system learns from its own
failures" to "the system notices its own failures". This module makes the
correction source an explicit, swappable, verifier-gated component so the two
can be run head to head instead of argued about.

Nothing here trusts the model. A self-generated repair becomes a training target
only if the hidden verifier passes it, exactly like a canonical reference.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from grove.models import Task, Verification

Verify = Callable[[Task, str], Verification]


@dataclass(frozen=True, slots=True)
class CorrectionProposal:
    """One candidate training target and the verifier's verdict on it."""

    task_id: str
    source: str
    response: str | None
    verification: Verification | None
    attempts: int
    generation_calls: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.verification is not None and self.verification.passed


class CorrectionSource(Protocol):
    name: str

    def propose(
        self, task: Task, failed_response: str, verify: Verify
    ) -> CorrectionProposal: ...


def repair_prompt(task: Task, failed_response: str, reason: str) -> str:
    """Deterministic repair instruction; no hidden test cases are revealed."""
    return (
        f"{task.prompt}\n\n"
        "Your previous answer was rejected by an automated verifier.\n"
        f"Previous answer:\n{failed_response}\n\n"
        f"Verifier feedback: {reason}\n\n"
        "Write a corrected answer. Output only the corrected answer."
    )


# Only the *class name* of an exception is safe to feed back: a message such as
# ``KeyError: 'secret'`` can carry a hidden payload value verbatim.
_STDERR_EXCEPTION_CLASS = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b", re.MULTILINE
)

# The only verification-detail facts a repair prompt may repeat. Everything
# else -- expected values, payloads, stderr message text -- stays hidden.
FEEDBACK_DETAIL_ALLOWLIST = (
    "cases",
    "cases_passed",
    "exit_code",
    "timed_out",
    "output_limited",
    "stderr_exception_class",
)


def repair_feedback(verification: Verification) -> str:
    """Honest failure detail for a repair prompt, without hidden-suite leakage.

    EXP-003 fed every retry the same generic sentence, so the model repaired
    blind. The verifier's ``reason`` strings are leak-free by construction
    ("candidate returned the wrong number of case results", "candidate exited
    with status 1", ...), and this adds only allowlisted facts on top: case
    pass counts, the exit/timeout/output class, and the *class name* of an
    exception found in stderr. It never repeats an expected value, a case
    payload, an ``expected`` detail field, or stderr message text, any of
    which could carry hidden-suite data.
    """
    parts = [verification.reason]
    details = verification.details or {}
    cases = details.get("cases")
    cases_passed = details.get("cases_passed")
    if isinstance(cases, int) and isinstance(cases_passed, int):
        parts.append(f"{cases_passed} of {cases} hidden cases passed")
    exit_code = details.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code:
        parts.append(f"your program exited with status {exit_code}")
    if details.get("timed_out"):
        parts.append("your program exceeded the wall-clock limit")
    if details.get("output_limited"):
        parts.append("your program exceeded the output limit")
    stderr = details.get("stderr_tail")
    if isinstance(stderr, str):
        classes = _STDERR_EXCEPTION_CLASS.findall(stderr)
        if classes:
            parts.append(f"stderr shows an unhandled {classes[-1]}")
    return "; ".join(parts)


def derive_attempt_seed(base_seed: int, task_id: str, attempt: int) -> int:
    """A distinct, reproducible sampling seed for one repair attempt.

    Derived rather than sequential so two tasks never share an attempt seed,
    and recorded in the proposal provenance so the run can be replayed.
    """
    digest = hashlib.sha256(f"{base_seed}:{task_id}:{attempt}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31)


class CanonicalReferenceSource:
    """Human-written reference solutions: the 2026-07-31 configuration."""

    name = "canonical-reference-v1"

    def __init__(self, references: dict[str, str]) -> None:
        self.references = dict(references)

    def propose(
        self, task: Task, failed_response: str, verify: Verify
    ) -> CorrectionProposal:
        reference = self.references.get(task.id)
        if reference is None:
            return CorrectionProposal(
                task_id=task.id,
                source=self.name,
                response=None,
                verification=None,
                attempts=0,
                generation_calls=0,
                provenance={"reason": "no reference solution registered"},
            )
        return CorrectionProposal(
            task_id=task.id,
            source=self.name,
            response=reference,
            verification=verify(task, reference),
            attempts=1,
            generation_calls=0,
            provenance={"origin": "human-authored reference"},
        )


class SelfRepairSource:
    """The deployed model repairs its own failure, under verifier gating.

    The task prompt is rewritten into a repair instruction but the task identity
    is preserved, so the hidden verifier suite selected for grading is the same
    one that rejected the original answer. Repairs are attempted up to
    ``attempts`` times and the first verified repair wins; unverified repairs are
    discarded, never used as training targets.

    EXP-003 showed why decoding and feedback are parameters. With the global
    greedy config every attempt was near-identical and all 60 generation calls
    were wasted. ``decoding`` (e.g. ``{"temperature": 0.8}``) samples repair
    attempts; ``seed`` makes them reproducible, with one derived integer seed
    per attempt recorded in provenance. The failed response is re-verified
    first so the repair prompt carries the verifier's actual failure detail
    (through ``repair_feedback``, which cannot leak hidden expected values).
    Grading itself never samples: ``verify`` is deterministic sandbox
    execution, and evaluation decoding elsewhere stays greedy.
    """

    name = "self-repair-v1"

    def __init__(
        self,
        backend: Any,
        *,
        attempts: int = 3,
        expert: Any = None,
        decoding: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.backend = backend
        self.attempts = attempts
        self.expert = expert
        self.decoding = dict(decoding) if decoding is not None else None
        self.seed = seed

    def _generate(self, task: Task, attempt: int) -> tuple[str, int | None]:
        if self.decoding is None and self.seed is None:
            return self.backend.generate(task, self.expert), None
        decoding = dict(self.decoding or {})
        attempt_seed: int | None = None
        if self.seed is not None:
            attempt_seed = derive_attempt_seed(self.seed, task.id, attempt)
            decoding["seed"] = attempt_seed
        return self.backend.generate(task, self.expert, decoding=decoding), attempt_seed

    def propose(
        self, task: Task, failed_response: str, verify: Verify
    ) -> CorrectionProposal:
        response = failed_response
        # Honest feedback: re-verify the failed answer so the first repair
        # prompt carries the verifier's real reason, not a generic sentence.
        # This is a verification, never a generation call.
        initial = verify(task, failed_response)
        reason = repair_feedback(initial)
        rejected: list[str] = []
        attempt_seeds: list[int | None] = []
        verification = initial
        generation_calls = 0
        for attempt in range(1, self.attempts + 1):
            prompt = repair_prompt(task, response, reason)
            response, attempt_seed = self._generate(
                replace(task, prompt=prompt), attempt
            )
            attempt_seeds.append(attempt_seed)
            generation_calls += 1
            verification = verify(task, response)
            if verification.passed:
                return CorrectionProposal(
                    task_id=task.id,
                    source=self.name,
                    response=response,
                    verification=verification,
                    attempts=attempt,
                    generation_calls=generation_calls,
                    provenance={
                        "origin": "model self-repair",
                        "rejected_attempts": rejected,
                        "expert_id": getattr(self.expert, "id", None),
                        "initial_feedback": repair_feedback(initial),
                        "decoding": dict(self.decoding) if self.decoding else None,
                        "base_seed": self.seed,
                        "attempt_seeds": attempt_seeds,
                    },
                )
            rejected.append(verification.reason)
            reason = repair_feedback(verification)
        return CorrectionProposal(
            task_id=task.id,
            source=self.name,
            response=None,
            verification=verification,
            attempts=self.attempts,
            generation_calls=generation_calls,
            provenance={
                "origin": "model self-repair",
                "rejected_attempts": rejected,
                "expert_id": getattr(self.expert, "id", None),
                "reason": "no attempt cleared the verifier",
                "initial_feedback": repair_feedback(initial),
                "decoding": dict(self.decoding) if self.decoding else None,
                "base_seed": self.seed,
                "attempt_seeds": attempt_seeds,
            },
        )


def correction_proposals(
    failures: Sequence[tuple[Task, str]],
    sources: Sequence[CorrectionSource],
    verify: Verify,
) -> dict[str, list[CorrectionProposal]]:
    """Every source's proposal for every failure, kept as objects.

    The comparison used to throw the proposals away and let the training loop
    ask the selected source again. With a nondeterministic source that means
    the reported yield describes corrections that were never trained on. Keep
    the objects so the caller can train on exactly what it measured.
    """
    proposals: dict[str, list[CorrectionProposal]] = {
        source.name: [] for source in sources
    }
    for task, failed_response in failures:
        for source in sources:
            proposals[source.name].append(source.propose(task, failed_response, verify))
    return proposals


def summarize_correction_sources(
    proposals: Mapping[str, Sequence[CorrectionProposal]],
    failures: Sequence[tuple[Task, str]],
) -> dict[str, Any]:
    """Verified yield, coverage and cost for each source over the same failures."""
    per_source = {
        name: {
            "failures": len(items),
            "verified": sum(item.accepted for item in items),
            "verified_rate": (
                sum(item.accepted for item in items) / len(items) if items else 0.0
            ),
            "mean_attempts": (
                sum(item.attempts for item in items) / len(items) if items else 0.0
            ),
            "generation_calls": sum(item.generation_calls for item in items),
            "verified_task_ids": sorted(
                item.task_id for item in items if item.accepted
            ),
        }
        for name, items in proposals.items()
    }
    names = list(proposals)
    coverage: dict[str, Any] = {}
    if len(names) == 2:
        left, right = (set(per_source[name]["verified_task_ids"]) for name in names)
        coverage = {
            "both": sorted(left & right),
            f"only_{names[0]}": sorted(left - right),
            f"only_{names[1]}": sorted(right - left),
            "neither": sorted(
                task.id for task, _ in failures if task.id not in (left | right)
            ),
        }
    return {
        "sources": names,
        "per_source": per_source,
        "total_generation_calls": sum(
            source["generation_calls"] for source in per_source.values()
        ),
        "coverage": coverage,
        "proposals": {
            name: [
                {
                    "task_id": item.task_id,
                    "accepted": item.accepted,
                    "attempts": item.attempts,
                    "generation_calls": item.generation_calls,
                    "response_sha256": (
                        hashlib.sha256(item.response.encode()).hexdigest()
                        if item.response is not None
                        else None
                    ),
                    "reason": (
                        item.verification.reason if item.verification else "no proposal"
                    ),
                    "decoding": item.provenance.get("decoding"),
                    "attempt_seeds": item.provenance.get("attempt_seeds"),
                }
                for item in items
            ]
            for name, items in proposals.items()
        },
    }


def compare_correction_sources(
    failures: Sequence[tuple[Task, str]],
    sources: Sequence[CorrectionSource],
    verify: Verify,
) -> dict[str, Any]:
    """Run every source over the same failures and report verified yield.

    This answers the cheap half of the audit's question -- can the model produce
    verifier-approved corrections at all, and where does it disagree with the
    human reference -- without paying for a second MLX training run. The
    expensive half (does training on self-repairs give the same expert?) is a
    predeclared protocol, not something this function claims to settle.
    """
    return summarize_correction_sources(
        correction_proposals(failures, sources, verify), failures
    )
