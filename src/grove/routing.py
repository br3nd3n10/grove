from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from grove.models import Cluster, Expert, Failure, RouteDecision, Task
from grove.text import cosine, task_profile, token_profile, tokens


class ProfileRouter:
    """External, removable router based on auditable expert profiles."""

    def __init__(self, minimum_score: float = 0.25) -> None:
        self.minimum_score = minimum_score

    def route(self, task: Task, experts: Sequence[Expert]) -> RouteDecision:
        ranked = sorted(
            ((self.score(task, expert), expert) for expert in experts),
            key=lambda item: (item[0], item[1].id),
            reverse=True,
        )
        if not ranked or ranked[0][0] < self.minimum_score:
            best = ranked[0][0] if ranked else 0.0
            return RouteDecision(
                None, best, "no expert profile cleared the routing threshold"
            )
        score, expert = ranked[0]
        return RouteDecision(
            expert.id, score, f"matched routing profile for {expert.name}"
        )

    def score(self, task: Task, expert: Expert) -> float:
        profile = expert.routing_profile
        expert_tags = set(profile.get("tags", []))
        task_tags = set(task.tags)
        tag_score = (
            len(expert_tags & task_tags) / len(expert_tags) if expert_tags else 0.0
        )

        keywords = set(profile.get("keywords", []))
        prompt_tokens = set(tokens(task.prompt))
        keyword_score = (
            len(keywords & prompt_tokens) / min(2, len(keywords)) if keywords else 0.0
        )
        keyword_score = min(keyword_score, 1.0)

        vector = profile.get("tokens", {})
        text_score = cosine(task_profile(task.prompt), vector) if vector else 0.0
        return max(tag_score, keyword_score, text_score)


class FingerprintClusterer:
    """Groups verifier failures by explicit failure type, falling back to text similarity."""

    def __init__(self, similarity_threshold: float = 0.55) -> None:
        self.similarity_threshold = similarity_threshold

    def cluster(self, failures: Sequence[Failure]) -> list[Cluster]:
        explicit: dict[str, list[Failure]] = defaultdict(list)
        untyped: list[Failure] = []
        for failure in failures:
            if failure.task.metadata.get("failure_type") or failure.task.tags:
                explicit[failure.fingerprint].append(failure)
            else:
                untyped.append(failure)

        groups = list(explicit.items())
        similarity_groups: list[tuple[str, list[Failure]]] = []
        for failure in untyped:
            candidate_profile = token_profile([failure.task.prompt])
            for _, members in similarity_groups:
                member_profile = token_profile(member.task.prompt for member in members)
                if (
                    cosine(candidate_profile, member_profile)
                    >= self.similarity_threshold
                ):
                    members.append(failure)
                    break
            else:
                similarity_groups.append((failure.fingerprint[:48], [failure]))
        groups.extend(similarity_groups)

        clusters: list[Cluster] = []
        for index, (label, members) in enumerate(groups, start=1):
            tags = sorted({tag for member in members for tag in member.task.tags})
            clusters.append(
                Cluster(
                    id=f"cluster_{index:03d}",
                    label=label,
                    failures=tuple(members),
                    profile={
                        "tags": tags,
                        "keywords": sorted(
                            token_profile(member.task.prompt for member in members)
                        )[:16],
                        "tokens": token_profile(
                            member.task.prompt for member in members
                        ),
                    },
                )
            )
        return sorted(
            clusters, key=lambda cluster: (-len(cluster.failures), cluster.label)
        )
