# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Incident catalog and deterministic incident selection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List

from .kubectl_tools import KubectlResult, KubectlTooling


Injector = Callable[[KubectlTooling, str, str], KubectlResult]


@dataclass(frozen=True)
class IncidentDefinition:
    id: str
    difficulty: str
    description: str
    injector: Injector


def inject_bad_image(tools: KubectlTooling, deployment: str, service: str) -> KubectlResult:
    del service
    return tools.patch(
        "deployment",
        deployment,
        '{"spec":{"template":{"spec":{"containers":[{"name":"app","image":"nginx:invalid-tag"}]}}}}',
        patch_type="strategic",
    )


def inject_broken_selector(tools: KubectlTooling, deployment: str, service: str) -> KubectlResult:
    del deployment
    return tools.patch(
        "service",
        service,
        '{"spec":{"selector":{"app":"does-not-exist"}}}',
        patch_type="merge",
    )


def inject_bad_readiness_probe(tools: KubectlTooling, deployment: str, service: str) -> KubectlResult:
    del service
    return tools.patch(
        "deployment",
        deployment,
        '{"spec":{"template":{"spec":{"containers":[{"name":"app","readinessProbe":{"httpGet":{"path":"/non-existent","port":80},"initialDelaySeconds":1,"periodSeconds":2}}]}}}}',
        patch_type="strategic",
    )


def inject_crash_loop(tools: KubectlTooling, deployment: str, service: str) -> KubectlResult:
    del service
    return tools.patch(
        "deployment",
        deployment,
        '{"spec":{"template":{"spec":{"containers":[{"name":"app","command":["/bin/sh","-c","exit 1"]}]}}}}',
        patch_type="strategic",
    )


def inject_cascading_failure(tools: KubectlTooling, deployment: str, service: str) -> KubectlResult:
    result = tools.patch(
        "service",
        service,
        '{"spec":{"selector":{"app":"does-not-exist"}}}',
        patch_type="merge",
    )
    tools.patch(
        "deployment",
        deployment,
        '{"spec":{"template":{"spec":{"containers":[{"name":"app","readinessProbe":{"httpGet":{"path":"/non-existent","port":80},"initialDelaySeconds":1,"periodSeconds":2}}]}}}}',
        patch_type="strategic",
    )
    return result


def inject_oom_killed_pod(tools: KubectlTooling, deployment: str, service: str) -> KubectlResult:
    del service
    # Keep the scenario realistic: aggressively low memory limits can force
    # workload instability under traffic and often manifest as OOMKilled.
    return tools.patch(
        "deployment",
        deployment,
        '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"16Mi"},"requests":{"memory":"16Mi"}}}]}}}}',
        patch_type="strategic",
    )


class IncidentManager:
    """Deterministic incident selector with difficulty support."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.catalog: List[IncidentDefinition] = [
            IncidentDefinition(
                id="broken_service_selector",
                difficulty="easy",
                description="Service selector does not match any running pods.",
                injector=inject_broken_selector,
            ),
            IncidentDefinition(
                id="bad_image_tag",
                difficulty="easy",
                description="Deployment image tag is invalid and pods fail to pull.",
                injector=inject_bad_image,
            ),
            IncidentDefinition(
                id="broken_readiness_probe",
                difficulty="medium",
                description="Readiness probe is misconfigured and prevents endpoints from becoming ready.",
                injector=inject_bad_readiness_probe,
            ),
            IncidentDefinition(
                id="crash_loop_container",
                difficulty="medium",
                description="Container exits immediately and enters CrashLoopBackOff.",
                injector=inject_crash_loop,
            ),
            IncidentDefinition(
                id="cascading_selector_and_readiness",
                difficulty="hard",
                description="Multiple interacting issues: broken service selector and readiness probe.",
                injector=inject_cascading_failure,
            ),
            IncidentDefinition(
                id="oom_killed_pod",
                difficulty="hard",
                description="Deployment memory limit is too low and causes OOMKilled instability.",
                injector=inject_oom_killed_pod,
            ),
        ]

    def choose(self, difficulty: str, episode: int, incident_id: str = "") -> IncidentDefinition:
        if incident_id:
            selected = self.get_by_id(incident_id)
            if selected is not None:
                return selected
        ordered = self._pool_for_difficulty(difficulty)
        rng = random.Random(self.seed + episode)
        return ordered[rng.randrange(0, len(ordered))]

    def get_by_id(self, incident_id: str) -> IncidentDefinition | None:
        for incident in self.catalog:
            if incident.id == incident_id:
                return incident
        return None

    def _pool_for_difficulty(self, difficulty: str) -> List[IncidentDefinition]:
        difficulty = (difficulty or "medium").lower()
        tiers: Dict[str, List[str]] = {
            "easy": ["easy"],
            "medium": ["easy", "medium"],
            "hard": ["easy", "medium", "hard"],
        }
        allowed = tiers.get(difficulty, ["easy", "medium"])
        pool = [incident for incident in self.catalog if incident.difficulty in allowed]
        if not pool:
            return self.catalog
        return pool
