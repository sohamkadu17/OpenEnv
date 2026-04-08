# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Kube SRE Gym environment."""

import json

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from typing import Callable


class KubeSreGymAction(Action):
    """Action for the Kube SRE Gym environment using kubectl tool wrappers."""

    thought: str = Field(
        default="",
        description="Agent reasoning for traceability.",
    )
    tool: str = Field(
        ...,
        description="Tool name to execute (for example: kubectl_get).",
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected tool.",
    )

    @field_validator("args", mode="before")
    @classmethod
    def parse_args_json(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("args must be a valid JSON object") from exc
            if not isinstance(parsed, dict):
                raise ValueError("args JSON must decode to an object")
            return parsed
        raise ValueError("args must be a dictionary or JSON object string")


class KubeSreGymObservation(Observation):
    """Compact, deterministic observation schema for SRE incident recovery."""

    pods: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Compact pod status list (name, status, restarts, reason).",
    )
    services: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Compact service status list.",
    )
    recent_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent events in compact form.",
    )
    endpoint_status: Optional[int] = Field(
        default=None,
        description="HTTP-like endpoint status for target service.",
    )
    incident_id: str = Field(default="", description="Current deterministic incident identifier.")
    difficulty: str = Field(default="medium", description="Current scenario difficulty.")
    step_count: int = Field(default=0, description="Current episode step count.")
    safety_violations: int = Field(default=0, description="Unsafe action count in current episode.")
    allowed_tools: List[str] = Field(
        default_factory=list,
        description="Whitelisted tool names accepted by step().",
    )
    last_tool: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured last tool execution output: stdout/stderr/exit_code/success.",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step-level contract info: success/error/safety_violation/metrics.",
    )


@dataclass(frozen=True)
class TaskDefinition:
    """Task metadata and deterministic grader contract used by benchmark runners."""

    task_id: str
    name: str
    description: str
    difficulty: str
    grader: Callable[["KubeSreGymObservation", bool, int, float], float]


SCORE_EPSILON = 0.01


def _clamp_open_unit_interval(value: float) -> float:
    return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, float(value)))


def _base_grader_score(obs: "KubeSreGymObservation") -> float:
    total_pods = len(obs.pods or [])
    running_pods = sum(
        1 for pod in (obs.pods or []) if str(pod.get("phase", "")) == "Running" and str(pod.get("ready", "")) == "1/1"
    )
    pod_ratio = (running_pods / total_pods) if total_pods > 0 else 0.0

    score = 0.12 + (0.50 * pod_ratio)
    if obs.endpoint_status == 200:
        score += 0.28
    if int(obs.safety_violations or 0) == 0:
        score += 0.10
    return _clamp_open_unit_interval(score)


def grade_fix_broken_service_selector(
    obs: "KubeSreGymObservation",
    done: bool,
    steps: int,
    cumulative_reward: float,
) -> float:
    del cumulative_reward
    score = _base_grader_score(obs)
    if not done:
        score -= 0.22
    score -= min(0.18, max(0, steps - 1) * 0.01)
    score -= min(0.25, int(obs.safety_violations or 0) * 0.05)
    return _clamp_open_unit_interval(score)


def grade_recover_crashloopbackoff_pod(
    obs: "KubeSreGymObservation",
    done: bool,
    steps: int,
    cumulative_reward: float,
) -> float:
    score = _base_grader_score(obs)
    if "crash" in (obs.incident_id or ""):
        score += 0.08
    if not done:
        score -= 0.18
    if cumulative_reward > 0:
        score += min(0.08, cumulative_reward * 0.03)
    score -= min(0.20, max(0, steps - 1) * 0.009)
    score -= min(0.25, int(obs.safety_violations or 0) * 0.05)
    return _clamp_open_unit_interval(score)


def grade_resolve_oomkilled_pod(
    obs: "KubeSreGymObservation",
    done: bool,
    steps: int,
    cumulative_reward: float,
) -> float:
    score = _base_grader_score(obs)
    if "oom" in (obs.incident_id or ""):
        score += 0.10
    if not done:
        score -= 0.20
    if cumulative_reward > 0:
        score += min(0.06, cumulative_reward * 0.02)
    score -= min(0.22, max(0, steps - 1) * 0.008)
    score -= min(0.30, int(obs.safety_violations or 0) * 0.06)
    return _clamp_open_unit_interval(score)


# Explicit in-file task catalog for validator discoverability.
TASK_CATALOG = [
    TaskDefinition(
        task_id="task_fix_broken_service_selector",
        name="Fix Broken Service Selector",
        description="Service selector does not match pod labels; restore service routing.",
        difficulty="easy",
        grader=grade_fix_broken_service_selector,
    ),
    TaskDefinition(
        task_id="task_recover_crashloopbackoff_pod",
        name="Recover CrashLoopBackOff Pod",
        description="Pod crashes from bad runtime configuration; recover with rollout or patch.",
        difficulty="medium",
        grader=grade_recover_crashloopbackoff_pod,
    ),
    TaskDefinition(
        task_id="task_resolve_oomkilled_pod",
        name="Resolve OOMKilled Pod",
        description="Pod is OOMKilled due to low memory limits; increase memory resources.",
        difficulty="hard",
        grader=grade_resolve_oomkilled_pod,
    ),
]
