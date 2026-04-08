# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Kube SRE Gym environment."""

import json

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass


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


# Task and Grader Definitions
# ===========================


@dataclass(frozen=True)
class TaskDefinition:
    """Explicit task definition with deterministic grader function."""

    task_id: str
    name: str
    description: str
    difficulty: str
    grader: Callable[["KubeSreGymObservation", bool, int, float], float]


def grade_easy_task(obs: "KubeSreGymObservation", done: bool, steps: int, cumulative_reward: float) -> float:
    """
    Grade easy tasks: single-incident recovery.
    Scoring: 1.0 if resolved, 0.0 otherwise, with step efficiency bonus.
    """
    if not done:
        return 0.0
    # Reward efficient solutions: max 1.0 at step <= 5, linear decay to 0.7 at step 15
    efficiency_bonus = max(0.7, 1.0 - (steps - 5) * 0.03)
    return min(1.0, efficiency_bonus)


def grade_medium_task(obs: "KubeSreGymObservation", done: bool, steps: int, cumulative_reward: float) -> float:
    """
    Grade medium tasks: multi-step diagnosis and recovery.
    Scoring: base 0.8 if resolved, efficiency adjustments, safe action bonus.
    """
    if not done:
        # Partial credit for good reasoning (positive cumulative reward)
        return min(0.4, max(0.0, cumulative_reward / 20.0))
    
    # Safety bonus: reward fewer violations
    safety_multiplier = 1.0 - (obs.safety_violations * 0.1)
    
    # Efficiency: best at step <= 8, decay to 0.6 at step 20
    efficiency_bonus = max(0.6, 1.0 - (steps - 8) * 0.025)
    
    return min(1.0, 0.8 * efficiency_bonus * safety_multiplier)


def grade_hard_task(obs: "KubeSreGymObservation", done: bool, steps: int, cumulative_reward: float) -> float:
    """
    Grade hard tasks: cascading failures, complex diagnostics.
    Scoring: base 0.6 if resolved, significant efficiency and safety penalties.
    """
    if not done:
        # Partial credit for good approach
        return min(0.2, max(0.0, cumulative_reward / 30.0))
    
    # Strict safety requirements on hard tasks
    if obs.safety_violations > 2:
        return 0.4  # Security violations heavily penalized
    
    safety_multiplier = 1.0 - (obs.safety_violations * 0.15)
    
    # Efficiency: best at step <= 12, decay to 0.5 at step 25
    efficiency_bonus = max(0.5, 1.0 - (steps - 12) * 0.02)
    
    return min(1.0, 0.6 * efficiency_bonus * safety_multiplier)


# Predefined Task Catalog
TASK_CATALOG = [
    TaskDefinition(
        task_id="task_easy_selector",
        name="Easy: Service Selector Fix",
        description="Diagnose and fix a broken service selector that prevents traffic routing.",
        difficulty="easy",
        grader=grade_easy_task,
    ),
    TaskDefinition(
        task_id="task_medium_readiness",
        name="Medium: Readiness Probe Recovery",
        description="Identify and repair a misconfigured readiness probe causing pod failures.",
        difficulty="medium",
        grader=grade_medium_task,
    ),
    TaskDefinition(
        task_id="task_hard_cascading",
        name="Hard: Cascading Failure Resolution",
        description="Resolve multiple interacting failures: broken selector and readiness probe.",
        difficulty="hard",
        grader=grade_hard_task,
    ),
]
