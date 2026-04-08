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


from kube_sre_gym.tasks import TASK_CATALOG, TaskDefinition
