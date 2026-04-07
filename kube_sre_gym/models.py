# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Kube SRE Gym environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
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


class KubeSreGymObservation(Observation):
    """Observation from the Kube SRE Gym environment."""

    phase: str = Field(default="OBSERVE", description="Current recommended loop phase.")
    tool_result: str = Field(default="", description="Compact output from the latest tool call.")
    tool_response: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured tool execution response with stdout/stderr metadata.",
    )
    allowed_tools: List[str] = Field(
        default_factory=list,
        description="Whitelisted tools available to the agent.",
    )
    namespace: str = Field(default="sre-gym", description="Scenario namespace.")
    endpoint: str = Field(default="", description="Target service endpoint URL.")
    endpoint_status_code: Optional[int] = Field(
        default=None,
        description="Latest observed endpoint status code.",
    )
    running_pods: int = Field(default=0, description="Number of running pods in scenario namespace.")
    total_pods: int = Field(default=0, description="Total pods in scenario namespace.")
    pod_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Compact pod-level health information.",
    )
    recent_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent warning and normal events for debugging signals.",
    )
    incident: str = Field(default="", description="Current injected incident identifier.")
    difficulty: str = Field(default="medium", description="Current scenario difficulty.")
    action_count: int = Field(default=0, description="Actions taken in the current episode.")
    safety_violations: int = Field(default=0, description="Count of unsafe actions attempted.")
