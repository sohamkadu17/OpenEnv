# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kube Sre Gym Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import KubeSreGymAction, KubeSreGymObservation


class KubeSreGymEnv(
    EnvClient[KubeSreGymAction, KubeSreGymObservation, State]
):
    """
    Client for the Kube Sre Gym Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with KubeSreGymEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.incident)
        ...
        ...     action = KubeSreGymAction(tool="kubectl_get", args={"resource": "pods"})
        ...     result = client.step(action)
        ...     print(result.observation.tool_result)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = KubeSreGymEnv.from_docker_image("kube_sre_gym-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(KubeSreGymAction(tool="kubectl_events", args={"limit": 5}))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: KubeSreGymAction) -> Dict:
        """
        Convert KubeSreGymAction to JSON payload for step message.

        Args:
            action: KubeSreGymAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "thought": action.thought,
            "tool": action.tool,
            "args": action.args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[KubeSreGymObservation]:
        """
        Parse server response into StepResult[KubeSreGymObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with KubeSreGymObservation
        """
        obs_data = payload.get("observation", {})
        observation = KubeSreGymObservation(
            phase=obs_data.get("phase", "OBSERVE"),
            tool_result=obs_data.get("tool_result", ""),
            tool_response=obs_data.get("tool_response", {}),
            allowed_tools=obs_data.get("allowed_tools", []),
            namespace=obs_data.get("namespace", "sre-gym"),
            endpoint=obs_data.get("endpoint", ""),
            endpoint_status_code=obs_data.get("endpoint_status_code"),
            running_pods=obs_data.get("running_pods", 0),
            total_pods=obs_data.get("total_pods", 0),
            pod_summaries=obs_data.get("pod_summaries", []),
            recent_events=obs_data.get("recent_events", []),
            incident=obs_data.get("incident", ""),
            difficulty=obs_data.get("difficulty", "medium"),
            action_count=obs_data.get("action_count", 0),
            safety_violations=obs_data.get("safety_violations", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
