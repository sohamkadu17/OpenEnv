# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kubernetes SRE incident simulation environment implementation."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import KubeSreGymAction, KubeSreGymObservation
except ImportError:
    from models import KubeSreGymAction, KubeSreGymObservation

try:
    from .incidents import IncidentManager
    from .kubectl_tools import KubectlResult, KubectlTooling
except ImportError:
    from incidents import IncidentManager
    from kubectl_tools import KubectlResult, KubectlTooling


class KubeSreGymEnvironment(Environment):
    """Environment that trains an agent to diagnose and recover Kubernetes failures."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    ALLOWED_TOOLS = [
        "kubectl_get",
        "kubectl_describe",
        "kubectl_logs",
        "kubectl_events",
        "kubectl_patch",
        "kubectl_apply_yaml",
        "kubectl_delete_pod",
        "kubectl_rollout_undo",
        "kubectl_exec",
    ]

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode = 0
        self._namespace = os.getenv("SRE_GYM_NAMESPACE", "sre-gym")
        self._app_name = os.getenv("SRE_GYM_APP_NAME", "sre-app")
        self._service_name = os.getenv("SRE_GYM_SERVICE_NAME", "sre-app")
        self._difficulty = os.getenv("SRE_GYM_DIFFICULTY", "medium").lower()
        self._seed = int(os.getenv("SRE_GYM_SEED", "42"))
        self._max_steps = int(os.getenv("SRE_GYM_MAX_STEPS", "40"))
        self._incident_id_override = os.getenv("SRE_GYM_INCIDENT_ID", "").strip()

        self._tools = KubectlTooling(namespace=self._namespace)
        self._incident_manager = IncidentManager(seed=self._seed)

        self._incident = ""
        self._incident_description = ""
        self._action_history: List[Dict[str, Any]] = []
        self._step_logs: List[Dict[str, Any]] = []
        self._safety_violations = 0
        self._health_cache: Optional[Dict[str, Any]] = None
        self._last_health: Dict[str, Any] = {
            "running_pods": 0,
            "total_pods": 0,
            "endpoint_status_code": None,
        }

    def reset(self) -> KubeSreGymObservation:
        self._episode += 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._action_history = []
        self._step_logs = []
        self._safety_violations = 0
        self._health_cache = None

        connectivity = self._check_cluster_connectivity()
        if not connectivity.ok:
            return self._build_observation(
                phase="OBSERVE",
                tool_result="Cluster connectivity check failed.",
                tool_response=connectivity,
                reward=-3.0,
                done=False,
            )

        try:
            self._setup_scenario()
        except Exception as exc:
            return self._build_observation(
                phase="OBSERVE",
                tool_result="Scenario setup failed.",
                tool_response=self._result_error(f"setup error: {exc}"),
                reward=-3.0,
                done=False,
            )

        incident = self._incident_manager.choose(
            self._difficulty,
            self._episode,
            incident_id=self._incident_id_override,
        )
        injection = incident.injector(self._tools, self._app_name, self._service_name)
        self._incident = incident.id
        self._incident_description = incident.description

        self._health_cache = self._collect_health(use_cache=False)
        if self._health_cache["total_pods"] == 0:
            return self._build_observation(
                phase="OBSERVE",
                tool_result="Scenario initialized, but workload is not observable yet.",
                tool_response=self._result_error("no pods discovered after reset"),
                reward=-1.0,
                done=False,
                health=self._health_cache,
            )

        reward = 0.5 if injection.ok else -2.0
        info = f"Scenario initialized with incident: {self._incident}"
        return self._build_observation(
            phase="OBSERVE",
            tool_result=info,
            tool_response=injection,
            reward=reward,
            done=False,
        )

    def step(self, action: KubeSreGymAction) -> KubeSreGymObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._health_cache = None
        try:
            result, safety_penalty = self._dispatch_tool(action)
            health = self._collect_health(use_cache=True)
            done = self._is_resolved(health) or self._state.step_count >= self._max_steps
            reward = self._compute_reward(result, safety_penalty, done, health)

            phase = "VERIFY" if result.ok else "ANALYZE"
            if done and self._is_resolved(health):
                phase = "RESOLVED"

            self._append_step_log(action, result, reward, health)
            self._action_history.append(
                {
                    "step": self._state.step_count,
                    "thought": action.thought,
                    "tool": action.tool,
                    "args": action.args,
                    "ok": result.ok,
                    "exit_code": result.exit_code,
                    "reward": reward,
                }
            )

            return self._build_observation(
                phase=phase,
                tool_result=self._compact_tool_text(result),
                tool_response=result,
                reward=reward,
                done=done,
                health=health,
                thought=action.thought,
            )
        except Exception as exc:
            error_result = self._result_error(f"step execution failed: {exc}")
            health = self._collect_health(use_cache=False)
            reward = -3.0
            self._append_step_log(action, error_result, reward, health)
            return self._build_observation(
                phase="ANALYZE",
                tool_result="Step failed due to environment exception.",
                tool_response=error_result,
                reward=reward,
                done=False,
                health=health,
                thought=action.thought,
            )

    def _check_cluster_connectivity(self) -> KubectlResult:
        return self._tools.cluster_info()

    def _setup_scenario(self) -> None:
        self._tools.apply_yaml(
            f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self._namespace}
"""
        )
        self._tools._run(
            [
                "kubectl",
                "delete",
                "deployment",
                self._app_name,
                "-n",
                self._namespace,
                "--ignore-not-found=true",
            ]
        )
        self._tools._run(
            [
                "kubectl",
                "delete",
                "service",
                self._service_name,
                "-n",
                self._namespace,
                "--ignore-not-found=true",
            ]
        )
        self._tools._run(
            [
                "kubectl",
                "delete",
                "pod",
                "-l",
                f"app={self._app_name}",
                "-n",
                self._namespace,
                "--ignore-not-found=true",
                "--wait=false",
            ]
        )

        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self._app_name}
  namespace: {self._namespace}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {self._app_name}
  template:
    metadata:
      labels:
        app: {self._app_name}
    spec:
      containers:
      - name: app
        image: nginx:1.27
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: {self._service_name}
  namespace: {self._namespace}
spec:
  selector:
    app: {self._app_name}
  ports:
  - port: 80
    targetPort: 80
"""
        self._tools.apply_yaml(manifest)
        self._wait_for_workload(timeout_seconds=90)

    def _wait_for_workload(self, timeout_seconds: int = 20) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            pod_result = self._tools.get_pod_summary()
            pods = ((pod_result.parsed or {}).get("pods") if pod_result.ok else []) or []
            if len(pods) > 0:
                return
            time.sleep(1)

    def _dispatch_tool(self, action: KubeSreGymAction) -> Tuple[KubectlResult, float]:
        tool = (action.tool or "").strip()
        args = action.args or {}

        if tool not in self.ALLOWED_TOOLS:
            return self._result_error("unsupported tool", [tool]), 0.0

        if tool == "kubectl_get":
            return (
                self._tools.get(
                    resource=str(args.get("resource", "pods")),
                    name=args.get("name"),
                    selector=args.get("selector"),
                    summary=bool(args.get("summary", True)),
                ),
                0.0,
            )

        if tool == "kubectl_describe":
            return (
                self._tools.describe(
                    resource=str(args.get("resource", "pod")),
                    name=str(args.get("name", "")),
                ),
                0.0,
            )

        if tool == "kubectl_logs":
            return (
                self._tools.logs(
                    pod=str(args.get("pod", "")),
                    container=args.get("container"),
                    tail=int(args.get("tail", 200)),
                ),
                0.0,
            )

        if tool == "kubectl_events":
            return self._tools.events(limit=int(args.get("limit", 20))), 0.0

        if tool == "kubectl_patch":
            return (
                self._tools.patch(
                    resource=str(args.get("resource", "")),
                    name=str(args.get("name", "")),
                    patch=str(args.get("patch", "")),
                    patch_type=str(args.get("patch_type", "merge")),
                ),
                0.0,
            )

        if tool == "kubectl_apply_yaml":
            return self._tools.apply_yaml(str(args.get("yaml", ""))), 0.0

        if tool == "kubectl_delete_pod":
            pod_name = str(args.get("pod", ""))
            penalty = self._danger_penalty_for_delete(pod_name)
            return self._tools.delete_pod(pod_name), penalty

        if tool == "kubectl_rollout_undo":
            return self._tools.rollout_undo(str(args.get("deployment", ""))), 0.0

        return self._tools.exec(pod=str(args.get("pod", "")), command=str(args.get("command", ""))), 0.0

    def _danger_penalty_for_delete(self, pod_name: str) -> float:
        if pod_name.startswith("coredns") or pod_name.startswith("kube-"):
            self._safety_violations += 1
            return -6.0
        if not pod_name.startswith(self._app_name):
            self._safety_violations += 1
            return -3.0
        return -0.5

    def _is_resolved(self, health: Optional[Dict[str, Any]] = None) -> bool:
        health = health or self._collect_health()
        return (
            health["total_pods"] > 0
            and health["running_pods"] == health["total_pods"]
            and health["endpoint_status_code"] == 200
        )

    def _collect_health(self, use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and self._health_cache is not None:
            return self._health_cache

        pod_result = self._tools.get_pod_summary()
        pod_summaries = ((pod_result.parsed or {}).get("pods") if pod_result.ok else []) or []
        total_pods = len(pod_summaries)
        running_pods = sum(1 for pod in pod_summaries if pod.get("phase") == "Running" and pod.get("ready") == "1/1")

        if not pod_result.ok and total_pods == 0:
            pod_summaries = []

        endpoint_result = self._tools.endpoint_health(self._service_name)
        endpoint_status = None
        if endpoint_result.ok:
            endpoint_status = 200 if (endpoint_result.stdout or "").strip() else 503

        events_result = self._tools.events(limit=10)
        recent_events = ((events_result.parsed or {}).get("events") if events_result.ok else []) or []

        health = {
            "pod_summaries": pod_summaries,
            "recent_events": recent_events,
            "running_pods": running_pods,
            "total_pods": total_pods,
            "endpoint_status_code": endpoint_status,
        }
        self._health_cache = health
        return health

    def _build_observation(
        self,
        phase: str,
        tool_result: str,
        tool_response: KubectlResult,
        reward: float,
        done: bool,
        health: Optional[Dict[str, Any]] = None,
        thought: str = "",
    ) -> KubeSreGymObservation:
        health = health or self._collect_health()
        self._last_health = {
            "running_pods": health["running_pods"],
            "total_pods": health["total_pods"],
            "endpoint_status_code": health["endpoint_status_code"],
        }

        return KubeSreGymObservation(
            phase=phase,
            tool_result=tool_result,
            tool_response=tool_response.to_dict(),
            allowed_tools=self.ALLOWED_TOOLS,
            namespace=self._namespace,
            endpoint=f"http://{self._service_name}.{self._namespace}.svc.cluster.local",
            endpoint_status_code=health["endpoint_status_code"],
            running_pods=health["running_pods"],
            total_pods=health["total_pods"],
            pod_summaries=health["pod_summaries"],
            recent_events=health["recent_events"],
            incident=self._incident,
            difficulty=self._difficulty,
            action_count=self._state.step_count,
            safety_violations=self._safety_violations,
            done=done,
            reward=reward,
            metadata={
                "episode": self._episode,
                "incident_description": self._incident_description,
                "thought": thought,
                "history_tail": self._action_history[-5:],
                "step_logs_tail": self._step_logs[-5:],
                "repro": {
                    "seed": self._seed,
                    "difficulty": self._difficulty,
                    "incident_id_override": self._incident_id_override,
                },
            },
        )

    def _compact_tool_text(self, result: KubectlResult) -> str:
        if result.ok and result.stdout:
            return result.stdout[:500]
        if result.stderr:
            return result.stderr[:500]
        return "tool executed with no output"

    def _compute_reward(
        self,
        result: KubectlResult,
        safety_penalty: float,
        done: bool,
        health: Dict[str, Any],
    ) -> float:
        reward = -0.1
        if result.ok:
            reward += 0.15
        else:
            reward -= 1.0

        prev_running = int(self._last_health.get("running_pods") or 0)
        prev_total = int(self._last_health.get("total_pods") or 0)
        prev_endpoint = self._last_health.get("endpoint_status_code")

        curr_running = health["running_pods"]
        curr_total = health["total_pods"]
        curr_endpoint = health["endpoint_status_code"]

        prev_ratio = (prev_running / prev_total) if prev_total else 0.0
        curr_ratio = (curr_running / curr_total) if curr_total else 0.0

        if curr_ratio > prev_ratio:
            reward += 2.5
        if prev_endpoint != 200 and curr_endpoint == 200:
            reward += 4.0

        reward += safety_penalty

        if done and self._is_resolved(health):
            reward += max(5.0, 20.0 - 0.5 * self._state.step_count)

        return reward

    def _append_step_log(
        self,
        action: KubeSreGymAction,
        result: KubectlResult,
        reward: float,
        health: Dict[str, Any],
    ) -> None:
        self._step_logs.append(
            {
                "step": self._state.step_count,
                "action": {
                    "thought": action.thought,
                    "tool": action.tool,
                    "args": action.args,
                },
                "tool_response": {
                    "success": result.ok,
                    "stdout": result.stdout[:800],
                    "stderr": result.stderr[:800],
                    "exit_code": result.exit_code,
                    "elapsed_ms": result.elapsed_ms,
                },
                "reward": reward,
                "state_summary": {
                    "running_pods": health.get("running_pods", 0),
                    "total_pods": health.get("total_pods", 0),
                    "endpoint_status_code": health.get("endpoint_status_code"),
                },
            }
        )

    @staticmethod
    def _result_error(message: str, command: Optional[List[str]] = None) -> KubectlResult:
        return KubectlResult(
            ok=False,
            command=command or ["kubectl"],
            exit_code=1,
            stdout="",
            stderr=message,
            elapsed_ms=0,
            parsed={},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
