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
    from ..tasks import SCORE_EPSILON, TaskDefinition, choose_task, get_tasks
except ImportError:
    from models import KubeSreGymAction, KubeSreGymObservation
    from tasks import SCORE_EPSILON, TaskDefinition, choose_task, get_tasks

try:
    from .incidents import IncidentManager
    from .kubectl_tools import KubectlResult, KubectlTooling
except ImportError:
    from incidents import IncidentManager
    from kubectl_tools import KubectlResult, KubectlTooling


class KubeSreGymEnvironment(Environment):
    """Environment that trains an agent to diagnose and recover Kubernetes failures."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    TASKS = get_tasks()

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

    TOOL_SCHEMAS: Dict[str, Dict[str, List[str]]] = {
        "kubectl_get": {"required": ["resource"], "optional": ["name", "selector", "summary"]},
        "kubectl_describe": {"required": ["resource", "name"], "optional": []},
        "kubectl_logs": {"required": ["pod"], "optional": ["container", "tail"]},
        "kubectl_events": {"required": [], "optional": ["limit"]},
        "kubectl_patch": {"required": ["resource", "name", "patch"], "optional": ["patch_type"]},
        "kubectl_apply_yaml": {"required": ["yaml"], "optional": []},
        "kubectl_delete_pod": {"required": ["pod"], "optional": []},
        "kubectl_rollout_undo": {"required": ["deployment"], "optional": []},
        "kubectl_exec": {"required": ["pod", "command"], "optional": []},
    }

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode = 0
        self._namespace = os.getenv("SRE_GYM_NAMESPACE", "sre-gym")
        self._app_name = os.getenv("SRE_GYM_APP_NAME", "sre-app")
        self._service_name = os.getenv("SRE_GYM_SERVICE_NAME", "sre-app")
        self._difficulty = os.getenv("SRE_GYM_DIFFICULTY", "medium").lower()
        self._task_id_override = os.getenv("SRE_GYM_TASK_ID", "").strip()
        self._seed = int(os.getenv("SRE_GYM_SEED", "42"))
        self._max_steps = int(os.getenv("SRE_GYM_MAX_STEPS", "40"))
        self._incident_id_override = os.getenv("SRE_GYM_INCIDENT_ID", "").strip()

        self._tools = KubectlTooling(namespace=self._namespace)
        self._incident_manager = IncidentManager(seed=self._seed)

        self._incident = ""
        self._incident_description = ""
        self._active_task: Optional[TaskDefinition] = None
        self._action_history: List[Dict[str, Any]] = []
        self._step_logs: List[Dict[str, Any]] = []
        self._safety_violations = 0
        self._health_cache: Optional[Dict[str, Any]] = None
        self._episode_start_time = time.time()
        self._resolved_step: Optional[int] = None
        self._last_health: Dict[str, Any] = {
            "running_pods": 0,
            "total_pods": 0,
            "endpoint_status_code": None,
        }

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Return plain task dictionaries with callable graders."""
        return get_tasks()

    def reset(self) -> KubeSreGymObservation:
        self._episode += 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._action_history = []
        self._step_logs = []
        self._safety_violations = 0
        self._health_cache = None
        self._episode_start_time = time.time()
        self._resolved_step = None

        connectivity = self._check_cluster_connectivity()
        if not connectivity.ok:
            return self._build_observation(
                phase="OBSERVE",
                tool_result="Cluster connectivity check failed.",
                tool_response=connectivity,
                reward=0.0,
                done=False,
            )

        try:
            self._setup_scenario()
        except Exception as exc:
            return self._build_observation(
                phase="OBSERVE",
                tool_result="Scenario setup failed.",
                tool_response=self._result_error(f"setup error: {exc}"),
                reward=0.0,
                done=False,
            )

        self._active_task = choose_task(self._difficulty, self._episode, task_id=self._task_id_override)
        task_init = self._active_task.init()
        self._difficulty = str(task_init.get("difficulty", self._active_task.difficulty)).lower()
        selected_incident_id = str(task_init.get("incident_id", "") or "")
        incident = self._incident_manager.choose(
            self._difficulty,
            self._episode,
            incident_id=self._incident_id_override or selected_incident_id,
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
                reward=0.1,
                done=False,
                health=self._health_cache,
            )

        reward = 0.2 if injection.ok else 0.0
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
            result, safety_penalty, safety_violation, error_msg = self._dispatch_tool(action)
            health = self._collect_health(use_cache=True)
            done = self._is_resolved(health) or self._state.step_count >= self._max_steps
            reward = self._compute_reward(result, safety_penalty, done, health)

            if done and self._is_resolved(health) and self._resolved_step is None:
                self._resolved_step = self._state.step_count

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
                step_success=result.ok,
                error=error_msg,
                safety_violation=safety_violation,
            )
        except Exception as exc:
            error_result = self._result_error(f"step execution failed: {exc}")
            health = self._collect_health(use_cache=False)
            reward = 0.0
            self._append_step_log(action, error_result, reward, health)
            return self._build_observation(
                phase="ANALYZE",
                tool_result="Step failed due to environment exception.",
                tool_response=error_result,
                reward=reward,
                done=False,
                health=health,
                thought=action.thought,
                step_success=False,
                error=str(exc),
                safety_violation=False,
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

    def _dispatch_tool(self, action: KubeSreGymAction) -> Tuple[KubectlResult, float, bool, Optional[str]]:
        tool = (action.tool or "").strip()
        args = action.args or {}

        if tool not in self.ALLOWED_TOOLS:
            return self._result_error("unsupported tool", [tool]), 0.0, False, "unsupported tool"

        is_valid, validation_error = self._validate_tool_args(tool, args)
        if not is_valid:
            return self._result_error(validation_error or "invalid arguments", [tool]), 0.0, False, validation_error

        if tool == "kubectl_get":
            return (
                self._tools.get(
                    resource=str(args.get("resource", "pods")),
                    name=args.get("name"),
                    selector=args.get("selector"),
                    summary=bool(args.get("summary", True)),
                ),
                0.0,
                False,
                None,
            )

        if tool == "kubectl_describe":
            return (
                self._tools.describe(
                    resource=str(args.get("resource", "pod")),
                    name=str(args.get("name", "")),
                ),
                0.0,
                False,
                None,
            )

        if tool == "kubectl_logs":
            return (
                self._tools.logs(
                    pod=str(args.get("pod", "")),
                    container=args.get("container"),
                    tail=int(args.get("tail", 200)),
                ),
                0.0,
                False,
                None,
            )

        if tool == "kubectl_events":
            return self._tools.events(limit=int(args.get("limit", 20))), 0.0, False, None

        if tool == "kubectl_patch":
            return (
                self._tools.patch(
                    resource=str(args.get("resource", "")),
                    name=str(args.get("name", "")),
                    patch=str(args.get("patch", "")),
                    patch_type=str(args.get("patch_type", "merge")),
                ),
                0.0,
                False,
                None,
            )

        if tool == "kubectl_apply_yaml":
            yaml_payload = str(args.get("yaml", ""))
            safety_violation = "delete namespace" in yaml_payload.lower()
            penalty = -0.25 if safety_violation else 0.0
            if safety_violation:
                self._safety_violations += 1
            return self._tools.apply_yaml(yaml_payload), penalty, safety_violation, None

        if tool == "kubectl_delete_pod":
            pod_name = str(args.get("pod", ""))
            penalty = self._danger_penalty_for_delete(pod_name)
            return self._tools.delete_pod(pod_name), penalty, penalty < -0.01, None

        if tool == "kubectl_rollout_undo":
            return self._tools.rollout_undo(str(args.get("deployment", ""))), 0.0, False, None

        exec_result = self._tools.exec(pod=str(args.get("pod", "")), command=str(args.get("command", "")))
        safety_violation = not exec_result.ok and "safety policy" in (exec_result.stderr or "").lower()
        if safety_violation:
            self._safety_violations += 1
        return exec_result, (-0.30 if safety_violation else 0.0), safety_violation, (exec_result.stderr if not exec_result.ok else None)

    def _validate_tool_args(self, tool: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        schema = self.TOOL_SCHEMAS.get(tool)
        if schema is None:
            return False, f"no schema registered for tool '{tool}'"

        required = set(schema.get("required", []))
        optional = set(schema.get("optional", []))
        allowed = required | optional
        provided = set(args.keys())

        missing = sorted(required - provided)
        unknown = sorted(provided - allowed)
        if missing:
            return False, f"missing required args for {tool}: {', '.join(missing)}"
        if unknown:
            return False, f"unknown args for {tool}: {', '.join(unknown)}"

        type_expectations: Dict[str, type] = {
            "summary": bool,
            "limit": int,
            "tail": int,
            "resource": str,
            "name": str,
            "selector": str,
            "patch": str,
            "patch_type": str,
            "yaml": str,
            "pod": str,
            "deployment": str,
            "command": str,
            "container": str,
        }
        for key, value in args.items():
            expected = type_expectations.get(key)
            if expected is not None and not isinstance(value, expected):
                return False, f"invalid type for '{key}' in {tool}: expected {expected.__name__}"
        return True, None

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

        service_result = self._tools.get_services_summary()
        services = ((service_result.parsed or {}).get("services") if service_result.ok else []) or []

        health = {
            "pods": pod_summaries,
            "services": services,
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
        step_success: bool = True,
        error: Optional[str] = None,
        safety_violation: bool = False,
    ) -> KubeSreGymObservation:
        health = health or self._collect_health()
        self._last_health = {
            "running_pods": health["running_pods"],
            "total_pods": health["total_pods"],
            "endpoint_status_code": health["endpoint_status_code"],
        }

        elapsed = max(0.0, time.time() - self._episode_start_time)
        resolved = self._is_resolved(health)
        efficiency = 0.0
        if resolved:
            efficiency = max(0.0, 1.0 - (self._state.step_count / max(1, self._max_steps)))

        task_score = self._grade_active_task(health)

        info = {
            "success": bool(step_success),
            "error": error,
            "safety_violation": bool(safety_violation),
            "metrics": {
                "total_steps": self._state.step_count,
                "time_to_recovery": elapsed if resolved else None,
                "unsafe_actions_count": self._safety_violations,
                "success_rate": 1.0 if resolved else 0.0,
                "efficiency_score": round(efficiency, 4),
                "task_score": round(task_score, 4),
                "task_id": (self._active_task.task_id if self._active_task else None),
            },
        }

        last_tool = {
            "stdout": tool_response.stdout,
            "stderr": tool_response.stderr,
            "exit_code": tool_response.exit_code,
            "success": tool_response.ok,
            "command": " ".join(tool_response.command),
            "elapsed_ms": tool_response.elapsed_ms,
        }

        return KubeSreGymObservation(
            pods=health["pods"],
            services=health["services"],
            recent_events=health["recent_events"],
            endpoint_status=health["endpoint_status_code"],
            incident_id=self._incident,
            difficulty=self._difficulty,
            step_count=self._state.step_count,
            safety_violations=self._safety_violations,
            allowed_tools=self.ALLOWED_TOOLS,
            last_tool=last_tool,
            info=info,
            done=done,
            reward=reward,
            metadata={
                "phase": phase,
                "tool_result": tool_result,
                "episode": self._episode,
                "incident_description": self._incident_description,
                "task_id": (self._active_task.task_id if self._active_task else None),
                "task_name": (self._active_task.name if self._active_task else None),
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
        reward = 0.05
        if result.ok:
            reward += 0.10
        else:
            reward -= 0.15

        prev_running = int(self._last_health.get("running_pods") or 0)
        prev_total = int(self._last_health.get("total_pods") or 0)
        prev_endpoint = self._last_health.get("endpoint_status_code")

        curr_running = health["running_pods"]
        curr_total = health["total_pods"]
        curr_endpoint = health["endpoint_status_code"]

        prev_ratio = (prev_running / prev_total) if prev_total else 0.0
        curr_ratio = (curr_running / curr_total) if curr_total else 0.0
        ratio_delta = max(0.0, curr_ratio - prev_ratio)

        if ratio_delta > 0:
            reward += min(0.40, ratio_delta * 0.40)
        if prev_endpoint != 200 and curr_endpoint == 200:
            reward += 0.35

        reward += safety_penalty * 0.05

        # Efficiency pressure: discourage long trajectories.
        reward -= min(0.15, self._state.step_count / max(1, self._max_steps) * 0.15)

        if done and self._is_resolved(health):
            reward = max(reward, 1.0)
        elif done:
            reward = min(reward, 0.20)

        return max(0.0, min(1.0, reward))

    def _grade_active_task(self, health: Dict[str, Any]) -> float:
        if self._active_task is None:
            return SCORE_EPSILON

        state = {
            "endpoint_status": health.get("endpoint_status_code"),
            "running_pods": health.get("running_pods", 0),
            "total_pods": health.get("total_pods", 0),
            "no_errors": True,
            "unsafe_actions": self._safety_violations,
            "step_count": self._state.step_count,
        }
        try:
            return float(self._active_task.grader(state, self._action_history))
        except Exception:
            return SCORE_EPSILON

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
