"""Episode graders for OpenEnv task registry discovery.

These functions are referenced directly from openenv.yaml.
They accept a trajectory dictionary and return a score in (0, 1).
"""

from __future__ import annotations

from typing import Any, Dict, List


def _clamp_open_unit_interval(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def _normalized_reward_score(rewards: List[float]) -> float:
    if not rewards:
        return 0.5
    avg_reward = sum(float(r) for r in rewards) / max(1, len(rewards))
    # Reward inputs are expected in [0,1] in this environment.
    return _clamp_open_unit_interval(avg_reward)


def _extract_history(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = [
        trajectory.get("history"),
        trajectory.get("actions"),
        trajectory.get("steps"),
        trajectory.get("step_logs"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            out: List[Dict[str, Any]] = []
            for item in candidate:
                if isinstance(item, dict):
                    out.append(item)
            return out
    return []


def _trajectory_score(trajectory: Dict[str, Any], base_shift: float, unsafe_penalty: float) -> float:
    rewards = trajectory.get("rewards") or trajectory.get("reward_trace") or []
    score = _normalized_reward_score(rewards)

    steps = int(trajectory.get("steps") or 0)
    max_steps = int(trajectory.get("max_steps") or 30)
    unsafe_actions = int(trajectory.get("unsafe_actions") or trajectory.get("safety_violations") or 0)

    if max_steps > 0 and steps > 0:
        score -= min(0.20, (steps / max_steps) * 0.20)

    score -= min(0.30, unsafe_actions * unsafe_penalty)
    score += base_shift
    return _clamp_open_unit_interval(score)


def _count_tool_usage(history: List[Dict[str, Any]], tool: str) -> int:
    count = 0
    for entry in history:
        if str(entry.get("tool") or entry.get("action_type") or "") == tool:
            count += 1
    return count


def _has_memory_patch(history: List[Dict[str, Any]]) -> bool:
    for entry in history:
        if str(entry.get("tool") or "") != "kubectl_patch":
            continue
        args = entry.get("args")
        args_text = str(args).lower()
        if "memory" in args_text or "mi" in args_text or "gi" in args_text:
            return True
    return False


def task_fix_broken_service_selector_grader(trajectory: Dict[str, Any] | None = None) -> float:
    data = trajectory or {}
    history = _extract_history(data)
    score = _trajectory_score(data, base_shift=0.05, unsafe_penalty=0.04)

    # Task-specific signal: selector task should use service patch at least once.
    if _count_tool_usage(history, "kubectl_patch") > 0:
        score += 0.10

    endpoint_status = data.get("endpoint_status") or data.get("final_endpoint_status")
    if endpoint_status == 200:
        score += 0.08
    return _clamp_open_unit_interval(score)


def task_recover_crashloopbackoff_pod_grader(trajectory: Dict[str, Any] | None = None) -> float:
    data = trajectory or {}
    history = _extract_history(data)
    score = _trajectory_score(data, base_shift=0.04, unsafe_penalty=0.05)

    # Task-specific signal: crash recovery should use rollout undo or deployment patch.
    used_rollout = _count_tool_usage(history, "kubectl_rollout_undo") > 0
    used_patch = _count_tool_usage(history, "kubectl_patch") > 0
    if used_rollout:
        score += 0.10
    elif used_patch:
        score += 0.06

    running_pods = int(data.get("running_pods") or 0)
    if running_pods > 0:
        score += 0.04
    return _clamp_open_unit_interval(score)


def task_resolve_oomkilled_pod_grader(trajectory: Dict[str, Any] | None = None) -> float:
    data = trajectory or {}
    history = _extract_history(data)
    score = _trajectory_score(data, base_shift=0.02, unsafe_penalty=0.06)

    # Task-specific signal: OOM recovery should include memory resource patch.
    if _has_memory_patch(history):
        score += 0.14

    endpoint_status = data.get("endpoint_status") or data.get("final_endpoint_status")
    if endpoint_status == 200:
        score += 0.06
    return _clamp_open_unit_interval(score)


# Backward-compatible aliases (difficulty-based naming)
def easy_grader(trajectory: Dict[str, Any] | None = None) -> float:
    return task_fix_broken_service_selector_grader(trajectory)


def medium_grader(trajectory: Dict[str, Any] | None = None) -> float:
    return task_recover_crashloopbackoff_pod_grader(trajectory)


def hard_grader(trajectory: Dict[str, Any] | None = None) -> float:
    return task_resolve_oomkilled_pod_grader(trajectory)
