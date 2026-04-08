"""Task registry and graders for kube_sre_gym."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


SCORE_EPSILON = 0.01


def _clamp_open_unit_interval(value: float) -> float:
    return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, float(value)))


def _base_health_score(state: Dict[str, Any]) -> float:
    running = int(state.get("running_pods") or 0)
    total = int(state.get("total_pods") or 0)
    endpoint = state.get("endpoint_status")
    no_errors = bool(state.get("no_errors", True))

    pod_ratio = (running / total) if total > 0 else 0.0
    score = 0.15 + (0.45 * pod_ratio)
    if endpoint == 200:
        score += 0.30
    if no_errors:
        score += 0.10
    return _clamp_open_unit_interval(score)


def _efficiency_penalty(step_count: int, max_steps: int) -> float:
    if max_steps <= 0:
        return 0.0
    return min(0.20, (max(0, step_count - 1) / max_steps) * 0.20)


def _safety_penalty(unsafe_actions: int) -> float:
    return min(0.25, max(0, unsafe_actions) * 0.05)


def init_fix_broken_service_selector() -> Dict[str, Any]:
    return {"incident_id": "broken_service_selector", "difficulty": "easy"}


def init_recover_crashloopbackoff_pod() -> Dict[str, Any]:
    return {"incident_id": "crash_loop_container", "difficulty": "medium"}


def init_resolve_oomkilled_pod() -> Dict[str, Any]:
    return {"incident_id": "oom_killed_pod", "difficulty": "hard"}


def grade_fix_broken_service_selector(state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    base = _base_health_score(state)
    step_count = int(state.get("step_count") or 0)
    unsafe_actions = int(state.get("unsafe_actions") or 0)
    score = base - _efficiency_penalty(step_count, 20) - _safety_penalty(unsafe_actions)
    return _clamp_open_unit_interval(score)


def grade_recover_crashloopbackoff_pod(state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    base = _base_health_score(state)
    step_count = int(state.get("step_count") or 0)
    unsafe_actions = int(state.get("unsafe_actions") or 0)

    used_recovery_tool = any(
        str(item.get("tool")) in {"kubectl_rollout_undo", "kubectl_patch"}
        for item in history
    )
    strategy_bonus = 0.10 if used_recovery_tool else 0.0
    score = base + strategy_bonus - _efficiency_penalty(step_count, 24) - _safety_penalty(unsafe_actions)
    return _clamp_open_unit_interval(score)


def grade_resolve_oomkilled_pod(state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    base = _base_health_score(state)
    step_count = int(state.get("step_count") or 0)
    unsafe_actions = int(state.get("unsafe_actions") or 0)

    memory_patch_detected = any(
        str(item.get("tool")) == "kubectl_patch"
        and "memory" in str(item.get("args", {})).lower()
        for item in history
    )
    strategy_bonus = 0.12 if memory_patch_detected else 0.0
    score = base + strategy_bonus - _efficiency_penalty(step_count, 30) - _safety_penalty(unsafe_actions)
    return _clamp_open_unit_interval(score)


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    description: str
    difficulty: str
    init: Callable[[], Dict[str, Any]]
    grader: Callable[[Dict[str, Any], List[Dict[str, Any]]], float]


TASK_CATALOG: List[TaskDefinition] = [
    TaskDefinition(
        task_id="task_fix_broken_service_selector",
        name="Fix Broken Service Selector",
        description="Service selector does not match pod labels; recover service routing.",
        difficulty="easy",
        init=init_fix_broken_service_selector,
        grader=grade_fix_broken_service_selector,
    ),
    TaskDefinition(
        task_id="task_recover_crashloopbackoff_pod",
        name="Recover CrashLoopBackOff Pod",
        description="Deployment enters CrashLoopBackOff due to bad runtime config.",
        difficulty="medium",
        init=init_recover_crashloopbackoff_pod,
        grader=grade_recover_crashloopbackoff_pod,
    ),
    TaskDefinition(
        task_id="task_resolve_oomkilled_pod",
        name="Resolve OOMKilled Pod",
        description="Pod is OOMKilled due to overly strict memory limits.",
        difficulty="hard",
        init=init_resolve_oomkilled_pod,
        grader=grade_resolve_oomkilled_pod,
    ),
]


# Explicit dict-like task schema for validator discoverability.
OPENENV_TASKS: List[Dict[str, Any]] = [
    {
        "id": task.task_id,
        "description": task.description,
        "difficulty": task.difficulty,
        "init": task.init,
        "grader": task.grader,
    }
    for task in TASK_CATALOG
]


def choose_task(difficulty: str, episode: int, task_id: str = "") -> TaskDefinition:
    if task_id:
        for task in TASK_CATALOG:
            if task.task_id == task_id:
                return task

    normalized = (difficulty or "medium").lower()
    tiers = {
        "easy": {"easy"},
        "medium": {"easy", "medium"},
        "hard": {"easy", "medium", "hard"},
    }
    allowed = tiers.get(normalized, {"easy", "medium"})
    pool = [task for task in TASK_CATALOG if task.difficulty in allowed]
    if not pool:
        pool = TASK_CATALOG

    rng = random.Random(42 + max(1, episode))
    return pool[rng.randrange(0, len(pool))]
