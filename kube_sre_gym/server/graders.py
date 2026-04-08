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


def easy_grader(trajectory: Dict[str, Any] | None = None) -> float:
    data = trajectory or {}
    return _trajectory_score(data, base_shift=0.08, unsafe_penalty=0.04)


def medium_grader(trajectory: Dict[str, Any] | None = None) -> float:
    data = trajectory or {}
    return _trajectory_score(data, base_shift=0.04, unsafe_penalty=0.05)


def hard_grader(trajectory: Dict[str, Any] | None = None) -> float:
    data = trajectory or {}
    return _trajectory_score(data, base_shift=0.02, unsafe_penalty=0.06)
