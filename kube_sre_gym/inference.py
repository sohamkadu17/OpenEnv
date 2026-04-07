"""Inference runner for the kube_sre_gym environment."""

import asyncio
import inspect
import json
import os
from typing import Dict, List, Optional, Tuple

from kube_sre_gym import KubeSreGymAction, KubeSreGymEnv
from kube_sre_gym.server.kube_sre_gym_environment import KubeSreGymEnvironment

IMAGE_NAME = os.getenv("IMAGE_NAME")
ENV_HTTP_URL = os.getenv("ENV_HTTP_URL", "http://127.0.0.1:8000")
TASK_NAME = os.getenv("TASK_NAME", "k8s-incident-recovery")
BENCHMARK = os.getenv("BENCHMARK", "kube_sre_gym")
MODEL_NAME = os.getenv("MODEL_NAME", "heuristic-policy")
MAX_STEPS = int(os.getenv("MAX_STEPS", "16"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def validate_prerequisites() -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if MAX_STEPS <= 0:
        issues.append("MAX_STEPS must be > 0")
    if IMAGE_NAME is None and not ENV_HTTP_URL:
        issues.append("Set IMAGE_NAME or ENV_HTTP_URL to create a client environment")
    return len(issues) == 0, issues


def choose_action(observation) -> Tuple[str, Dict]:
    incident = getattr(observation, "incident", "") or ""
    endpoint_status = getattr(observation, "endpoint_status_code", None)
    pods = getattr(observation, "pod_summaries", []) or []

    if endpoint_status == 200:
        return "kubectl_get", {"resource": "pods", "summary": True}

    if "selector" in incident:
        return "kubectl_patch", {
            "resource": "service",
            "name": "sre-app",
            "patch": '{"spec":{"selector":{"app":"sre-app"}}}',
        }

    for pod in pods:
        reason = str(pod.get("reason", ""))
        if reason in {"ErrImagePull", "ImagePullBackOff", "CrashLoopBackOff"}:
            return "kubectl_rollout_undo", {"deployment": "sre-app"}

    return "kubectl_events", {"limit": 10}


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def run_with_client() -> Tuple[bool, int, float, List[float]]:
    if IMAGE_NAME:
        env = await _maybe_await(KubeSreGymEnv.from_docker_image(IMAGE_NAME))
    else:
        env = KubeSreGymEnv(base_url=ENV_HTTP_URL)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        result = await _maybe_await(env.reset())
        for step in range(1, MAX_STEPS + 1):
            tool, args = choose_action(result.observation)
            action = KubeSreGymAction(thought="policy-step", tool=tool, args=args)
            result = await _maybe_await(env.step(action))

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps({"tool": tool, "args": args}, separators=(",", ":"))
            tool_response = getattr(result.observation, "tool_response", {}) or {}
            error = tool_response.get("stderr") if not tool_response.get("success", True) else None
            log_step(step, action_str, reward, bool(result.done), error)

            if result.done:
                break

        score = 1.0 if bool(result.done) else min(max(sum(rewards) / 20.0, 0.0), 1.0)
        success = score >= 0.8 or bool(result.done)
        return success, steps_taken, score, rewards
    finally:
        try:
            await _maybe_await(env.close())
        except Exception:
            pass


def run_inprocess_fallback() -> Tuple[bool, int, float, List[float]]:
    env = KubeSreGymEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    result = type("ResetResult", (), {"observation": env.reset(), "done": False})
    for step in range(1, MAX_STEPS + 1):
        tool, args = choose_action(result.observation)
        obs = env.step(KubeSreGymAction(thought="policy-step", tool=tool, args=args))

        reward = float(obs.reward or 0.0)
        done = bool(obs.done)
        rewards.append(reward)
        steps_taken = step

        action_str = json.dumps({"tool": tool, "args": args}, separators=(",", ":"))
        tool_response = getattr(obs, "tool_response", {}) or {}
        error = tool_response.get("stderr") if not tool_response.get("success", True) else None
        log_step(step, action_str, reward, done, error)

        result = type("StepResult", (), {"observation": obs, "done": done})
        if done:
            break

    score = 1.0 if bool(result.done) else min(max(sum(rewards) / 20.0, 0.0), 1.0)
    success = score >= 0.8 or bool(result.done)
    return success, steps_taken, score, rewards


async def main() -> None:
    ok, issues = validate_prerequisites()
    if not ok:
        raise RuntimeError("; ".join(issues))

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    success = False
    steps_taken = 0
    score = 0.0
    rewards: List[float] = []

    try:
        success, steps_taken, score, rewards = await run_with_client()
    except Exception:
        success, steps_taken, score, rewards = run_inprocess_fallback()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())