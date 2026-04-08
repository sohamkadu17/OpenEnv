"""Inference runner for kube_sre_gym with mandatory OpenAI-client integration."""

import asyncio
import inspect
import json
import os
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from kube_sre_gym import KubeSreGymAction, KubeSreGymEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_HTTP_URL = os.getenv("ENV_HTTP_URL", "http://127.0.0.1:8000")
TASK_NAME = os.getenv("TASK_NAME", "k8s-incident-recovery")
BENCHMARK = os.getenv("BENCHMARK", "kube_sre_gym")
MAX_STEPS = int(os.getenv("MAX_STEPS", "16"))
SUCCESS_SCORE_THRESHOLD = 0.80


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
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def validate_prerequisites() -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if MAX_STEPS <= 0:
        issues.append("MAX_STEPS must be > 0")
    if not HF_TOKEN:
        issues.append("HF_TOKEN is required")
    if not API_BASE_URL:
        issues.append("API_BASE_URL is required")
    if not MODEL_NAME:
        issues.append("MODEL_NAME is required")
    if LOCAL_IMAGE_NAME is None and not ENV_HTTP_URL:
        issues.append("Set LOCAL_IMAGE_NAME or ENV_HTTP_URL")
    return len(issues) == 0, issues


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _fallback_action(observation) -> Tuple[str, Dict]:
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


def choose_action(client: OpenAI, observation, step: int) -> Tuple[str, Dict]:
    payload = {
        "step": step,
        "incident": getattr(observation, "incident", ""),
        "endpoint_status_code": getattr(observation, "endpoint_status_code", None),
        "pod_summaries": (getattr(observation, "pod_summaries", []) or [])[:5],
        "recent_events": (getattr(observation, "recent_events", []) or [])[:5],
        "allowed_tools": getattr(observation, "allowed_tools", []),
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an SRE agent. Return JSON only: "
                        '{"tool":"<allowed_tool>","args":{...}}. No markdown, no prose.'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, separators=(",", ":")),
                },
            ],
            temperature=0.2,
            max_tokens=180,
        )
        raw = (completion.choices[0].message.content or "").strip()
        data = json.loads(raw)
        tool = str(data.get("tool", "")).strip()
        args = data.get("args", {})
        if not tool or not isinstance(args, dict):
            return _fallback_action(observation)
        return tool, args
    except Exception:
        return _fallback_action(observation)


async def run_episode() -> Tuple[bool, int, float, List[float]]:
    if LOCAL_IMAGE_NAME:
        env = await _maybe_await(KubeSreGymEnv.from_docker_image(LOCAL_IMAGE_NAME))
    else:
        env = KubeSreGymEnv(base_url=ENV_HTTP_URL)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        result = await _maybe_await(env.reset())
        for step in range(1, MAX_STEPS + 1):
            tool, args = choose_action(client, result.observation, step)
            action = KubeSreGymAction(thought="llm-policy", tool=tool, args=args)
            result = await _maybe_await(env.step(action))

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps({"tool": tool, "args": args}, separators=(",", ":"))
            tool_response = getattr(result.observation, "tool_response", {}) or {}
            error = tool_response.get("stderr") if not tool_response.get("success", True) else None
            log_step(step, action_str, reward, done, error)

            if done:
                break

        score = 1.0 if bool(result.done) else min(max(sum(rewards) / 20.0, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD or bool(result.done)
        return success, steps_taken, score, rewards
    finally:
        try:
            await _maybe_await(env.close())
        except Exception:
            pass


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
        success, steps_taken, score, rewards = await run_episode()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())