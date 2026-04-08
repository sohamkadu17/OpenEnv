"""Inference runner for the kube_sre_gym environment with multi-task benchmarking."""

import asyncio
import inspect
import json
import os
from typing import Dict, List, Optional, Tuple

from kube_sre_gym import KubeSreGymAction, KubeSreGymEnv
from kube_sre_gym.server.kube_sre_gym_environment import KubeSreGymEnvironment
from kube_sre_gym.models import TASK_CATALOG, TaskDefinition

IMAGE_NAME = os.getenv("IMAGE_NAME")
ENV_HTTP_URL = os.getenv("ENV_HTTP_URL", "http://127.0.0.1:8000")
TASK_NAME = os.getenv("TASK_NAME", "k8s-incident-recovery")
BENCHMARK = os.getenv("BENCHMARK", "kube_sre_gym")
MODEL_NAME = os.getenv("MODEL_NAME", "heuristic-policy")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
MAX_STEPS_HARD = int(os.getenv("MAX_STEPS_HARD", "30"))  # Harder tasks get more steps


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_task_start(task_id: str, task_name: str, difficulty: str) -> None:
    print(f"[TASK_START] task_id={task_id} task_name={task_name} difficulty={difficulty}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_task_end(task_id: str, success: bool, steps: int, graded_score: float, cumulative_reward: float) -> None:
    print(
        f"[TASK_END] task_id={task_id} success={str(success).lower()} steps={steps} graded_score={graded_score:.3f} cumulative_reward={cumulative_reward:.2f}",
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


async def run_task_episode(
    task_def: TaskDefinition,
    max_steps: int,
    use_client: bool = False,
) -> Tuple[bool, int, float, float]:
    """
    Run a single task episode and return (success, steps, graded_score, cumulative_reward).
    
    graded_score: 0.0-1.0 from the task's grader function
    cumulative_reward: sum of all step rewards
    """
    log_task_start(task_def.task_id, task_def.name, task_def.difficulty)
    
    rewards: List[float] = []
    steps_taken = 0
    done = False
    final_obs = None
    safety_violations = 0

    try:
        if use_client:
            if IMAGE_NAME:
                env = await _maybe_await(KubeSreGymEnv.from_docker_image(IMAGE_NAME))
            else:
                env = KubeSreGymEnv(base_url=ENV_HTTP_URL)
        else:
            env = KubeSreGymEnvironment()

        try:
            result = await _maybe_await(env.reset()) if use_client else env.reset()
            final_obs = result if not use_client else result.observation
            
            for step in range(1, max_steps + 1):
                tool, args = choose_action(final_obs if not use_client else final_obs)
                action = KubeSreGymAction(thought="policy-step", tool=tool, args=args)
                
                if use_client:
                    result = await _maybe_await(env.step(action))
                    final_obs = result.observation
                    done = result.done
                else:
                    final_obs = env.step(action)
                    done = getattr(final_obs, "done", False)

                reward = float(getattr(final_obs, "reward", 0.0) or 0.0)
                rewards.append(reward)
                steps_taken = step
                safety_violations = getattr(final_obs, "safety_violations", 0)

                action_str = json.dumps({"tool": tool, "args": args}, separators=(",", ":"))
                tool_response = getattr(final_obs, "tool_response", {}) or {}
                error = tool_response.get("stderr") if not tool_response.get("success", True) else None
                log_step(step, action_str, reward, done, error)

                if done:
                    break
        finally:
            if use_client:
                try:
                    await _maybe_await(env.close())
                except Exception:
                    pass

        cumulative_reward = sum(rewards)
        graded_score = task_def.grader(final_obs, done, steps_taken, cumulative_reward)
        success = graded_score >= 0.7  # Success threshold
        
        log_task_end(task_def.task_id, success, steps_taken, graded_score, cumulative_reward)
        return success, steps_taken, graded_score, cumulative_reward
        
    except Exception as e:
        print(f"[ERROR] Task {task_def.task_id} failed with exception: {e}", flush=True)
        log_task_end(task_def.task_id, False, steps_taken, 0.0, sum(rewards))
        return False, steps_taken, 0.0, sum(rewards)


async def run_all_tasks(use_client: bool = False) -> Tuple[List[Dict], float, bool]:
    """
    Run all 3 tasks (easy, medium, hard) and return per-task results + aggregate score.
    
    Returns: (per_task_results, aggregate_score, overall_success)
    """
    per_task_results = []
    aggregate_scores = []
    
    for task_def in TASK_CATALOG:
        # Adjust max steps based on difficulty
        max_steps = MAX_STEPS if task_def.difficulty != "hard" else MAX_STEPS_HARD
        
        success, steps, graded_score, cumulative_reward = await run_task_episode(
            task_def, 
            max_steps,
            use_client=use_client,
        )
        
        result = {
            "task_id": task_def.task_id,
            "name": task_def.name,
            "difficulty": task_def.difficulty,
            "success": success,
            "steps": steps,
            "graded_score": graded_score,
            "cumulative_reward": cumulative_reward,
        }
        per_task_results.append(result)
        aggregate_scores.append(graded_score)
    
    # Aggregate: weighted by difficulty (easy: 1x, medium: 1.5x, hard: 2x)
    weights = {
        "easy": 1.0,
        "medium": 1.5,
        "hard": 2.0,
    }
    weighted_sum = sum(
        result["graded_score"] * weights[result["difficulty"]]
        for result in per_task_results
    )
    total_weight = sum(weights[result["difficulty"]] for result in per_task_results)
    aggregate_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    overall_success = all(result["success"] for result in per_task_results)
    
    return per_task_results, aggregate_score, overall_success


def run_inprocess_fallback() -> Tuple[bool, int, float, List[float]]:
    """Legacy single-task in-process runner for compatibility."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        per_task_results, aggregate_score, overall_success = loop.run_until_complete(
            run_all_tasks(use_client=False)
        )
        total_steps = sum(r["steps"] for r in per_task_results)
        total_rewards = []
        for r in per_task_results:
            total_rewards.extend([r["cumulative_reward"] / max(1, r["steps"])] * r["steps"])
        return overall_success, total_steps, aggregate_score, total_rewards
    finally:
        loop.close()


async def main() -> None:
    ok, issues = validate_prerequisites()
    if not ok:
        raise RuntimeError("; ".join(issues))

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Try client mode first (Docker/HTTP)
        per_task_results, aggregate_score, overall_success = await run_all_tasks(use_client=True)
    except Exception as e:
        print(f"[INFO] Client mode failed ({e}), falling back to in-process", flush=True)
        overall_success, _, aggregate_score, _ = run_inprocess_fallback()
        per_task_results = []  # Not populated in fallback for now
    
    # Print summary
    print("\n" + "="*70, flush=True)
    print(f"BENCHMARK RESULTS: {BENCHMARK} ({MODEL_NAME})", flush=True)
    print("="*70, flush=True)
    
    if per_task_results:
        for result in per_task_results:
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            print(
                f"  {status} | {result['name']:40s} | "
                f"Score: {result['graded_score']:.3f} | Steps: {result['steps']:2d} | "
                f"Reward: {result['cumulative_reward']:7.2f}",
                flush=True,
            )
    
    print("-"*70, flush=True)
    print(
        f"AGGREGATE SCORE: {aggregate_score:.3f} | "
        f"OVERALL: {'PASS' if overall_success else 'FAIL'}",
        flush=True,
    )
    print("="*70 + "\n", flush=True)
    
    log_end(success=overall_success, steps=0, score=aggregate_score, rewards=[])


if __name__ == "__main__":
    asyncio.run(main())