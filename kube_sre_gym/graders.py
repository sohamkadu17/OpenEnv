"""Root-level grader re-export for robust import resolution.

Some validators import graders via module:function paths relative to project root.
Expose both difficulty-based and task-specific grader names.
"""

try:
    from kube_sre_gym.tasks import (
        task_fix_broken_service_selector_grader,
        task_recover_crashloopbackoff_pod_grader,
        task_resolve_oomkilled_pod_grader,
    )
except Exception:
    from tasks import (  # type: ignore
        task_fix_broken_service_selector_grader,
        task_recover_crashloopbackoff_pod_grader,
        task_resolve_oomkilled_pod_grader,
    )


def easy_grader(trajectory=None):
    return task_fix_broken_service_selector_grader(trajectory, None)


def medium_grader(trajectory=None):
    return task_recover_crashloopbackoff_pod_grader(trajectory, None)


def hard_grader(trajectory=None):
    return task_resolve_oomkilled_pod_grader(trajectory, None)


__all__ = [
    "easy_grader",
    "medium_grader",
    "hard_grader",
    "task_fix_broken_service_selector_grader",
    "task_recover_crashloopbackoff_pod_grader",
    "task_resolve_oomkilled_pod_grader",
]
