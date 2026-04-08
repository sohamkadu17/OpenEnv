"""Root-level grader re-export for robust import resolution.

Some validators import graders via module:function paths relative to project root.
"""

try:
    from kube_sre_gym.server.graders import easy_grader, medium_grader, hard_grader
except Exception:
    from server.graders import easy_grader, medium_grader, hard_grader  # type: ignore

__all__ = ["easy_grader", "medium_grader", "hard_grader"]
