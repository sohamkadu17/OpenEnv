# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kube Sre Gym environment server components."""

# Lazy imports to avoid requiring openenv at import time
def __getattr__(name):
    if name == "KubeSreGymEnvironment":
        from .kube_sre_gym_environment import KubeSreGymEnvironment
        return KubeSreGymEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["KubeSreGymEnvironment"]
