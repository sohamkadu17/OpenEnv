# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Kube Sre Gym Environment.
Usage: python -m server.app
"""

from openenv.core.env_server.http_server import create_app

# Import environment components
try:
    from ..models import KubeSreGymAction, KubeSreGymObservation
    from .kube_sre_gym_environment import KubeSreGymEnvironment
except (ImportError, ValueError):
    from models import KubeSreGymAction, KubeSreGymObservation
    try:
        from kube_sre_gym_environment import KubeSreGymEnvironment
    except ImportError:
        from server.kube_sre_gym_environment import KubeSreGymEnvironment

# Create FastAPI app with web interface
app = create_app(
    KubeSreGymEnvironment,
    KubeSreGymAction,
    KubeSreGymObservation,
    env_name="kube_sre_gym",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kube SRE Gym Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
