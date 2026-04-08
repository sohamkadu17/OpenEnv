# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Kube Sre Gym environment server."""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# Import environment components
try:
    from ..models import KubeSreGymAction, KubeSreGymObservation
    from .custom_web_ui import build_custom_gradio_ui
    from .kube_sre_gym_environment import KubeSreGymEnvironment
except (ImportError, ValueError):
    from models import KubeSreGymAction, KubeSreGymObservation
    try:
        from custom_web_ui import build_custom_gradio_ui
        from kube_sre_gym_environment import KubeSreGymEnvironment
    except ImportError:
        from server.custom_web_ui import build_custom_gradio_ui
        from server.kube_sre_gym_environment import KubeSreGymEnvironment

# Enable web UI by default for local development while preserving explicit overrides.
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

# Create the app with web interface and README integration
app = create_app(
    KubeSreGymEnvironment,
    KubeSreGymAction,
    KubeSreGymObservation,
    env_name="kube_sre_gym",
    max_concurrent_envs=1,
    gradio_builder=build_custom_gradio_ui,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
