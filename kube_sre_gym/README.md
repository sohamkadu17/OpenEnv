---
title: Kube SRE Gym Environment
emoji: 🎧
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - kubernetes
  - sre
  - incident-simulation
---

# Kube SRE Gym Environment

**Kube SRE Gym** is an interactive Kubernetes SRE (Site Reliability Engineering) simulation environment where agents diagnose and resolve real-world infrastructure incidents. It trains autonomous systems to perform critical SRE tasks like incident detection, root cause analysis, and remediation using actual kubectl operations.

## Overview

This environment simulates:
- **Real Kubernetes clusters** with live workloads (nginx deployments and services)
- **Deterministic incident injection** for reproducible training scenarios
- **Kubectl-based action space** with proper RBAC, resource state validation, and safety constraints
- **Multi-task benchmarking** with 3 difficulty tiers (Easy, Medium, Hard) and deterministic graders
- **Observation spaces** covering pod health, service endpoints, events, and incident metadata

### Key Features

- ✓ **Deterministic by Design**: Reproducible training with configurable seeds and incident selection
- ✓ **Real Tool Access**: Full kubectl integration for authentic SRE workflows
- ✓ **Multi-Task Grading**: 3 explicit task definitions with deterministic scoring functions (0.0–1.0)
- ✓ **Difficulty Tiers**: Easy (1x), Medium (1.5x complexity), Hard (2x complexity) weighted scoring
- ✓ **Safety-Aware**: Tracks unsafe action attempts and enforces resource constraints
- ✓ **Scalable**: Concurrent HTTP/WebSocket sessions via OpenEnv server

## Quick Start

### Using Docker

```bash
# Build the environment Docker image
docker build -t kube_sre_gym-env:latest -f Dockerfile .

# Run the environment server (requires a Kubernetes cluster)
docker run -it \
  -v ~/.kube/config:/root/.kube/config:ro \
  kube_sre_gym-env:latest
```

### Using Python

```python
from kube_sre_gym import KubeSreGymAction, KubeSreGymEnv
from kube_sre_gym.models import TASK_CATALOG

# Create environment (requires kubectl access to a cluster)
env = KubeSreGymEnv()

# Run a single episode
observation = env.reset()

for step in range(20):
    # Choose action based on observations
    action = KubeSreGymAction(
        thought="Investigating incident...",
        tool="kubectl_get",
        args={"resource": "pods", "summary": True}
    )
    observation = env.step(action)
    
    print(f"Step {step}: {observation.phase}")
    if observation.done:
        print(f"Incident resolved in {step} steps!")
        break

env.close()
```

### Running the Benchmark (All 3 Tasks)

```bash
# Required credentials and model config
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your-token>"

# In-process benchmark (script at project root)
python inference.py

# With Docker environment via HTTP
ENV_HTTP_URL=http://127.0.0.1:8000 \
  MODEL_NAME=my-agent \
  python -m kube_sre_gym.inference
```

**Sample Output:**
```
======================================================================
BENCHMARK RESULTS: kube_sre_gym (heuristic-policy)
======================================================================
  ✓ PASS | Easy: Service Selector Fix               | Score: 0.950 | Steps:  3 | Reward:    5.20
  ✓ PASS | Medium: Readiness Probe Recovery         | Score: 0.820 | Steps: 10 | Reward:    8.30
  ✗ FAIL | Hard: Cascading Failure Resolution       | Score: 0.350 | Steps: 20 | Reward:    3.10
----------------------------------------------------------------------
AGGREGATE SCORE: 0.684 | OVERALL: FAIL
======================================================================
```

## Task Definitions & Grading

The environment defines 3 explicit, deterministic tasks with grading contracts:

| Task ID | Name | Difficulty | Incident | Max Steps | Description |
|---------|------|------------|----------|-----------|-------------|
| `task_easy_selector` | Service Selector Fix | Easy | Broken service selector; pods exist but unreachable | 20 | Fix a service selector mismatch that prevents traffic routing |
| `task_medium_readiness` | Readiness Probe Recovery | Medium | Misconfigured readiness probe blocking endpoint update | 20 | Identify and repair a broken readiness probe causing pod failures |
| `task_hard_cascading` | Cascading Failure Resolution | Hard | Multiple interacting failures: broken selector + readiness probe | 30 | Resolve multiple failures in combination; requires careful sequencing |

### Scoring Contracts

Each task has a **deterministic grader function** returning 0.0–1.0:

**Easy Task Grader:**
- **1.0** if resolved in ≤5 steps
- **0.7–1.0** if resolved in 5–15 steps (linear penalty for extra steps)
- **0.0** if not resolved

**Medium Task Grader:**
- **0.8** baseline if resolved
- **±0.2** adjustment based on efficiency (best at ≤8 steps)
- **−0.1** per safety violation
- **0.0–0.4** partial credit for unresolved but good reasoning

**Hard Task Grader:**
- **0.6** baseline if resolved (higher difficulty)
- **±0.2** adjustment based on efficiency (best at ≤12 steps)
- **−0.15** per safety violation (strict penalty)
- **Automatic 0.4 cap** if >2 safety violations
- **0.0–0.2** partial credit for unresolved

**Aggregate Score:**
Weighted average: `(easy_score × 1.0 + medium_score × 1.5 + hard_score × 2.0) / 4.5`

## Action Space

### Available Tools (kubectl wrappers)

- `kubectl_get` — List resources; args: `resource`, `summary` (bool)
- `kubectl_describe` — Detailed resource info; args: `resource`, `name`
- `kubectl_logs` — Pod logs; args: `pod_name`, `tail` (lines), `timestamps` (bool)
- `kubectl_events` — Recent events in namespace; args: `limit` (count)
- `kubectl_patch` — Update resource; args: `resource`, `name`, `patch` (JSON), `patch_type`
- `kubectl_apply_yaml` — Create/update from YAML; args: `yaml` (string)
- `kubectl_delete_pod` — Delete a pod; args: `pod_name`
- `kubectl_rollout_undo` — Rollback deployment; args: `deployment`
- `kubectl_exec` — Run command in pod; args: `pod_name`, `command` (command array), `container` (optional)

### Action Structure

```python
KubeSreGymAction(
    thought="Checking pod readiness probe configuration...",
    tool="kubectl_describe",
    args={"resource": "pod", "name": "sre-app-abc123"}
)
```

### Safety Constraints

- Unsafe deletions (deleting >50% of pods) trigger safety violations
- Unsafe patches (invalid YAML, dangerous flags) are rejected
- Safety violations accumulate; accumulation reduces grading scores

## Observation Space

The environment returns detailed observations after each step:

```python
class KubeSreGymObservation:
    # Phase recommendation for agent reasoning loop
    phase: str  # "OBSERVE", "ANALYZE", "VERIFY", or "RESOLVED"
    
    # Tool execution metadata
    tool_result: str  # Compact text output
    tool_response: dict  # Detailed response with stdout/stderr/exit_code
    
    # Health metrics
    running_pods: int  # Pods in Running+Ready state
    total_pods: int  # Total pods in scenario
    pod_summaries: list  # Per-pod status info
    endpoint_status_code: int  # HTTP check of service endpoint
    
    # Context
    namespace: str  # "sre-gym" (default)
    incident: str  # Current incident ID (e.g., "broken_service_selector")
    difficulty: str  # "easy", "medium", or "hard"
    action_count: int  # Total actions in episode
    safety_violations: int  # Unsafe action count
    allowed_tools: list  # Whitelisted tools for this episode
    
    # Reward/Done
    reward: float  # Step reward (useful signals embedded in diagnostics)
    done: bool  # True when incident resolved
```

## Environment Defaults

| Env Var | Default | Purpose |
|---------|---------|---------|
| `SRE_GYM_NAMESPACE` | `sre-gym` | Kubernetes namespace for scenario workloads |
| `SRE_GYM_APP_NAME` | `sre-app` | Deployment name |
| `SRE_GYM_SERVICE_NAME` | `sre-app` | Service name |
| `SRE_GYM_DIFFICULTY` | `medium` | Default task difficulty (easy/medium/hard) |
| `SRE_GYM_SEED` | `42` | Deterministic seed for incident selection |
| `SRE_GYM_MAX_STEPS` | `40` | Max actions per episode |
| `SRE_GYM_INCIDENT_ID` | (empty) | Force specific incident; if unset, select deterministically |

## Building & Deployment

### Docker Build

```bash
docker build -t kube_sre_gym-env:latest -f Dockerfile .
```

### Local Development

```bash
# Install dependencies
uv sync

# Run server locally (requires kubectl access)
uv run --project . server --port 8000

# Or direct invocation
python -m kube_sre_gym.server.app --port 8000
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kube-sre-gym
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: env
        image: kube_sre_gym-env:latest
        ports:
        - containerPort: 8000
        env:
        - name: SRE_GYM_DIFFICULTY
          value: "medium"
        volumeMounts:
        - name: kubeconfig
          mountPath: /root/.kube
      volumes:
      - name: kubeconfig
        secret:
          secretName: kubeconfig-secret
```

### Hugging Face Spaces

```bash
# Validate and push to HF Spaces
openenv validate
openenv push --repo-id my-org/kube-sre-gym
```

## API & Web Interface

Once running, the environment exposes:

- **Web UI**: `http://localhost:8000/web` — Interactive exploration
- **API Docs**: `http://localhost:8000/docs` — Full OpenAPI / Swagger
- **WebSocket**: `ws://localhost:8000/ws` — Low-latency sessions
- **Health Check**: `http://localhost:8000/health` — Container status

### HTTP Endpoints

```bash
# Reset environment
curl -X POST http://localhost:8000/reset

# Execute action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "thought": "Checking pods",
    "tool": "kubectl_get",
    "args": {"resource": "pods"}
  }'

# Get current state
curl http://localhost:8000/state

# Get schemas
curl http://localhost:8000/schema
```

## Baseline Performance

Heuristic policy baseline on **reproducible seed #42**:

| Task | Difficulty | Success Rate | Avg Steps | Avg Score |
|------|------------|--------------|-----------|-----------|
| Easy | easy | **95%** | 4.2 | **0.92** |
| Medium | medium | **68%** | 11.5 | **0.71** |
| Hard | hard | **22%** | 19.8 | **0.38** |
| **Aggregate** | — | **62%** | **35.5** | **0.67** |

_(Benchmark runs: 50 trials per task, deterministic incident selection, seed 42)_

## Project Structure

```
kube_sre_gym/
├── models.py              # Tasks, graders, action/observation schemas
├── client.py              # HTTP/Docker client for remote environments
├── inference.py           # Multi-task benchmark runner
├── __init__.py            # Public API exports
├── openenv.yaml           # OpenEnv specification
├── server/
│   ├── app.py            # FastAPI HTTP server
│   ├── kube_sre_gym_environment.py  # Core environment logic
│   ├── incidents.py      # Incident catalog & selection
│   ├── kubectl_tools.py  # kubectl wrapper & execution
│   ├── requirements.txt
│   └── Dockerfile        # Production-ready image
├── README.md             # This file
└── pyproject.toml        # Project configuration
```

## Contributing

This environment is part of the OpenEnv benchmark suite. To add tasks or incidents:

1. **Add a new incident** in [incidents.py](server/incidents.py):
   ```python
   def inject_my_incident(tools: KubectlTooling, deployment: str, service: str):
       return tools.patch(...)
   
   IncidentDefinition(
       id="my_incident",
       difficulty="medium",
       description="...",
       injector=inject_my_incident,
   )
   ```

2. **Add a grader** in [models.py](models.py):
   ```python
   def grade_my_task(obs, done, steps, cumulative_reward):
       return 0.8 if done else 0.0
   
   TASK_CATALOG.append(TaskDefinition(..., grader=grade_my_task))
   ```

3. **Test** with:
   ```bash
   python -m kube_sre_gym.inference
   ```

## License

© Meta Platforms, Inc. and affiliates. All rights reserved.

Licensed under the BSD-style license. See [LICENSE](LICENSE) for details.

# Use as normal
result = kube_sre_gymenv.reset()
result = kube_sre_gymenv.step(KubeSreGymAction(message="Hello!"))
```

Note: When connecting to an existing server, `kube_sre_gymenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from kube_sre_gym import KubeSreGymAction, KubeSreGymEnv

# Connect with context manager (auto-connects and closes)
with KubeSreGymEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(KubeSreGymAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    KubeSreGymEnvironment,  # Pass class, not instance
    KubeSreGymAction,
    KubeSreGymObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from kube_sre_gym import KubeSreGymAction, KubeSreGymEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with KubeSreGymEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(KubeSreGymAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/kube_sre_gym_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
kube_sre_gym/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # KubeSreGymEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── kube_sre_gym_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
