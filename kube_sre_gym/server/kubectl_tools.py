# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kubectl tooling layer with structured responses and compact summaries."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional


@dataclass
class KubectlResult:
    """Structured output for kubectl command execution."""

    ok: bool
    command: List[str]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_ms: int
    parsed: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.ok,
            "ok": self.ok,
            "command": " ".join(self.command),
            "exit_code": self.exit_code,
            "stdout": KubectlTooling._truncate(self.stdout, limit=4000),
            "stderr": KubectlTooling._truncate(self.stderr, limit=2000),
            "elapsed_ms": self.elapsed_ms,
            "parsed": self.parsed or {},
        }


class KubectlTooling:
    """Safe kubectl wrapper constrained to a scenario namespace."""

    def __init__(self, namespace: str, timeout_seconds: int = 45):
        self.namespace = namespace
        self.timeout_seconds = timeout_seconds
        # Auto-enable mock mode on Hugging Face Spaces unless explicitly disabled.
        mock_env = os.getenv("SRE_GYM_MOCK_MODE", "").strip().lower()
        if mock_env in {"1", "true", "yes", "on"}:
            self.mock_mode = True
        elif mock_env in {"0", "false", "no", "off"}:
            self.mock_mode = False
        else:
            self.mock_mode = bool(os.getenv("SPACE_ID"))

        self._mock_state = {
            "selector_ok": True,
            "readiness_ok": True,
            "image_ok": True,
            "crash_loop": False,
        }

    def get(self, resource: str, name: Optional[str] = None, selector: Optional[str] = None, summary: bool = True) -> KubectlResult:
        if resource == "pods" and summary:
            return self.get_pod_summary(selector=selector)

        cmd = ["kubectl", "get", resource, "-n", self.namespace]
        if name:
            cmd.append(name)
        if selector:
            cmd.extend(["-l", selector])
        cmd.extend(["-o", "wide"])
        return self._run(cmd)

    def cluster_info(self) -> KubectlResult:
        return self._run(["kubectl", "cluster-info"])

    def describe(self, resource: str, name: str) -> KubectlResult:
        if not resource or not name:
            return self._error_result("describe requires resource and name")
        return self._run(["kubectl", "describe", resource, name, "-n", self.namespace])

    def logs(self, pod: str, container: Optional[str] = None, tail: int = 200) -> KubectlResult:
        if not pod:
            return self._error_result("logs requires pod name")
        cmd = ["kubectl", "logs", pod, "-n", self.namespace, "--tail", str(tail)]
        if container:
            cmd.extend(["-c", container])
        return self._run(cmd)

    def events(self, limit: int = 20) -> KubectlResult:
        result = self._run(
            [
                "kubectl",
                "get",
                "events",
                "-n",
                self.namespace,
                "--sort-by=.metadata.creationTimestamp",
                "-o",
                "json",
            ]
        )
        if not result.ok:
            return result

        parsed = self._parse_json(result.stdout)
        if parsed is None:
            return self._error_result("failed to parse events response as JSON")
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        compact = []
        for event in items[-limit:]:
            compact.append(
                {
                    "type": event.get("type", ""),
                    "reason": event.get("reason", ""),
                    "object": event.get("involvedObject", {}).get("name", ""),
                    "message": (event.get("message", "") or "")[:180],
                    "count": event.get("count", 1),
                    "last_timestamp": event.get("lastTimestamp") or event.get("eventTime") or "",
                }
            )
        result.parsed = {"events": compact}
        result.stdout = self._truncate(json.dumps(compact, separators=(",", ":"), ensure_ascii=True), limit=5000)
        return result

    def patch(self, resource: str, name: str, patch: str, patch_type: str = "merge") -> KubectlResult:
        if not resource or not name:
            return self._error_result("patch requires resource and name")
        if not patch.strip():
            return self._error_result("patch payload is empty")
        return self._run(
            [
                "kubectl",
                "patch",
                resource,
                name,
                "-n",
                self.namespace,
                "--type",
                patch_type,
                "-p",
                patch,
            ]
        )

    def apply_yaml(self, yaml_text: str) -> KubectlResult:
        if not yaml_text.strip():
            return KubectlResult(
                ok=False,
                command=["kubectl", "apply", "-f", "<empty>"],
                exit_code=1,
                stdout="",
                stderr="yaml payload is empty",
                elapsed_ms=0,
                parsed={},
            )

        with NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            tmp.write(yaml_text)
            manifest_path = Path(tmp.name)

        try:
            return self._run(["kubectl", "apply", "-f", str(manifest_path)])
        finally:
            manifest_path.unlink(missing_ok=True)

    def delete_pod(self, pod: str) -> KubectlResult:
        if not pod:
            return self._error_result("delete_pod requires pod name")
        return self._run(["kubectl", "delete", "pod", pod, "-n", self.namespace, "--wait=false"])

    def rollout_undo(self, deployment: str) -> KubectlResult:
        if not deployment:
            return self._error_result("rollout_undo requires deployment name")
        return self._run(["kubectl", "rollout", "undo", f"deployment/{deployment}", "-n", self.namespace])

    def exec(self, pod: str, command: str) -> KubectlResult:
        if not pod:
            return self._error_result("exec requires pod name")
        if not command.strip():
            return self._error_result("exec requires a command")

        blocked = [
            "--privileged",
            " rm ",
            "rm ",
            ";rm",
            "&&rm",
            "mount ",
            "umount ",
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "mkfs",
            "fdisk",
            "dd if=",
            "chroot",
            "systemctl",
            "rm -rf /",
            "rm -rf /*",
            "chmod 777 /",
            "echo 1 > /proc/sys",
        ]
        lower = command.lower()
        if any(token in lower for token in blocked):
            return KubectlResult(
                ok=False,
                command=["kubectl", "exec", pod, "--", command],
                exit_code=1,
                stdout="",
                stderr="exec command rejected by safety policy",
                elapsed_ms=0,
                parsed={},
            )
        return self._run(["kubectl", "exec", pod, "-n", self.namespace, "--", "sh", "-c", command])

    def get_services_summary(self) -> KubectlResult:
        result = self._run(["kubectl", "get", "svc", "-n", self.namespace, "-o", "json"])
        if not result.ok:
            return result

        parsed = self._parse_json(result.stdout)
        if parsed is None:
            return self._error_result("failed to parse service list response as JSON")

        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        services: List[Dict] = []
        for item in items:
            spec = item.get("spec", {}) or {}
            ports = spec.get("ports", []) or []
            services.append(
                {
                    "name": item.get("metadata", {}).get("name", ""),
                    "type": spec.get("type", "ClusterIP"),
                    "cluster_ip": spec.get("clusterIP", ""),
                    "ports": [p.get("port") for p in ports if p.get("port") is not None],
                }
            )

        result.parsed = {"services": services}
        result.stdout = self._truncate(json.dumps(services, separators=(",", ":"), ensure_ascii=True), limit=5000)
        return result

    def get_pod_summary(self, selector: Optional[str] = None) -> KubectlResult:
        cmd = ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"]
        if selector:
            cmd.extend(["-l", selector])
        result = self._run(cmd)
        if not result.ok:
            return result

        parsed = self._parse_json(result.stdout)
        if parsed is None:
            return self._error_result("failed to parse pod list response as JSON")
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        summaries: List[Dict] = []
        for item in items:
            statuses = item.get("status", {}).get("containerStatuses", []) or []
            restarts = sum(int(c.get("restartCount", 0) or 0) for c in statuses)
            ready = sum(1 for c in statuses if c.get("ready"))
            total = len(statuses)

            reason = item.get("status", {}).get("reason", "")
            if not reason:
                for c in statuses:
                    waiting = (c.get("state", {}) or {}).get("waiting") or {}
                    terminated = (c.get("state", {}) or {}).get("terminated") or {}
                    reason = waiting.get("reason") or terminated.get("reason") or reason

            summaries.append(
                {
                    "name": item.get("metadata", {}).get("name", ""),
                    "phase": item.get("status", {}).get("phase", "Unknown"),
                    "ready": f"{ready}/{total}",
                    "restarts": restarts,
                    "reason": reason or "",
                }
            )

        result.parsed = {"pods": summaries}
        result.stdout = self._truncate(json.dumps(summaries, separators=(",", ":"), ensure_ascii=True), limit=5000)
        return result

    def endpoint_health(self, service_name: str) -> KubectlResult:
        return self._run(
            [
                "kubectl",
                "get",
                "endpoints",
                service_name,
                "-n",
                self.namespace,
                "-o",
                "jsonpath={.subsets[*].addresses[*].ip}",
            ]
        )

    def _run(self, command: List[str]) -> KubectlResult:
        if self.mock_mode:
            return self._run_mock(command)

        start = time.time()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            elapsed_ms = int((time.time() - start) * 1000)
            return KubectlResult(
                ok=proc.returncode == 0,
                command=command,
                exit_code=proc.returncode,
                stdout=(proc.stdout or "").strip(),
                stderr=(proc.stderr or "").strip(),
                elapsed_ms=elapsed_ms,
                parsed={},
            )
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            return KubectlResult(
                ok=False,
                command=command,
                exit_code=124,
                stdout=(exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
                stderr=f"kubectl command timed out after {self.timeout_seconds}s",
                elapsed_ms=elapsed_ms,
                parsed={},
            )
        except Exception as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            return KubectlResult(
                ok=False,
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"kubectl execution error: {exc}",
                elapsed_ms=elapsed_ms,
                parsed={},
            )

    def _run_mock(self, command: List[str]) -> KubectlResult:
        start = time.time()

        def _result(ok: bool, stdout: str = "", stderr: str = "", exit_code: int = 0, parsed: Optional[Dict] = None) -> KubectlResult:
            return KubectlResult(
                ok=ok,
                command=command,
                exit_code=0 if ok else exit_code,
                stdout=stdout,
                stderr=stderr,
                elapsed_ms=int((time.time() - start) * 1000),
                parsed=parsed or {},
            )

        joined = " ".join(command)
        if command[:2] == ["kubectl", "cluster-info"]:
            return _result(True, stdout="Kubernetes control plane is running (mock)")

        if command[:2] == ["kubectl", "apply"]:
            # Applying baseline manifests resets healthy state before incident injection.
            self._mock_state.update(
                {
                    "selector_ok": True,
                    "readiness_ok": True,
                    "image_ok": True,
                    "crash_loop": False,
                }
            )
            return _result(True, stdout="resources configured (mock)")

        if len(command) >= 3 and command[1] == "delete":
            return _result(True, stdout="deleted (mock)")

        if command[:2] == ["kubectl", "rollout"] and len(command) >= 3 and command[2] == "undo":
            self._mock_state.update(
                {
                    "selector_ok": True,
                    "readiness_ok": True,
                    "image_ok": True,
                    "crash_loop": False,
                }
            )
            return _result(True, stdout="deployment rolled back (mock)")

        if command[:2] == ["kubectl", "patch"] and len(command) >= 4:
            resource = command[2]
            patch_payload = command[-1] if command else ""

            if resource == "service":
                if "does-not-exist" in patch_payload:
                    self._mock_state["selector_ok"] = False
                if '"app":"sre-app"' in patch_payload:
                    self._mock_state["selector_ok"] = True

            if resource == "deployment":
                if "invalid-tag" in patch_payload:
                    self._mock_state["image_ok"] = False
                if "/non-existent" in patch_payload:
                    self._mock_state["readiness_ok"] = False
                if "exit 1" in patch_payload:
                    self._mock_state["crash_loop"] = True

                # Best-effort positive remediations.
                if '"image":"nginx:1.27"' in patch_payload:
                    self._mock_state["image_ok"] = True
                if '"path":"/"' in patch_payload or '"path":"/index.html"' in patch_payload:
                    self._mock_state["readiness_ok"] = True
                if "readinessProbe" not in patch_payload and "command" in patch_payload and "exit 1" not in patch_payload:
                    self._mock_state["crash_loop"] = False

            return _result(True, stdout="patched (mock)")

        if command[:3] == ["kubectl", "get", "pods"] and "-o" in command and "json" in command:
            running = self._mock_state["image_ok"] and not self._mock_state["crash_loop"]
            ready = running and self._mock_state["readiness_ok"]

            reason = ""
            phase = "Running"
            if not self._mock_state["image_ok"]:
                reason = "ErrImagePull"
                phase = "Pending"
            elif self._mock_state["crash_loop"]:
                reason = "CrashLoopBackOff"
                phase = "Running"
            elif running and not ready:
                reason = "ReadinessFailed"

            payload = {
                "items": [
                    {
                        "metadata": {"name": "sre-app-7f9d9f8b7d-abcde"},
                        "status": {
                            "phase": phase,
                            "reason": reason,
                            "containerStatuses": [
                                {
                                    "ready": bool(ready),
                                    "restartCount": 2 if self._mock_state["crash_loop"] else 0,
                                    "state": {
                                        "waiting": {"reason": reason} if reason in {"ErrImagePull", "CrashLoopBackOff"} else {},
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
            return _result(True, stdout=json.dumps(payload), parsed=payload)

        if command[:3] == ["kubectl", "get", "events"] and "-o" in command and "json" in command:
            warning = None
            if not self._mock_state["image_ok"]:
                warning = "Failed to pull image"
            elif self._mock_state["crash_loop"]:
                warning = "Back-off restarting failed container"
            elif not self._mock_state["readiness_ok"]:
                warning = "Readiness probe failed"

            items = []
            if warning:
                items.append(
                    {
                        "type": "Warning",
                        "reason": "Unhealthy",
                        "message": warning,
                        "count": 1,
                        "involvedObject": {"name": "sre-app-7f9d9f8b7d-abcde"},
                        "lastTimestamp": "2026-01-01T00:00:00Z",
                    }
                )
            payload = {"items": items}
            return _result(True, stdout=json.dumps(payload), parsed=payload)

        if command[:3] == ["kubectl", "get", "svc"] and "-o" in command and "json" in command:
            payload = {
                "items": [
                    {
                        "metadata": {"name": "sre-app"},
                        "spec": {
                            "type": "ClusterIP",
                            "clusterIP": "10.96.0.25",
                            "ports": [{"port": 80}],
                        },
                    }
                ]
            }
            return _result(True, stdout=json.dumps(payload), parsed=payload)

        if command[:3] == ["kubectl", "get", "endpoints"]:
            endpoint_ok = (
                self._mock_state["selector_ok"]
                and self._mock_state["image_ok"]
                and not self._mock_state["crash_loop"]
                and self._mock_state["readiness_ok"]
            )
            return _result(True, stdout="10.0.0.10" if endpoint_ok else "")

        if command[:2] == ["kubectl", "describe"]:
            return _result(True, stdout=f"describe output (mock): {joined}")

        if command[:2] == ["kubectl", "logs"]:
            return _result(True, stdout="mock logs: application output")

        if command[:2] == ["kubectl", "exec"]:
            return _result(True, stdout="mock exec completed")

        return _result(False, stderr=f"unsupported mock kubectl command: {joined}", exit_code=1)

    @staticmethod
    def _truncate(text: str, limit: int = 8000) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _parse_json(payload: str) -> Optional[Dict]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _error_result(message: str) -> KubectlResult:
        return KubectlResult(
            ok=False,
            command=["kubectl"],
            exit_code=1,
            stdout="",
            stderr=message,
            elapsed_ms=0,
            parsed={},
        )
