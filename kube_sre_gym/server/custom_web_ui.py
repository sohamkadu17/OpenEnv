"""Custom Gradio UI builder for the Kube SRE Gym environment."""

import json
from typing import Any, Dict, Tuple

import gradio as gr


TOOL_PRESETS: Dict[str, Dict[str, str]] = {
    "Inspect Pods": {
        "thought": "Inspect pod health first",
        "tool": "kubectl_get",
        "args": '{"resource":"pods","summary":true}',
    },
    "Inspect Events": {
        "thought": "Inspect recent warning events",
        "tool": "kubectl_events",
        "args": '{"limit":10}',
    },
    "Fix Service Selector": {
        "thought": "Restore service selector to app=sre-app",
        "tool": "kubectl_patch",
        "args": '{"resource":"service","name":"sre-app","patch":"{\\"spec\\":{\\"selector\\":{\\"app\\":\\"sre-app\\"}}}"}',
    },
    "Rollback Deployment": {
        "thought": "Rollback deployment to previous revision",
        "tool": "kubectl_rollout_undo",
        "args": '{"deployment":"sre-app"}',
    },
}


def _to_json_text(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=True)
    except Exception:
        return str(value)


def _safe_parse_args(args_text: str) -> Dict[str, Any]:
    if not args_text or not args_text.strip():
        return {}
    parsed = json.loads(args_text)
    if not isinstance(parsed, dict):
        raise ValueError("Args JSON must decode to an object")
    return parsed


def _result_to_outputs(result: Dict[str, Any], action_name: str) -> Tuple[str, str, str, str]:
    reward = result.get("reward")
    done = bool(result.get("done", False))
    observation = result.get("observation", {})
    status = f"{action_name} complete." if not done else f"{action_name} complete. Episode resolved."
    return status, str(reward), str(done), _to_json_text(observation)


def build_custom_gradio_ui(
    web_manager: Any,
    action_fields: Any,
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    del action_fields, metadata, is_chat_env, quick_start_md

    with gr.Blocks() as demo:
        gr.Markdown("## Kube SRE Mission Control")
        gr.Markdown(
            "Run guided recovery actions with validated JSON args and quick action presets."
        )

        with gr.Row():
            with gr.Column(scale=5):
                thought = gr.Textbox(
                    label="Thought",
                    placeholder="Explain what you are trying to verify or fix",
                )
                tool = gr.Dropdown(
                    label="Tool",
                    choices=sorted({v["tool"] for v in TOOL_PRESETS.values()}),
                    value="kubectl_get",
                    allow_custom_value=True,
                )
                args = gr.Code(
                    label="Args (JSON object)",
                    language="json",
                    value='{"resource":"pods","summary":true}',
                )

                with gr.Row():
                    step_button = gr.Button("Execute Step", variant="primary")
                    reset_button = gr.Button("Reset Episode")
                    state_button = gr.Button("Get State")

                gr.Markdown("### Quick Actions")
                with gr.Row():
                    preset_pods = gr.Button("Inspect Pods")
                    preset_events = gr.Button("Inspect Events")
                with gr.Row():
                    preset_selector = gr.Button("Fix Service Selector")
                    preset_undo = gr.Button("Rollback Deployment")

            with gr.Column(scale=6):
                status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                with gr.Row():
                    reward_text = gr.Textbox(label="Reward", value="-", interactive=False)
                    done_text = gr.Textbox(label="Done", value="False", interactive=False)

                observation_json = gr.Code(label="Observation", language="json", value="{}")
                state_json = gr.Code(label="Environment State", language="json", value="{}")

        async def on_reset() -> Tuple[str, str, str, str, str]:
            result = await web_manager.reset_environment({})
            status, reward, done, obs_json = _result_to_outputs(result, "Reset")
            state = web_manager.get_state()
            return status, reward, done, obs_json, _to_json_text(state)

        async def on_step(thought_value: str, tool_value: str, args_value: str) -> Tuple[str, str, str, str, str]:
            try:
                payload = {
                    "thought": thought_value or "",
                    "tool": (tool_value or "").strip(),
                    "args": _safe_parse_args(args_value),
                }
                result = await web_manager.step_environment(payload)
                status, reward, done, obs_json = _result_to_outputs(result, "Step")
                state = web_manager.get_state()
                return status, reward, done, obs_json, _to_json_text(state)
            except Exception as exc:
                state = web_manager.get_state()
                return f"Error: {exc}", "-", "False", "{}", _to_json_text(state)

        def on_get_state() -> str:
            return _to_json_text(web_manager.get_state())

        def use_preset(name: str) -> Tuple[str, str, str]:
            preset = TOOL_PRESETS[name]
            return preset["thought"], preset["tool"], preset["args"]

        reset_button.click(
            fn=on_reset,
            outputs=[status_text, reward_text, done_text, observation_json, state_json],
        )
        step_button.click(
            fn=on_step,
            inputs=[thought, tool, args],
            outputs=[status_text, reward_text, done_text, observation_json, state_json],
        )
        state_button.click(fn=on_get_state, outputs=[state_json])

        preset_pods.click(
            fn=lambda: use_preset("Inspect Pods"),
            outputs=[thought, tool, args],
        )
        preset_events.click(
            fn=lambda: use_preset("Inspect Events"),
            outputs=[thought, tool, args],
        )
        preset_selector.click(
            fn=lambda: use_preset("Fix Service Selector"),
            outputs=[thought, tool, args],
        )
        preset_undo.click(
            fn=lambda: use_preset("Rollback Deployment"),
            outputs=[thought, tool, args],
        )

    demo.title = f"{title} - Mission Control"
    return demo
