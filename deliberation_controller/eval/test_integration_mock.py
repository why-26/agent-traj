"""Mock integration test for controller bridge + process_step contract."""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping


ADAPTIVE_SCRIPTS_DIR = "/data/wanghy/adaptive_deliberation_controller_eval/scripts"
AGENT_TRAJ_PROJECT_ROOT = "/data/wanghy/agent_traj"
DEFAULT_CONTROLLER_PATH = "/data/wanghy/agent_traj/deliberation_controller/checkpoints/best_controller.pt"
DEFAULT_REF_DIST_PATH = (
    "/data/wanghy/agent_traj/deliberation_controller/data/ref_dist_hotpotqa_qwen3.json"
)
FALLBACK_REF_DIST_PATH = (
    "/data/wanghy/agent_traj/deliberation_controller/data/reference_distribution_hotpotqa_qwen3_full.json"
)
LOG_PATH = "/data/wanghy/agent_traj/deliberation_controller/eval/test_integration_mock.log"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_import() -> None:
    if AGENT_TRAJ_PROJECT_ROOT not in sys.path:
        sys.path.insert(0, AGENT_TRAJ_PROJECT_ROOT)
    if ADAPTIVE_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, ADAPTIVE_SCRIPTS_DIR)


def build_mock_steps() -> List[Dict[str, object]]:
    # Thought length gradually increases; repeated search query simulates failure loop.
    return [
        {
            "step_id": 0,
            "thought": "先搜关键实体。",
            "action_type": "search",
            "action_input": {"query": "hotpotqa sample entity"},
            "observation": '[{"title":"A","snippet":"weak hit"}]',
            "tokens_input": 420,
            "tokens_output": 180,
        },
        {
            "step_id": 1,
            "thought": "结果不够，我再搜同一个关键词并尝试补充。为了确保不遗漏，我会重复验证。",
            "action_type": "search",
            "action_input": {"query": "hotpotqa sample entity"},
            "observation": "[]",
            "tokens_input": 560,
            "tokens_output": 240,
        },
        {
            "step_id": 2,
            "thought": "仍然没有关键信息。我继续重复同一搜索，逐条比对上下文，并记录失败证据以便下一步调整。",
            "action_type": "search",
            "action_input": {"query": "hotpotqa sample entity"},
            "observation": "[]",
            "tokens_input": 700,
            "tokens_output": 310,
        },
        {
            "step_id": 3,
            "thought": "我再查一次同样的词，并扩展推理链，尝试从边角信息拼出答案，虽然目前证据依旧稀疏。",
            "action_type": "search",
            "action_input": {"query": "hotpotqa sample entity"},
            "observation": "",
            "tokens_input": 860,
            "tokens_output": 370,
        },
        {
            "step_id": 4,
            "thought": (
                "最后整理：如果仍无新证据，我会给出当前最可能结论并停止无效重复。"
                "我将基于已有线索生成一个候选答案。"
            ),
            "action_type": "respond",
            "action_input": {"content": "Candidate answer placeholder"},
            "observation": "",
            "tokens_input": 940,
            "tokens_output": 420,
        },
    ]


def print_controller_path_info(controller_obj: object) -> None:
    cls = controller_obj.__class__
    cls_name = cls.__name__
    module_name = cls.__module__
    if cls_name == "_Adapter":
        route = "fallback(agent_wrapper)"
    elif "agent_controller" in module_name:
        route = "main(agent_controller)"
    else:
        route = f"unknown({module_name}.{cls_name})"
    log(f"Controller route detected: {route}")
    log(f"Controller concrete type: {module_name}.{cls_name}")


def required_step_fields() -> List[str]:
    return [
        "step_id",
        "thought",
        "action_type",
        "action_input",
        "observation",
        "tokens_input",
        "tokens_output",
    ]


def main() -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")  # truncate old log

    ensure_import()
    from controller_integration import apply_controller_result, create_controller

    ref_dist_path = DEFAULT_REF_DIST_PATH
    if not os.path.exists(ref_dist_path):
        log(f"Given ref_dist path not found: {ref_dist_path}")
        log(f"Fallback to existing file: {FALLBACK_REF_DIST_PATH}")
        ref_dist_path = FALLBACK_REF_DIST_PATH

    log("Creating controller ...")
    controller = create_controller(
        controller_path=DEFAULT_CONTROLLER_PATH,
        ref_dist_path=ref_dist_path,
        gate_threshold=0.5,
        k=5,
        signal_dim=5,
    )
    print_controller_path_info(controller)

    mock_steps = build_mock_steps()
    mock_history_messages: List[Mapping[str, str]] = [
        {"role": "system", "content": "You are a helpful QA agent."},
        {"role": "user", "content": "Mock HotpotQA question"},
    ]

    log("Running process_step on 5 mock steps ...")
    for step in mock_steps:
        log(f"--- step_id={step['step_id']} ---")
        try:
            result = controller.process_step(step)
            log(f"process_step result.action = {result.get('action')}")
            log(f"process_step result.modified_prompt = {repr(result.get('modified_prompt'))[:300]}")
            log(f"process_step result.extracted_answer = {repr(result.get('extracted_answer'))}")

            # Also test apply_controller_result compatibility
            try:
                new_history, stop_answer, action_name = apply_controller_result(result, mock_history_messages)
                mock_history_messages = new_history
                log(
                    "apply_controller_result OK: "
                    f"action={action_name}, stop_answer={repr(stop_answer)}, history_len={len(new_history)}"
                )
            except Exception:
                log("apply_controller_result ERROR")
                log(traceback.format_exc())

        except KeyError as e:
            log(f"process_step KeyError: {e}")
            log(f"step_data keys provided: {sorted(step.keys())}")
            log(f"expected step_data keys: {required_step_fields()}")
            log(traceback.format_exc())
        except Exception:
            log("process_step ERROR")
            log(f"step_data keys provided: {sorted(step.keys())}")
            log(f"expected step_data keys: {required_step_fields()}")
            log(traceback.format_exc())

    log(f"Done. Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
