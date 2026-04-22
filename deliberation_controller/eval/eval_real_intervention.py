"""Offline simulation of real intervention on trajectory replay."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping

from deliberation_controller.data.prepare_dataset import SplitConfig, is_success_trajectory, split_trajectories
from deliberation_controller.intervene.agent_wrapper import AgentWithController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate controller with offline intervention replay.")
    parser.add_argument("--trajectories_path", required=True, help="Path to full trajectories JSON.")
    parser.add_argument("--controller_path", required=True, help="Path to best_controller.pt.")
    parser.add_argument("--reference_dist_path", required=True, help="Path to reference distribution JSON.")
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--output_path", required=True, help="Path to save evaluation result JSON.")
    parser.add_argument(
        "--use_all_as_test",
        action="store_true",
        help="Use all trajectories as the test set (no train/val/test split).",
    )
    parser.add_argument("--dataset_name", default="", help="Optional dataset name stored in output JSON.")
    return parser.parse_args()


def load_trajectories(path: str) -> List[Mapping[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "steps" in data:
            return [data]
        return list(data.values())
    raise ValueError(f"Unsupported trajectory format: {type(data)}")


def normalize_text(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`“”‘’\[\]\(\)\{\}\.,:;!?]", "", s)
    return s


def to_answer_list(ground_truth: object) -> List[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, list):
        return [str(x).strip() for x in ground_truth if str(x).strip()]
    text = str(ground_truth).strip()
    if not text:
        return []
    candidates = [text]
    if "|" in text:
        candidates.extend([x.strip() for x in text.split("|") if x.strip()])
    return list(dict.fromkeys(candidates))


def answer_match(predicted: str, ground_truth: object) -> bool:
    pred = normalize_text(predicted or "")
    if not pred:
        return False
    gt_list = [normalize_text(x) for x in to_answer_list(ground_truth)]
    for gt in gt_list:
        if not gt:
            continue
        if gt in pred or pred in gt:
            return True
    return False


def step_tokens(step: Mapping[str, object]) -> float:
    return float(step.get("tokens_input", 0) or 0) + float(step.get("tokens_output", 0) or 0)


def trajectory_total_tokens(traj: Mapping[str, object]) -> float:
    total = float(traj.get("total_input_tokens", 0) or 0) + float(traj.get("total_output_tokens", 0) or 0)
    if total > 0:
        return total
    steps = traj.get("steps", [])
    if isinstance(steps, list):
        return sum(step_tokens(s) for s in steps)
    return 0.0


def run_offline_intervention_eval(
    trajectories_path: str,
    controller_path: str,
    reference_dist_path: str,
    gate_threshold: float,
    output_path: str,
    use_all_as_test: bool = False,
    dataset_name: str = "",
) -> Dict[str, object]:
    trajectories = load_trajectories(trajectories_path)
    if use_all_as_test:
        test_trajs = trajectories
    else:
        split = split_trajectories(trajectories, SplitConfig(seed=42))
        test_trajs = split["test"]

    wrapper = AgentWithController(
        controller_path=controller_path,
        reference_dist_path=reference_dist_path,
        agent_config={
            "model_type": "thinking",
            "gate_threshold": gate_threshold,
            "window_size": 5,
            "device": "cpu",
        },
    )

    no_control_success = 0
    no_control_tokens = 0.0
    controller_success = 0
    controller_tokens = 0.0
    intervention_distribution: Counter[str] = Counter()
    per_trajectory_log: List[Dict[str, object]] = []

    for traj in test_trajs:
        wrapper.reset_episode()
        steps = traj.get("steps", [])
        if not isinstance(steps, list):
            steps = []

        gt = traj.get("ground_truth", "")
        traj_no_control_success = bool(is_success_trajectory(traj))
        traj_no_control_token = trajectory_total_tokens(traj)
        no_control_success += int(traj_no_control_success)
        no_control_tokens += traj_no_control_token

        stop_triggered = False
        stop_step = -1
        extracted_answer = ""
        traj_controller_success = traj_no_control_success
        token_until_stop = 0.0

        per_step_actions: List[Dict[str, object]] = []
        running_token = 0.0
        for idx, step in enumerate(steps):
            running_token += step_tokens(step)
            result = wrapper.process_step(step)
            per_step_actions.append(
                {
                    "step_idx": idx,
                    "action": result.get("action"),
                    "intervention_log": result.get("intervention_log", ""),
                }
            )
            if result.get("action") == "stop":
                stop_triggered = True
                stop_step = idx
                extracted_answer = str(result.get("extracted_answer") or "")
                token_until_stop = running_token
                traj_controller_success = answer_match(extracted_answer, gt)
                break

        summary = wrapper.get_intervention_summary()
        intervention_distribution.update(summary["intervention_distribution"])
        estimated_saving = float(summary.get("estimated_token_saving", 0.0))

        base_token = token_until_stop if stop_triggered else traj_no_control_token
        traj_controller_token = max(0.0, base_token - estimated_saving)
        controller_tokens += traj_controller_token
        controller_success += int(traj_controller_success)

        per_trajectory_log.append(
            {
                "task_id": traj.get("task_id"),
                "ground_truth": gt,
                "no_control_success": traj_no_control_success,
                "controller_success": traj_controller_success,
                "no_control_token": traj_no_control_token,
                "controller_token": traj_controller_token,
                "stop_triggered": stop_triggered,
                "stop_step": stop_step,
                "extracted_answer": extracted_answer,
                "answer_match": answer_match(extracted_answer, gt) if stop_triggered else None,
                "estimated_token_saving": estimated_saving,
                "intervention_summary": summary,
                "per_step_actions": per_step_actions,
            }
        )

    n = max(len(test_trajs), 1)
    no_control_success_rate = no_control_success / n
    controller_success_rate = controller_success / n
    no_control_avg_token = no_control_tokens / n
    controller_avg_token = controller_tokens / n
    token_saving_rate = (no_control_tokens - controller_tokens) / max(no_control_tokens, 1e-8)

    results = {
        "dataset_name": dataset_name,
        "trajectories_path": trajectories_path,
        "num_test_trajectories": len(test_trajs),
        "no_control": {
            "success_rate": no_control_success_rate,
            "avg_token": no_control_avg_token,
            "total_token": no_control_tokens,
        },
        "controller": {
            "success_rate": controller_success_rate,
            "avg_token": controller_avg_token,
            "total_token": controller_tokens,
        },
        "token_saving_rate": token_saving_rate,
        "intervention_distribution": dict(intervention_distribution),
        "per_trajectory_log": per_trajectory_log,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def main() -> None:
    args = parse_args()
    results = run_offline_intervention_eval(
        trajectories_path=args.trajectories_path,
        controller_path=args.controller_path,
        reference_dist_path=args.reference_dist_path,
        gate_threshold=args.gate_threshold,
        output_path=args.output_path,
        use_all_as_test=args.use_all_as_test,
        dataset_name=args.dataset_name,
    )
    print("Offline intervention evaluation completed.")
    print(f"  dataset:                   {results.get('dataset_name', '')}")
    print(f"  test trajectories:         {results['num_test_trajectories']}")
    print(f"  no_control success_rate:   {results['no_control']['success_rate']:.4f}")
    print(f"  controller success_rate:   {results['controller']['success_rate']:.4f}")
    print(f"  no_control avg_token:      {results['no_control']['avg_token']:.2f}")
    print(f"  controller avg_token:      {results['controller']['avg_token']:.2f}")
    print(f"  token_saving_rate:         {results['token_saving_rate']:.4f}")
    print(f"  intervention_distribution: {results['intervention_distribution']}")
    print(f"  saved to:                  {args.output_path}")


if __name__ == "__main__":
    main()

