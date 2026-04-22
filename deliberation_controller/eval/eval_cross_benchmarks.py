"""Batch runner for cross-domain intervention evaluation with local normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from deliberation_controller.data.normalizer import (
    build_reference_distribution,
    load_trajectories,
    save_reference_distribution,
)
from deliberation_controller.eval.eval_real_intervention import run_offline_intervention_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-domain intervention evaluations using per-dataset local reference distributions."
    )
    parser.add_argument("--controller_path", required=True)
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", default="/data/wanghy/agent_traj/deliberation_controller/eval")

    # 7 cross-domain groups (Layer-2 + Layer-3)
    parser.add_argument("--hotpotqa_gpt41_path", required=True)
    parser.add_argument("--tau_retail_qwen_path", required=True)
    parser.add_argument("--tau_retail_gpt41_path", required=True)
    parser.add_argument("--tau_airline_qwen_path", required=True)
    parser.add_argument("--tau_airline_gpt41_path", required=True)
    parser.add_argument("--bamboogle_qwen_path", required=True)
    parser.add_argument("--bamboogle_gpt41_path", required=True)
    return parser.parse_args()


def compact_result(dataset_name: str, path: str, result: Dict[str, object]) -> Dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "result_path": path,
        "num_test_trajectories": result["num_test_trajectories"],
        "no_control_success_rate": result["no_control"]["success_rate"],
        "controller_success_rate": result["controller"]["success_rate"],
        "no_control_avg_token": result["no_control"]["avg_token"],
        "controller_avg_token": result["controller"]["avg_token"],
        "token_saving_rate": result["token_saving_rate"],
        "intervention_distribution": result["intervention_distribution"],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, str]] = [
        # Layer 2 (cross model)
        {"name": "hotpotqa_gpt4_1", "path": args.hotpotqa_gpt41_path},
        # Layer 3 (cross benchmark)
        {"name": "tau_retail_qwen3_4b", "path": args.tau_retail_qwen_path},
        {"name": "tau_retail_gpt4_1", "path": args.tau_retail_gpt41_path},
        {"name": "tau_airline_qwen3_4b", "path": args.tau_airline_qwen_path},
        {"name": "tau_airline_gpt4_1", "path": args.tau_airline_gpt41_path},
        {"name": "bamboogle_qwen3_4b", "path": args.bamboogle_qwen_path},
        {"name": "bamboogle_gpt4_1", "path": args.bamboogle_gpt41_path},
    ]

    summary_rows: List[Dict[str, object]] = []
    for job in jobs:
        dataset_name = job["name"]
        output_path = out_dir / f"results_cross_{dataset_name}_local_ref.json"
        local_ref_path = out_dir / f"ref_dist_{dataset_name}_local_ref.json"
        print(f"\n=== Running {dataset_name} (local ref) ===")

        trajectories = load_trajectories(job["path"])
        local_ref_dist = build_reference_distribution(trajectories)
        save_reference_distribution(local_ref_dist, local_ref_path)

        result = run_offline_intervention_eval(
            trajectories_path=job["path"],
            controller_path=args.controller_path,
            reference_dist_path=str(local_ref_path),
            gate_threshold=args.gate_threshold,
            output_path=str(output_path),
            use_all_as_test=True,
            dataset_name=dataset_name,
        )
        row = compact_result(dataset_name, str(output_path), result)
        summary_rows.append(row)
        print(
            f"done {dataset_name}: "
            f"no_control_sr={row['no_control_success_rate']:.4f}, "
            f"controller_sr={row['controller_success_rate']:.4f}, "
            f"token_saving={row['token_saving_rate']:.4f}, "
            f"local_ref={local_ref_path.name}"
        )

    summary = {
        "controller_path": args.controller_path,
        "reference_mode": "per_dataset_local_reference_distribution",
        "gate_threshold": args.gate_threshold,
        "num_experiments": len(summary_rows),
        "results": summary_rows,
    }
    summary_path = out_dir / "results_summary_local_ref.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nAll experiments finished.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
