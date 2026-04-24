"""Threshold sweep runner for Pareto analysis (success vs token saving)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from deliberation_controller.eval.eval_real_intervention import run_offline_intervention_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gate-threshold sweep on HotpotQA and TAU-bench Retail."
    )
    parser.add_argument("--controller_path", required=True, help="Path to best_controller.pt")
    parser.add_argument("--reference_dist_path", required=True, help="Path to reference distribution JSON")
    parser.add_argument("--hotpot_path", required=True, help="Path to HotpotQA trajectories JSON")
    parser.add_argument("--tau_retail_path", required=True, help="Path to TAU-bench Retail trajectories JSON")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
        help="Gate thresholds to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/wanghy/agent_traj/deliberation_controller/eval",
        help="Directory to save sweep results.",
    )
    return parser.parse_args()


def row_to_markdown(row: Dict[str, object]) -> str:
    return (
        f"| {row['dataset']} | {row['threshold']:.1f} | {row['num_test_trajectories']} | "
        f"{row['controller_success_rate']:.4f} | {row['token_saving_rate']:.4f} |"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "name": "HotpotQA",
            "path": args.hotpot_path,
            "use_all_as_test": False,
        },
        {
            "name": "TAU-bench Retail",
            "path": args.tau_retail_path,
            "use_all_as_test": True,
        },
    ]

    rows: List[Dict[str, object]] = []
    for ds in datasets:
        for threshold in args.thresholds:
            safe_name = ds["name"].lower().replace(" ", "_").replace("-", "_")
            output_path = out_dir / f"results_threshold_{safe_name}_t{threshold:.1f}.json"
            print(f"Running {ds['name']} @ gate_threshold={threshold:.1f}")

            result = run_offline_intervention_eval(
                trajectories_path=ds["path"],
                controller_path=args.controller_path,
                reference_dist_path=args.reference_dist_path,
                gate_threshold=float(threshold),
                output_path=str(output_path),
                use_all_as_test=bool(ds["use_all_as_test"]),
                dataset_name=ds["name"],
            )

            row = {
                "dataset": ds["name"],
                "threshold": float(threshold),
                "num_test_trajectories": int(result["num_test_trajectories"]),
                "controller_success_rate": float(result["controller"]["success_rate"]),
                "token_saving_rate": float(result["token_saving_rate"]),
                "no_control_success_rate": float(result["no_control"]["success_rate"]),
                "output_path": str(output_path),
            }
            rows.append(row)
            print(
                f"  done: success_rate={row['controller_success_rate']:.4f}, "
                f"token_saving={row['token_saving_rate']:.4f}"
            )

    summary = {
        "controller_path": args.controller_path,
        "reference_dist_path": args.reference_dist_path,
        "thresholds": [float(x) for x in args.thresholds],
        "rows": rows,
    }

    json_path = out_dir / "results_threshold_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "results_threshold_sweep.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "threshold",
                "num_test_trajectories",
                "controller_success_rate",
                "token_saving_rate",
                "no_control_success_rate",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "results_threshold_sweep.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Dataset | Threshold | N | Success Rate | Token Saving |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(row_to_markdown(row) + "\n")

    print("\nPareto table:")
    print("| Dataset | Threshold | N | Success Rate | Token Saving |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(row_to_markdown(row))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved MD:   {md_path}")


if __name__ == "__main__":
    main()
