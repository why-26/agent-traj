"""Evaluate the single-head 5-way controller ablation checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deliberation_controller.model.single_head_controller import CLASS_ID_TO_NAME, SingleHeadTemporalController
from deliberation_controller.train.train_single_head import (
    DEFAULT_DATA_PATH,
    DEFAULT_SAVE_DIR,
    DUAL_HEAD_BASELINE,
    WindowDataset,
    compute_inverse_frequency_weights,
    delta_pp,
    evaluate,
    format_metrics,
    load_data,
    pct,
    print_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-head 5-way controller ablation.")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", default=str(Path(DEFAULT_SAVE_DIR) / "best_single_head_controller.pt"))
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, signal_dim, num_steps = load_data(args.data_path)
    class_weights = compute_inverse_frequency_weights(data["train"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    model = SingleHeadTemporalController(
        signal_dim=signal_dim,
        num_steps=num_steps,
        hidden_dim=int(ckpt_args.get("hidden_dim", 32)),
        head_hidden_dim=int(ckpt_args.get("head_hidden_dim", 16)),
        nhead=int(ckpt_args.get("nhead", 4)),
        ff_dim=int(ckpt_args.get("ff_dim", 192)),
        num_layers=int(ckpt_args.get("num_layers", 2)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = DataLoader(WindowDataset(data["test"]), batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, test_loader, device, class_weights)

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data_path}")
    print(f"Params:     {checkpoint.get('n_params', 'unknown')} total, {checkpoint.get('head_params', 'unknown')} head")
    print("\nTest Results:")
    print(format_metrics(metrics))
    print_comparison(metrics)

    output = {
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "overall_accuracy": metrics.overall_accuracy,
        "gate_precision": metrics.gate_precision,
        "gate_recall": metrics.gate_recall,
        "gate_f1": metrics.gate_f1,
        "per_class": {CLASS_ID_TO_NAME[k]: v for k, v in metrics.per_class.items()},
        "dual_head_baseline": DUAL_HEAD_BASELINE,
        "comparison_delta_pp": {
            "overall_accuracy": (metrics.overall_accuracy - DUAL_HEAD_BASELINE["overall_accuracy"]) * 100,
            "gate_f1": (metrics.gate_f1 - DUAL_HEAD_BASELINE["gate_f1"]) * 100,
        },
    }
    out_path = Path(args.checkpoint).with_name("single_head_eval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved eval metrics: {out_path}")


if __name__ == "__main__":
    main()
