"""Evaluate rule-based baseline with the same metrics as train_sl.py."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from deliberation_controller.model.rule_baseline import (
    COMPRESS,
    CONTINUE,
    MODESWITCH,
    REDIRECT,
    STOP,
    RuleBasedController,
)

ACTION_NAMES = {0: "Compress", 1: "Redirect", 2: "ModeSwitch", 3: "Stop"}
CONTINUE_CLASS_ID = 4

RULE_TO_ACTION_ID = {
    COMPRESS: 0,
    REDIRECT: 1,
    MODESWITCH: 2,
    STOP: 3,
}
RULE_TO_OVERALL_ID = {
    CONTINUE: CONTINUE_CLASS_ID,
    COMPRESS: 0,
    REDIRECT: 1,
    MODESWITCH: 2,
    STOP: 3,
}


@dataclass
class EvalResult:
    gate_accuracy: float
    action_accuracy: float
    overall_accuracy: float
    action_precision: Dict[int, float]
    action_recall: Dict[int, float]
    action_support: Dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-based baseline.")
    parser.add_argument("--data_path", required=True, help="Path to prepared dataset JSON.")
    return parser.parse_args()


def get_raw_signals(sample: Mapping[str, object]) -> Sequence[Sequence[float]] | None:
    # Support a few possible key names if dataset is regenerated with raw features.
    for key in ("signals_raw", "raw_signals", "signals_original"):
        if key in sample:
            return sample[key]  # type: ignore[return-value]
    meta = sample.get("meta", {})
    if isinstance(meta, Mapping):
        for key in ("signals_raw", "raw_signals"):
            if key in meta:
                return meta[key]  # type: ignore[return-value]
    return None


def print_dataset_stats(test_samples: List[Mapping[str, object]]) -> None:
    gate_counter = Counter(int(x["gate_label"]) for x in test_samples)
    action_counter = Counter(int(x["action_label"]) for x in test_samples if int(x["gate_label"]) == 1)
    total = len(test_samples)
    gate_pos = gate_counter.get(1, 0)
    gate_ratio = (gate_pos / total * 100.0) if total else 0.0
    print("Dataset statistics:")
    print(f"  test: {total} samples")
    print(f"    gate distribution: {dict(gate_counter)} (gate=1 ratio={gate_ratio:.2f}%)")
    print(f"    action distribution (gate=1 only): {dict(action_counter)}")


def evaluate_rule_baseline(test_samples: List[Mapping[str, object]]) -> EvalResult:
    controller = RuleBasedController()
    gate_correct = 0
    gate_total = 0
    action_correct = 0
    action_total = 0
    overall_correct = 0
    overall_total = 0

    tp = Counter()
    pred_count = Counter()
    true_count = Counter()
    warned_missing_raw = False

    for sample in test_samples:
        signals_norm = sample["signals"]
        signals_raw = get_raw_signals(sample)
        if signals_raw is None and not warned_missing_raw:
            print(
                "[WARN] Raw signals not found in dataset. "
                "Redirect rule will fallback to normalized consecutive_failure_count."
            )
            warned_missing_raw = True

        pred_rule = controller.decide(signals_norm=signals_norm, signals_raw=signals_raw)
        pred_gate = 0 if pred_rule == CONTINUE else 1

        true_gate = int(sample["gate_label"])
        true_action = int(sample["action_label"])

        gate_correct += int(pred_gate == true_gate)
        gate_total += 1

        true_overall = CONTINUE_CLASS_ID if true_gate == 0 else true_action
        pred_overall = RULE_TO_OVERALL_ID[pred_rule]
        overall_correct += int(pred_overall == true_overall)
        overall_total += 1

        if true_gate == 1:
            # For action metrics, evaluate action rules directly (Head-2 style).
            pred_action_rule = controller.decide_action(signals_norm=signals_norm, signals_raw=signals_raw)
            pred_action = RULE_TO_ACTION_ID[pred_action_rule]
            action_correct += int(pred_action == true_action)
            action_total += 1

            for cls_id in ACTION_NAMES:
                pred_cls = pred_action == cls_id
                true_cls = true_action == cls_id
                pred_count[cls_id] += int(pred_cls)
                true_count[cls_id] += int(true_cls)
                tp[cls_id] += int(pred_cls and true_cls)

    action_precision = {}
    action_recall = {}
    action_support = {}
    for cls_id in ACTION_NAMES:
        p_denom = pred_count[cls_id]
        r_denom = true_count[cls_id]
        action_precision[cls_id] = (tp[cls_id] / p_denom) if p_denom else 0.0
        action_recall[cls_id] = (tp[cls_id] / r_denom) if r_denom else 0.0
        action_support[cls_id] = r_denom

    return EvalResult(
        gate_accuracy=gate_correct / max(gate_total, 1),
        action_accuracy=action_correct / max(action_total, 1),
        overall_accuracy=overall_correct / max(overall_total, 1),
        action_precision=action_precision,
        action_recall=action_recall,
        action_support=action_support,
    )


def format_action_metrics(result: EvalResult) -> str:
    lines = []
    for cls_id, cls_name in ACTION_NAMES.items():
        p = result.action_precision[cls_id] * 100.0
        r = result.action_recall[cls_id] * 100.0
        support = result.action_support[cls_id]
        lines.append(f"    {cls_name:<10} precision={p:6.2f}% recall={r:6.2f}% support={support}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_samples = data.get("test", [])
    print_dataset_stats(test_samples)

    result = evaluate_rule_baseline(test_samples)
    print("\nTest Results:")
    print("  loss:            N/A")
    print(f"  gate_accuracy:   {result.gate_accuracy:.4f}")
    print(f"  action_accuracy: {result.action_accuracy:.4f}")
    print(f"  overall_accuracy:{result.overall_accuracy:.4f}")
    print("  Action class metrics:")
    print(format_action_metrics(result))


if __name__ == "__main__":
    main()

