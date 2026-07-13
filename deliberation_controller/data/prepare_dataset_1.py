"""Prepare sliding-window dataset for Deliberation Controller supervised training."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

from .normalizer import (
    PercentileNormalizer,
    build_reference_distribution,
    load_trajectories,
    save_reference_distribution,
)
from .signal_extractor import extract_all_step_signals, signals_to_vector

ACTION_COMPRESS = 0
ACTION_REDIRECT = 1
ACTION_MODESWITCH = 2
ACTION_STOP = 3

DEFAULT_GATE_SIGNALS: Tuple[str, ...] = (
    "thought_length_mean",
    "thought_length_var",
    "tokens_per_step",
)
INPUT_SIGNAL_NAMES: Tuple[str, ...] = DEFAULT_GATE_SIGNALS
GATE_PERCENTILE = 90.0
ACTION_OSCILLATION_PERCENTILE = 75.0
ACTION_STOP_MEAN_PERCENTILE = 90.0
ACTION_STOP_VAR_PERCENTILE = 85.0
ACTION_REDIRECT_FAILURE_COUNT = 5.0


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


def is_success_trajectory(traj: Mapping[str, object]) -> bool:
    """Infer success flag across TAU-bench / HotpotQA / Bamboogle formats."""
    if "success" in traj and traj["success"] is not None:
        return bool(traj["success"])
    if "reward" in traj and traj["reward"] is not None:
        try:
            return float(traj["reward"]) == 1.0
        except Exception:
            return bool(traj["reward"])
    metrics = traj.get("Metrics")
    if isinstance(metrics, Mapping):
        acc = metrics.get("acc")
        if acc is not None:
            try:
                return float(acc) == 1.0
            except Exception:
                return bool(acc)
        valid = metrics.get("is_valid_answer")
        if valid is not None:
            return bool(valid)
    return False


def split_trajectories(
    trajectories: Sequence[Mapping[str, object]],
    split_cfg: SplitConfig,
) -> Dict[str, List[Mapping[str, object]]]:
    trajectories = list(trajectories)
    rng = random.Random(split_cfg.seed)
    rng.shuffle(trajectories)

    total = len(trajectories)
    train_end = int(total * split_cfg.train_ratio)
    val_end = train_end + int(total * split_cfg.val_ratio)

    return {
        "train": trajectories[:train_end],
        "val": trajectories[train_end:val_end],
        "test": trajectories[val_end:],
    }


def assign_gate_label(
    is_success: bool,
    next_step_signals_norm: Mapping[str, float],
    tier1_signals: Sequence[str] = DEFAULT_GATE_SIGNALS,
    gate_percentile_threshold: float = GATE_PERCENTILE,
) -> int:
    if is_success:
        return 0
    for name in tier1_signals:
        if float(next_step_signals_norm.get(name, 0.0)) > gate_percentile_threshold:
            return 1
    return 0


def assign_action_label(
    next_step_signals_raw: Mapping[str, float],
    next_step_signals_norm: Mapping[str, float],
) -> int:
    # Rule 1: reasoning oscillation first (thinking-model specific).
    if float(next_step_signals_norm.get("decision_oscillation", 0.0)) > ACTION_OSCILLATION_PERCENTILE:
        return ACTION_MODESWITCH

    # Rule 2: severe reasoning inflation.
    if (
        float(next_step_signals_norm.get("thought_length_mean", 0.0)) > ACTION_STOP_MEAN_PERCENTILE
        and float(next_step_signals_norm.get("thought_length_var", 0.0)) > ACTION_STOP_VAR_PERCENTILE
    ):
        return ACTION_STOP

    # Rule 3: mechanical retries with stricter threshold.
    if (
        float(next_step_signals_raw.get("consecutive_failure_count", 0.0))
        >= ACTION_REDIRECT_FAILURE_COUNT
    ):
        return ACTION_REDIRECT

    # Rule 4: fallback.
    return ACTION_COMPRESS


def build_samples_for_trajectory(
    traj: Mapping[str, object],
    normalizer: PercentileNormalizer,
    window_size: int = 5,
) -> List[MutableMapping[str, object]]:
    steps = traj.get("steps", [])
    if not isinstance(steps, list) or len(steps) <= window_size:
        return []

    raw_signal_matrix = extract_all_step_signals(steps)
    norm_signal_matrix = [normalizer.normalize_signal_dict(sig) for sig in raw_signal_matrix]
    success_flag = is_success_trajectory(traj)

    samples: List[MutableMapping[str, object]] = []
    for start in range(0, len(steps) - window_size):
        next_idx = start + window_size
        window = norm_signal_matrix[start:next_idx]
        next_signals_norm = norm_signal_matrix[next_idx]
        next_signals_raw = raw_signal_matrix[next_idx]

        gate_label = assign_gate_label(success_flag, next_signals_norm)
        action_label = (
            assign_action_label(next_signals_raw, next_signals_norm)
            if gate_label == 1
            else -100
        )

        samples.append(
            {
                "signals": [
                    signals_to_vector(step_sig, signal_names=INPUT_SIGNAL_NAMES)
                    for step_sig in window
                ],
                "gate_label": gate_label,
                "action_label": action_label,
                "meta": {
                    "task_id": traj.get("task_id"),
                    "target_step_idx": next_idx,
                    "is_success_trajectory": success_flag,
                },
            }
        )
    return samples


def prepare_dataset(
    trajectories: Sequence[Mapping[str, object]],
    window_size: int = 5,
    split_cfg: SplitConfig = SplitConfig(),
    reference_distribution: Mapping[str, Sequence[float]] | None = None,
) -> Dict[str, object]:
    if reference_distribution is None:
        reference_distribution = build_reference_distribution(trajectories)
    normalizer = PercentileNormalizer.from_distribution(reference_distribution)

    split_trajs = split_trajectories(trajectories, split_cfg)
    dataset: Dict[str, object] = {
        "train": [],
        "val": [],
        "test": [],
        "meta": {
            "window_size": window_size,
            "signal_order": list(INPUT_SIGNAL_NAMES),
            "action_map": {
                "Compress": ACTION_COMPRESS,
                "Redirect": ACTION_REDIRECT,
                "ModeSwitch": ACTION_MODESWITCH,
                "Stop": ACTION_STOP,
            },
        },
    }

    for split_name in ("train", "val", "test"):
        split_samples: List[MutableMapping[str, object]] = []
        for traj in split_trajs[split_name]:
            split_samples.extend(
                build_samples_for_trajectory(
                    traj=traj,
                    normalizer=normalizer,
                    window_size=window_size,
                )
            )
        dataset[split_name] = split_samples

    return dataset


def save_dataset(dataset: Mapping[str, object], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Deliberation Controller dataset.")
    parser.add_argument("--input", required=True, help="Path to trajectory JSON file.")
    parser.add_argument("--output", required=True, help="Path to output dataset JSON file.")
    parser.add_argument(
        "--reference-distribution",
        default="reference_distribution.json",
        help="Path to reference distribution JSON file.",
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-reference", action="store_true")
    args = parser.parse_args()

    trajectories = load_trajectories(args.input)

    ref_path = Path(args.reference_distribution)
    if args.rebuild_reference or not ref_path.exists():
        reference_distribution = build_reference_distribution(trajectories)
        save_reference_distribution(reference_distribution, ref_path)
    else:
        with open(ref_path, "r", encoding="utf-8") as f:
            reference_distribution = json.load(f)

    split_cfg = SplitConfig(seed=args.seed)
    dataset = prepare_dataset(
        trajectories=trajectories,
        window_size=args.window_size,
        split_cfg=split_cfg,
        reference_distribution=reference_distribution,
    )
    save_dataset(dataset, args.output)

    print("Dataset prepared.")
    print(f"  train samples: {len(dataset['train'])}")
    print(f"  val samples:   {len(dataset['val'])}")
    print(f"  test samples:  {len(dataset['test'])}")
    print(f"  saved to:      {args.output}")
    print(f"  reference:     {ref_path}")


if __name__ == "__main__":
    main()
