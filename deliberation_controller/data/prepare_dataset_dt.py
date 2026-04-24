"""Prepare DT-style dataset with return-to-go for controller training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Tuple

from deliberation_controller.data.prepare_dataset import is_success_trajectory


def load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def step_token_count(step: Mapping[str, object]) -> float:
    return float(step.get("tokens_input", 0) or 0) + float(step.get("tokens_output", 0) or 0)


def build_trajectory_rtg_map(
    trajectories: List[Mapping[str, object]],
    token_penalty: float = 0.001,
) -> Dict[str, List[float]]:
    """
    Reward design:
      r_t = -token_penalty * step_token_count_t
      terminal step adds success reward (1/0)
    """
    rtg_map: Dict[str, List[float]] = {}
    for traj in trajectories:
        task_id = str(traj.get("task_id"))
        steps = traj.get("steps", [])
        if not isinstance(steps, list):
            steps = []

        rewards = [-token_penalty * step_token_count(s) for s in steps]
        if rewards:
            rewards[-1] += 1.0 if is_success_trajectory(traj) else 0.0

        rtg = [0.0] * len(rewards)
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running += rewards[i]
            rtg[i] = running
        rtg_map[task_id] = rtg
    return rtg_map


def build_label_lookup(
    prepared_dataset: Mapping[str, object],
) -> Dict[Tuple[str, int], int]:
    """
    Map (task_id, target_step_idx) -> 5-class action id for the "next decision".
    0=Continue, 1=Compress, 2=Redirect, 3=ModeSwitch, 4=Stop
    """
    lookup: Dict[Tuple[str, int], int] = {}
    for split in ("train", "val", "test"):
        for sample in prepared_dataset.get(split, []):
            meta = sample.get("meta", {})
            task_id = str(meta.get("task_id"))
            target_step_idx = int(meta.get("target_step_idx", -1))
            gate = int(sample.get("gate_label", 0))
            action = int(sample.get("action_label", -100))
            cls = 0 if gate == 0 else (action + 1)
            lookup[(task_id, target_step_idx)] = cls
    return lookup


def build_dt_sample(
    sample: Mapping[str, object],
    action_lookup: Mapping[Tuple[str, int], int],
    rtg_map: Mapping[str, List[float]],
) -> MutableMapping[str, object]:
    signals = sample["signals"]
    k = len(signals)
    meta = sample.get("meta", {})
    task_id = str(meta.get("task_id"))
    target_step_idx = int(meta.get("target_step_idx", 0))
    start_step_idx = target_step_idx - k

    traj_rtg = rtg_map.get(task_id, [])
    rtg_seq: List[float] = []
    action_seq: List[int] = []

    for step_idx in range(start_step_idx, target_step_idx):
        if 0 <= step_idx < len(traj_rtg):
            rtg_seq.append(float(traj_rtg[step_idx]))
        else:
            rtg_seq.append(0.0)

        # action_t approximated by labeled next decision at (t+1)
        action_cls = int(action_lookup.get((task_id, step_idx + 1), 0))
        action_seq.append(action_cls)

    return {
        "rtg": rtg_seq,
        "signals": signals,
        "actions": action_seq,
        "gate_label": int(sample["gate_label"]),
        "action_label": int(sample["action_label"]),
        "meta": dict(meta),
    }


def prepare_dt_dataset(
    prepared_dataset: Mapping[str, object],
    trajectories: List[Mapping[str, object]],
    token_penalty: float = 0.001,
) -> Dict[str, object]:
    action_lookup = build_label_lookup(prepared_dataset)
    rtg_map = build_trajectory_rtg_map(trajectories, token_penalty=token_penalty)

    out: Dict[str, object] = {
        "train": [],
        "val": [],
        "test": [],
        "meta": {
            "window_size": int(prepared_dataset.get("meta", {}).get("window_size", 5)),
            "signal_order": list(prepared_dataset.get("meta", {}).get("signal_order", [])),
            "action_vocab": {
                "Continue": 0,
                "Compress": 1,
                "Redirect": 2,
                "ModeSwitch": 3,
                "Stop": 4,
            },
            "reward_formula": "r_t = -0.001 * step_token_count; terminal += success(1/0)",
            "token_penalty": token_penalty,
        },
    }

    for split in ("train", "val", "test"):
        out[split] = [
            build_dt_sample(sample, action_lookup=action_lookup, rtg_map=rtg_map)
            for sample in prepared_dataset.get(split, [])
        ]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DT controller dataset with return-to-go.")
    parser.add_argument(
        "--prepared_data_path",
        required=True,
        help="Path to existing SL prepared dataset JSON (e.g. hotpotqa_qwen3_full_dataset_v2_rules.json).",
    )
    parser.add_argument(
        "--trajectories_path",
        required=True,
        help="Path to full raw trajectories JSON (for reward/RTG calculation).",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--token_penalty", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = load_json(args.prepared_data_path)
    trajectories = load_json(args.trajectories_path)
    if not isinstance(prepared, dict):
        raise ValueError("prepared_data_path must point to a dict JSON with train/val/test keys.")
    if not isinstance(trajectories, list):
        raise ValueError("trajectories_path must point to a list JSON.")

    dt_dataset = prepare_dt_dataset(
        prepared_dataset=prepared,
        trajectories=trajectories,
        token_penalty=args.token_penalty,
    )
    save_json(dt_dataset, args.output_path)
    print("DT dataset prepared.")
    print(f"  train samples: {len(dt_dataset['train'])}")
    print(f"  val samples:   {len(dt_dataset['val'])}")
    print(f"  test samples:  {len(dt_dataset['test'])}")
    print(f"  saved to:      {args.output_path}")


if __name__ == "__main__":
    main()
