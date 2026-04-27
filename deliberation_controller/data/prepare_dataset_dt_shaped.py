"""Build reward-shaped DT datasets (v2/v3/v4) with unchanged state/action."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


ACTION_ID_TO_NAME = {
    0: "Continue",
    1: "Compress",
    2: "Redirect",
    3: "ModeSwitch",
    4: "Stop",
}


def load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: object, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_success_trajectory(traj: Mapping[str, object]) -> bool:
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
    return False


def step_token_count(step: Mapping[str, object]) -> float:
    return float(step.get("tokens_input", 0) or 0) + float(step.get("tokens_output", 0) or 0)


def build_action_lookup_from_sl(sl_dataset: Mapping[str, object]) -> Dict[Tuple[str, int], int]:
    """Map (task_id, target_step_idx) -> joint action id in {0..4}."""
    lookup: Dict[Tuple[str, int], int] = {}
    for split in ("train", "val", "test"):
        arr = sl_dataset.get(split, [])
        if not isinstance(arr, list):
            continue
        for sample in arr:
            if not isinstance(sample, Mapping):
                continue
            meta = sample.get("meta", {})
            if not isinstance(meta, Mapping):
                continue
            task_id = str(meta.get("task_id"))
            step_idx = int(meta.get("target_step_idx", -1))
            gate = int(sample.get("gate_label", 0))
            action = int(sample.get("action_label", -100))
            joint = 0 if gate == 0 else (action + 1)
            lookup[(task_id, step_idx)] = joint
    return lookup


def build_rtg_map_v2(
    trajectories: Sequence[Mapping[str, object]],
) -> Dict[str, List[float]]:
    rtg_map: Dict[str, List[float]] = {}
    for traj in trajectories:
        task_id = str(traj.get("task_id"))
        steps = traj.get("steps", [])
        if not isinstance(steps, list) or not steps:
            rtg_map[task_id] = []
            continue

        success = is_success_trajectory(traj)
        terminal = 1.0 if success else -1.0

        rewards = [-0.0002 * step_token_count(s) for s in steps]
        rewards[-1] += terminal

        rtg = [0.0] * len(rewards)
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running += rewards[i]
            rtg[i] = running
        rtg_map[task_id] = rtg
    return rtg_map


def build_rtg_map_v3(
    trajectories: Sequence[Mapping[str, object]],
    action_lookup: Mapping[Tuple[str, int], int],
) -> Dict[str, List[float]]:
    rtg_map: Dict[str, List[float]] = {}

    for traj in trajectories:
        task_id = str(traj.get("task_id"))
        steps = traj.get("steps", [])
        if not isinstance(steps, list) or not steps:
            rtg_map[task_id] = []
            continue

        success = is_success_trajectory(traj)
        terminal = 2.0 if success else -1.0

        rewards: List[float] = []
        for t in range(len(steps)):
            action_id = int(action_lookup.get((task_id, t), 0))
            action_name = ACTION_ID_TO_NAME.get(action_id, "Continue")

            if action_name == "Continue":
                bonus = 0.0
            elif action_name == "Stop":
                bonus = 0.30 if success else -0.30
            else:
                bonus = 0.05 if success else -0.03
            rewards.append(bonus)

        rewards[-1] += terminal

        rtg = [0.0] * len(rewards)
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running += rewards[i]
            rtg[i] = running
        rtg_map[task_id] = rtg
    return rtg_map


def build_rtg_map_v4(
    trajectories: Sequence[Mapping[str, object]],
) -> Dict[str, List[float]]:
    rtg_map: Dict[str, List[float]] = {}
    for traj in trajectories:
        task_id = str(traj.get("task_id"))
        steps = traj.get("steps", [])
        if not isinstance(steps, list) or not steps:
            rtg_map[task_id] = []
            continue

        success = is_success_trajectory(traj)
        terminal = 1.0 if success else -0.5

        total_tokens = float(traj.get("total_input_tokens", 0) or 0) + float(
            traj.get("total_output_tokens", 0) or 0
        )
        if total_tokens <= 0:
            total_tokens = sum(step_token_count(s) for s in steps)

        efficiency = 1.0 - total_tokens / 8000.0
        efficiency = max(-0.5, min(0.5, efficiency))
        final_return = terminal + 0.3 * efficiency

        rewards = [0.0 for _ in steps]
        rewards[-1] += final_return

        rtg = [0.0] * len(rewards)
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running += rewards[i]
            rtg[i] = running
        rtg_map[task_id] = rtg
    return rtg_map


def apply_rtg_to_base_dt(
    base_dt: Mapping[str, object],
    rtg_map: Mapping[str, List[float]],
) -> Dict[str, object]:
    out = copy.deepcopy(base_dt)

    for split in ("train", "val", "test"):
        arr = out.get(split, [])
        if not isinstance(arr, list):
            continue

        for sample in arr:
            meta = sample.get("meta", {})
            if not isinstance(meta, Mapping):
                continue
            task_id = str(meta.get("task_id"))
            target_step_idx = int(meta.get("target_step_idx", 0))
            k = len(sample.get("signals", []))
            start = target_step_idx - k

            traj_rtg = rtg_map.get(task_id, [])
            seq: List[float] = []
            for step_idx in range(start, target_step_idx):
                if 0 <= step_idx < len(traj_rtg):
                    seq.append(float(traj_rtg[step_idx]))
                else:
                    seq.append(0.0)
            sample["rtg"] = seq

    meta = out.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    out["meta"] = meta
    return out


def summarize_dataset(dataset: Mapping[str, object]) -> None:
    for split in ("train", "val", "test"):
        arr = dataset.get(split, [])
        if not isinstance(arr, list):
            continue
        if not arr:
            print(f"  {split}: 0 samples")
            continue
        rtg0 = arr[0].get("rtg", [])
        print(f"  {split}: {len(arr)} samples | rtg_len={len(rtg0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DT reward-shaping variants (v2/v3/v4).")
    parser.add_argument(
        "--sl_data_path",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules.json",
    )
    parser.add_argument(
        "--base_dt_path",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules_dt.json",
    )
    parser.add_argument(
        "--trajectories_path",
        default="/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/qwen3-4b-thinking-hotpotqa-all.triples.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/wanghy/agent_traj/deliberation_controller/data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sl_data = load_json(args.sl_data_path)
    base_dt = load_json(args.base_dt_path)
    trajectories = load_json(args.trajectories_path)

    if not isinstance(sl_data, Mapping):
        raise ValueError("sl_data_path must be dict json.")
    if not isinstance(base_dt, Mapping):
        raise ValueError("base_dt_path must be dict json.")
    if not isinstance(trajectories, list):
        raise ValueError("trajectories_path must be list json.")

    action_lookup = build_action_lookup_from_sl(sl_data)

    rtg_v2 = build_rtg_map_v2(trajectories)
    rtg_v3 = build_rtg_map_v3(trajectories, action_lookup)
    rtg_v4 = build_rtg_map_v4(trajectories)

    ds_v2 = apply_rtg_to_base_dt(base_dt, rtg_v2)
    ds_v3 = apply_rtg_to_base_dt(base_dt, rtg_v3)
    ds_v4 = apply_rtg_to_base_dt(base_dt, rtg_v4)

    for name, ds in (("v2", ds_v2), ("v3", ds_v3), ("v4", ds_v4)):
        meta = ds.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        meta["reward_variant"] = name
        ds["meta"] = meta

    out_dir = Path(args.output_dir)
    p2 = out_dir / "hotpotqa_qwen3_dt_reward_v2.json"
    p3 = out_dir / "hotpotqa_qwen3_dt_reward_v3.json"
    p4 = out_dir / "hotpotqa_qwen3_dt_reward_v4.json"

    save_json(ds_v2, p2)
    save_json(ds_v3, p3)
    save_json(ds_v4, p4)

    print("Saved reward-shaped DT datasets:")
    print(f"  {p2}")
    print(f"  {p3}")
    print(f"  {p4}")

    print("\\nDataset summary:")
    print("v2:")
    summarize_dataset(ds_v2)
    print("v3:")
    summarize_dataset(ds_v3)
    print("v4:")
    summarize_dataset(ds_v4)


if __name__ == "__main__":
    main()
