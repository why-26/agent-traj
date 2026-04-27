"""Prepare pairwise preference dataset for DPO controller training."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


@dataclass
class PairRecord:
    task_id: str
    step_idx: int
    state: List[List[float]]
    gate_chosen: int
    action_chosen: int
    gate_rejected: int
    action_rejected: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "step_idx": self.step_idx,
            "state": self.state,
            "gate_chosen": self.gate_chosen,
            "action_chosen": self.action_chosen,
            "gate_rejected": self.gate_rejected,
            "action_rejected": self.action_rejected,
        }


def load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_input_samples(obj: object) -> List[Mapping[str, object]]:
    """
    Supported input layout:
      {"train": [...], "val": [...], "test": [...]} from prepare_dataset.py
      OR a plain list of sample dicts.
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, Mapping)]
    if isinstance(obj, Mapping):
        samples: List[Mapping[str, object]] = []
        for split in ("train", "val", "test"):
            v = obj.get(split)
            if isinstance(v, list):
                samples.extend(x for x in v if isinstance(x, Mapping))
        if samples:
            return samples
    raise ValueError("Unsupported input format. Expected prepare_dataset-style dict or list of samples.")


def joint_action_from_sample(sample: Mapping[str, object]) -> Tuple[int, int]:
    gate = int(sample.get("gate_label", 0))
    action = int(sample.get("action_label", -100))
    if gate == 0:
        return 0, 0
    # action: 0..3 -> joint 1..4
    return 1, action + 1


def trajectory_key(sample: Mapping[str, object]) -> Tuple[str, str]:
    meta = sample.get("meta", {})
    if not isinstance(meta, Mapping):
        return "unknown_task", "unknown_traj"

    task_id = str(meta.get("task_id", "unknown_task"))
    # Prefer explicit trajectory key if future datasets provide it.
    traj_id = meta.get("trajectory_id")
    if traj_id is None:
        traj_id = meta.get("traj_id")
    if traj_id is None:
        traj_id = task_id
    return task_id, str(traj_id)


def is_success_sample(sample: Mapping[str, object]) -> bool:
    meta = sample.get("meta", {})
    if isinstance(meta, Mapping):
        return bool(meta.get("is_success_trajectory", False))
    return False


def sample_step_idx(sample: Mapping[str, object], fallback_idx: int) -> int:
    meta = sample.get("meta", {})
    if isinstance(meta, Mapping) and meta.get("target_step_idx") is not None:
        try:
            return int(meta.get("target_step_idx"))
        except Exception:
            return fallback_idx
    return fallback_idx


def build_pairs_task_strict(samples: Sequence[Mapping[str, object]]) -> List[PairRecord]:
    # group: task_id -> trajectory_id -> list[samples ordered by step]
    grouped: Dict[str, Dict[str, List[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        task_id, traj_id = trajectory_key(s)
        grouped[task_id][traj_id].append(s)

    for task_id, traj_map in grouped.items():
        for traj_id, seq in traj_map.items():
            seq.sort(key=lambda x: sample_step_idx(x, 10**9))

    out: List[PairRecord] = []

    for task_id, traj_map in grouped.items():
        success_trajs: List[List[Mapping[str, object]]] = []
        fail_trajs: List[List[Mapping[str, object]]] = []

        for _, seq in traj_map.items():
            if not seq:
                continue
            if is_success_sample(seq[0]):
                success_trajs.append(seq)
            else:
                fail_trajs.append(seq)

        # Strictly follow requested rule: only task with both S and F
        if not success_trajs or not fail_trajs:
            continue

        for s_traj in success_trajs:
            for f_traj in fail_trajs:
                align_len = min(len(s_traj), len(f_traj))
                for t in range(align_len):
                    s = s_traj[t]
                    f = f_traj[t]

                    gate_c, action_c = joint_action_from_sample(s)
                    gate_r, action_r = joint_action_from_sample(f)
                    if action_c == action_r:
                        continue

                    out.append(
                        PairRecord(
                            task_id=task_id,
                            step_idx=sample_step_idx(s, t),
                            state=[list(map(float, row)) for row in s.get("signals", [])],
                            gate_chosen=gate_c,
                            action_chosen=action_c,
                            gate_rejected=gate_r,
                            action_rejected=action_r,
                        )
                    )

    return out


def build_pairs_global_by_step(
    samples: Sequence[Mapping[str, object]],
    seed: int = 42,
) -> List[PairRecord]:
    """
    Fallback pairing mode for datasets without multi-seed same-task trajectories.

    Strategy:
    - group successful and failed samples by step index
    - pair within each step bucket
    - state/chosen come from success sample, rejected from fail sample
    """
    rng = random.Random(seed)
    succ_by_step: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    fail_by_step: Dict[int, List[Mapping[str, object]]] = defaultdict(list)

    for i, s in enumerate(samples):
        step = sample_step_idx(s, i)
        if is_success_sample(s):
            succ_by_step[step].append(s)
        else:
            fail_by_step[step].append(s)

    out: List[PairRecord] = []
    for step_idx in sorted(set(succ_by_step.keys()) & set(fail_by_step.keys())):
        s_list = succ_by_step[step_idx][:]
        f_list = fail_by_step[step_idx][:]
        rng.shuffle(s_list)
        rng.shuffle(f_list)
        n = min(len(s_list), len(f_list))
        for i in range(n):
            s = s_list[i]
            f = f_list[i]
            gate_c, action_c = joint_action_from_sample(s)
            gate_r, action_r = joint_action_from_sample(f)
            if action_c == action_r:
                continue
            meta = s.get("meta", {})
            task_id = str(meta.get("task_id", "unknown_task")) if isinstance(meta, Mapping) else "unknown_task"
            out.append(
                PairRecord(
                    task_id=task_id,
                    step_idx=step_idx,
                    state=[list(map(float, row)) for row in s.get("signals", [])],
                    gate_chosen=gate_c,
                    action_chosen=action_c,
                    gate_rejected=gate_r,
                    action_rejected=action_r,
                )
            )
    return out


def split_pairs(
    pairs: Sequence[PairRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    arr = [p.to_dict() for p in pairs]
    rng = random.Random(seed)
    rng.shuffle(arr)

    n = len(arr)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = arr[:n_train]
    val = arr[n_train : n_train + n_val]
    test = arr[n_train + n_val :]
    return train, val, test


def pair_stats(pairs: Sequence[PairRecord]) -> Tuple[int, Dict[str, int]]:
    c = Counter((p.action_chosen, p.action_rejected) for p in pairs)
    return len(pairs), {f"{a}->{b}": int(v) for (a, b), v in sorted(c.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs from prepared controller samples.")
    parser.add_argument(
        "--input_path",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules.json",
        help="Input prepared dataset path (train/val/test dict or list).",
    )
    parser.add_argument(
        "--output_path",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_dpo_pairs.json",
        help="Output all-pairs JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--pair_mode",
        choices=["task_strict", "global_by_step"],
        default="task_strict",
        help=(
            "task_strict: only same-task success/fail pairs (requested DPO design). "
            "global_by_step: fallback when task_strict yields 0 pairs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = load_json(args.input_path)
    samples = normalize_input_samples(raw)
    if args.pair_mode == "task_strict":
        pairs = build_pairs_task_strict(samples)
    else:
        pairs = build_pairs_global_by_step(samples, seed=args.seed)

    total_pairs, combo = pair_stats(pairs)

    print(f"Input samples: {len(samples)}")
    print(f"Total DPO pairs: {total_pairs}")
    print("Chosen->Rejected action combo counts:")
    if combo:
        for k, v in combo.items():
            print(f"  {k}: {v}")
    else:
        print("  (none)")

    save_json([p.to_dict() for p in pairs], args.output_path)

    train, val, test = split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    out_base = Path(args.output_path)
    save_json(train, out_base.with_name(out_base.stem + "_train.json"))
    save_json(val, out_base.with_name(out_base.stem + "_val.json"))
    save_json(test, out_base.with_name(out_base.stem + "_test.json"))

    print(f"Saved all pairs: {args.output_path}")
    print(f"Saved train: {out_base.with_name(out_base.stem + '_train.json')}")
    print(f"Saved val:   {out_base.with_name(out_base.stem + '_val.json')}")
    print(f"Saved test:  {out_base.with_name(out_base.stem + '_test.json')}")
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
