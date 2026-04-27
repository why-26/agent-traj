"""Evaluate SL, Vanilla DT, and DT reward-shaping variants; generate markdown summary."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from deliberation_controller.model.controller import DeliberationController
from deliberation_controller.model.controller_dt import DeliberationDecisionTransformer


CLASS_NAMES = {
    0: "Continue",
    1: "Compress",
    2: "Redirect",
    3: "ModeSwitch",
    4: "Stop",
}


@dataclass
class EvalPack:
    gate_accuracy: float
    action_accuracy: float
    overall_accuracy: float
    prf: Dict[int, Dict[str, float]]


class SLSamples(Dataset):
    def __init__(self, arr: Sequence[Mapping[str, object]]) -> None:
        self.arr = list(arr)

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.arr[idx]
        return {
            "signals": torch.tensor(x["signals"], dtype=torch.float32),
            "gate_label": torch.tensor(int(x["gate_label"]), dtype=torch.long),
            "action_label": torch.tensor(int(x["action_label"]), dtype=torch.long),
        }


class DTSamples(Dataset):
    def __init__(self, arr: Sequence[Mapping[str, object]]) -> None:
        self.arr = list(arr)

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.arr[idx]
        return {
            "rtg": torch.tensor(x["rtg"], dtype=torch.float32),
            "signals": torch.tensor(x["signals"], dtype=torch.float32),
            "actions": torch.tensor(x["actions"], dtype=torch.long),
            "gate_label": torch.tensor(int(x["gate_label"]), dtype=torch.long),
            "action_label": torch.tensor(int(x["action_label"]), dtype=torch.long),
        }


def safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def summarize_metrics(y_true: List[int], y_pred: List[int], gate_true: List[int]) -> EvalPack:
    gate_pred = [0 if p == 0 else 1 for p in y_pred]
    gate_acc = sum(1 for t, p in zip(gate_true, gate_pred) if t == p) / max(len(gate_true), 1)

    gated_idx = [i for i, t in enumerate(gate_true) if t == 1]
    action_acc = (
        sum(1 for i in gated_idx if y_true[i] == y_pred[i]) / len(gated_idx)
        if gated_idx
        else 0.0
    )

    overall_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)

    prf: Dict[int, Dict[str, float]] = {}
    for cls in CLASS_NAMES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        p, r, f1 = safe_prf(tp, fp, fn)
        prf[cls] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": sum(1 for t in y_true if t == cls),
        }

    return EvalPack(gate_acc, action_acc, overall_acc, prf)


def eval_sl(
    model: DeliberationController,
    samples: Sequence[Mapping[str, object]],
    device: torch.device,
    gate_threshold: float,
    batch_size: int,
) -> EvalPack:
    loader = DataLoader(SLSamples(samples), batch_size=batch_size, shuffle=False)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    gate_true: List[int] = []

    with torch.no_grad():
        for b in loader:
            x = b["signals"].to(device)
            g = b["gate_label"].to(device)
            a = b["action_label"].to(device)

            gp, action_logits = model(x)
            pg = (gp >= gate_threshold).long()
            pa = torch.argmax(action_logits, dim=-1)

            true_cls = torch.where(g == 1, a + 1, torch.zeros_like(a))
            pred_cls = torch.where(pg == 1, pa + 1, torch.zeros_like(pa))

            y_true.extend(int(v) for v in true_cls.cpu().tolist())
            y_pred.extend(int(v) for v in pred_cls.cpu().tolist())
            gate_true.extend(int(v) for v in g.cpu().tolist())

    return summarize_metrics(y_true, y_pred, gate_true)


def eval_dt(
    model: DeliberationDecisionTransformer,
    samples: Sequence[Mapping[str, object]],
    device: torch.device,
    gate_threshold: float,
    batch_size: int,
) -> EvalPack:
    loader = DataLoader(DTSamples(samples), batch_size=batch_size, shuffle=False)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    gate_true: List[int] = []

    with torch.no_grad():
        for b in loader:
            rtg = b["rtg"].to(device)
            signals = b["signals"].to(device)
            actions = b["actions"].to(device)
            g = b["gate_label"].to(device)
            a = b["action_label"].to(device)

            gp, action_logits = model(rtg, signals, actions)
            pg = (gp >= gate_threshold).long()
            pa = torch.argmax(action_logits, dim=-1)

            true_cls = torch.where(g == 1, a + 1, torch.zeros_like(a))
            pred_cls = torch.where(pg == 1, pa + 1, torch.zeros_like(pa))

            y_true.extend(int(v) for v in true_cls.cpu().tolist())
            y_pred.extend(int(v) for v in pred_cls.cpu().tolist())
            gate_true.extend(int(v) for v in g.cpu().tolist())

    return summarize_metrics(y_true, y_pred, gate_true)


def load_state(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_shape(sl_data: Mapping[str, object]) -> Tuple[int, int]:
    for split in ("train", "val", "test"):
        arr = sl_data.get(split)
        if isinstance(arr, list) and arr:
            return len(arr[0]["signals"][0]), len(arr[0]["signals"])
    raise ValueError("Cannot infer shape from SL dataset.")


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def build_markdown(
    results: Mapping[str, EvalPack],
) -> str:
    sl = results["SL"]
    lines: List[str] = []
    lines.append("# DT Reward Shaping Results")
    lines.append("")
    lines.append("| 模型 | Overall | Gate Acc | Action Acc | Continue F1 | Compress F1 | ModeSwitch F1 | Stop F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    order = ["SL", "Vanilla DT", "DT v2", "DT v3", "DT v4"]
    for name in order:
        r = results[name]
        lines.append(
            "| {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                name,
                f"{r.overall_accuracy:.4f}",
                f"{r.gate_accuracy:.4f}",
                f"{r.action_accuracy:.4f}",
                r.prf[0]["f1"],
                r.prf[1]["f1"],
                r.prf[3]["f1"],
                r.prf[4]["f1"],
            )
        )

    lines.append("")
    lines.append("## Per-class P/R/F1")
    lines.append("")
    lines.append("| Action | Model | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cls_id, cls_name in CLASS_NAMES.items():
        for name in order:
            m = results[name].prf[cls_id]
            lines.append(
                f"| {cls_name} | {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {int(m['support'])} |"
            )

    lines.append("")
    lines.append("## Discussion")
    lines.append("")
    vanilla = results["Vanilla DT"]
    for name in ["DT v2", "DT v3", "DT v4"]:
        r = results[name]
        d_overall = r.overall_accuracy - vanilla.overall_accuracy
        d_stop = r.prf[4]["f1"] - vanilla.prf[4]["f1"]
        d_comp = r.prf[1]["f1"] - vanilla.prf[1]["f1"]
        d_mode = r.prf[3]["f1"] - vanilla.prf[3]["f1"]
        lines.append(
            f"- {name} vs Vanilla DT: overall {d_overall:+.4f}, Stop F1 {d_stop:+.4f}, "
            f"Compress F1 {d_comp:+.4f}, ModeSwitch F1 {d_mode:+.4f}."
        )

    d_sl = vanilla.overall_accuracy - sl.overall_accuracy
    lines.append(f"- Vanilla DT vs SL overall delta: {d_sl:+.4f}.")
    lines.append("- 以上为直接评估结果，不做手工筛选或重加权。")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DT reward shaping variants.")
    parser.add_argument(
        "--sl_data",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules.json",
    )
    parser.add_argument(
        "--dt_data_vanilla",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules_dt.json",
    )
    parser.add_argument(
        "--dt_data_v2",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_dt_reward_v2.json",
    )
    parser.add_argument(
        "--dt_data_v3",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_dt_reward_v3.json",
    )
    parser.add_argument(
        "--dt_data_v4",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_dt_reward_v4.json",
    )

    parser.add_argument(
        "--sl_ckpt",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/best_controller.pt",
    )
    parser.add_argument(
        "--dt_ckpt_vanilla",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dt/best_controller_dt.pt",
    )
    parser.add_argument(
        "--dt_ckpt_v2",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dt_reward_v2/best_controller_dt.pt",
    )
    parser.add_argument(
        "--dt_ckpt_v3",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dt_reward_v3/best_controller_dt.pt",
    )
    parser.add_argument(
        "--dt_ckpt_v4",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dt_reward_v4/best_controller_dt.pt",
    )

    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output_md",
        default="/data/wanghy/agent_traj/deliberation_controller/eval/results_dt_shaping.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    sl_data = load_json(args.sl_data)
    dt_vanilla = load_json(args.dt_data_vanilla)
    dt_v2 = load_json(args.dt_data_v2)
    dt_v3 = load_json(args.dt_data_v3)
    dt_v4 = load_json(args.dt_data_v4)

    if not all(isinstance(x, Mapping) for x in [sl_data, dt_vanilla, dt_v2, dt_v3, dt_v4]):
        raise ValueError("All dataset files must be dict JSON with train/val/test.")

    signal_dim, num_steps = infer_shape(sl_data)

    sl_model = DeliberationController(signal_dim=signal_dim, hidden_dim=64, num_steps=num_steps, num_actions=4).to(device)
    dt_model_vanilla = DeliberationDecisionTransformer(signal_dim=signal_dim, num_steps=num_steps, hidden_dim=64, nhead=4, num_layers=3, ff_dim=128, num_actions=4).to(device)
    dt_model_v2 = DeliberationDecisionTransformer(signal_dim=signal_dim, num_steps=num_steps, hidden_dim=64, nhead=4, num_layers=3, ff_dim=128, num_actions=4).to(device)
    dt_model_v3 = DeliberationDecisionTransformer(signal_dim=signal_dim, num_steps=num_steps, hidden_dim=64, nhead=4, num_layers=3, ff_dim=128, num_actions=4).to(device)
    dt_model_v4 = DeliberationDecisionTransformer(signal_dim=signal_dim, num_steps=num_steps, hidden_dim=64, nhead=4, num_layers=3, ff_dim=128, num_actions=4).to(device)

    load_state(sl_model, args.sl_ckpt, device)
    load_state(dt_model_vanilla, args.dt_ckpt_vanilla, device)
    load_state(dt_model_v2, args.dt_ckpt_v2, device)
    load_state(dt_model_v3, args.dt_ckpt_v3, device)
    load_state(dt_model_v4, args.dt_ckpt_v4, device)

    results: Dict[str, EvalPack] = {}
    results["SL"] = eval_sl(sl_model, sl_data["test"], device, args.gate_threshold, args.batch_size)
    results["Vanilla DT"] = eval_dt(dt_model_vanilla, dt_vanilla["test"], device, args.gate_threshold, args.batch_size)
    results["DT v2"] = eval_dt(dt_model_v2, dt_v2["test"], device, args.gate_threshold, args.batch_size)
    results["DT v3"] = eval_dt(dt_model_v3, dt_v3["test"], device, args.gate_threshold, args.batch_size)
    results["DT v4"] = eval_dt(dt_model_v4, dt_v4["test"], device, args.gate_threshold, args.batch_size)

    md = build_markdown(results)
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(md)
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
