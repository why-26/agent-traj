"""Evaluate DPO controller and compare against SL / DT baselines."""

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

CLASS_NAMES = {0: "Continue", 1: "Compress", 2: "Redirect", 3: "ModeSwitch", 4: "Stop"}


@dataclass
class EvalPack:
    gate_accuracy: float
    action_accuracy: float
    overall_accuracy: float
    prf: Dict[int, Dict[str, float]]


def safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


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


def eval_sl_like(
    model: DeliberationController,
    samples: Sequence[Mapping[str, object]],
    device: torch.device,
    gate_threshold: float,
    batch_size: int = 256,
) -> EvalPack:
    loader = DataLoader(SLSamples(samples), batch_size=batch_size, shuffle=False)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    gate_true: List[int] = []
    gate_pred: List[int] = []

    with torch.no_grad():
        for b in loader:
            signals = b["signals"].to(device)
            g = b["gate_label"].to(device)
            a = b["action_label"].to(device)

            gp, action_logits = model(signals)
            pg = (gp >= gate_threshold).long()
            pa = torch.argmax(action_logits, dim=-1)

            true_cls = torch.where(g == 1, a + 1, torch.zeros_like(a))
            pred_cls = torch.where(pg == 1, pa + 1, torch.zeros_like(pa))

            y_true.extend(int(v) for v in true_cls.cpu().tolist())
            y_pred.extend(int(v) for v in pred_cls.cpu().tolist())
            gate_true.extend(int(v) for v in g.cpu().tolist())
            gate_pred.extend(int(v) for v in pg.cpu().tolist())

    return summarize_metrics(y_true, y_pred, gate_true)


def eval_dt(
    model: DeliberationDecisionTransformer,
    samples: Sequence[Mapping[str, object]],
    device: torch.device,
    gate_threshold: float,
    batch_size: int = 256,
) -> EvalPack:
    loader = DataLoader(DTSamples(samples), batch_size=batch_size, shuffle=False)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    gate_true: List[int] = []
    gate_pred: List[int] = []

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
            gate_pred.extend(int(v) for v in pg.cpu().tolist())

    return summarize_metrics(y_true, y_pred, gate_true)


def summarize_metrics(y_true: List[int], y_pred: List[int], gate_true: List[int]) -> EvalPack:
    gate_pred = [0 if p == 0 else 1 for p in y_pred]
    gate_acc = sum(1 for t, p in zip(gate_true, gate_pred) if t == p) / max(len(gate_true), 1)

    gated_idx = [i for i, t in enumerate(gate_true) if t == 1]
    if gated_idx:
        action_acc = sum(1 for i in gated_idx if y_true[i] == y_pred[i]) / len(gated_idx)
    else:
        action_acc = 0.0

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


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def infer_signal_shape(sl_data: Mapping[str, object]) -> Tuple[int, int]:
    for split in ("train", "val", "test"):
        arr = sl_data.get(split)
        if isinstance(arr, list) and arr:
            return len(arr[0]["signals"][0]), len(arr[0]["signals"])
    raise ValueError("Cannot infer signal shape from SL data.")


def build_md_report(sl: EvalPack, dt: EvalPack, dpo: EvalPack) -> str:
    lines: List[str] = []
    lines.append("# DPO vs SL vs DT")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | Gate Acc | Action Acc | Overall Acc |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| SL Dual-Head | {format_pct(sl.gate_accuracy)} | {format_pct(sl.action_accuracy)} | {format_pct(sl.overall_accuracy)} |")
    lines.append(f"| DT | {format_pct(dt.gate_accuracy)} | {format_pct(dt.action_accuracy)} | {format_pct(dt.overall_accuracy)} |")
    lines.append(f"| DPO | {format_pct(dpo.gate_accuracy)} | {format_pct(dpo.action_accuracy)} | {format_pct(dpo.overall_accuracy)} |")
    lines.append("")

    lines.append("## Per-Class P/R/F1")
    lines.append("")
    lines.append("| Action | Model | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cls in CLASS_NAMES:
        n = CLASS_NAMES[cls]
        ms = sl.prf[cls]
        md = dt.prf[cls]
        mp = dpo.prf[cls]
        lines.append(f"| {n} | SL | {ms['precision']:.4f} | {ms['recall']:.4f} | {ms['f1']:.4f} | {int(ms['support'])} |")
        lines.append(f"| {n} | DT | {md['precision']:.4f} | {md['recall']:.4f} | {md['f1']:.4f} | {int(md['support'])} |")
        lines.append(f"| {n} | DPO | {mp['precision']:.4f} | {mp['recall']:.4f} | {mp['f1']:.4f} | {int(mp['support'])} |")

    lines.append("")
    lines.append("## DPO vs SL Action-Level Delta (F1)")
    lines.append("")
    lines.append("| Action | DPO F1 | SL F1 | Delta | Verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for cls in CLASS_NAMES:
        d = dpo.prf[cls]["f1"] - sl.prf[cls]["f1"]
        verdict = "stronger" if d > 0 else ("weaker" if d < 0 else "same")
        lines.append(
            f"| {CLASS_NAMES[cls]} | {dpo.prf[cls]['f1']:.4f} | {sl.prf[cls]['f1']:.4f} | {d:+.4f} | {verdict} |"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DPO model and compare with SL/DT.")
    parser.add_argument(
        "--sl_data",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules.json",
    )
    parser.add_argument(
        "--dt_data",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules_dt.json",
    )
    parser.add_argument(
        "--sl_ckpt",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/best_controller.pt",
    )
    parser.add_argument(
        "--dt_ckpt",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dt/best_controller_dt.pt",
    )
    parser.add_argument(
        "--dpo_ckpt",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dpo/best_controller_dpo.pt",
    )
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--output_md",
        default="/data/wanghy/agent_traj/deliberation_controller/eval/results_dpo_vs_sl_dt.md",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    sl_data = load_json(args.sl_data)
    dt_data = load_json(args.dt_data)
    if not isinstance(sl_data, Mapping) or not isinstance(dt_data, Mapping):
        raise ValueError("Dataset json must be mapping with train/val/test splits.")

    sl_test = sl_data.get("test", [])
    dt_test = dt_data.get("test", [])
    if not isinstance(sl_test, list) or not isinstance(dt_test, list):
        raise ValueError("Missing test split in dataset files.")

    signal_dim, num_steps = infer_signal_shape(sl_data)

    sl_model = DeliberationController(signal_dim=signal_dim, hidden_dim=64, num_steps=num_steps, num_actions=4).to(device)
    dpo_model = DeliberationController(signal_dim=signal_dim, hidden_dim=64, num_steps=num_steps, num_actions=4).to(device)
    dt_model = DeliberationDecisionTransformer(
        signal_dim=signal_dim,
        num_steps=num_steps,
        hidden_dim=64,
        nhead=4,
        num_layers=3,
        ff_dim=128,
        num_actions=4,
    ).to(device)

    load_state(sl_model, args.sl_ckpt, device)
    load_state(dt_model, args.dt_ckpt, device)
    load_state(dpo_model, args.dpo_ckpt, device)

    sl_eval = eval_sl_like(sl_model, sl_test, device, args.gate_threshold, args.batch_size)
    dt_eval = eval_dt(dt_model, dt_test, device, args.gate_threshold, args.batch_size)
    dpo_eval = eval_sl_like(dpo_model, sl_test, device, args.gate_threshold, args.batch_size)

    report = build_md_report(sl_eval, dt_eval, dpo_eval)
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
