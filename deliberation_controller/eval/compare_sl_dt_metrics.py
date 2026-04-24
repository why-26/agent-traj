"""Compare SL dual-head and DT controller metrics on test split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Tuple

import torch

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
class BinaryPR:
    precision: float
    recall: float


def load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_multiclass_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for cls_id in CLASS_NAMES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_id and p == cls_id)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls_id and p == cls_id)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_id and p != cls_id)
        p, r, f1 = safe_prf(tp, fp, fn)
        out[cls_id] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": sum(1 for t in y_true if t == cls_id),
        }
    return out


def compute_gate_pr(y_true_gate: List[int], y_pred_gate: List[int]) -> BinaryPR:
    tp = sum(1 for t, p in zip(y_true_gate, y_pred_gate) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true_gate, y_pred_gate) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_gate, y_pred_gate) if t == 1 and p == 0)
    p, r, _ = safe_prf(tp, fp, fn)
    return BinaryPR(precision=p, recall=r)


def true_class(gate_label: int, action_label: int) -> int:
    # 0..4 => Continue/Compress/Redirect/ModeSwitch/Stop
    return 0 if gate_label == 0 else (action_label + 1)


def batched(items: List[Mapping[str, object]], batch_size: int) -> Iterator[List[Mapping[str, object]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def sl_predict(
    model: DeliberationController,
    sample: Mapping[str, object],
    gate_threshold: float,
    device: torch.device,
) -> Tuple[int, int]:
    x = torch.tensor([sample["signals"]], dtype=torch.float32, device=device)
    with torch.no_grad():
        gate_prob, action_logits = model(x)
    gate = 1 if float(gate_prob[0].item()) >= gate_threshold else 0
    if gate == 0:
        return 0, gate
    action = int(torch.argmax(action_logits[0]).item())  # 0..3
    return action + 1, gate


def dt_predict(
    model: DeliberationDecisionTransformer,
    sample: Mapping[str, object],
    gate_threshold: float,
    device: torch.device,
) -> Tuple[int, int]:
    rtg = torch.tensor([sample["rtg"]], dtype=torch.float32, device=device)
    signals = torch.tensor([sample["signals"]], dtype=torch.float32, device=device)
    actions = torch.tensor([sample["actions"]], dtype=torch.long, device=device)
    with torch.no_grad():
        gate_prob, action_logits = model(rtg, signals, actions)
    gate = 1 if float(gate_prob[0].item()) >= gate_threshold else 0
    if gate == 0:
        return 0, gate
    action = int(torch.argmax(action_logits[0]).item())  # 0..3
    return action + 1, gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SL and DT per-class metrics.")
    parser.add_argument(
        "--sl_ckpt",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/best_controller.pt",
    )
    parser.add_argument(
        "--dt_ckpt",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dt/best_controller_dt.pt",
    )
    parser.add_argument(
        "--sl_data",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules.json",
    )
    parser.add_argument(
        "--dt_data",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules_dt.json",
    )
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Device for evaluation.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def to_markdown(
    sl_metrics: Dict[int, Dict[str, float]],
    dt_metrics: Dict[int, Dict[str, float]],
    sl_gate: BinaryPR,
    dt_gate: BinaryPR,
) -> str:
    lines = []
    lines.append("| Action | Model | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cls_id in CLASS_NAMES:
        name = CLASS_NAMES[cls_id]
        m1 = sl_metrics[cls_id]
        m2 = dt_metrics[cls_id]
        lines.append(
            f"| {name} | SL Dual-Head | {m1['precision']:.4f} | {m1['recall']:.4f} | {m1['f1']:.4f} | {int(m1['support'])} |"
        )
        lines.append(
            f"| {name} | DT Dual-Head | {m2['precision']:.4f} | {m2['recall']:.4f} | {m2['f1']:.4f} | {int(m2['support'])} |"
        )

    lines.append("")
    lines.append("| Metric | SL Dual-Head | DT Dual-Head |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Gate Precision | {sl_gate.precision:.4f} | {dt_gate.precision:.4f} |")
    lines.append(f"| Gate Recall | {sl_gate.recall:.4f} | {dt_gate.recall:.4f} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    sl_data = load_json(args.sl_data)
    dt_data = load_json(args.dt_data)
    sl_test = sl_data["test"]
    dt_test = dt_data["test"]
    if len(sl_test) != len(dt_test):
        raise ValueError(f"Test size mismatch: sl={len(sl_test)} dt={len(dt_test)}")

    sl_model = DeliberationController(signal_dim=5, hidden_dim=64, num_steps=5, num_actions=4).to(device)
    dt_model = DeliberationDecisionTransformer(
        signal_dim=5,
        num_steps=5,
        hidden_dim=64,
        nhead=4,
        num_layers=3,
        ff_dim=128,
        num_actions=4,
    ).to(device)

    sl_ckpt = torch.load(args.sl_ckpt, map_location=device)
    dt_ckpt = torch.load(args.dt_ckpt, map_location=device)
    sl_state = sl_ckpt["model_state_dict"] if isinstance(sl_ckpt, dict) and "model_state_dict" in sl_ckpt else sl_ckpt
    dt_state = dt_ckpt["model_state_dict"] if isinstance(dt_ckpt, dict) and "model_state_dict" in dt_ckpt else dt_ckpt
    sl_model.load_state_dict(sl_state, strict=True)
    dt_model.load_state_dict(dt_state, strict=True)
    sl_model.eval()
    dt_model.eval()

    y_true = [true_class(int(s["gate_label"]), int(s["action_label"])) for s in sl_test]
    gate_true = [int(s["gate_label"]) for s in sl_test]

    y_sl: List[int] = []
    gate_sl: List[int] = []
    for batch in batched(sl_test, args.batch_size):
        signals = torch.tensor([s["signals"] for s in batch], dtype=torch.float32, device=device)
        with torch.no_grad():
            gate_prob, action_logits = sl_model(signals)
        pred_gate = (gate_prob >= args.gate_threshold).long()
        pred_action = torch.argmax(action_logits, dim=-1)
        pred_cls = torch.where(pred_gate == 1, pred_action + 1, torch.zeros_like(pred_action))
        y_sl.extend(int(x) for x in pred_cls.cpu().tolist())
        gate_sl.extend(int(x) for x in pred_gate.cpu().tolist())

    y_dt: List[int] = []
    gate_dt: List[int] = []
    for batch in batched(dt_test, args.batch_size):
        rtg = torch.tensor([s["rtg"] for s in batch], dtype=torch.float32, device=device)
        signals = torch.tensor([s["signals"] for s in batch], dtype=torch.float32, device=device)
        actions = torch.tensor([s["actions"] for s in batch], dtype=torch.long, device=device)
        with torch.no_grad():
            gate_prob, action_logits = dt_model(rtg, signals, actions)
        pred_gate = (gate_prob >= args.gate_threshold).long()
        pred_action = torch.argmax(action_logits, dim=-1)
        pred_cls = torch.where(pred_gate == 1, pred_action + 1, torch.zeros_like(pred_action))
        y_dt.extend(int(x) for x in pred_cls.cpu().tolist())
        gate_dt.extend(int(x) for x in pred_gate.cpu().tolist())

    sl_metrics = compute_multiclass_metrics(y_true, y_sl)
    dt_metrics = compute_multiclass_metrics(y_true, y_dt)
    sl_gate = compute_gate_pr(gate_true, gate_sl)
    dt_gate = compute_gate_pr(gate_true, gate_dt)

    md = to_markdown(sl_metrics, dt_metrics, sl_gate, dt_gate)
    print(md)

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nSaved markdown to: {out}")


if __name__ == "__main__":
    main()
