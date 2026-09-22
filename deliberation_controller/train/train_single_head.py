"""Train the single-head 5-way controller ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deliberation_controller.model.single_head_controller import (
    CLASS_ID_TO_NAME,
    SingleHeadTemporalController,
)

DEFAULT_DATA_PATH = (
    "/data/wanghy/agent_traj/deliberation_controller/data/"
    "hotpotqa_qwen3_full_dataset_v2_rules.json"
)
DEFAULT_SAVE_DIR = (
    "/data/wanghy/agent_traj/deliberation_controller/checkpoints/"
    "single_head_ablation"
)
DUAL_HEAD_BASELINE = {
    "overall_accuracy": 0.9312,
    "gate_f1": 0.8163,
    "per_action_f1": {
        "Compress": 0.9651,
        "Redirect": 0.0,
        "ModeSwitch": 0.9123,
        "Stop": 0.6111,
    },
}


class WindowDataset(Dataset):
    def __init__(self, samples: List[Mapping[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "signals": torch.tensor(item["signals"], dtype=torch.float32),
            "gate_label": torch.tensor(int(item["gate_label"]), dtype=torch.long),
            "action_label": torch.tensor(int(item["action_label"]), dtype=torch.long),
        }


@dataclass
class Metrics:
    loss: float
    overall_accuracy: float
    gate_precision: float
    gate_recall: float
    gate_f1: float
    per_class: Dict[int, Dict[str, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_from_sample(item: Mapping[str, object]) -> int:
    if int(item["gate_label"]) == 0:
        return 0
    return int(item["action_label"]) + 1


def load_data(data_path: str) -> Tuple[dict, int, int]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta", {}) if isinstance(data, Mapping) else {}
    signal_dim = len(meta.get("signal_order", [])) if isinstance(meta, Mapping) else 0
    num_steps = int(meta.get("window_size", 0)) if isinstance(meta, Mapping) else 0
    probe = next((data[s][0] for s in ("train", "val", "test") if data.get(s)), None)
    if probe is not None:
        signals = probe.get("signals", [])
        if not num_steps:
            num_steps = len(signals)
        if signals and not signal_dim:
            signal_dim = len(signals[0])
    if signal_dim <= 0 or num_steps <= 0:
        raise ValueError("Could not infer signal_dim/num_steps from dataset.")
    return data, signal_dim, num_steps


def make_loaders(data: Mapping[str, object], batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(WindowDataset(data["train"]), batch_size=batch_size, shuffle=True),
        DataLoader(WindowDataset(data["val"]), batch_size=batch_size, shuffle=False),
        DataLoader(WindowDataset(data["test"]), batch_size=batch_size, shuffle=False),
    )


def compute_inverse_frequency_weights(samples: List[Mapping[str, object]]) -> torch.Tensor:
    counts = Counter(target_from_sample(x) for x in samples)
    total = sum(counts.values())
    num_classes = len(CLASS_ID_TO_NAME)
    weights = [total / (num_classes * max(counts.get(i, 0), 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def print_dataset_stats(data: Mapping[str, object]) -> None:
    print("Dataset statistics:")
    for split in ("train", "val", "test"):
        samples = data[split]
        target_counts = Counter(target_from_sample(x) for x in samples)
        task_ids = {x.get("meta", {}).get("task_id") for x in samples if isinstance(x.get("meta", {}), Mapping)}
        print(f"  {split}: {len(samples)} entries | {len(task_ids)} trajectories")
        print(
            "    class distribution: "
            + str({CLASS_ID_TO_NAME[i]: target_counts.get(i, 0) for i in CLASS_ID_TO_NAME})
        )


def prf(tp: int, pred: int, true: int) -> Tuple[float, float, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / true if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(
    model: SingleHeadTemporalController,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    true_counts = Counter()
    pred_counts = Counter()
    tp_counts = Counter()
    gate_tp = gate_pred = gate_true = 0

    with torch.no_grad():
        for batch in loader:
            signals = batch["signals"].to(device)
            gate_label = batch["gate_label"].to(device)
            action_label = batch["action_label"].to(device)
            logits = model(signals)
            targets = model.build_targets(gate_label, action_label)
            loss = model.compute_loss(logits, gate_label, action_label, class_weights=class_weights)
            preds = torch.argmax(logits, dim=-1)

            batch_size = signals.size(0)
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            correct += int((preds == targets).sum().item())

            for cls_id in CLASS_ID_TO_NAME:
                pred_mask = preds == cls_id
                true_mask = targets == cls_id
                pred_counts[cls_id] += int(pred_mask.sum().item())
                true_counts[cls_id] += int(true_mask.sum().item())
                tp_counts[cls_id] += int((pred_mask & true_mask).sum().item())

            pred_gate = preds != 0
            true_gate = targets != 0
            gate_tp += int((pred_gate & true_gate).sum().item())
            gate_pred += int(pred_gate.sum().item())
            gate_true += int(true_gate.sum().item())

    per_class: Dict[int, Dict[str, float]] = {}
    for cls_id in CLASS_ID_TO_NAME:
        p, r, f1 = prf(tp_counts[cls_id], pred_counts[cls_id], true_counts[cls_id])
        per_class[cls_id] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": float(true_counts[cls_id]),
        }
    gate_p, gate_r, gate_f1 = prf(gate_tp, gate_pred, gate_true)
    return Metrics(
        loss=total_loss / max(total, 1),
        overall_accuracy=correct / max(total, 1),
        gate_precision=gate_p,
        gate_recall=gate_r,
        gate_f1=gate_f1,
        per_class=per_class,
    )


def format_metrics(metrics: Metrics) -> str:
    lines = [
        f"  loss:             {metrics.loss:.4f}",
        f"  overall_accuracy: {metrics.overall_accuracy:.4f}",
        f"  gate_precision:   {metrics.gate_precision:.4f}",
        f"  gate_recall:      {metrics.gate_recall:.4f}",
        f"  gate_f1:          {metrics.gate_f1:.4f}",
        "  Per-class P/R/F1:",
    ]
    for cls_id, name in CLASS_ID_TO_NAME.items():
        item = metrics.per_class[cls_id]
        lines.append(
            f"    {name:<10} P={item['precision']:.4f} "
            f"R={item['recall']:.4f} F1={item['f1']:.4f} support={int(item['support'])}"
        )
    return "\n".join(lines)


def write_metrics_json(metrics: Metrics, path: Path, extra: Mapping[str, object]) -> None:
    payload = {
        **extra,
        "loss": metrics.loss,
        "overall_accuracy": metrics.overall_accuracy,
        "gate_precision": metrics.gate_precision,
        "gate_recall": metrics.gate_recall,
        "gate_f1": metrics.gate_f1,
        "per_class": {CLASS_ID_TO_NAME[k]: v for k, v in metrics.per_class.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_history_csv(history: List[Mapping[str, float]], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def pct(x: float) -> str:
    return f"{x * 100:.2f}"


def delta_pp(single: float, dual: float) -> str:
    return f"{(single - dual) * 100:+.2f}pp"


def print_comparison(metrics: Metrics) -> None:
    print("\nComparison table:")
    print("| Metric | Dual-head (V3b) | Single-head | Δ |")
    print("|---|---:|---:|---:|")
    print(
        f"| Overall Accuracy | {pct(DUAL_HEAD_BASELINE['overall_accuracy'])} | "
        f"{pct(metrics.overall_accuracy)} | "
        f"{delta_pp(metrics.overall_accuracy, DUAL_HEAD_BASELINE['overall_accuracy'])} |"
    )
    print(
        f"| Gate F1 | {pct(DUAL_HEAD_BASELINE['gate_f1'])} | "
        f"{pct(metrics.gate_f1)} | {delta_pp(metrics.gate_f1, DUAL_HEAD_BASELINE['gate_f1'])} |"
    )
    for cls_id in (1, 2, 3, 4):
        name = CLASS_ID_TO_NAME[cls_id]
        dual = DUAL_HEAD_BASELINE["per_action_f1"][name]
        single = metrics.per_class[cls_id]["f1"]
        print(f"| {name} F1 | {pct(dual)} | {pct(single)} | {delta_pp(single, dual)} |")


def train_one_epoch(
    model: SingleHeadTemporalController,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        signals = batch["signals"].to(device)
        gate_label = batch["gate_label"].to(device)
        action_label = batch["action_label"].to(device)
        logits = model(signals)
        loss = model.compute_loss(logits, gate_label, action_label, class_weights=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = signals.size(0)
        total_loss += float(loss.item()) * batch_size
        total += batch_size
    return total_loss / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-head 5-way controller ablation.")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--head_hidden_dim", type=int, default=16)
    parser.add_argument("--ff_dim", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data, signal_dim, num_steps = load_data(args.data_path)
    print_dataset_stats(data)
    train_loader, val_loader, test_loader = make_loaders(data, args.batch_size)

    class_weights = compute_inverse_frequency_weights(data["train"]).to(device)
    print("Class weights (inverse frequency):")
    print({CLASS_ID_TO_NAME[i]: round(float(class_weights[i].item()), 6) for i in CLASS_ID_TO_NAME})

    model = SingleHeadTemporalController(
        signal_dim=signal_dim,
        num_steps=num_steps,
        hidden_dim=args.hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        nhead=args.nhead,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(
        f"Model: single_head | signal_dim={signal_dim} | num_steps={num_steps} | "
        f"hidden_dim={args.hidden_dim} | head_hidden_dim={args.head_hidden_dim} | "
        f"trainable parameters={n_params} | head parameters={head_params}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = -1.0
    best_epoch = -1
    no_improve = 0
    history: List[Mapping[str, float]] = []
    best_path = save_dir / "best_single_head_controller.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = evaluate(model, val_loader, device, class_weights)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "val_overall_accuracy": val_metrics.overall_accuracy,
            "val_gate_f1": val_metrics.gate_f1,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_overall_acc={val_metrics.overall_accuracy:.4f} | "
            f"val_gate_f1={val_metrics.gate_f1:.4f}"
        )

        if val_metrics.overall_accuracy > best_val:
            best_val = val_metrics.overall_accuracy
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_overall_accuracy": best_val,
                    "args": vars(args),
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "class_id_to_name": CLASS_ID_TO_NAME,
                    "signal_dim": signal_dim,
                    "num_steps": num_steps,
                    "n_params": n_params,
                    "head_params": head_params,
                },
                best_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}: val overall_accuracy did not improve "
                    f"for {args.patience} epochs."
                )
                break

    print(f"Best model from epoch {best_epoch} with val overall_accuracy={best_val:.4f}")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, class_weights)
    print("\nTest Results:")
    print(format_metrics(test_metrics))
    print_comparison(test_metrics)

    write_history_csv(history, save_dir / "training_history.csv")
    write_metrics_json(
        test_metrics,
        save_dir / "single_head_metrics.json",
        {
            "best_epoch": best_epoch,
            "best_val_overall_accuracy": best_val,
            "checkpoint": str(best_path),
            "data_path": args.data_path,
            "n_params": n_params,
            "head_params": head_params,
            "dual_head_baseline": DUAL_HEAD_BASELINE,
        },
    )
    print(f"\nSaved checkpoint: {best_path}")
    print(f"Saved metrics:    {save_dir / 'single_head_metrics.json'}")
    print(f"Saved history:    {save_dir / 'training_history.csv'}")


if __name__ == "__main__":
    main()
