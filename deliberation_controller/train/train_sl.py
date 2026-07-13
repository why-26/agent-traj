"""Supervised learning training entry for Deliberation Controller."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from deliberation_controller.model.controller import DeliberationController
from deliberation_controller.model.controller_mlp import DeliberationMLPController
from deliberation_controller.model.controller_single_head import DeliberationSingleHeadController

ACTION_NAMES = {0: "Compress", 1: "Redirect", 2: "ModeSwitch", 3: "Stop"}
CONTINUE_CLASS_ID = 4


class TrajectoryWindowDataset(Dataset):
    """Dataset over sliding-window controller samples."""

    def __init__(self, samples: List[Mapping[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        signals = torch.tensor(item["signals"], dtype=torch.float32)
        gate_label = torch.tensor(float(item["gate_label"]), dtype=torch.float32)
        action_label = torch.tensor(int(item["action_label"]), dtype=torch.long)
        return {
            "signals": signals,
            "gate_label": gate_label,
            "action_label": action_label,
        }


@dataclass
class EvalResult:
    loss: float
    gate_accuracy: float
    gate_f1: float
    action_accuracy: float
    overall_accuracy: float
    stop_precision: float
    action_precision: Dict[int, float]
    action_recall: Dict[int, float]
    action_support: Dict[int, int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_action_class_weights(
    train_samples: List[Mapping[str, object]],
    device: torch.device,
) -> torch.Tensor:
    """Strict inverse frequency, normalized so weights.sum() == n_classes."""
    train_actions = [
        int(e["action_label"])
        for e in train_samples
        if int(e["action_label"]) != -100
    ]
    counts = Counter(train_actions)
    n_classes = 4

    weights = torch.zeros(n_classes)
    for cls in range(n_classes):
        n = counts.get(cls, 1)
        weights[cls] = 1.0 / n
    weights = weights / weights.sum() * n_classes

    action_names = ["Compress", "Redirect", "ModeSwitch", "Stop"]
    print("Class weights (strict inverse frequency):")
    for name, w, c in zip(action_names, weights.tolist(), [counts.get(i, 0) for i in range(4)]):
        print(f"  {name}: weight={w:.3f} (count={c})")
    return weights.to(device)


def compute_dual_head_loss(
    gate_prob: torch.Tensor,
    action_logits: torch.Tensor,
    gate_label: torch.Tensor,
    action_label: torch.Tensor,
    action_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gate_loss = F.binary_cross_entropy(gate_prob, gate_label.float())
    action_loss = F.cross_entropy(
        action_logits,
        action_label,
        weight=action_weights,
        ignore_index=-100,
        reduction="mean",
    )
    return gate_loss + action_loss


def build_true_overall_class(gate_label: torch.Tensor, action_label: torch.Tensor) -> torch.Tensor:
    """Map labels to 5-way target class: {0..3 actions, 4 Continue}."""
    is_gate = gate_label == 1
    out = torch.full_like(action_label, CONTINUE_CLASS_ID)
    out[is_gate] = action_label[is_gate]
    return out


def build_pred_overall_class(
    gate_prob: torch.Tensor,
    action_logits: torch.Tensor,
    gate_threshold: float,
) -> torch.Tensor:
    """Map model outputs to 5-way prediction class."""
    pred_gate = gate_prob >= gate_threshold
    pred_action = torch.argmax(action_logits, dim=-1)
    out = torch.full_like(pred_action, CONTINUE_CLASS_ID)
    out[pred_gate] = pred_action[pred_gate]
    return out


def single_head_to_legacy_overall_class(pred_single_class: torch.Tensor) -> torch.Tensor:
    """
    Convert single-head class ids to legacy overall class ids used by metrics.

    single-head classes: 0=Continue,1=Compress,2=Redirect,3=ModeSwitch,4=Stop
    legacy overall ids: 4=Continue,0=Compress,1=Redirect,2=ModeSwitch,3=Stop
    """
    out = pred_single_class.clone()
    continue_mask = pred_single_class == 0
    out[continue_mask] = CONTINUE_CLASS_ID
    out[~continue_mask] = pred_single_class[~continue_mask] - 1
    return out


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    gate_threshold: float,
    model_type: str = "attention",
    action_weights: Optional[torch.Tensor] = None,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    gate_correct = 0
    gate_total = 0
    gate_tp = gate_fp = gate_fn = 0
    action_correct = 0
    action_total = 0
    overall_correct = 0
    overall_total = 0

    tp = Counter()
    pred_count = Counter()
    true_count = Counter()

    with torch.no_grad():
        for batch in dataloader:
            signals = batch["signals"].to(device)
            gate_label = batch["gate_label"].to(device)
            action_label = batch["action_label"].to(device)

            if model_type == "single_head":
                class_logits = model(signals)
                loss = model.compute_loss(class_logits, gate_label, action_label)
                pred_single = torch.argmax(class_logits, dim=-1)
                pred_overall = single_head_to_legacy_overall_class(pred_single)
                pred_gate = (pred_overall != CONTINUE_CLASS_ID).float()
            else:
                gate_prob, action_logits = model(signals)
                loss = compute_dual_head_loss(
                    gate_prob,
                    action_logits,
                    gate_label,
                    action_label,
                    action_weights=action_weights,
                )
                pred_gate = (gate_prob >= gate_threshold).float()
                pred_overall = build_pred_overall_class(gate_prob, action_logits, gate_threshold)

            batch_size = signals.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            gate_correct += int((pred_gate == gate_label).sum().item())
            gate_total += batch_size
            gate_true = (gate_label >= 0.5).long()
            gate_pred = (pred_gate >= 0.5).long()
            gate_tp += int(((gate_pred == 1) & (gate_true == 1)).sum().item())
            gate_fp += int(((gate_pred == 1) & (gate_true == 0)).sum().item())
            gate_fn += int(((gate_pred == 0) & (gate_true == 1)).sum().item())

            gate_mask = gate_label == 1
            if gate_mask.any():
                pred_action = pred_overall[gate_mask]
                true_action = action_label[gate_mask]
                action_correct += int((pred_action == true_action).sum().item())
                action_total += int(gate_mask.sum().item())

                for cls_id in ACTION_NAMES:
                    pred_cls = pred_action == cls_id
                    true_cls = true_action == cls_id
                    pred_count[cls_id] += int(pred_cls.sum().item())
                    true_count[cls_id] += int(true_cls.sum().item())
                    tp[cls_id] += int((pred_cls & true_cls).sum().item())

            true_overall = build_true_overall_class(gate_label.long(), action_label)
            overall_correct += int((true_overall == pred_overall).sum().item())
            overall_total += batch_size

    action_precision = {}
    action_recall = {}
    action_support = {}
    for cls_id in ACTION_NAMES:
        p_denom = pred_count[cls_id]
        r_denom = true_count[cls_id]
        action_precision[cls_id] = (tp[cls_id] / p_denom) if p_denom else 0.0
        action_recall[cls_id] = (tp[cls_id] / r_denom) if r_denom else 0.0
        action_support[cls_id] = r_denom

    avg_loss = total_loss / max(total_samples, 1)
    gate_acc = gate_correct / max(gate_total, 1)
    gate_precision = gate_tp / (gate_tp + gate_fp) if (gate_tp + gate_fp) else 0.0
    gate_recall = gate_tp / (gate_tp + gate_fn) if (gate_tp + gate_fn) else 0.0
    gate_f1 = (
        2.0 * gate_precision * gate_recall / (gate_precision + gate_recall)
        if (gate_precision + gate_recall)
        else 0.0
    )
    action_acc = action_correct / max(action_total, 1)
    overall_acc = overall_correct / max(overall_total, 1)
    stop_precision = action_precision.get(3, 0.0)
    return EvalResult(
        loss=avg_loss,
        gate_accuracy=gate_acc,
        gate_f1=gate_f1,
        action_accuracy=action_acc,
        overall_accuracy=overall_acc,
        stop_precision=stop_precision,
        action_precision=action_precision,
        action_recall=action_recall,
        action_support=action_support,
    )


def print_dataset_stats(dataset_splits: Mapping[str, List[Mapping[str, object]]]) -> None:
    print("Dataset statistics:")
    for split_name in ("train", "val", "test"):
        samples = dataset_splits[split_name]
        gate_counter = Counter(int(x["gate_label"]) for x in samples)
        action_counter = Counter(int(x["action_label"]) for x in samples if int(x["gate_label"]) == 1)
        total = len(samples)
        gate_pos = gate_counter.get(1, 0)
        gate_ratio = (gate_pos / total * 100.0) if total else 0.0
        print(f"  {split_name}: {total} samples")
        print(f"    gate distribution: {dict(gate_counter)} (gate=1 ratio={gate_ratio:.2f}%)")
        print(f"    action distribution (gate=1 only): {dict(action_counter)}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str = "attention",
    action_weights: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        signals = batch["signals"].to(device)
        gate_label = batch["gate_label"].to(device)
        action_label = batch["action_label"].to(device)

        if model_type == "single_head":
            class_logits = model(signals)
            loss = model.compute_loss(class_logits, gate_label, action_label)
        else:
            gate_prob, action_logits = model(signals)
            loss = compute_dual_head_loss(
                gate_prob,
                action_logits,
                gate_label,
                action_label,
                action_weights=action_weights,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = signals.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def create_dataloaders(
    data_path: str,
    batch_size: int,
    seed: Optional[int] = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Dict[str, List[Mapping[str, object]]],
    int,
    int,
]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_samples = data.get("train", [])
    val_samples = data.get("val", [])
    test_samples = data.get("test", [])

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        TrajectoryWindowDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(TrajectoryWindowDataset(val_samples), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TrajectoryWindowDataset(test_samples), batch_size=batch_size, shuffle=False)
    signal_dim = 0
    num_steps = 0
    meta = data.get("meta", {})
    if isinstance(meta, dict) and isinstance(meta.get("signal_order"), list):
        signal_dim = len(meta["signal_order"])
    if isinstance(meta, dict) and isinstance(meta.get("window_size"), int):
        num_steps = int(meta["window_size"])

    probe = None
    for split in (train_samples, val_samples, test_samples):
        if split:
            probe = split[0]
            break
    if probe is not None:
        probe_signals = probe.get("signals", [])
        if isinstance(probe_signals, list) and probe_signals:
            if num_steps == 0:
                num_steps = len(probe_signals)
            first_step = probe_signals[0]
            if isinstance(first_step, list) and signal_dim == 0:
                signal_dim = len(first_step)

    if signal_dim <= 0 or num_steps <= 0:
        raise ValueError("Failed to infer signal_dim/num_steps from dataset.")

    return (
        train_loader,
        val_loader,
        test_loader,
        {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        },
        signal_dim,
        num_steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Deliberation Controller (Supervised Learning).")
    parser.add_argument("--data_path", required=True, help="Path to prepared dataset JSON.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument(
        "--model_type",
        type=str,
        default="attention",
        choices=("attention", "mlp", "single_head"),
        help="Controller backbone type.",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights = strict inverse frequency for action CE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def build_model(model_type: str, signal_dim: int, num_steps: int) -> nn.Module:
    if model_type == "attention":
        return DeliberationController(
            signal_dim=signal_dim,
            hidden_dim=64,
            num_steps=num_steps,
            num_actions=4,
        )
    if model_type == "mlp":
        return DeliberationMLPController(
            signal_dim=signal_dim,
            hidden_dim=64,
            num_steps=num_steps,
            num_actions=4,
        )
    if model_type == "single_head":
        return DeliberationSingleHeadController(
            signal_dim=signal_dim,
            hidden_dim=64,
            num_steps=num_steps,
            num_classes=5,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def format_action_metrics(result: EvalResult) -> str:
    lines = []
    for cls_id, cls_name in ACTION_NAMES.items():
        p = result.action_precision[cls_id] * 100.0
        r = result.action_recall[cls_id] * 100.0
        support = result.action_support[cls_id]
        lines.append(f"    {cls_name:<10} precision={p:6.2f}% recall={r:6.2f}% support={support}")
    return "\n".join(lines)


def eval_result_to_dict(result: EvalResult) -> Dict[str, float]:
    return {
        "loss": result.loss,
        "gate_accuracy": result.gate_accuracy,
        "val_gate_f1": result.gate_f1,
        "action_accuracy": result.action_accuracy,
        "val_overall_accuracy": result.overall_accuracy,
        "val_stop_precision": result.stop_precision,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = str(Path(args.save_dir) / "best_controller.pt")
    training_log_path = str(Path(args.save_dir) / "training_log.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (seed={args.seed}, use_class_weights={args.use_class_weights})")

    train_loader, val_loader, test_loader, splits, signal_dim, num_steps = create_dataloaders(
        args.data_path,
        args.batch_size,
        seed=args.seed,
    )
    print_dataset_stats(splits)

    action_weights: Optional[torch.Tensor] = None
    if args.use_class_weights:
        action_weights = build_action_class_weights(splits["train"], device)

    model = build_model(args.model_type, signal_dim=signal_dim, num_steps=num_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {args.model_type} | signal_dim={signal_dim} | "
        f"num_steps={num_steps} | trainable parameters: {n_params}"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_overall = -1.0
    best_epoch = -1
    best_epoch_metrics: Dict[str, float] = {}
    no_improve_epochs = 0
    epoch_history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            model_type=args.model_type,
            action_weights=action_weights,
        )
        val_result = evaluate(
            model,
            val_loader,
            device,
            args.gate_threshold,
            model_type=args.model_type,
            action_weights=action_weights,
        )

        epoch_metrics = eval_result_to_dict(val_result)
        epoch_history.append({"epoch": epoch, "train_loss": train_loss, **epoch_metrics})

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_result.loss:.4f} | "
            f"gate_acc={val_result.gate_accuracy:.4f} | "
            f"gate_f1={val_result.gate_f1:.4f} | "
            f"action_acc={val_result.action_accuracy:.4f} | "
            f"overall_acc={val_result.overall_accuracy:.4f} | "
            f"stop_prec={val_result.stop_precision:.4f}"
        )

        if val_result.overall_accuracy > best_val_overall:
            best_val_overall = val_result.overall_accuracy
            best_epoch = epoch
            best_epoch_metrics = epoch_metrics
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_overall_accuracy": val_result.overall_accuracy,
                    "val_gate_f1": val_result.gate_f1,
                    "val_stop_precision": val_result.stop_precision,
                    "args": vars(args),
                },
                best_model_path,
            )
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"val overall_accuracy did not improve for {args.patience} epochs."
            )
            break

    training_log = {
        "data_path": args.data_path,
        "seed": args.seed,
        "use_class_weights": args.use_class_weights,
        "best_epoch": best_epoch,
        "best_epoch_metrics": best_epoch_metrics,
        "epoch_history": epoch_history,
    }
    with open(training_log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    print(f"Wrote training log: {training_log_path}")

    print(f"Best model from epoch {best_epoch} with val overall_accuracy={best_val_overall:.4f}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_result = evaluate(
        model,
        test_loader,
        device,
        args.gate_threshold,
        model_type=args.model_type,
        action_weights=action_weights,
    )
    print("\nTest Results:")
    print(f"  loss:            {test_result.loss:.4f}")
    print(f"  gate_accuracy:   {test_result.gate_accuracy:.4f}")
    print(f"  action_accuracy: {test_result.action_accuracy:.4f}")
    print(f"  overall_accuracy:{test_result.overall_accuracy:.4f}")
    print("  Action class metrics:")
    print(format_action_metrics(test_result))


if __name__ == "__main__":
    main()
