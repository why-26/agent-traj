"""Train Decision Transformer version of Deliberation Controller."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from deliberation_controller.model.controller_dt import DeliberationDecisionTransformer

ACTION_NAMES = {0: "Compress", 1: "Redirect", 2: "ModeSwitch", 3: "Stop"}
CONTINUE_CLASS_ID = 4


class DTWindowDataset(Dataset):
    def __init__(self, samples: List[Mapping[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "rtg": torch.tensor(item["rtg"], dtype=torch.float32),
            "signals": torch.tensor(item["signals"], dtype=torch.float32),
            "actions": torch.tensor(item["actions"], dtype=torch.long),
            "gate_label": torch.tensor(float(item["gate_label"]), dtype=torch.float32),
            "action_label": torch.tensor(int(item["action_label"]), dtype=torch.long),
        }


@dataclass
class EvalResult:
    loss: float
    gate_accuracy: float
    action_accuracy: float
    overall_accuracy: float


def build_true_overall_class(gate_label: torch.Tensor, action_label: torch.Tensor) -> torch.Tensor:
    is_gate = gate_label == 1
    out = torch.full_like(action_label, CONTINUE_CLASS_ID)
    out[is_gate] = action_label[is_gate]
    return out


def build_pred_overall_class(
    gate_prob: torch.Tensor,
    action_logits: torch.Tensor,
    gate_threshold: float,
) -> torch.Tensor:
    pred_gate = gate_prob >= gate_threshold
    pred_action = torch.argmax(action_logits, dim=-1)
    out = torch.full_like(pred_action, CONTINUE_CLASS_ID)
    out[pred_gate] = pred_action[pred_gate]
    return out


def evaluate(
    model: DeliberationDecisionTransformer,
    dataloader: DataLoader,
    device: torch.device,
    gate_threshold: float,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    gate_correct = 0
    action_correct = 0
    action_total = 0
    overall_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            rtg = batch["rtg"].to(device)
            signals = batch["signals"].to(device)
            actions = batch["actions"].to(device)
            gate_label = batch["gate_label"].to(device)
            action_label = batch["action_label"].to(device)

            gate_prob, action_logits = model(rtg, signals, actions)
            loss = model.compute_loss(gate_prob, action_logits, gate_label, action_label)

            bsz = signals.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

            pred_gate = (gate_prob >= gate_threshold).float()
            gate_correct += int((pred_gate == gate_label).sum().item())

            gate_mask = gate_label == 1
            if gate_mask.any():
                pred_action = torch.argmax(action_logits[gate_mask], dim=-1)
                true_action = action_label[gate_mask]
                action_correct += int((pred_action == true_action).sum().item())
                action_total += int(gate_mask.sum().item())

            true_overall = build_true_overall_class(gate_label.long(), action_label)
            pred_overall = build_pred_overall_class(gate_prob, action_logits, gate_threshold)
            overall_correct += int((true_overall == pred_overall).sum().item())

    return EvalResult(
        loss=total_loss / max(total_samples, 1),
        gate_accuracy=gate_correct / max(total_samples, 1),
        action_accuracy=action_correct / max(action_total, 1),
        overall_accuracy=overall_correct / max(total_samples, 1),
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


def create_dataloaders(
    data_path: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[Mapping[str, object]]], int, int]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_samples = data.get("train", [])
    val_samples = data.get("val", [])
    test_samples = data.get("test", [])

    train_loader = DataLoader(DTWindowDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DTWindowDataset(val_samples), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(DTWindowDataset(test_samples), batch_size=batch_size, shuffle=False)

    signal_dim = 0
    num_steps = 0
    meta = data.get("meta", {})
    if isinstance(meta, dict) and isinstance(meta.get("signal_order"), list):
        signal_dim = len(meta["signal_order"])
    if isinstance(meta, dict) and isinstance(meta.get("window_size"), int):
        num_steps = int(meta["window_size"])

    if signal_dim <= 0 and train_samples:
        signal_dim = len(train_samples[0]["signals"][0])
    if num_steps <= 0 and train_samples:
        num_steps = len(train_samples[0]["signals"])
    if signal_dim <= 0 or num_steps <= 0:
        raise ValueError("Failed to infer signal_dim/num_steps from DT dataset.")

    return (
        train_loader,
        val_loader,
        test_loader,
        {"train": train_samples, "val": val_samples, "test": test_samples},
        signal_dim,
        num_steps,
    )


def train_one_epoch(
    model: DeliberationDecisionTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        rtg = batch["rtg"].to(device)
        signals = batch["signals"].to(device)
        actions = batch["actions"].to(device)
        gate_label = batch["gate_label"].to(device)
        action_label = batch["action_label"].to(device)

        gate_prob, action_logits = model(rtg, signals, actions)
        loss = model.compute_loss(gate_prob, action_logits, gate_label, action_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsz = signals.size(0)
        running_loss += loss.item() * bsz
        total_samples += bsz
    return running_loss / max(total_samples, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DT controller.")
    parser.add_argument("--data_path", required=True, help="Path to DT-prepared dataset JSON.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = str(Path(args.save_dir) / "best_controller_dt.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, splits, signal_dim, num_steps = create_dataloaders(
        args.data_path,
        args.batch_size,
    )
    print_dataset_stats(splits)

    model = DeliberationDecisionTransformer(
        signal_dim=signal_dim,
        num_steps=num_steps,
        hidden_dim=64,
        nhead=4,
        num_layers=3,
        ff_dim=128,
        num_actions=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: decision_transformer | signal_dim={signal_dim} | "
        f"num_steps={num_steps} | trainable parameters: {n_params}"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_overall = -1.0
    best_epoch = -1
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_result = evaluate(model, val_loader, device, args.gate_threshold)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_result.loss:.4f} | "
            f"gate_acc={val_result.gate_accuracy:.4f} | "
            f"action_acc={val_result.action_accuracy:.4f} | "
            f"overall_acc={val_result.overall_accuracy:.4f}"
        )

        if val_result.overall_accuracy > best_val_overall:
            best_val_overall = val_result.overall_accuracy
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_overall_accuracy": val_result.overall_accuracy,
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

    print(f"Best model from epoch {best_epoch} with val overall_accuracy={best_val_overall:.4f}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_result = evaluate(model, test_loader, device, args.gate_threshold)
    print("\nTest Results:")
    print(f"  loss:            {test_result.loss:.4f}")
    print(f"  gate_accuracy:   {test_result.gate_accuracy:.4f}")
    print(f"  action_accuracy: {test_result.action_accuracy:.4f}")
    print(f"  overall_accuracy:{test_result.overall_accuracy:.4f}")


if __name__ == "__main__":
    main()
