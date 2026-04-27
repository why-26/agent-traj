"""Train Deliberation Controller with Direct Preference Optimization (DPO)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from deliberation_controller.model.controller import DeliberationController
from deliberation_controller.model.controller_dpo import (
    dpo_loss,
    forward_with_logits,
    joint_logp,
    split_joint_action,
)

CLASS_NAMES = {0: "Continue", 1: "Compress", 2: "Redirect", 3: "ModeSwitch", 4: "Stop"}


class DPOPairDataset(Dataset):
    def __init__(self, samples: List[Mapping[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "state": torch.tensor(item["state"], dtype=torch.float32),
            "action_chosen": torch.tensor(int(item["action_chosen"]), dtype=torch.long),
            "action_rejected": torch.tensor(int(item["action_rejected"]), dtype=torch.long),
        }


class SLEvalDataset(Dataset):
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
class DPOEvalResult:
    loss: float
    margin: float
    chosen_hit_rate: float


@dataclass
class ClassifEvalResult:
    gate_accuracy: float
    action_accuracy: float
    overall_accuracy: float
    prf: Dict[int, Dict[str, float]]


def load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def load_dpo_splits(path_all_pairs: str) -> Tuple[List[Mapping[str, object]], List[Mapping[str, object]], List[Mapping[str, object]]]:
    base = Path(path_all_pairs)
    train_path = base.with_name(base.stem + "_train.json")
    val_path = base.with_name(base.stem + "_val.json")
    test_path = base.with_name(base.stem + "_test.json")

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            "DPO split files not found. Run prepare_dataset_dpo.py first to generate "
            f"{train_path.name}/{val_path.name}/{test_path.name}."
        )

    train = load_json(train_path)
    val = load_json(val_path)
    test = load_json(test_path)
    if not isinstance(train, list) or not isinstance(val, list) or not isinstance(test, list):
        raise ValueError("DPO split files must be JSON lists.")
    return train, val, test


def infer_signal_shape_from_sl_data(sl_data_path: str) -> Tuple[int, int]:
    data = load_json(sl_data_path)
    if not isinstance(data, Mapping):
        raise ValueError("SL data must be dict with train/val/test.")

    for split in ("train", "val", "test"):
        arr = data.get(split)
        if isinstance(arr, list) and arr:
            sample = arr[0]
            signals = sample["signals"]
            return len(signals[0]), len(signals)
    raise ValueError("Cannot infer signal_dim/num_steps from SL data.")


def load_sl_checkpoint_weights(model: DeliberationController, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def evaluate_dpo_objective(
    policy: DeliberationController,
    reference: DeliberationController,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
) -> DPOEvalResult:
    policy.eval()
    reference.eval()

    total_loss = 0.0
    total_margin = 0.0
    total_hit = 0.0
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch["state"].to(device)
            chosen = batch["action_chosen"].to(device)
            rejected = batch["action_rejected"].to(device)

            gate_c, action_c = split_joint_action(chosen)
            gate_r, action_r = split_joint_action(rejected)

            out_p = forward_with_logits(policy, state)
            out_ref = forward_with_logits(reference, state)

            p_c = joint_logp(out_p, gate_c, action_c)
            p_r = joint_logp(out_p, gate_r, action_r)
            r_c = joint_logp(out_ref, gate_c, action_c)
            r_r = joint_logp(out_ref, gate_r, action_r)

            m = dpo_loss(p_c, p_r, r_c, r_r, beta=beta)
            bsz = state.size(0)
            total_loss += float(m.loss.item()) * bsz
            total_margin += float(m.dpo_margin_mean.item()) * bsz
            total_hit += float(m.chosen_hit_rate.item()) * bsz
            n += bsz

    if n == 0:
        return DPOEvalResult(loss=0.0, margin=0.0, chosen_hit_rate=0.0)
    return DPOEvalResult(
        loss=total_loss / n,
        margin=total_margin / n,
        chosen_hit_rate=total_hit / n,
    )


def train_one_epoch(
    policy: DeliberationController,
    reference: DeliberationController,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
) -> DPOEvalResult:
    policy.train()
    reference.eval()

    total_loss = 0.0
    total_margin = 0.0
    total_hit = 0.0
    n = 0

    for batch in dataloader:
        state = batch["state"].to(device)
        chosen = batch["action_chosen"].to(device)
        rejected = batch["action_rejected"].to(device)

        gate_c, action_c = split_joint_action(chosen)
        gate_r, action_r = split_joint_action(rejected)

        out_p = forward_with_logits(policy, state)
        with torch.no_grad():
            out_ref = forward_with_logits(reference, state)

        p_c = joint_logp(out_p, gate_c, action_c)
        p_r = joint_logp(out_p, gate_r, action_r)
        r_c = joint_logp(out_ref, gate_c, action_c)
        r_r = joint_logp(out_ref, gate_r, action_r)

        m = dpo_loss(p_c, p_r, r_c, r_r, beta=beta)

        optimizer.zero_grad()
        m.loss.backward()
        optimizer.step()

        bsz = state.size(0)
        total_loss += float(m.loss.item()) * bsz
        total_margin += float(m.dpo_margin_mean.item()) * bsz
        total_hit += float(m.chosen_hit_rate.item()) * bsz
        n += bsz

    if n == 0:
        return DPOEvalResult(loss=0.0, margin=0.0, chosen_hit_rate=0.0)
    return DPOEvalResult(
        loss=total_loss / n,
        margin=total_margin / n,
        chosen_hit_rate=total_hit / n,
    )


def evaluate_classification(
    model: DeliberationController,
    sl_test_samples: List[Mapping[str, object]],
    device: torch.device,
    gate_threshold: float,
    batch_size: int = 256,
) -> ClassifEvalResult:
    loader = DataLoader(SLEvalDataset(sl_test_samples), batch_size=batch_size, shuffle=False)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    gate_true: List[int] = []
    gate_pred: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["signals"].to(device)
            g = batch["gate_label"].to(device)
            a = batch["action_label"].to(device)

            gate_prob, action_logits = model(x)
            pg = (gate_prob >= gate_threshold).long()
            pa = torch.argmax(action_logits, dim=-1)

            true_cls = torch.where(g == 1, a + 1, torch.zeros_like(a))
            pred_cls = torch.where(pg == 1, pa + 1, torch.zeros_like(pa))

            y_true.extend(int(v) for v in true_cls.cpu().tolist())
            y_pred.extend(int(v) for v in pred_cls.cpu().tolist())
            gate_true.extend(int(v) for v in g.cpu().tolist())
            gate_pred.extend(int(v) for v in pg.cpu().tolist())

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

    return ClassifEvalResult(
        gate_accuracy=gate_acc,
        action_accuracy=action_acc,
        overall_accuracy=overall_acc,
        prf=prf,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPO controller from pairwise preferences.")
    parser.add_argument(
        "--dpo_pairs_path",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_dpo_pairs.json",
        help="Base path of DPO pairs (expects *_train/_val/_test split files).",
    )
    parser.add_argument(
        "--sl_data_path",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules.json",
        help="SL dataset path for final classification eval.",
    )
    parser.add_argument(
        "--sl_checkpoint",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/best_controller.pt",
        help="Reference and warm-start checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument(
        "--save_dir",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dpo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = str(Path(args.save_dir) / "best_controller_dpo.pt")

    train_pairs, val_pairs, test_pairs = load_dpo_splits(args.dpo_pairs_path)
    print(f"DPO pairs: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    if len(train_pairs) == 0:
        raise ValueError(
            "No DPO training pairs found. Check task-level success/fail pairing source data."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    signal_dim, num_steps = infer_signal_shape_from_sl_data(args.sl_data_path)
    print(f"Model shape inferred from SL data: signal_dim={signal_dim}, num_steps={num_steps}")

    policy = DeliberationController(signal_dim=signal_dim, hidden_dim=64, num_steps=num_steps, num_actions=4).to(device)
    reference = DeliberationController(signal_dim=signal_dim, hidden_dim=64, num_steps=num_steps, num_actions=4).to(device)
    load_sl_checkpoint_weights(policy, args.sl_checkpoint, device)
    load_sl_checkpoint_weights(reference, args.sl_checkpoint, device)

    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    train_loader = DataLoader(DPOPairDataset(train_pairs), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DPOPairDataset(val_pairs), batch_size=args.batch_size, shuffle=False)

    best_hit = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_res = train_one_epoch(policy, reference, train_loader, optimizer, device, beta=args.beta)
        val_res = evaluate_dpo_objective(policy, reference, val_loader, device, beta=args.beta)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_res.loss:.6f} | train_margin={train_res.margin:.6f} | train_hit={train_res.chosen_hit_rate:.4f} | "
            f"val_loss={val_res.loss:.6f} | val_margin={val_res.margin:.6f} | val_hit={val_res.chosen_hit_rate:.4f}"
        )

        if val_res.chosen_hit_rate > best_hit:
            best_hit = val_res.chosen_hit_rate
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_chosen_hit_rate": best_hit,
                    "args": vars(args),
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: val chosen_hit_rate did not improve for {args.patience} epochs."
            )
            break

    print(f"Best model from epoch {best_epoch}, val chosen_hit_rate={best_hit:.4f}")

    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["model_state_dict"], strict=True)

    sl_data = load_json(args.sl_data_path)
    if not isinstance(sl_data, Mapping) or not isinstance(sl_data.get("test"), list):
        raise ValueError("SL data path must contain dict with 'test' list.")

    classif = evaluate_classification(
        policy,
        sl_test_samples=sl_data["test"],
        device=device,
        gate_threshold=args.gate_threshold,
        batch_size=max(args.batch_size, 256),
    )

    print("\nDPO Model Classification Eval on SL Test Split:")
    print(f"  gate_accuracy:    {classif.gate_accuracy:.4f}")
    print(f"  action_accuracy:  {classif.action_accuracy:.4f}")
    print(f"  overall_accuracy: {classif.overall_accuracy:.4f}")
    print("  per-class P/R/F1:")
    for cls_id, name in CLASS_NAMES.items():
        m = classif.prf[cls_id]
        print(
            f"    {name:<10} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} support={int(m['support'])}"
        )


if __name__ == "__main__":
    main()
