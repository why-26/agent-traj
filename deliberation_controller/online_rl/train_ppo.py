from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from deliberation_controller.model.controller import DeliberationController
from deliberation_controller.model.controller_dt import DeliberationDecisionTransformer
from deliberation_controller.online_rl.env_base import BaseEnv
from deliberation_controller.online_rl.replay_env import ReplayEnv
from deliberation_controller.online_rl.search_o1_env import SearchO1Env

ACTION_NAMES = {0: "Continue", 1: "Compress", 2: "Redirect", 3: "ModeSwitch", 4: "Stop"}


class SLTestDataset(Dataset):
    def __init__(self, samples: Sequence[Mapping[str, object]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.samples[idx]
        return {
            "signals": torch.tensor(x["signals"], dtype=torch.float32),
            "gate_label": torch.tensor(int(x["gate_label"]), dtype=torch.long),
            "action_label": torch.tensor(int(x["action_label"]), dtype=torch.long),
        }


class DTTestDataset(Dataset):
    def __init__(self, samples: Sequence[Mapping[str, object]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.samples[idx]
        return {
            "rtg": torch.tensor(x["rtg"], dtype=torch.float32),
            "signals": torch.tensor(x["signals"], dtype=torch.float32),
            "actions": torch.tensor(x["actions"], dtype=torch.long),
            "gate_label": torch.tensor(int(x["gate_label"]), dtype=torch.long),
            "action_label": torch.tensor(int(x["action_label"]), dtype=torch.long),
        }


@dataclass
class EvalMetrics:
    gate_accuracy: float
    action_accuracy: float
    overall_accuracy: float
    prf: Dict[int, Dict[str, float]]
    pred_action_dist: Dict[int, float]


class PPOController(nn.Module):
    """Dual-head actor + value head, reusing existing attention architecture."""

    def __init__(
        self,
        signal_dim: int = 5,
        num_steps: int = 5,
        hidden_dim: int = 64,
        nhead: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = signal_dim
        self.seq_len = num_steps

        self.input_proj = nn.Linear(signal_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_steps, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(hidden_dim)

        # same as SL dual-head actor
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Linear(hidden_dim, 4)
        # critic head
        self.value_head = nn.Linear(hidden_dim, 1)

        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.positional_encoding[:, : x.size(1), :]
        h = self.encoder(h)
        return self.post_norm(h.mean(dim=1))

    @staticmethod
    def joint_log_probs(gate_logit: torch.Tensor, action_logits: torch.Tensor) -> torch.Tensor:
        # 5-way action log-probs: Continue + 4 interventions
        log_p_continue = F.logsigmoid(-gate_logit)
        log_p_gate = F.logsigmoid(gate_logit)
        log_p_action = F.log_softmax(action_logits, dim=-1)
        log_interventions = log_p_gate.unsqueeze(1) + log_p_action
        return torch.cat([log_p_continue.unsqueeze(1), log_interventions], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        gate_logit = self.gate_head(z).squeeze(-1)
        action_logits = self.action_head(z)
        value = self.value_head(z).squeeze(-1)
        logp = self.joint_log_probs(gate_logit, action_logits)
        probs = torch.exp(logp)
        return {
            "gate_logit": gate_logit,
            "action_logits": action_logits,
            "joint_log_probs": logp,
            "joint_probs": probs,
            "value": value,
        }


def load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "avg_reward",
                "success_rate",
                "avg_total_tokens",
                "avg_episode_length",
                "action_dist_continue",
                "action_dist_compress",
                "action_dist_redirect",
                "action_dist_modeswitch",
                "action_dist_stop",
                "kl_divergence",
                "value_loss",
                "policy_loss",
                "offline_overall_accuracy",
                "offline_gate_accuracy",
                "offline_action_accuracy",
            ]
        )


def append_csv_row(path: Path, row: List[object]) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def summarize_metrics(y_true: List[int], y_pred: List[int], gate_true: List[int]) -> EvalMetrics:
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
    for cls in range(5):
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

    pred_cnt = Counter(y_pred)
    tot = max(len(y_pred), 1)
    pred_dist = {k: pred_cnt.get(k, 0) / tot for k in range(5)}

    return EvalMetrics(
        gate_accuracy=gate_acc,
        action_accuracy=action_acc,
        overall_accuracy=overall_acc,
        prf=prf,
        pred_action_dist=pred_dist,
    )


def evaluate_policy_on_sl_test(
    model: PPOController,
    sl_test_samples: Sequence[Mapping[str, object]],
    device: torch.device,
    batch_size: int = 256,
) -> EvalMetrics:
    loader = DataLoader(SLTestDataset(sl_test_samples), batch_size=batch_size, shuffle=False)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    gate_true: List[int] = []

    with torch.no_grad():
        for b in loader:
            x = b["signals"].to(device)
            g = b["gate_label"].to(device)
            a = b["action_label"].to(device)

            out = model(x)
            pred = torch.argmax(out["joint_probs"], dim=-1)
            true_cls = torch.where(g == 1, a + 1, torch.zeros_like(a))

            y_true.extend(int(v) for v in true_cls.cpu().tolist())
            y_pred.extend(int(v) for v in pred.cpu().tolist())
            gate_true.extend(int(v) for v in g.cpu().tolist())

    return summarize_metrics(y_true, y_pred, gate_true)


def evaluate_sl_baseline(
    ckpt_path: str,
    sl_test_samples: Sequence[Mapping[str, object]],
    signal_dim: int,
    num_steps: int,
    device: torch.device,
) -> EvalMetrics:
    model = DeliberationController(signal_dim=signal_dim, hidden_dim=64, num_steps=num_steps, num_actions=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    loader = DataLoader(SLTestDataset(sl_test_samples), batch_size=256, shuffle=False)
    y_true: List[int] = []
    y_pred: List[int] = []
    gate_true: List[int] = []

    with torch.no_grad():
        for b in loader:
            x = b["signals"].to(device)
            g = b["gate_label"].to(device)
            a = b["action_label"].to(device)
            gate_prob, action_logits = model(x)
            pred_gate = (gate_prob >= 0.5).long()
            pred_action = torch.argmax(action_logits, dim=-1) + 1
            pred = torch.where(pred_gate == 1, pred_action, torch.zeros_like(pred_action))
            true_cls = torch.where(g == 1, a + 1, torch.zeros_like(a))
            y_true.extend(int(v) for v in true_cls.cpu().tolist())
            y_pred.extend(int(v) for v in pred.cpu().tolist())
            gate_true.extend(int(v) for v in g.cpu().tolist())

    return summarize_metrics(y_true, y_pred, gate_true)


def evaluate_dt_baseline(
    ckpt_path: str,
    dt_data_path: str,
    signal_dim: int,
    num_steps: int,
    device: torch.device,
) -> EvalMetrics:
    data = load_json(dt_data_path)
    test = data["test"]
    model = DeliberationDecisionTransformer(
        signal_dim=signal_dim,
        num_steps=num_steps,
        hidden_dim=64,
        nhead=4,
        num_layers=3,
        ff_dim=128,
        num_actions=4,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    loader = DataLoader(DTTestDataset(test), batch_size=256, shuffle=False)
    y_true: List[int] = []
    y_pred: List[int] = []
    gate_true: List[int] = []

    with torch.no_grad():
        for b in loader:
            rtg = b["rtg"].to(device)
            sig = b["signals"].to(device)
            acts = b["actions"].to(device)
            g = b["gate_label"].to(device)
            a = b["action_label"].to(device)
            gp, action_logits = model(rtg, sig, acts)
            pred_gate = (gp >= 0.5).long()
            pred_action = torch.argmax(action_logits, dim=-1) + 1
            pred = torch.where(pred_gate == 1, pred_action, torch.zeros_like(pred_action))
            true_cls = torch.where(g == 1, a + 1, torch.zeros_like(a))
            y_true.extend(int(v) for v in true_cls.cpu().tolist())
            y_pred.extend(int(v) for v in pred.cpu().tolist())
            gate_true.extend(int(v) for v in g.cpu().tolist())

    return summarize_metrics(y_true, y_pred, gate_true)


def build_env(args: argparse.Namespace) -> BaseEnv:
    if args.env_type == "replay":
        return ReplayEnv(dataset_path=args.sl_data_path, trajectories_path=args.trajectories_path, seed=args.seed)
    if args.env_type == "search_o1":
        return SearchO1Env()  # TODO backend integration
    raise ValueError(f"Unknown env_type: {args.env_type}")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def write_report(
    report_path: Path,
    metrics_csv: Path,
    sl_eval: EvalMetrics,
    dt_eval: EvalMetrics,
    ppo_eval: EvalMetrics,
    sl_dist: Dict[int, float],
    ppo_dist: Dict[int, float],
) -> None:
    rows = []
    if metrics_csv.exists():
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    lines: List[str] = []
    lines.append("# PPO Replay Training Report")
    lines.append("")
    lines.append("## Training curves")
    lines.append("")
    lines.append("| Iter | avg_reward | success_rate | avg_total_tokens | avg_episode_length | KL | value_loss | policy_loss | offline_overall |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['iteration']} | {float(r['avg_reward']):.4f} | {float(r['success_rate']):.4f} | "
            f"{float(r['avg_total_tokens']):.2f} | {float(r['avg_episode_length']):.2f} | {float(r['kl_divergence']):.6f} | "
            f"{float(r['value_loss']):.6f} | {float(r['policy_loss']):.6f} | {float(r['offline_overall_accuracy']):.4f} |"
        )

    lines.append("")
    lines.append("## Final comparison table")
    lines.append("")
    lines.append("| Model | Overall | Gate Acc | Action Acc | Continue F1 | Compress F1 | ModeSwitch F1 | Stop F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in [("SL", sl_eval), ("DT", dt_eval), ("PPO", ppo_eval)]:
        lines.append(
            f"| {name} | {m.overall_accuracy:.4f} | {m.gate_accuracy:.4f} | {m.action_accuracy:.4f} | "
            f"{m.prf[0]['f1']:.4f} | {m.prf[1]['f1']:.4f} | {m.prf[3]['f1']:.4f} | {m.prf[4]['f1']:.4f} |"
        )

    lines.append("")
    lines.append("## Action distribution shift")
    lines.append("")
    lines.append("| Action | SL trigger ratio | PPO trigger ratio | Delta |")
    lines.append("|---|---:|---:|---:|")
    for i in range(5):
        s = sl_dist.get(i, 0.0)
        p = ppo_dist.get(i, 0.0)
        lines.append(f"| {ACTION_NAMES[i]} | {s:.4f} | {p:.4f} | {p - s:+.4f} |")

    lines.append("")
    lines.append("## Discussion")
    lines.append("")
    lines.append(
        f"- PPO vs SL overall delta: {ppo_eval.overall_accuracy - sl_eval.overall_accuracy:+.4f}."
    )
    lines.append(
        f"- PPO vs DT overall delta: {ppo_eval.overall_accuracy - dt_eval.overall_accuracy:+.4f}."
    )
    lines.append(
        f"- Stop F1 delta (PPO-SL): {ppo_eval.prf[4]['f1'] - sl_eval.prf[4]['f1']:+.4f}."
    )
    lines.append("- Result is reported directly from replay PPO + offline test evaluation without manual reweighting.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online RL PPO training for deliberation controller.")
    p.add_argument("--env_type", choices=["replay", "search_o1"], default="replay")
    p.add_argument(
        "--sl_data_path",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules.json",
    )
    p.add_argument(
        "--dt_data_path",
        default="/data/wanghy/agent_traj/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules_dt.json",
    )
    p.add_argument(
        "--trajectories_path",
        default="/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/qwen3-4b-thinking-hotpotqa-all.triples.json",
    )
    p.add_argument(
        "--sl_checkpoint",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/best_controller.pt",
    )
    p.add_argument(
        "--dt_checkpoint",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/dt/best_controller_dt.pt",
    )
    p.add_argument(
        "--save_dir",
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/ppo_replay",
    )
    p.add_argument(
        "--report_path",
        default="/data/wanghy/agent_traj/deliberation_controller/eval/results_ppo_replay.md",
    )

    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--episodes_per_iter", type=int, default=64)
    p.add_argument("--max_episode_steps", type=int, default=20)
    p.add_argument("--ppo_epochs", type=int, default=3)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--kl_beta", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef", type=float, default=0.001)
    p.add_argument("--update_batch_size", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = Path(args.save_dir)
    metrics_csv = save_dir / "train_metrics.csv"
    save_csv_header(metrics_csv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sl_data = load_json(args.sl_data_path)
    if not isinstance(sl_data, Mapping):
        raise ValueError("SL data must be dict json")
    sl_test = sl_data["test"]

    signal_dim = len(sl_test[0]["signals"][0])
    num_steps = len(sl_test[0]["signals"])

    env = build_env(args)
    if isinstance(env, ReplayEnv):
        sanity = env.run_random_episodes(num_episodes=100, max_steps=args.max_episode_steps)
        print(
            "Replay sanity (100 random episodes): "
            f"success_rate={sanity['success_rate']:.4f}, avg_reward={sanity['avg_reward']:.4f}, "
            f"avg_len={sanity['avg_episode_length']:.2f}"
        )

    model = PPOController(signal_dim=signal_dim, num_steps=num_steps).to(device)
    ref_model = PPOController(signal_dim=signal_dim, num_steps=num_steps).to(device)

    # warm start from SL checkpoint
    ckpt = torch.load(args.sl_checkpoint, map_location=device)
    sl_state = ckpt["model_state_dict"] if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sl_state, strict=False)
    if missing:
        print(f"[init] missing keys (expected value_head): {missing}")
    if unexpected:
        print(f"[init] unexpected keys: {unexpected}")
    ref_model.load_state_dict(model.state_dict(), strict=True)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_overall = -1.0
    best_iter = -1
    no_improve = 0

    # Track action distribution shift / hacking
    prev_action_dist: Dict[int, float] = {i: 0.2 for i in range(5)}

    for it in range(1, args.iterations + 1):
        states: List[np.ndarray] = []
        actions: List[int] = []
        old_logprobs: List[float] = []
        values: List[float] = []
        rewards: List[float] = []
        dones: List[float] = []

        ep_rewards: List[float] = []
        ep_success: List[float] = []
        ep_tokens: List[float] = []
        ep_lens: List[int] = []
        action_counter = Counter()

        for _ in range(args.episodes_per_iter):
            state, info = env.reset()
            done = False
            ep_r = 0.0
            ep_len = 0
            last_info = info

            while not done and ep_len < args.max_episode_steps:
                st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    out = model(st)
                    dist = Categorical(probs=out["joint_probs"])
                    act = int(dist.sample().item())
                    logp = float(dist.log_prob(torch.tensor([act], device=device)).item())
                    val = float(out["value"].item())

                next_state, reward, done, step_info = env.step(act)

                states.append(np.asarray(state, dtype=np.float32))
                actions.append(act)
                old_logprobs.append(logp)
                values.append(val)
                rewards.append(float(reward))
                dones.append(1.0 if done else 0.0)

                action_counter[act] += 1
                ep_r += float(reward)
                ep_len += 1
                state = next_state
                last_info = step_info

            ep_rewards.append(ep_r)
            ep_success.append(float(bool(last_info.get("success", False))))
            ep_tokens.append(float(last_info.get("total_tokens", 8000.0)))
            ep_lens.append(ep_len)

        # tensors
        states_t = torch.tensor(np.stack(states, axis=0), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_logp_t = torch.tensor(old_logprobs, dtype=torch.float32, device=device)
        values_t = torch.tensor(values, dtype=torch.float32, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        adv_t, ret_t = compute_gae(rewards_t, values_t, dones_t, gamma=args.gamma, lam=args.gae_lambda)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO updates
        policy_loss_meter = 0.0
        value_loss_meter = 0.0
        kl_meter = 0.0
        n_updates = 0

        n = states_t.size(0)
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.update_batch_size):
                idx = perm[start : start + args.update_batch_size]

                b_states = states_t[idx]
                b_actions = actions_t[idx]
                b_old_logp = old_logp_t[idx]
                b_adv = adv_t[idx]
                b_ret = ret_t[idx]

                out = model(b_states)
                logp_all = out["joint_log_probs"]
                new_logp = logp_all.gather(1, b_actions.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(new_logp - b_old_logp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(out["value"], b_ret)
                entropy = Categorical(probs=out["joint_probs"]).entropy().mean()

                with torch.no_grad():
                    ref_out = ref_model(b_states)
                kl = F.kl_div(
                    out["joint_log_probs"],
                    ref_out["joint_probs"],
                    reduction="batchmean",
                    log_target=False,
                )

                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy
                    + args.kl_beta * kl
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                policy_loss_meter += float(policy_loss.item())
                value_loss_meter += float(value_loss.item())
                kl_meter += float(kl.item())
                n_updates += 1

        # iteration metrics
        total_actions = max(sum(action_counter.values()), 1)
        action_dist = {i: action_counter.get(i, 0) / total_actions for i in range(5)}
        avg_reward = float(np.mean(ep_rewards) if ep_rewards else 0.0)
        success_rate = float(np.mean(ep_success) if ep_success else 0.0)
        avg_tokens = float(np.mean(ep_tokens) if ep_tokens else 0.0)
        avg_ep_len = float(np.mean(ep_lens) if ep_lens else 0.0)

        kl_avg = kl_meter / max(n_updates, 1)
        value_loss_avg = value_loss_meter / max(n_updates, 1)
        policy_loss_avg = policy_loss_meter / max(n_updates, 1)

        offline_metrics = EvalMetrics(0.0, 0.0, 0.0, {i: {"precision": 0, "recall": 0, "f1": 0, "support": 0} for i in range(5)}, {i: 0 for i in range(5)})
        if it % args.eval_every == 0:
            offline_metrics = evaluate_policy_on_sl_test(model, sl_test, device=device)

            if offline_metrics.overall_accuracy > best_overall:
                best_overall = offline_metrics.overall_accuracy
                best_iter = it
                no_improve = 0
                torch.save(
                    {
                        "iteration": it,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_overall_accuracy": best_overall,
                        "args": vars(args),
                    },
                    save_dir / "best_controller_ppo.pt",
                )
            else:
                no_improve += 1

        if it % 50 == 0:
            torch.save(
                {
                    "iteration": it,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                save_dir / f"iter_{it}.pt",
            )

        # reward hacking monitoring (every 10 iterations)
        if it % 10 == 0:
            print(
                f"Iter {it:03d} | avg_reward={avg_reward:.4f} | sr={success_rate:.4f} | "
                f"avg_tokens={avg_tokens:.1f} | ep_len={avg_ep_len:.2f} | "
                f"action_dist={{{', '.join([f'{ACTION_NAMES[k]}:{action_dist[k]:.3f}' for k in range(5)])}}} | "
                f"KL={kl_avg:.6f} | V={value_loss_avg:.6f} | P={policy_loss_avg:.6f} | "
                f"offline_overall={offline_metrics.overall_accuracy:.4f}"
            )

            # Early warning: Stop/Compress collapse.
            for bad_action in (1, 4):  # Compress / Stop
                freq = action_dist.get(bad_action, 0.0)
                if freq > 0.5 or freq == 0.0:
                    print(
                        f"[warning] action {ACTION_NAMES[bad_action]} frequency={freq:.4f}; "
                        "possible reward hacking. Early stopping triggered."
                    )
                    no_improve = max(no_improve, args.early_stop_patience)

        append_csv_row(
            metrics_csv,
            [
                it,
                avg_reward,
                success_rate,
                avg_tokens,
                avg_ep_len,
                action_dist[0],
                action_dist[1],
                action_dist[2],
                action_dist[3],
                action_dist[4],
                kl_avg,
                value_loss_avg,
                policy_loss_avg,
                offline_metrics.overall_accuracy,
                offline_metrics.gate_accuracy,
                offline_metrics.action_accuracy,
            ],
        )

        prev_action_dist = action_dist

        if no_improve >= args.early_stop_patience:
            print(
                f"Early stopping at iteration {it}: offline overall did not improve for "
                f"{args.early_stop_patience} eval checks or reward-hacking warning triggered."
            )
            break

    # final eval/report
    best_ckpt_path = save_dir / "best_controller_ppo.pt"
    if best_ckpt_path.exists():
        best = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best["model_state_dict"], strict=True)

    ppo_eval = evaluate_policy_on_sl_test(model, sl_test, device=device)
    sl_eval = evaluate_sl_baseline(args.sl_checkpoint, sl_test, signal_dim, num_steps, device)
    dt_eval = evaluate_dt_baseline(args.dt_checkpoint, args.dt_data_path, signal_dim, num_steps, device)

    write_report(
        report_path=Path(args.report_path),
        metrics_csv=metrics_csv,
        sl_eval=sl_eval,
        dt_eval=dt_eval,
        ppo_eval=ppo_eval,
        sl_dist=sl_eval.pred_action_dist,
        ppo_dist=ppo_eval.pred_action_dist,
    )

    print(f"Best offline overall={best_overall:.4f} at iter={best_iter}")
    print(f"Saved best checkpoint: {best_ckpt_path}")
    print(f"Saved training csv: {metrics_csv}")
    print(f"Saved report: {args.report_path}")

    env.close()


if __name__ == "__main__":
    main()
