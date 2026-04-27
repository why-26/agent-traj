"""DPO utilities on top of the existing Dual-Head Attention controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from deliberation_controller.model.controller import DeliberationController


@dataclass
class DPOBatchMetrics:
    loss: torch.Tensor
    dpo_margin_mean: torch.Tensor
    chosen_hit_rate: torch.Tensor


def forward_with_logits(model: DeliberationController, signals: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Forward pass that returns gate logits + action logits while reusing the same architecture.

    DeliberationController.forward() returns gate_prob, action_logits, so for DPO we expose
    logits explicitly via the underlying layers.
    """
    if signals.dim() != 3:
        raise ValueError(f"signals must have shape [B,K,D], got {tuple(signals.shape)}")

    hidden = model.input_proj(signals)
    hidden = hidden + model.positional_encoding[:, : signals.size(1), :]
    hidden = model.encoder(hidden)
    pooled = model.post_norm(hidden.mean(dim=1))

    gate_logit = model.gate_head(pooled).squeeze(-1)
    action_logits = model.action_head(pooled)
    return {
        "gate_logit": gate_logit,
        "action_logits": action_logits,
    }


def split_joint_action(joint_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert joint 5-way action into (gate_label, action_label_4way_for_head).

    Joint ids:
      0 = Continue  -> gate=0, action label ignored
      1 = Compress  -> gate=1, action=0
      2 = Redirect  -> gate=1, action=1
      3 = ModeSwitch-> gate=1, action=2
      4 = Stop      -> gate=1, action=3
    """
    if joint_action.dtype not in (torch.int32, torch.int64):
        joint_action = joint_action.long()

    gate = (joint_action > 0).float()
    # safe index for gather; masked out later when gate==0
    action = torch.clamp(joint_action - 1, min=0, max=3)
    return gate, action


def joint_logp(
    model_output: Mapping[str, torch.Tensor],
    gate_label: torch.Tensor,
    action_label_4way: torch.Tensor,
) -> torch.Tensor:
    """
    Joint log-probability log pi(gate, action | s).

    gate=0 (Continue) ignores action term.
    """
    gate_logit = model_output["gate_logit"]
    action_logits = model_output["action_logits"]

    gate_logp = -F.binary_cross_entropy_with_logits(gate_logit, gate_label.float(), reduction="none")

    action_logp_all = F.log_softmax(action_logits, dim=-1)
    chosen_action_logp = action_logp_all.gather(1, action_label_4way.unsqueeze(1)).squeeze(1)

    return gate_logp + chosen_action_logp * gate_label.float()


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> DPOBatchMetrics:
    """
    L_DPO = -log sigma( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )
    """
    advantage = (policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)
    logits = beta * advantage
    loss = -F.logsigmoid(logits).mean()

    # Diagnostics
    with torch.no_grad():
        dpo_margin_mean = advantage.mean()
        chosen_hit_rate = (policy_chosen_logp > policy_rejected_logp).float().mean()

    return DPOBatchMetrics(
        loss=loss,
        dpo_margin_mean=dpo_margin_mean,
        chosen_hit_rate=chosen_hit_rate,
    )
