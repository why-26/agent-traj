"""Dual-Head Temporal Attention controller."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_ID_TO_NAME = {
    0: "Compress",
    1: "Redirect",
    2: "ModeSwitch",
    3: "Stop",
}


class DualHeadTemporalAttention(nn.Module):
    """Temporal attention model with gate/action heads."""

    def __init__(
        self,
        input_dim: int = 5,
        seq_len: int = 5,
        hidden_dim: int = 64,
        nhead: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        num_actions: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_actions = num_actions

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(hidden_dim)

        self.gate_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Linear(hidden_dim, num_actions)

        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [batch, K, input_dim].
        Returns:
            dict containing gate logit/prob and action logits/probs.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x rank 3, got shape={tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got last dim={x.size(-1)}"
            )
        if x.size(1) > self.seq_len:
            raise ValueError(f"Input length {x.size(1)} exceeds configured seq_len={self.seq_len}")

        hidden = self.input_proj(x)
        hidden = hidden + self.positional_encoding[:, : x.size(1), :]
        hidden = self.encoder(hidden)
        pooled = self.post_norm(hidden.mean(dim=1))

        gate_logit = self.gate_head(pooled).squeeze(-1)
        action_logits = self.action_head(pooled)
        gate_prob = torch.sigmoid(gate_logit)
        action_probs = F.softmax(action_logits, dim=-1)

        return {
            "gate_logit": gate_logit,
            "gate_prob": gate_prob,
            "action_logits": action_logits,
            "action_probs": action_probs,
        }

    @torch.no_grad()
    def decide(
        self,
        signal_window: torch.Tensor | Sequence[Sequence[float]],
        threshold: float = 0.5,
        device: torch.device | str | None = None,
    ) -> Dict[str, object]:
        """Inference helper returning Continue vs one intervention action."""
        if not isinstance(signal_window, torch.Tensor):
            x = torch.tensor(signal_window, dtype=torch.float32)
        else:
            x = signal_window.to(dtype=torch.float32)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("signal_window must have shape [K, D] or [B, K, D]")

        was_training = self.training
        self.eval()
        if device is not None:
            x = x.to(device)
            self.to(device)

        outputs = self.forward(x)
        gate_probs = outputs["gate_prob"]
        action_probs = outputs["action_probs"]

        decisions = []
        for i in range(x.size(0)):
            gate_prob = float(gate_probs[i].item())
            if gate_prob < threshold:
                decisions.append(
                    {
                        "gate_prob": gate_prob,
                        "decision": "Continue",
                        "action_id": -1,
                        "action_name": "Continue",
                    }
                )
                continue

            action_id = int(torch.argmax(action_probs[i]).item())
            decisions.append(
                {
                    "gate_prob": gate_prob,
                    "decision": ACTION_ID_TO_NAME[action_id],
                    "action_id": action_id,
                    "action_name": ACTION_ID_TO_NAME[action_id],
                }
            )

        if was_training:
            self.train()

        return decisions[0] if len(decisions) == 1 else {"batch_decisions": decisions}


def compute_multitask_loss(
    model_outputs: Mapping[str, torch.Tensor],
    gate_labels: torch.Tensor,
    action_labels: torch.Tensor,
    lambda_action: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Loss = BCE(gate) + lambda * CE(action|gate=1).
    For action labels, use -100 as ignore index for gate=0 samples.
    """
    gate_logits = model_outputs["gate_logit"]
    action_logits = model_outputs["action_logits"]

    gate_labels = gate_labels.float()
    gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_labels)

    valid_action_mask = action_labels != -100
    if valid_action_mask.any():
        action_loss = F.cross_entropy(action_logits[valid_action_mask], action_labels[valid_action_mask])
    else:
        action_loss = torch.zeros((), device=action_logits.device, dtype=action_logits.dtype)

    total = gate_loss + lambda_action * action_loss
    return {"loss": total, "loss_gate": gate_loss, "loss_action": action_loss}


class DeliberationController(DualHeadTemporalAttention):
    """Compatibility wrapper exposing the requested SL training interface."""

    def __init__(
        self,
        signal_dim: int = 5,
        hidden_dim: int = 64,
        num_steps: int = 5,
        num_actions: int = 4,
        **kwargs: object,
    ) -> None:
        super().__init__(
            input_dim=signal_dim,
            seq_len=num_steps,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = super().forward(x)
        return outputs["gate_prob"], outputs["action_logits"]

    def compute_loss(
        self,
        gate_prob: torch.Tensor,
        action_logits: torch.Tensor,
        gate_label: torch.Tensor,
        action_label: torch.Tensor,
    ) -> torch.Tensor:
        gate_label = gate_label.float()
        gate_loss = F.binary_cross_entropy(gate_prob, gate_label)
        action_loss = F.cross_entropy(action_logits, action_label, ignore_index=-100)
        return gate_loss + action_loss
