"""MLP baseline controller with dual heads."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_ID_TO_NAME = {
    0: "Compress",
    1: "Redirect",
    2: "ModeSwitch",
    3: "Stop",
}


class DeliberationMLPController(nn.Module):
    """
    Flattened-window MLP controller.

    Input: [B, K, 5] -> flatten to [B, K*5]
    Backbone: 25 -> 64 -> 64 with ReLU
    Heads:
      - gate: 64 -> 1 (sigmoid prob)
      - action: 64 -> 4 (logits)
    """

    def __init__(
        self,
        signal_dim: int = 5,
        hidden_dim: int = 64,
        num_steps: int = 5,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.signal_dim = signal_dim
        self.num_steps = num_steps
        self.num_actions = num_actions
        input_dim = signal_dim * num_steps

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x rank 3 [B, K, D], got {tuple(x.shape)}")
        if x.size(1) != self.num_steps or x.size(2) != self.signal_dim:
            raise ValueError(
                f"Expected input shape [B, {self.num_steps}, {self.signal_dim}], got {tuple(x.shape)}"
            )

        h = x.reshape(x.size(0), -1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        gate_logit = self.gate_head(h).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logit)
        action_logits = self.action_head(h)
        return gate_prob, action_logits

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

    @torch.no_grad()
    def decide(
        self,
        signal_window: torch.Tensor | Sequence[Sequence[float]],
        threshold: float = 0.5,
        device: torch.device | str | None = None,
    ) -> dict[str, object]:
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

        gate_prob, action_logits = self.forward(x)
        gate_prob_val = float(gate_prob[0].item())
        if gate_prob_val < threshold:
            result = {
                "gate_prob": gate_prob_val,
                "decision": "Continue",
                "action_id": -1,
                "action_name": "Continue",
            }
        else:
            action_id = int(torch.argmax(action_logits[0]).item())
            result = {
                "gate_prob": gate_prob_val,
                "decision": ACTION_ID_TO_NAME[action_id],
                "action_id": action_id,
                "action_name": ACTION_ID_TO_NAME[action_id],
            }

        if was_training:
            self.train()
        return result
