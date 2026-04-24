"""Single-head 5-class controller baseline."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Single-head class order:
# 0=Continue, 1=Compress, 2=Redirect, 3=ModeSwitch, 4=Stop
CLASS_ID_TO_NAME = {
    0: "Continue",
    1: "Compress",
    2: "Redirect",
    3: "ModeSwitch",
    4: "Stop",
}


class DeliberationSingleHeadController(nn.Module):
    """Temporal attention backbone with a single 5-way softmax head."""

    def __init__(
        self,
        signal_dim: int = 5,
        hidden_dim: int = 64,
        num_steps: int = 5,
        nhead: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.signal_dim = signal_dim
        self.num_steps = num_steps
        self.num_classes = num_classes

        self.input_proj = nn.Linear(signal_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_steps, hidden_dim))
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
        self.classifier = nn.Linear(hidden_dim, num_classes)

        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x rank 3 [B, K, D], got {tuple(x.shape)}")
        if x.size(1) != self.num_steps or x.size(2) != self.signal_dim:
            raise ValueError(
                f"Expected input shape [B, {self.num_steps}, {self.signal_dim}], got {tuple(x.shape)}"
            )

        h = self.input_proj(x)
        h = h + self.positional_encoding[:, : x.size(1), :]
        h = self.encoder(h)
        h = self.post_norm(h.mean(dim=1))
        return self.classifier(h)

    def _build_targets_from_gate_action(
        self,
        gate_label: torch.Tensor,
        action_label: torch.Tensor,
    ) -> torch.Tensor:
        # Single-head target: Continue=0, actions are shifted by +1.
        targets = torch.zeros_like(action_label, dtype=torch.long)
        gate_mask = gate_label == 1
        targets[gate_mask] = action_label[gate_mask] + 1
        return targets

    def compute_loss(
        self,
        logits: torch.Tensor,
        gate_label: torch.Tensor,
        action_label: torch.Tensor,
    ) -> torch.Tensor:
        targets = self._build_targets_from_gate_action(gate_label, action_label)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def decide(
        self,
        signal_window: torch.Tensor | Sequence[Sequence[float]],
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

        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        cls_id = int(torch.argmax(probs[0]).item())

        if was_training:
            self.train()

        return {
            "class_id": cls_id,
            "decision": CLASS_ID_TO_NAME[cls_id],
            "prob": float(probs[0, cls_id].item()),
        }
