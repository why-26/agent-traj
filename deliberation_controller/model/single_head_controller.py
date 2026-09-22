"""Single-head 5-way temporal controller for ablation experiments.

This variant intentionally does not replace the V3b dual-head controller. It
keeps the same temporal-attention stage and predicts Continue plus the four
intervention actions with one softmax head.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "Continue",
    1: "Compress",
    2: "Redirect",
    3: "ModeSwitch",
    4: "Stop",
}
CLASS_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CLASS_ID_TO_NAME.items()}


class SingleHeadTemporalController(nn.Module):
    """Temporal encoder plus one MLP classifier over 5 actions.

    Architecture:
      - 2-layer Transformer encoder with 32-d hidden states by default.
      - Single MLP head: 32 -> 16 -> 5.
      - Class order: Continue, Compress, Redirect, ModeSwitch, Stop.
    """

    def __init__(
        self,
        signal_dim: int = 5,
        num_steps: int = 5,
        hidden_dim: int = 32,
        head_hidden_dim: int = 16,
        nhead: int = 4,
        ff_dim: int = 192,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.signal_dim = signal_dim
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
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
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, num_classes),
        )

        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 5-way logits for x with shape [batch, num_steps, signal_dim]."""
        if x.dim() != 3:
            raise ValueError(f"Expected x rank 3 [B, K, D], got {tuple(x.shape)}")
        if x.size(-1) != self.signal_dim:
            raise ValueError(f"Expected signal_dim={self.signal_dim}, got {x.size(-1)}")
        if x.size(1) > self.num_steps:
            raise ValueError(f"Input length {x.size(1)} exceeds num_steps={self.num_steps}")

        hidden = self.input_proj(x)
        hidden = hidden + self.positional_encoding[:, : x.size(1), :]
        hidden = self.encoder(hidden)
        pooled = self.post_norm(hidden.mean(dim=1))
        return self.classifier(pooled)

    @staticmethod
    def build_targets(gate_label: torch.Tensor, action_label: torch.Tensor) -> torch.Tensor:
        """Map dual-head labels to 5-way targets.

        Dataset labels use action ids 0..3 when gate=1 and -100 when gate=0.
        Single-head targets use 0=Continue and 1..4 for the four actions.
        """
        targets = torch.zeros_like(action_label, dtype=torch.long)
        gate_mask = gate_label.long() == 1
        targets[gate_mask] = action_label[gate_mask].long() + 1
        return targets

    def compute_loss(
        self,
        logits: torch.Tensor,
        gate_label: torch.Tensor,
        action_label: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = self.build_targets(gate_label, action_label)
        return F.cross_entropy(logits, targets, weight=class_weights)

    @torch.no_grad()
    def decide(
        self,
        signal_window: torch.Tensor | Sequence[Sequence[float]],
        device: torch.device | str | None = None,
    ) -> dict[str, object]:
        """Inference helper using argmax over the 5-way softmax."""
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
        class_id = int(torch.argmax(probs[0]).item())

        if was_training:
            self.train()

        return {
            "class_id": class_id,
            "decision": CLASS_ID_TO_NAME[class_id],
            "prob": float(probs[0, class_id].item()),
        }
