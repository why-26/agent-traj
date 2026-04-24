"""Decision Transformer controller with dual-head outputs."""

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


class DeliberationDecisionTransformer(nn.Module):
    """
    DT-style controller.

    Inputs:
      - rtg: [B, K]
      - states(signals): [B, K, D]
      - actions: [B, K] where 0=Continue, 1..4 are intervention classes.
    """

    def __init__(
        self,
        signal_dim: int = 5,
        num_steps: int = 5,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        ff_dim: int = 128,
        num_actions: int = 4,
        action_vocab_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.signal_dim = signal_dim
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.action_vocab_size = action_vocab_size
        self.num_actions = num_actions
        self.seq_token_len = 3 * num_steps

        self.rtg_embed = nn.Linear(1, hidden_dim)
        self.state_embed = nn.Linear(signal_dim, hidden_dim)
        self.action_embed = nn.Embedding(action_vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(self.seq_token_len, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.memory_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.post_norm = nn.LayerNorm(hidden_dim)

        self.gate_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Linear(hidden_dim, num_actions)

        nn.init.trunc_normal_(self.memory_token, std=0.02)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        # Upper-triangular -inf mask for autoregressive decoding.
        mask = torch.full((length, length), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        rtg: torch.Tensor,
        signals: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rtg.dim() != 2:
            raise ValueError(f"rtg must be [B, K], got {tuple(rtg.shape)}")
        if signals.dim() != 3:
            raise ValueError(f"signals must be [B, K, D], got {tuple(signals.shape)}")
        if actions.dim() != 2:
            raise ValueError(f"actions must be [B, K], got {tuple(actions.shape)}")
        if signals.size(1) != self.num_steps or rtg.size(1) != self.num_steps or actions.size(1) != self.num_steps:
            raise ValueError("Input sequence length does not match configured num_steps")
        if signals.size(2) != self.signal_dim:
            raise ValueError("Input signal_dim mismatch")

        bsz = signals.size(0)
        device = signals.device

        rtg_tok = self.rtg_embed(rtg.unsqueeze(-1))
        state_tok = self.state_embed(signals)
        action_tok = self.action_embed(actions.clamp(min=0, max=self.action_vocab_size - 1))

        # Interleave as (R_t, s_t, a_t) for each timestep t.
        tokens = torch.stack((rtg_tok, state_tok, action_tok), dim=2).reshape(
            bsz, self.seq_token_len, self.hidden_dim
        )
        pos = torch.arange(self.seq_token_len, device=device).unsqueeze(0)
        tokens = tokens + self.pos_embed(pos)

        memory = self.memory_token.expand(bsz, -1, -1)
        tgt_mask = self._causal_mask(self.seq_token_len, device=device)
        hidden = self.decoder(tgt=tokens, memory=memory, tgt_mask=tgt_mask)
        hidden = self.post_norm(hidden)

        # Predict control decision from the last state token position.
        # token index for s_{K-1} is 3*(K-1)+1
        state_pos = 3 * (self.num_steps - 1) + 1
        h = hidden[:, state_pos, :]

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
        rtg_window: torch.Tensor | Sequence[float],
        signal_window: torch.Tensor | Sequence[Sequence[float]],
        action_window: torch.Tensor | Sequence[int],
        threshold: float = 0.5,
        device: torch.device | str | None = None,
    ) -> dict[str, object]:
        rtg_t = torch.tensor(rtg_window, dtype=torch.float32) if not isinstance(rtg_window, torch.Tensor) else rtg_window
        sig_t = (
            torch.tensor(signal_window, dtype=torch.float32)
            if not isinstance(signal_window, torch.Tensor)
            else signal_window
        )
        act_t = (
            torch.tensor(action_window, dtype=torch.long)
            if not isinstance(action_window, torch.Tensor)
            else action_window
        )

        if rtg_t.dim() == 1:
            rtg_t = rtg_t.unsqueeze(0)
        if sig_t.dim() == 2:
            sig_t = sig_t.unsqueeze(0)
        if act_t.dim() == 1:
            act_t = act_t.unsqueeze(0)

        was_training = self.training
        self.eval()
        if device is not None:
            self.to(device)
            rtg_t = rtg_t.to(device)
            sig_t = sig_t.to(device)
            act_t = act_t.to(device)

        gate_prob, action_logits = self.forward(rtg_t, sig_t, act_t)
        gate_value = float(gate_prob[0].item())
        if gate_value < threshold:
            result = {
                "gate_prob": gate_value,
                "decision": "Continue",
                "action_id": -1,
                "action_name": "Continue",
            }
        else:
            action_id = int(torch.argmax(action_logits[0]).item())
            result = {
                "gate_prob": gate_value,
                "decision": ACTION_ID_TO_NAME[action_id],
                "action_id": action_id,
                "action_name": ACTION_ID_TO_NAME[action_id],
            }

        if was_training:
            self.train()
        return result
