"""Wrapper that injects controller decisions into an agent step loop."""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, List, Mapping, Sequence

import torch

from deliberation_controller.data.normalizer import PercentileNormalizer
from deliberation_controller.data.signal_extractor import extract_step_signals, signals_to_vector
from deliberation_controller.intervene.intervention import InterventionExecutor
from deliberation_controller.model.controller import DeliberationController

# 0=Continue, 1=Compress, 2=Redirect, 3=ModeSwitch, 4=Stop
ACTION_NAMES = {
    0: "continue",
    1: "compress",
    2: "redirect",
    3: "mode_switch",
    4: "stop",
}


class AgentWithController:
    """Controller-augmented agent loop wrapper."""

    def __init__(
        self,
        controller_path: str,
        reference_dist_path: str,
        agent_config: Mapping[str, object],
    ) -> None:
        self.agent_config = dict(agent_config)
        self.k = int(self.agent_config.get("window_size", 5))
        self.gate_threshold = float(self.agent_config.get("gate_threshold", 0.5))
        self.device = torch.device(self.agent_config.get("device", "cpu"))

        self.controller = DeliberationController(
            signal_dim=5,
            hidden_dim=64,
            num_steps=self.k,
            num_actions=4,
        )
        ckpt = torch.load(controller_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.controller.load_state_dict(state_dict, strict=True)
        self.controller.to(self.device)
        self.controller.eval()

        self.normalizer = PercentileNormalizer.from_json(reference_dist_path)
        self.intervention_executor = InterventionExecutor()

        self.reset_episode()

    def reset_episode(self) -> None:
        self.history: List[Dict[str, object]] = []
        self.signal_buffer_norm: Deque[List[float]] = deque(maxlen=self.k)
        self.signal_buffer_raw: Deque[List[float]] = deque(maxlen=self.k)
        self.intervention_counts: Counter[str] = Counter()
        self.intervention_logs: List[Dict[str, object]] = []
        self.total_steps = 0
        self.total_estimated_token_saving = 0.0

    def _predict_decision(self, signal_window_norm: Sequence[Sequence[float]]) -> int:
        x = torch.tensor([signal_window_norm], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            gate_prob, action_logits = self.controller(x)
        gate_value = float(gate_prob[0].item())
        if gate_value < self.gate_threshold:
            return 0
        action_id = int(torch.argmax(action_logits[0], dim=-1).item())  # 0..3
        return action_id + 1  # -> 1..4

    def process_step(self, step: Mapping[str, object]) -> Dict[str, object]:
        """
        Receive a finished agent step and return intervention result.
        """
        self.total_steps += 1
        self.history.append(dict(step))

        step_idx = len(self.history) - 1
        raw_signal_dict = extract_step_signals(self.history, step_idx)
        norm_signal_dict = self.normalizer.normalize_signal_dict(raw_signal_dict)

        raw_signal_vec = signals_to_vector(raw_signal_dict)
        norm_signal_vec = signals_to_vector(norm_signal_dict)
        self.signal_buffer_raw.append(raw_signal_vec)
        self.signal_buffer_norm.append(norm_signal_vec)

        if len(self.signal_buffer_norm) < self.k:
            decision = 0
        else:
            decision = self._predict_decision(list(self.signal_buffer_norm))

        result = self.intervention_executor.execute(
            decision=decision,
            history=self.history,
            agent_config=self.agent_config,
        )

        # Offline token saving estimator for compress.
        estimated_saving = 0.0
        if decision == 1:
            history_token = 0.0
            for s in self.history:
                history_token += float(s.get("tokens_input", 0) or 0) + float(s.get("tokens_output", 0) or 0)
            estimated_saving = 0.3 * history_token
            result["estimated_token_saving"] = estimated_saving

        self.total_estimated_token_saving += float(estimated_saving)
        action_name = ACTION_NAMES.get(decision, "continue")
        self.intervention_counts[action_name] += 1

        self.intervention_logs.append(
            {
                "step_idx": step_idx,
                "decision": decision,
                "action": action_name,
                "raw_signals": raw_signal_dict,
                "norm_signals": norm_signal_dict,
                "intervention_log": result.get("intervention_log", ""),
            }
        )
        return result

    def get_intervention_summary(self) -> Dict[str, object]:
        total_interventions = self.total_steps - self.intervention_counts.get("continue", 0)
        return {
            "total_steps": self.total_steps,
            "intervention_count": total_interventions,
            "intervention_distribution": dict(self.intervention_counts),
            "estimated_token_saving": float(self.total_estimated_token_saving),
            "intervention_logs": list(self.intervention_logs),
        }

