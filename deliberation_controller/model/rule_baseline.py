"""Rule-based baseline controller for deliberation intervention."""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Output id mapping required by user:
# 0=Continue, 1=Compress, 2=Redirect, 3=ModeSwitch, 4=Stop
CONTINUE = 0
COMPRESS = 1
REDIRECT = 2
MODESWITCH = 3
STOP = 4

SIGNAL_INDEX = {
    "thought_length_mean": 0,
    "thought_length_var": 1,
    "tokens_per_step": 2,
    "decision_oscillation": 3,
    "consecutive_failure_count": 4,
}


class RuleBasedController:
    """Rule baseline with a DeliberationController-like decide interface."""

    def __init__(
        self,
        gate_threshold: float = 75.0,
        mode_switch_threshold: float = 75.0,
        stop_mean_threshold: float = 90.0,
        stop_var_threshold: float = 85.0,
        redirect_failure_threshold: float = 5.0,
    ) -> None:
        self.gate_threshold = gate_threshold
        self.mode_switch_threshold = mode_switch_threshold
        self.stop_mean_threshold = stop_mean_threshold
        self.stop_var_threshold = stop_var_threshold
        self.redirect_failure_threshold = redirect_failure_threshold

    def _to_array(self, signals: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(signals, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(f"Expected signals shape [K,5], got {arr.shape}")
        return arr

    def should_intervene(self, signals_norm: Sequence[Sequence[float]]) -> bool:
        arr = self._to_array(signals_norm)
        last = arr[-1]
        return not (
            last[SIGNAL_INDEX["thought_length_mean"]] <= self.gate_threshold
            and last[SIGNAL_INDEX["thought_length_var"]] <= self.gate_threshold
            and last[SIGNAL_INDEX["tokens_per_step"]] <= self.gate_threshold
        )

    def decide_action(
        self,
        signals_norm: Sequence[Sequence[float]],
        signals_raw: Sequence[Sequence[float]] | None = None,
    ) -> int:
        """
        Return intervention action id in {1,2,3,4} (Compress/Redirect/ModeSwitch/Stop).
        """
        norm_arr = self._to_array(signals_norm)
        raw_arr = self._to_array(signals_raw) if signals_raw is not None else norm_arr
        last_norm = norm_arr[-1]
        last_raw = raw_arr[-1]

        if last_norm[SIGNAL_INDEX["decision_oscillation"]] > self.mode_switch_threshold:
            return MODESWITCH
        if (
            last_norm[SIGNAL_INDEX["thought_length_mean"]] > self.stop_mean_threshold
            and last_norm[SIGNAL_INDEX["thought_length_var"]] > self.stop_var_threshold
        ):
            return STOP
        if (
            last_raw[SIGNAL_INDEX["consecutive_failure_count"]]
            >= self.redirect_failure_threshold
        ):
            return REDIRECT
        return COMPRESS

    def decide(
        self,
        signals_norm: Sequence[Sequence[float]],
        signals_raw: Sequence[Sequence[float]] | None = None,
    ) -> int:
        """
        Args:
            signals_norm: [K,5] normalized percentile signals.
            signals_raw: [K,5] raw signals. Required for strict Redirect rule.
        Returns:
            int in {0,1,2,3,4}:
            0=Continue, 1=Compress, 2=Redirect, 3=ModeSwitch, 4=Stop.
        """
        if not self.should_intervene(signals_norm):
            return CONTINUE
        return self.decide_action(signals_norm=signals_norm, signals_raw=signals_raw)

