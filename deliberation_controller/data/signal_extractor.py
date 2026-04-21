"""Step-level cumulative signal extraction for controller training/inference."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

SIGNAL_NAMES: List[str] = [
    "thought_length_mean",
    "thought_length_var",
    "tokens_per_step",
    "decision_oscillation",
    "consecutive_failure_count",
]

_OSCILLATION_MARKERS: Sequence[str] = (
    r"\bwait\b",
    r"\bactually\b",
    r"\bbut wait\b",
    r"\bhold on\b",
    r"\bhmm\b",
    r"\bhowever\b",
    r"\binstead\b",
    r"\bcorrection\b",
    r"\bon second thought\b",
    r"\bmaybe not\b",
    r"不对",
    r"等等",
    r"但是",
    r"其实",
    r"重新考虑",
    r"再想想",
)
_OSCILLATION_PATTERN = re.compile("|".join(_OSCILLATION_MARKERS), flags=re.IGNORECASE)


def _get_text(step: Mapping[str, object], key: str) -> str:
    value = step.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _get_action(step: Mapping[str, object]) -> str:
    action = step.get("action_type")
    if action is None:
        action = step.get("action", "")
    return _get_text({"action": action}, "action").strip().lower()


def _get_step_tokens(step: Mapping[str, object]) -> float:
    token_in = float(step.get("tokens_input", 0) or 0)
    token_out = float(step.get("tokens_output", 0) or 0)
    return token_in + token_out


def _is_failure_observation(observation: str) -> bool:
    text = observation.strip()
    if not text:
        return True
    lower = text.lower()
    if lower in {"[]", "{}", "none", "null", "no results found"}:
        return True
    if len(text) < 5:
        return True
    if lower.startswith("error"):
        return True
    return False


def _compute_consecutive_failure_count(steps: Iterable[Mapping[str, object]]) -> float:
    max_streak = 0
    streak = 0
    prev_failed_action = ""

    for step in steps:
        action = _get_action(step)
        observation = _get_text(step, "observation")
        is_failed = bool(action) and _is_failure_observation(observation)

        if is_failed:
            if action == prev_failed_action:
                streak += 1
            else:
                streak = 1
            prev_failed_action = action
            max_streak = max(max_streak, streak)
        else:
            streak = 0
            prev_failed_action = ""

    return float(max_streak)


def _compute_decision_oscillation(steps: Iterable[Mapping[str, object]]) -> float:
    total_words = 0
    total_matches = 0

    for step in steps:
        thought = _get_text(step, "thought")
        if not thought.strip():
            continue
        total_words += len(thought.split())
        total_matches += len(_OSCILLATION_PATTERN.findall(thought))

    if total_words == 0:
        return 0.0
    return float(total_matches / total_words * 100.0)


def extract_step_signals(steps: Sequence[Mapping[str, object]], step_idx: int) -> Dict[str, float]:
    """Compute cumulative 5-dim signals up to and including `step_idx`."""
    if step_idx < 0 or step_idx >= len(steps):
        raise IndexError(f"step_idx={step_idx} out of range for {len(steps)} steps")

    prefix_steps = steps[: step_idx + 1]
    # 应该改成（和 Phase 1 一致）
    thought_lengths = [len(_get_text(step, "thought")) for step in prefix_steps 
                   if len(_get_text(step, "thought")) > 20]
    tokens = [_get_step_tokens(step) for step in prefix_steps]

    mean_len = float(np.mean(thought_lengths)) if thought_lengths else 0.0
    var_len = float(np.var(thought_lengths)) if len(thought_lengths) > 1 else 0.0
    tokens_per_step = float(np.mean(tokens)) if tokens else 0.0
    oscillation = _compute_decision_oscillation(prefix_steps)
    failure_streak = _compute_consecutive_failure_count(prefix_steps)

    return {
        "thought_length_mean": mean_len,
        "thought_length_var": var_len,
        "tokens_per_step": tokens_per_step,
        "decision_oscillation": oscillation,
        "consecutive_failure_count": failure_streak,
    }


def extract_all_step_signals(steps: Sequence[Mapping[str, object]]) -> List[Dict[str, float]]:
    """Compute cumulative signals for every step in a trajectory."""
    return [extract_step_signals(steps, idx) for idx in range(len(steps))]


def signals_to_vector(signals: Mapping[str, float]) -> List[float]:
    """Convert signal dict to fixed-order vector for model input."""
    return [float(signals[name]) for name in SIGNAL_NAMES]


def extract_signals_for_trajectory(trajectory: Mapping[str, object]) -> List[Dict[str, float]]:
    """Convenience wrapper for a trajectory dict containing `steps`."""
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("trajectory['steps'] must be a list")
    return extract_all_step_signals(steps)
