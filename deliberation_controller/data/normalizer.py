"""Percentile-based signal normalization utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .signal_extractor import SIGNAL_NAMES, extract_all_step_signals


def load_trajectories(path: str | Path) -> List[Mapping[str, object]]:
    """Load trajectories from a JSON list/dict file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "steps" in data:
            return [data]
        return list(data.values())
    raise ValueError(f"Unsupported trajectory JSON format: {type(data)}")


def build_reference_distribution(
    trajectories: Iterable[Mapping[str, object]],
) -> Dict[str, List[float]]:
    """Build sorted value lists for each signal from all trajectory steps."""
    pooled: Dict[str, List[float]] = {name: [] for name in SIGNAL_NAMES}

    for traj in trajectories:
        steps = traj.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue
        for signal_dict in extract_all_step_signals(steps):
            for name in SIGNAL_NAMES:
                pooled[name].append(float(signal_dict.get(name, 0.0)))

    for name in SIGNAL_NAMES:
        pooled[name].sort()
    return pooled


def save_reference_distribution(distribution: Mapping[str, Sequence[float]], path: str | Path) -> None:
    output = {name: [float(v) for v in values] for name, values in distribution.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


@dataclass
class PercentileNormalizer:
    """CDF lookup normalizer backed by sorted reference values."""

    reference_distribution: Dict[str, np.ndarray]

    @classmethod
    def from_distribution(
        cls,
        distribution: Mapping[str, Sequence[float]],
    ) -> "PercentileNormalizer":
        arrays: Dict[str, np.ndarray] = {}
        for name in SIGNAL_NAMES:
            values = distribution.get(name, [])
            arrays[name] = np.asarray(sorted(float(v) for v in values), dtype=np.float64)
        return cls(reference_distribution=arrays)

    @classmethod
    def from_json(cls, path: str | Path) -> "PercentileNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_distribution(data)

    def to_json(self, path: str | Path) -> None:
        save_reference_distribution(
            {name: arr.tolist() for name, arr in self.reference_distribution.items()},
            path,
        )

    def normalize_value(self, signal_name: str, raw_value: float) -> float:
        values = self.reference_distribution.get(signal_name)
        if values is None or values.size == 0:
            return 0.0

        idx = int(np.searchsorted(values, raw_value, side="right"))
        return float(idx / values.size * 100.0)

    def normalize_signal_dict(self, signal_dict: Mapping[str, float]) -> Dict[str, float]:
        return {
            name: self.normalize_value(name, float(signal_dict.get(name, 0.0)))
            for name in SIGNAL_NAMES
        }


def build_and_save_reference_distribution(
    trajectory_path: str | Path,
    output_path: str | Path,
) -> Dict[str, List[float]]:
    trajectories = load_trajectories(trajectory_path)
    distribution = build_reference_distribution(trajectories)
    save_reference_distribution(distribution, output_path)
    return distribution

