from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .env_base import BaseEnv


def _joint_action(gate_label: int, action_label: int) -> int:
    return 0 if int(gate_label) == 0 else int(action_label) + 1


@dataclass
class StepSample:
    state: np.ndarray
    task_id: str
    step_idx: int
    action: int
    success: bool


class ReplayEnv(BaseEnv):
    """Replay-based simulator environment backed by offline dataset samples."""

    def __init__(
        self,
        dataset_path: str,
        trajectories_path: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.rng = random.Random(seed)
        np.random.seed(seed)

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples: List[Mapping[str, object]] = []
        for split in ("train", "val", "test"):
            split_arr = data.get(split, []) if isinstance(data, Mapping) else []
            if isinstance(split_arr, list):
                samples.extend(split_arr)

        if not samples:
            raise ValueError(f"No samples found in {dataset_path}")

        self.samples: List[StepSample] = []
        self.traj_steps: Dict[str, List[Tuple[int, int]]] = {}
        self.traj_success: Dict[str, bool] = {}
        self.traj_total_tokens: Dict[str, float] = {}

        for idx, sample in enumerate(samples):
            meta = sample.get("meta", {}) if isinstance(sample, Mapping) else {}
            if not isinstance(meta, Mapping):
                continue
            task_id = str(meta.get("task_id"))
            step_idx = int(meta.get("target_step_idx", -1))
            success = bool(meta.get("is_success_trajectory", False))

            state = np.asarray(sample["signals"], dtype=np.float32)
            action = _joint_action(int(sample["gate_label"]), int(sample["action_label"]))

            self.samples.append(
                StepSample(
                    state=state,
                    task_id=task_id,
                    step_idx=step_idx,
                    action=action,
                    success=success,
                )
            )
            self.traj_steps.setdefault(task_id, []).append((step_idx, idx))
            self.traj_success[task_id] = success

        for task_id in self.traj_steps:
            self.traj_steps[task_id].sort(key=lambda x: x[0])

        self.state_matrix = np.stack([s.state.reshape(-1) for s in self.samples], axis=0)
        self.action_array = np.asarray([s.action for s in self.samples], dtype=np.int64)
        action_counts = np.bincount(self.action_array, minlength=5).astype(np.float64)
        self.action_prior = action_counts / max(action_counts.sum(), 1.0)

        if trajectories_path and Path(trajectories_path).exists():
            with open(trajectories_path, "r", encoding="utf-8") as f:
                trajectories = json.load(f)
            if isinstance(trajectories, list):
                for t in trajectories:
                    task_id = str(t.get("task_id"))
                    total_tokens = float(t.get("total_input_tokens", 0) or 0) + float(
                        t.get("total_output_tokens", 0) or 0
                    )
                    self.traj_total_tokens[task_id] = total_tokens
                    # Prefer raw trajectory success when available.
                    metrics = t.get("Metrics")
                    if isinstance(metrics, Mapping) and metrics.get("acc") is not None:
                        try:
                            self.traj_success[task_id] = float(metrics["acc"]) >= 1.0
                        except Exception:
                            self.traj_success[task_id] = bool(metrics["acc"])

        self._current_state: Optional[np.ndarray] = None
        self._current_task_id: Optional[str] = None
        self._current_step_pos: int = 0
        self._episode_done: bool = False

    def _pick_initial_index(self) -> int:
        # Pick a random trajectory and start from its first available step.
        task_id = self.rng.choice(list(self.traj_steps.keys()))
        return self.traj_steps[task_id][0][1]

    def reset(self):
        init_idx = self._pick_initial_index()
        s = self.samples[init_idx]

        self._current_state = s.state.copy()
        self._current_task_id = s.task_id
        self._current_step_pos = 0
        self._episode_done = False

        info = {
            "task_id": s.task_id,
            "step_idx": s.step_idx,
            "success": self.traj_success.get(s.task_id, s.success),
            "total_tokens": self.traj_total_tokens.get(s.task_id, 8000.0),
        }
        return self._current_state.copy(), info

    def _match_sample(self, action: int) -> Tuple[int, bool, float]:
        if self._current_state is None:
            raise RuntimeError("Call reset() before step().")

        q = self._current_state.reshape(-1)
        diff = self.state_matrix - q[None, :]
        dists = np.einsum("ij,ij->i", diff, diff)

        action_mask = self.action_array == int(action)
        if np.any(action_mask):
            cand_idx = np.where(action_mask)[0]
            local_best = int(cand_idx[np.argmin(dists[cand_idx])])
            return local_best, False, float(dists[local_best])

        # Approximate nearest neighbor without action match.
        best = int(np.argmin(dists))
        return best, True, float(dists[best])

    def _terminal_reward(self, task_id: str) -> float:
        success = bool(self.traj_success.get(task_id, False))
        total_tokens = float(self.traj_total_tokens.get(task_id, 8000.0))

        reward = 1.0 if success else -0.5
        reward += max(0.0, 0.3 * (1.0 - total_tokens / 8000.0))
        return float(reward)

    def step(self, action: int):
        if self._episode_done:
            raise RuntimeError("Episode already done. Call reset().")

        match_idx, approx, distance = self._match_sample(action)
        matched = self.samples[match_idx]

        traj = self.traj_steps.get(matched.task_id, [])
        # locate matched step in trajectory
        pos = 0
        for i, (_, idx) in enumerate(traj):
            if idx == match_idx:
                pos = i
                break

        done = pos >= len(traj) - 1

        if done:
            next_state = matched.state.copy()
            reward = self._terminal_reward(matched.task_id)
            self._episode_done = True
        else:
            next_idx = traj[pos + 1][1]
            next_state = self.samples[next_idx].state.copy()
            reward = 0.0

        self._current_state = next_state
        self._current_task_id = matched.task_id
        self._current_step_pos = pos + 1

        info = {
            "task_id": matched.task_id,
            "matched_step_idx": matched.step_idx,
            "matched_action": matched.action,
            "approximate_match": bool(approx),
            "distance_l2_sq": distance,
            "success": bool(self.traj_success.get(matched.task_id, matched.success)),
            "total_tokens": float(self.traj_total_tokens.get(matched.task_id, 8000.0)),
            "forced_stop": False,
        }
        return next_state.copy(), float(reward), bool(done), info

    def close(self) -> None:
        self._current_state = None
        self._current_task_id = None
        self._episode_done = True

    def run_random_episodes(self, num_episodes: int = 100, max_steps: int = 20) -> Dict[str, float]:
        success_cnt = 0
        rewards: List[float] = []
        lengths: List[int] = []

        for _ in range(num_episodes):
            self.reset()
            ep_reward = 0.0
            ep_len = 0
            done = False
            while not done and ep_len < max_steps:
                action = int(np.random.choice(5, p=self.action_prior))
                _, r, done, info = self.step(action)
                ep_reward += r
                ep_len += 1
                if done:
                    success_cnt += int(bool(info.get("success", False)))
            rewards.append(ep_reward)
            lengths.append(ep_len)

        return {
            "episodes": float(num_episodes),
            "success_rate": float(success_cnt / max(num_episodes, 1)),
            "avg_reward": float(np.mean(rewards) if rewards else 0.0),
            "avg_episode_length": float(np.mean(lengths) if lengths else 0.0),
        }
