from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from deliberation_controller.data.normalizer import PercentileNormalizer
from deliberation_controller.data.signal_extractor import extract_step_signals, signals_to_vector
from deliberation_controller.intervene.intervention import InterventionExecutor
from deliberation_controller.online_rl.env_base import BaseEnv


ACTION_NAMES = {
    0: "continue",
    1: "compress",
    2: "redirect",
    3: "mode_switch",
    4: "stop",
}


class SearchO1Runner(Protocol):
    """Minimal search-o1 runner interface used by SearchO1Env.

    Concrete implementations can call local search-o1 code or a remote API.
    """

    def reset(self, question: str, meta: Mapping[str, Any]) -> Dict[str, Any]:
        """Create an initial runner state.

        Expected return keys (best effort):
        - messages: list[dict]
        - prompt: optional str
        """

    def run_step(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        """Run one generation+tool step.

        Expected return keys:
        - thought: str
        - action_type: str (search/respond/fetch_url/...)
        - action_input: dict
        - observation: str
        - tokens_input: int
        - tokens_output: int
        Optional:
        - done: bool
        - final_answer: str
        - messages: updated message list (if omitted env updates messages itself)
        """


class HTTPSearchO1Runner:
    """HTTP adapter for a remote search-o1 service.

    Service contract (recommended):
    - POST {endpoint}/reset with {question, meta}
    - POST {endpoint}/run_step with {state}
    """

    def __init__(self, endpoint: str, timeout_sec: float = 120.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        import requests  # local import to avoid hard dependency when using mock runner

        url = f"{self.endpoint}/{path.lstrip('/')}"
        resp = requests.post(url, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        out = resp.json()
        if not isinstance(out, dict):
            raise ValueError(f"HTTP runner expected dict response from {url}, got: {type(out)}")
        return out

    def reset(self, question: str, meta: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post_json("reset", {"question": question, "meta": dict(meta)})

    def run_step(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        return self._post_json("run_step", {"state": dict(state)})


class MockSearchO1Runner:
    """Deterministic mock runner for local dry runs.

    This does not call any model/API. It generates synthetic search-o1-like steps.
    """

    def reset(self, question: str, meta: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": question},
            ],
            "prompt": question,
            "step_idx": 0,
        }

    def run_step(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        step_idx = int(state.get("step_idx", 0))
        if step_idx == 0:
            out = {
                "thought": "Need evidence, search first.",
                "action_type": "search",
                "action_input": {"query": "entity background"},
                "observation": "BEGIN_SEARCH_RESULT weak snippet END_SEARCH_RESULT",
                "tokens_input": 420,
                "tokens_output": 160,
                "done": False,
            }
        elif step_idx == 1:
            out = {
                "thought": "Try another query and compare clues.",
                "action_type": "search",
                "action_input": {"query": "entity relation"},
                "observation": "BEGIN_SEARCH_RESULT better snippet END_SEARCH_RESULT",
                "tokens_input": 520,
                "tokens_output": 210,
                "done": False,
            }
        elif step_idx == 2:
            out = {
                "thought": "I can now answer: \\boxed{mock_answer}",
                "action_type": "respond",
                "action_input": {"content": "mock_answer"},
                "observation": "",
                "tokens_input": 640,
                "tokens_output": 190,
                "done": True,
                "final_answer": "mock_answer",
            }
        else:
            out = {
                "thought": "No further action.",
                "action_type": "respond",
                "action_input": {"content": "mock_answer"},
                "observation": "",
                "tokens_input": 80,
                "tokens_output": 20,
                "done": True,
                "final_answer": "mock_answer",
            }

        # Return updated runner state so env can feed it back on next call.
        next_state = dict(state)
        next_state["step_idx"] = step_idx + 1
        out["runner_state"] = next_state
        return out


@dataclass
class EpisodeItem:
    task_id: str
    question: str
    ground_truth: List[str]
    raw: Mapping[str, Any]


class SearchO1Env(BaseEnv):
    """Online RL env backed by search-o1 step loop.

    This env is environment-agnostic from PPO's perspective and only exposes BaseEnv.
    In production, inject `HTTPSearchO1Runner(endpoint=...)` or a local runner wrapper.
    """

    def __init__(
        self,
        questions_path: Optional[str] = None,
        reference_dist_path: Optional[str] = None,
        search_o1_endpoint: Optional[str] = None,
        runner: Optional[SearchO1Runner] = None,
        seed: int = 42,
        k: int = 5,
        signal_dim: int = 5,
        max_episode_steps: int = 20,
        model_type: str = "thinking",
    ) -> None:
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.k = int(k)
        self.signal_dim = int(signal_dim)
        self.max_episode_steps = int(max_episode_steps)

        self.questions_path = questions_path or os.environ.get(
            "SEARCH_O1_QUESTIONS_PATH",
            "/data/wanghy/adaptive_deliberation_controller_eval/data/QA_Datasets/hotpotqa.json",
        )
        self.reference_dist_path = reference_dist_path or os.environ.get(
            "SEARCH_O1_REF_DIST_PATH",
            "/data/wanghy/agent_traj/deliberation_controller/data/reference_distribution_hotpotqa_qwen3_full.json",
        )

        self.normalizer = PercentileNormalizer.from_json(self.reference_dist_path)
        self.intervention_executor = InterventionExecutor()
        self.agent_config: Dict[str, Any] = {
            "model_type": model_type,
            "window_size": self.k,
            "signal_dim": self.signal_dim,
        }

        self.items: List[EpisodeItem] = self._load_items(self.questions_path)
        if not self.items:
            raise ValueError(f"No question items loaded from {self.questions_path}")

        # runner selection:
        # 1) explicit runner
        # 2) HTTP endpoint from arg/env var SEARCH_O1_API_ENDPOINT
        # 3) local mock for dry run
        if runner is not None:
            self.runner = runner
        else:
            endpoint = (search_o1_endpoint or os.environ.get("SEARCH_O1_API_ENDPOINT", "")).strip()
            self.runner = HTTPSearchO1Runner(endpoint) if endpoint else MockSearchO1Runner()

        self._done: bool = True
        self._episode_item: Optional[EpisodeItem] = None
        self._runner_state: Dict[str, Any] = {}
        self._messages: List[Dict[str, str]] = []
        self._history_steps: List[Dict[str, Any]] = []
        self._signal_buffer: List[List[float]] = []
        self._total_tokens: float = 0.0
        self._step_count: int = 0
        self._final_answer: str = ""

    @staticmethod
    def _to_list_str(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if value is None:
            return []
        s = str(value).strip()
        return [s] if s else []

    def _load_items(self, path: str) -> List[EpisodeItem]:
        if not os.path.exists(path):
            # Keep environment runnable even before dataset path is configured.
            return [
                EpisodeItem(
                    task_id="mock_task_0",
                    question="Mock HotpotQA question for SearchO1Env dry run.",
                    ground_truth=["mock_answer"],
                    raw={"question": "Mock HotpotQA question for SearchO1Env dry run."},
                )
            ]

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"questions_path must be a list json: {path}")

        out: List[EpisodeItem] = []
        for i, x in enumerate(data):
            if not isinstance(x, Mapping):
                continue
            q = str(x.get("question", x.get("Question", ""))).strip()
            if not q:
                continue

            gt = self._to_list_str(x.get("ground_truth"))
            if not gt:
                gt = self._to_list_str(x.get("answer", x.get("Answer", "")))

            task_id = str(x.get("task_id", i))
            out.append(EpisodeItem(task_id=task_id, question=q, ground_truth=gt, raw=x))
        return out

    @staticmethod
    def _normalize_message(msg: Mapping[str, Any]) -> Dict[str, str]:
        return {
            "role": str(msg.get("role", "user")),
            "content": str(msg.get("content", "")),
        }

    def _empty_state(self) -> np.ndarray:
        return np.zeros((self.k, self.signal_dim), dtype=np.float32)

    def _build_state(self) -> np.ndarray:
        if not self._history_steps:
            return self._empty_state()

        sig_dict = extract_step_signals(self._history_steps, len(self._history_steps) - 1)
        sig_norm = self.normalizer.normalize_signal_dict(sig_dict)
        vec = signals_to_vector(sig_norm)
        self._signal_buffer.append([float(v) for v in vec])
        if len(self._signal_buffer) > self.k:
            self._signal_buffer = self._signal_buffer[-self.k :]

        # left-pad to K
        pad_len = self.k - len(self._signal_buffer)
        if pad_len > 0:
            padded = [[0.0] * self.signal_dim for _ in range(pad_len)] + list(self._signal_buffer)
        else:
            padded = list(self._signal_buffer)
        return np.asarray(padded, dtype=np.float32)

    @staticmethod
    def _match_answer(pred: str, gts: Sequence[str]) -> bool:
        p = (pred or "").strip().lower()
        if not p:
            return False
        for g in gts:
            gg = (g or "").strip().lower()
            if not gg:
                continue
            if gg in p or p in gg:
                return True
        return False

    def _terminal_reward(self, success: bool) -> float:
        reward = 1.0 if success else -0.5
        reward += max(0.0, 0.3 * (1.0 - self._total_tokens / 8000.0))
        return float(reward)

    def _apply_action_to_messages(self, action: int) -> Dict[str, Any]:
        if action == 0:
            return {"action": "continue", "modified_prompt": None, "extracted_answer": None}

        result = self.intervention_executor.execute(
            decision=int(action),
            history=self._history_steps,
            agent_config=self.agent_config,
        )

        act = str(result.get("action", "continue"))
        modified = result.get("modified_prompt")

        if act == "compress" and isinstance(modified, str):
            # keep first system if exists, then provide compressed context as user
            system_msgs = [m for m in self._messages if m.get("role") == "system"]
            first_user = next((m for m in self._messages if m.get("role") == "user"), None)
            new_msgs: List[Dict[str, str]] = []
            if system_msgs:
                new_msgs.append(system_msgs[0])
            if first_user is not None:
                new_msgs.append(first_user)
            new_msgs.append({"role": "user", "content": f"[COMPRESSED CONTEXT] {modified}"})
            self._messages = new_msgs

        elif act == "redirect" and isinstance(modified, str):
            self._messages.append({"role": "user", "content": f"[SYSTEM INSTRUCTION] {modified}"})

        elif act == "mode_switch" and isinstance(modified, str):
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages[-1]["content"] = f"{self._messages[-1]['content']}\n\n{modified}\n\n/no_think"
            else:
                self._messages.append({"role": "user", "content": f"{modified}\n\n/no_think"})

        return {
            "action": act,
            "modified_prompt": modified,
            "extracted_answer": result.get("extracted_answer"),
        }

    def reset(self) -> Tuple[Any, Dict]:
        self._episode_item = self.rng.choice(self.items)
        self._history_steps = []
        self._signal_buffer = []
        self._total_tokens = 0.0
        self._step_count = 0
        self._final_answer = ""
        self._done = False

        runner_init = self.runner.reset(self._episode_item.question, {"task_id": self._episode_item.task_id})
        if not isinstance(runner_init, Mapping):
            raise ValueError(f"Runner reset() must return dict-like, got {type(runner_init)}")

        msgs = runner_init.get("messages", [{"role": "user", "content": self._episode_item.question}])
        if not isinstance(msgs, list):
            msgs = [{"role": "user", "content": self._episode_item.question}]
        self._messages = [self._normalize_message(m) for m in msgs if isinstance(m, Mapping)]

        self._runner_state = dict(runner_init)
        self._runner_state["messages"] = self._messages

        state = self._empty_state()
        info = {
            "task_id": self._episode_item.task_id,
            "question": self._episode_item.question,
            "ground_truth": list(self._episode_item.ground_truth),
            "forced_stop": False,
            "used_mock_runner": isinstance(self.runner, MockSearchO1Runner),
        }
        return state, info

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode already done. Call reset().")
        if self._episode_item is None:
            raise RuntimeError("Call reset() before step().")

        act = int(action)
        if act < 0 or act > 4:
            raise ValueError(f"Invalid action={action}, expected 0..4")

        intervention = self._apply_action_to_messages(act)
        forced_stop = intervention.get("action") == "stop"

        if forced_stop:
            pred = str(intervention.get("extracted_answer") or "")
            self._final_answer = pred
            success = self._match_answer(pred, self._episode_item.ground_truth)
            reward = self._terminal_reward(success)
            self._done = True
            info = {
                "task_id": self._episode_item.task_id,
                "forced_stop": True,
                "success": bool(success),
                "final_answer": pred,
                "ground_truth": list(self._episode_item.ground_truth),
                "total_tokens": float(self._total_tokens),
                "step_count": self._step_count,
                "intervention": intervention,
            }
            return self._build_state(), reward, True, info

        # Continue online generation for one step.
        self._runner_state["messages"] = self._messages
        out = self.runner.run_step(self._runner_state)
        if not isinstance(out, Mapping):
            raise ValueError(f"Runner run_step() must return dict-like, got {type(out)}")

        thought = str(out.get("thought", ""))
        action_type = str(out.get("action_type", "respond"))
        action_input = out.get("action_input", {})
        if not isinstance(action_input, Mapping):
            action_input = {"content": str(action_input)}
        observation = str(out.get("observation", ""))
        tokens_input = int(out.get("tokens_input", 0) or 0)
        tokens_output = int(out.get("tokens_output", 0) or 0)

        step_data: Dict[str, Any] = {
            "step_id": self._step_count,
            "thought": thought,
            "action_type": action_type,
            "action_input": dict(action_input),
            "observation": observation,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }
        self._history_steps.append(step_data)
        self._step_count += 1
        self._total_tokens += float(tokens_input + tokens_output)

        # Keep messages/history aligned with run_search_o1 style.
        if thought:
            self._messages.append({"role": "assistant", "content": thought})
        if observation:
            self._messages.append({"role": "tool", "content": observation})

        done = bool(out.get("done", False)) or self._step_count >= self.max_episode_steps
        final_answer = str(out.get("final_answer", "") or "")

        if done:
            if not final_answer:
                # best-effort extraction from latest respond action
                if action_type.strip().lower() == "respond":
                    final_answer = str(action_input.get("content", "") or "")
                if not final_answer:
                    final_answer = thought
            self._final_answer = final_answer
            success = self._match_answer(final_answer, self._episode_item.ground_truth)
            reward = self._terminal_reward(success)
            self._done = True
        else:
            reward = 0.0
            success = False

        runner_state = out.get("runner_state")
        if isinstance(runner_state, Mapping):
            self._runner_state = dict(runner_state)
        self._runner_state["messages"] = self._messages

        info = {
            "task_id": self._episode_item.task_id,
            "forced_stop": False,
            "success": bool(success),
            "final_answer": self._final_answer if done else "",
            "ground_truth": list(self._episode_item.ground_truth),
            "total_tokens": float(self._total_tokens),
            "step_count": self._step_count,
            "intervention": intervention,
            "runner_done": bool(out.get("done", False)),
        }
        return self._build_state(), float(reward), bool(done), info

    def close(self) -> None:
        self._done = True
        self._episode_item = None
        self._runner_state = {}
        self._messages = []
        self._history_steps = []
        self._signal_buffer = []
        self._total_tokens = 0.0
        self._step_count = 0
        self._final_answer = ""


def _dry_run() -> None:
    """Dry run with mock runner to validate BaseEnv contract locally."""
    env = SearchO1Env(runner=MockSearchO1Runner(), max_episode_steps=6)
    state, info = env.reset()
    print("[dry-run] reset state_shape=", np.asarray(state).shape, "task_id=", info.get("task_id"))

    # apply a small action sequence: continue -> redirect -> continue -> stop
    actions = [0, 2, 0, 4]
    for i, a in enumerate(actions):
        ns, r, d, inf = env.step(a)
        print(
            f"[dry-run] step={i} action={ACTION_NAMES.get(a,a)} reward={r:.4f} done={d} "
            f"state_shape={np.asarray(ns).shape} success={inf.get('success')}"
        )
        if d:
            print("[dry-run] terminal info:", {k: inf.get(k) for k in ["forced_stop", "final_answer", "ground_truth", "total_tokens"]})
            break

    env.close()
    print("[dry-run] done")


if __name__ == "__main__":
    _dry_run()
