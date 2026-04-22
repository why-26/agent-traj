"""Intervention execution logic for controller decisions."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence

ACTION_NAME = {
    0: "continue",
    1: "compress",
    2: "redirect",
    3: "mode_switch",
    4: "stop",
}


class InterventionExecutor:
    """Apply intervention policies based on controller decision."""

    _BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")

    @staticmethod
    def _get_text(step: Mapping[str, object], key: str) -> str:
        value = step.get(key)
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    @staticmethod
    def _get_action_type(step: Mapping[str, object]) -> str:
        v = step.get("action_type", step.get("action", ""))
        return (v if isinstance(v, str) else str(v)).strip()

    @staticmethod
    def _is_invalid_observation(observation: str) -> bool:
        text = observation.strip()
        if not text:
            return True
        lowered = text.lower()
        if lowered in {"[]", "{}", "none", "null", "no results found"}:
            return True
        return len(text) < 8

    def _format_step(self, idx: int, step: Mapping[str, object]) -> str:
        thought = self._get_text(step, "thought")
        action_type = self._get_action_type(step)
        observation = self._get_text(step, "observation")
        return (
            f"[Step {idx}] action_type={action_type}\n"
            f"thought: {thought}\n"
            f"observation: {observation}\n"
        )

    def _build_compressed_prompt(self, history: Sequence[Mapping[str, object]]) -> tuple[str, str]:
        if not history:
            return "", "compress: empty history, nothing to compress."

        first_step = history[0]
        first_req = self._get_text(first_step, "thought")
        if not first_req:
            first_req = self._get_text(first_step, "observation")

        tool_result_lines: List[str] = []
        omitted_respond = 0
        for idx, step in enumerate(history):
            action_type = self._get_action_type(step).lower()
            observation = self._get_text(step, "observation").strip()
            if action_type == "respond":
                omitted_respond += 1
            if action_type and action_type != "respond" and observation:
                tool_result_lines.append(f"- step {idx} [{action_type}]: {observation}")

        recent = history[-3:]
        recent_text = "\n".join(self._format_step(len(history) - len(recent) + i, s) for i, s in enumerate(recent))
        summary = (
            f"中间冗余段摘要: 共省略 {max(omitted_respond - 1, 0)} 条连续 respond 交互，"
            "保留关键工具返回结果与最近上下文。"
        )
        compressed_prompt = (
            "【首步需求】\n"
            f"{first_req}\n\n"
            "【关键工具返回】\n"
            f"{chr(10).join(tool_result_lines) if tool_result_lines else '- 无显式工具返回'}\n\n"
            "【摘要】\n"
            f"{summary}\n\n"
            "【最近3步完整历史】\n"
            f"{recent_text}"
        )
        log = f"compress: kept first request + {len(tool_result_lines)} tool results + last 3 steps."
        return compressed_prompt, log

    def _build_redirect_text(self, history: Sequence[Mapping[str, object]]) -> tuple[str, str]:
        if not history:
            msg = "请尝试换一种解题路径，优先使用与问题更相关的工具。"
            return msg, "redirect: empty history fallback."

        streak = 0
        last_tool = ""
        for step in reversed(history):
            tool = self._get_action_type(step).strip()
            obs = self._get_text(step, "observation")
            if not tool:
                break
            if streak == 0:
                last_tool = tool
            if tool != last_tool:
                break
            if not self._is_invalid_observation(obs):
                break
            streak += 1

        if streak >= 2 and last_tool:
            msg = (
                f"你已连续 {streak} 次使用 {last_tool} 未获得有效结果，"
                "请尝试更换搜索策略或使用不同的工具。"
            )
            return msg, f"redirect: repeated tool={last_tool}, streak={streak}."

        msg = "当前路径信息增益较低，请切换检索关键词或改用其他工具继续。"
        return msg, "redirect: generic guidance."

    def _extract_answer_from_history(self, history: Sequence[Mapping[str, object]]) -> tuple[str, str]:
        recent = history[-5:] if len(history) >= 5 else history

        for step in reversed(recent):
            thought = self._get_text(step, "thought")
            m = self._BOXED_PATTERN.search(thought)
            if m and m.group(1).strip():
                answer = m.group(1).strip()
                return answer, "stop: extracted boxed answer from recent thought."

        for step in reversed(recent):
            thought = self._get_text(step, "thought").strip()
            if thought:
                pieces = re.split(r"[。.!?]\s*", thought)
                pieces = [p.strip() for p in pieces if p.strip()]
                if pieces:
                    return pieces[-1], "stop: extracted final sentence from recent thought."

        for step in reversed(recent):
            obs = self._get_text(step, "observation").strip()
            if obs:
                return obs, "stop: fallback to recent observation."

        return "", "stop: no answer found in history."

    def execute(
        self,
        decision: int,
        history: Sequence[Mapping[str, object]],
        agent_config: Mapping[str, object],
    ) -> Dict[str, object]:
        action = ACTION_NAME.get(decision, "continue")
        result: Dict[str, object] = {
            "action": action,
            "modified_prompt": None,
            "extracted_answer": None,
            "intervention_log": f"{action}: no-op",
            "estimated_token_saving": 0.0,
        }

        if decision == 0:
            result["intervention_log"] = "continue: no intervention."
            return result

        if decision == 1:
            compressed_prompt, log = self._build_compressed_prompt(history)
            result["modified_prompt"] = compressed_prompt
            result["intervention_log"] = log
            return result

        if decision == 2:
            redirect_text, log = self._build_redirect_text(history)
            result["modified_prompt"] = redirect_text
            result["intervention_log"] = log
            return result

        if decision == 3:
            model_type = str(agent_config.get("model_type", "thinking")).strip().lower()
            if model_type == "thinking":
                prompt = "请不要进行过多的内部推理验证，直接基于已有信息给出你的最佳答案。"
                result["modified_prompt"] = prompt
                result["intervention_log"] = "mode_switch: applied concise-answer instruction."
                return result

            # non-thinking model fallback to redirect
            redirect_text, log = self._build_redirect_text(history)
            result["action"] = "redirect"
            result["modified_prompt"] = redirect_text
            result["intervention_log"] = f"mode_switch->redirect fallback: {log}"
            return result

        if decision == 4:
            answer, log = self._extract_answer_from_history(history)
            result["extracted_answer"] = answer
            result["intervention_log"] = log
            return result

        result["intervention_log"] = f"unknown decision={decision}, treated as continue."
        return result

