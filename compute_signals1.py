"""
Overthinking Signal Computation & Statistical Validation (V3)
==============================================================
包含第二轮（已验证）+ 第三轮（新增）信号
在所有 benchmark × 模型 数据上跑 Mann-Whitney U 检验
输出完整 p 值汇总表

用法：
  python compute_signals_v3.py --config config.json
  或直接修改底部的 DATA_FILES 字典运行
"""

import json
import re
import numpy as np
from scipy.stats import mannwhitneyu
from collections import Counter
import argparse
import os


# ============================================================
# 1. 成功/失败判断（适配不同 benchmark 格式）
# ============================================================

def is_success(traj):
    """判断轨迹是否成功，适配 TAU-bench / HotpotQA / Bamboogle"""
    # HotpotQA / Bamboogle 格式
    if "Metrics" in traj:
        return traj["Metrics"].get("acc", 0) == 1
    # TAU-bench 格式
    if "reward" in traj:
        return traj["reward"] == 1
    # 其他
    if "success" in traj:
        return traj["success"] == True
    return False


# ============================================================
# 2. 信号计算函数
# ============================================================

# ---------- 第二轮信号（已验证） ----------

def sig_thought_length_mean(steps):
    """每步 thought 的平均字符长度"""
    lengths = [len(s.get("thought") or "") for s in steps
               if (s.get("thought") or "") and len(s.get("thought") or "") > 20]
    return float(np.mean(lengths)) if lengths else None


def sig_thought_length_var(steps):
    """thought 长度的方差"""
    lengths = [len(s.get("thought") or "") for s in steps
               if (s.get("thought") or "") and len(s.get("thought") or "") > 20]
    return float(np.var(lengths)) if len(lengths) >= 2 else None


def sig_tokens_per_step(steps):
    """每步平均 token 消耗"""
    total = sum((s.get("tokens_input", 0) or 0) + (s.get("tokens_output", 0) or 0)
                for s in steps)
    return total / len(steps) if steps else None


# ---------- 第三轮信号（新增，基于轨迹分析） ----------

def sig_action_repetition_ratio(steps):
    """
    连续相同 action_type 的步骤占比
    检测: Type B 机械重试型 overthinking
    例: search,search,search,respond,search → 连续重复3次/总5步 = 0.6
    """
    if len(steps) < 2:
        return None
    consecutive_repeats = 0
    for i in range(1, len(steps)):
        curr_action = steps[i].get("action_type") or steps[i].get("action", "")
        prev_action = steps[i-1].get("action_type") or steps[i-1].get("action", "")
        if curr_action and prev_action and curr_action == prev_action:
            consecutive_repeats += 1
    return consecutive_repeats / (len(steps) - 1)


def sig_consecutive_failure_count(steps):
    """
    连续调用同一 tool 且返回空/无效结果的最大次数
    检测: Type B 机械重试型 (如 search_direct_flight 连续21次返回空)
    """
    max_streak = 0
    current_streak = 0
    prev_action = None

    for s in steps:
        action = s.get("action_type") or s.get("action", "") or ""
        obs = s.get("observation") or ""

        # 判断 observation 是否为空/无效
        obs_stripped = obs.strip()
        is_empty = (
            obs_stripped in ["", "[]", "null", "None", "{}"] or
            len(obs_stripped) < 5 or
            obs_stripped == "No results found" or
            obs_stripped.startswith("Error")
        )

        if action and action == prev_action and is_empty:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
        prev_action = action

    return max_streak


def sig_respond_ratio(steps):
    """
    respond 类型步骤占总步数的比例
    检测: Type A 对话循环型 overthinking
    成功轨迹 respond 比例低(如3/8=0.375), 失败轨迹 respond 比例高(如16/30=0.53)
    """
    if not steps:
        return None
    respond_count = 0
    for s in steps:
        action = (s.get("action_type") or s.get("action", "") or "").lower()
        if action in ["respond", "response", "reply", "answer"]:
            respond_count += 1
    return respond_count / len(steps)


def sig_decision_oscillation(steps):
    """
    thought 中自我否定/转折词的频率 (per 100 words)
    检测: thought 层面的犹豫不决
    注意: 基于轨迹分析，GPT-4.1 几乎不出现此类词，qwen3-4b 可能有效
    """
    oscillation_markers = [
        r'\bwait\b', r'\bactually\b', r'\bno,', r'\bbut wait\b',
        r'\bhold on\b', r'\blet me reconsider\b', r'\bhmm\b',
        r'\bhowever\b', r'\binstead\b', r'\bcorrection\b',
        r'\bI was wrong\b', r'\blet me re-check\b', r'\blet me think again\b',
        r'\bon second thought\b', r'\bmaybe not\b',
        # 中文
        r'不对', r'等等', r'但是', r'其实', r'重新考虑', r'再想想'
    ]
    pattern = '|'.join(oscillation_markers)

    total_words = 0
    total_matches = 0
    for s in steps:
        thought = s.get("thought") or ""
        if len(thought) < 20:
            continue
        matches = len(re.findall(pattern, thought, re.IGNORECASE))
        words = len(thought.split())
        total_matches += matches
        total_words += words

    if total_words < 10:
        return None
    return total_matches / (total_words / 100)  # per 100 words


def sig_reasoning_repetition(steps):
    """
    跨步骤的 thought 内容重复度（相邻步骤的词汇重叠率均值）
    检测: 推理内容重复（同样的思路反复出现）
    """
    thoughts = [s.get("thought") or "" for s in steps
                if (s.get("thought") or "") and len(s.get("thought") or "") > 20]
    if len(thoughts) < 2:
        return None

    overlaps = []
    for i in range(1, len(thoughts)):
        words_prev = set(thoughts[i-1].lower().split())
        words_curr = set(thoughts[i].lower().split())
        if words_curr:
            overlap = len(words_prev & words_curr) / len(words_curr)
            overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else None


def sig_think_act_coherence(steps):
    """
    thought 中提到的 action/tool 名称与实际 action 的不一致率
    检测: 思维说"应该查询X"但行动做了"respond"
    简化版: 当 thought 包含工具相关关键词但 action 是 respond 时计为不一致
    """
    tool_keywords = [
        'search', 'query', 'find', 'look up', 'check', 'get',
        'fetch', 'retrieve', 'call', 'api', 'tool',
        '查询', '搜索', '获取', '调用',
        'get_order', 'get_user', 'get_product', 'get_reservation',
        'modify', 'cancel', 'transfer', 'book'
    ]
    pattern = '|'.join([re.escape(kw) for kw in tool_keywords])

    misaligned = 0
    total = 0
    for s in steps:
        thought = (s.get("thought") or "").lower()
        action = (s.get("action_type") or s.get("action", "") or "").lower()
        if not thought or len(thought) < 20:
            continue
        total += 1

        # thought 提到了工具操作但实际 action 是 respond
        thought_mentions_tool = bool(re.search(pattern, thought, re.IGNORECASE))
        action_is_respond = action in ["respond", "response", "reply", "answer"]

        if thought_mentions_tool and action_is_respond:
            misaligned += 1

    return misaligned / max(total, 1)


# ============================================================
# 3. 信号注册表
# ============================================================

SIGNALS = {
    # 第二轮
    "thought_length_mean": sig_thought_length_mean,
    "thought_length_var": sig_thought_length_var,
    "tokens_per_step": sig_tokens_per_step,
    # 第三轮 - 行为模式信号
    "action_repetition_ratio": sig_action_repetition_ratio,
    "consecutive_failure_count": sig_consecutive_failure_count,
    "respond_ratio": sig_respond_ratio,
    # 第三轮 - 推理内容信号
    "decision_oscillation": sig_decision_oscillation,
    "reasoning_repetition": sig_reasoning_repetition,
    "think_act_coherence": sig_think_act_coherence,
}


# ============================================================
# 4. 主流程
# ============================================================

def compute_all_signals(traj):
    """对一条轨迹计算所有信号"""
    steps = traj.get("steps", [])
    if not steps:
        return {}
    results = {}
    for name, func in SIGNALS.items():
        try:
            val = func(steps)
            results[name] = val
        except Exception as e:
            results[name] = None
    return results


def run_analysis(filepath, dataset_name):
    """对一个数据集跑全部信号 + 统计检验"""
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name}")
    print(f"  File: {filepath}")
    print(f"{'='*70}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # 计算信号
    for traj in data:
        traj["_signals"] = compute_all_signals(traj)
        traj["_success"] = is_success(traj)

    success = [d for d in data if d["_success"]]
    failure = [d for d in data if not d["_success"]]

    print(f"\n  总条数: {len(data)}, 成功: {len(success)}, 失败: {len(failure)}, "
          f"成功率: {len(success)/len(data)*100:.1f}%")

    if len(success) < 5 or len(failure) < 5:
        print(f"  ⚠️ 成功或失败样本不足5条，跳过统计检验")
        return None

    # 统计检验
    results = {}
    print(f"\n  {'信号':<30} {'成功组均值':>12} {'失败组均值':>12} {'p值':>10} {'显著性':>6} {'Effect':>8}")
    print(f"  {'-'*88}")

    for sig_name in SIGNALS.keys():
        vals_success = [d["_signals"][sig_name] for d in success
                       if d["_signals"].get(sig_name) is not None]
        vals_failure = [d["_signals"][sig_name] for d in failure
                       if d["_signals"].get(sig_name) is not None]

        if len(vals_success) < 3 or len(vals_failure) < 3:
            print(f"  {sig_name:<30} {'样本不足':>12} {'':>12} {'N/A':>10} {'':>6} {'':>8}")
            results[sig_name] = {"p": None, "significant": False}
            continue

        # Mann-Whitney U 检验
        stat, p_val = mannwhitneyu(vals_success, vals_failure, alternative='two-sided')

        # Effect size (Cohen's d)
        mean_s = np.mean(vals_success)
        mean_f = np.mean(vals_failure)
        pooled_std = np.sqrt((np.var(vals_success) + np.var(vals_failure)) / 2)
        cohens_d = abs(mean_s - mean_f) / pooled_std if pooled_std > 0 else 0

        sig_marker = "✅" if p_val < 0.05 else ("⚠️" if p_val < 0.1 else "❌")

        print(f"  {sig_name:<30} {mean_s:>12.2f} {mean_f:>12.2f} {p_val:>10.4f} {sig_marker:>6} {cohens_d:>8.3f}")

        results[sig_name] = {
            "p": p_val,
            "significant": p_val < 0.05,
            "mean_success": mean_s,
            "mean_failure": mean_f,
            "cohens_d": cohens_d
        }

    return results


def print_summary_table(all_results):
    """打印跨数据集的 p 值汇总表"""
    datasets = list(all_results.keys())
    signals = list(SIGNALS.keys())

    print(f"\n\n{'='*100}")
    print(f"  跨数据集 P 值汇总表")
    print(f"{'='*100}")

    # 表头
    header = f"  {'信号':<30}"
    for ds in datasets:
        short_name = ds[:20]
        header += f" {short_name:>20}"
    print(header)
    print(f"  {'-'*len(header)}")

    # 每行
    for sig in signals:
        row = f"  {sig:<30}"
        sig_count = 0  # 跨数据集显著的次数
        for ds in datasets:
            if all_results[ds] is None:
                row += f" {'skip':>20}"
            elif all_results[ds][sig]["p"] is None:
                row += f" {'N/A':>20}"
            else:
                p = all_results[ds][sig]["p"]
                marker = "✅" if p < 0.05 else ("⚠️" if p < 0.1 else "")
                row += f" {p:>16.4f} {marker:>3}"
                if p < 0.05:
                    sig_count += 1
        row += f"  [{sig_count}/{len(datasets)}]"
        print(row)

    print(f"\n  ✅ = p < 0.05    ⚠️ = p < 0.10    [X/Y] = X个数据集显著/共Y个")


# ============================================================
# 5. 运行
# ============================================================

if __name__ == "__main__":
    # ===== 修改这里的路径 =====
    DATA_FILES = {
        # "数据集名称": "轨迹文件路径",
        "TAU_Retail_qwen3":   "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Retail/tool-calling-qwen3-4b-0.0_range_0--1_user-deepseek-chat-llm_0327153644_triples.json",
        "TAU_Retail_GPT41":   "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Retail/retail-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0330210746_triples.json",
        "TAU_Airline_qwen3":  "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Airline/airline-tool-calling-qwen3-4b-0.0_range_0--1_user-gpt-4.1-llm_0409223338_triples.json",
        "TAU_Airline_GPT41":  "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Airline/airline-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0409224328_triples.json",
        "HotpotQA_qwen3":     "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/qwen3-4b-thinking-hotpotqa-all.triples.json",
        "HotpotQA_GPT41":     "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/hotpotqa-gpt-4.1-test.4.10,0.55.triples.json",
        "Bamboogle_qwen3":    "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark3-Bamboogle/bamboogle-qwen3-4b-thinking-test.4.10,15.53.triples.json",
        "Bamboogle_GPT41":    "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark3-Bamboogle/bamboogle-gpt-4.1-test.4.10,15.23.triples.json",
    }

    if not DATA_FILES:
        print("请在 DATA_FILES 中填入你的轨迹文件路径后运行！")
        print("示例：")
        print('  DATA_FILES = {')
        print('      "TAU_Retail_qwen3": "/data/wanghy/agent_traj/tau_retail_qwen3.json",')
        print('      "TAU_Retail_GPT41": "/data/wanghy/agent_traj/tau_retail_gpt41.json",')
        print('  }')
        exit(1)

    # 逐个数据集跑
    all_results = {}
    for name, path in DATA_FILES.items():
        if not os.path.exists(path):
            print(f"\n⚠️ 文件不存在: {path}")
            all_results[name] = None
            continue
        result = run_analysis(path, name)
        all_results[name] = result

    # 汇总表
    print_summary_table(all_results)

    print(f"\n\n{'='*70}")
    print("  完成！将以上结果截图保存即可用于开会汇报。")
    print(f"{'='*70}")