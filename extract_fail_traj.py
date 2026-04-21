import json, random, numpy as np
import os

random.seed(42)

# ===== 改成你的文件路径 =====
DATA_FILES = {
    "TAU_Retail_qwen3":   "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Retail/tool-calling-qwen3-4b-0.0_range_0--1_user-deepseek-chat-llm_0327153644_triples.json",
    "TAU_Retail_GPT41":   "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Retail/retail-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0330210746_triples.json",
    "TAU_Airline_qwen3":  "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Airline/airline-tool-calling-qwen3-4b-0.0_range_0--1_user-gpt-4.1-llm_0409223338_triples.json",
    "TAU_Airline_GPT41":  "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Airline/airline-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0409224328_triples.json",
    "HotpotQA_qwen3":     "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/qwen3-4b-thinking-hotpotqa-all.triples.json",
    "HotpotQA_GPT41":     "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/hotpotqa-gpt-4.1-test.4.10,0.55.triples.json",
    "Bamboogle_qwen3":    "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark3-Bamboogle/bamboogle-qwen3-4b-thinking-test.4.10,15.53.triples.json",
    "Bamboogle_GPT41":    "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark3-Bamboogle/bamboogle-gpt-4.1-test.4.10,15.23.triples.json",
}

# 每组采样多少条失败轨迹
SAMPLE_COUNTS = {
    "TAU_Retail_qwen3": 15, "TAU_Retail_GPT41": 15,
    "TAU_Airline_qwen3": 10, "TAU_Airline_GPT41": 10,
    "HotpotQA_qwen3": 15, "HotpotQA_GPT41": 15,
    "Bamboogle_qwen3": 10, "Bamboogle_GPT41": 10,
}

def is_success(traj):
    if "Metrics" in traj:
        return traj["Metrics"].get("acc", 0) == 1
    elif "reward" in traj:
        return traj["reward"] == 1
    return traj.get("success", False)

def compute_signals(traj):
    steps = traj.get("steps", [])
    if not steps:
        return {}
    
    # thought_length_mean
    lengths = [len(s.get("thought") or "") for s in steps 
               if len(s.get("thought") or "") > 20]
    thought_mean = float(np.mean(lengths)) if lengths else 0
    
    # thought_length_var
    thought_var = float(np.var(lengths)) if len(lengths) >= 2 else 0
    
    # tokens_per_step
    total_tok = sum((s.get("tokens_input",0) or 0) + (s.get("tokens_output",0) or 0) 
                     for s in steps)
    tokens_per = total_tok / len(steps)
    
    # decision_oscillation
    import re
    pattern = r'\b(wait|actually|no,|but wait|hold on|let me reconsider|hmm|however|instead|correction)\b'
    total_words, total_matches = 0, 0
    for s in steps:
        thought = s.get("thought") or ""
        if len(thought) < 20: continue
        total_matches += len(re.findall(pattern, thought, re.IGNORECASE))
        total_words += len(thought.split())
    oscillation = total_matches / (total_words / 100) if total_words > 10 else 0
    
    # consecutive_failure_count
    max_streak, streak, prev_action = 0, 0, None
    for s in steps:
        action = s.get("action_type") or s.get("action", "") or ""
        obs = (s.get("observation") or "").strip()
        is_empty = obs in ["", "[]", "null", "None", "{}"] or len(obs) < 5
        if action and action == prev_action and is_empty:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
        prev_action = action
    
    return {
        "thought_length_mean": round(thought_mean, 1),
        "thought_length_var": round(thought_var, 1),
        "tokens_per_step": round(tokens_per, 1),
        "decision_oscillation": round(oscillation, 3),
        "consecutive_failure_count": max_streak,
    }

# 采样
sampled = []
for name, path in DATA_FILES.items():
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        continue
    data = json.load(open(path))
    failures = [d for d in data if not is_success(d)]
    n = min(SAMPLE_COUNTS.get(name, 10), len(failures))
    picked = random.sample(failures, n)
    
    for d in picked:
        d["_source"] = name
        d["_signals"] = compute_signals(d)
    sampled.extend(picked)
    print(f"{name}: 失败{len(failures)}条, 采样{n}条")

print(f"\n总采样: {len(sampled)}条")
json.dump(sampled, open("sampled_for_annotation.json", "w"), 
          indent=2, ensure_ascii=False)