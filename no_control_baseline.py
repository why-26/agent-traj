import json, numpy as np, os

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

def is_success(traj):
    if "Metrics" in traj:
        return traj["Metrics"].get("acc", 0) == 1
    elif "reward" in traj:
        return traj["reward"] == 1
    return traj.get("success", False)

print(f"{'数据集':<25} {'success_rate':>12} {'avg_token':>12} {'avg_steps':>10}")
print("-" * 65)

for name, path in DATA_FILES.items():
    if not os.path.exists(path):
        print(f"{name:<25} 文件不存在")
        continue
    data = json.load(open(path))
    
    success_rate = sum(1 for d in data if is_success(d)) / len(data)
    avg_token = np.mean([
        (d.get("total_input_tokens", 0) or 0) + (d.get("total_output_tokens", 0) or 0)
        for d in data
    ])
    avg_steps = np.mean([d.get("total_steps", len(d.get("steps", []))) for d in data])
    
    print(f"{name:<25} {success_rate:>12.4f} {avg_token:>12.1f} {avg_steps:>10.2f}")