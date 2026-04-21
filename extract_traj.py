import json

# 改成你的路径和筛出来的 task_id
config = {
    "tau_retail": {
        "file": "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Retail/retail-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0330210746_triples.json",
        "ids": [105, 34, 60]
    },
    "tau_airline": {
        "file": "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Airline/airline-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0409224328_triples.json",
        "ids": [10, 25]
    },
    "hotpotqa": {
        "file": "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/hotpotqa-gpt-4.1-test.4.10,0.55.triples.json",
        "ids": [364, 263]
    },
    "bamboogle": {
       "file": "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark3-Bamboogle/bamboogle-gpt-4.1-test.4.10,15.23.triples.json",
        "ids": [50, 78]
    },
}

selected = []
for name, cfg in config.items():
    data = json.load(open(cfg["file"]))
    for d in data:
        if d["task_id"] in cfg["ids"]:
            d["_benchmark"] = name
            selected.append(d)

print(f"共抽出 {len(selected)} 条轨迹")
json.dump(selected, open("selected_trajectories.json", "w"), indent=2, ensure_ascii=False)