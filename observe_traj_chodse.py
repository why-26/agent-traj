import json, numpy as np
import sys

def analyze(filepath, benchmark_name):
    data = json.load(open(filepath))
    
    for d in data:
        thoughts = [s.get("thought", "") or "" for s in d["steps"]]
        valid = [len(t) for t in thoughts if len(t) > 20]
        d["thought_length_mean"] = np.mean(valid) if valid else 0
        
        # 适配不同 benchmark 的成功判断
        if "Metrics" in d:
            d["_success"] = d["Metrics"].get("acc", 0) == 1
        elif "reward" in d:
            d["_success"] = d["reward"] == 1
        else:
            d["_success"] = d.get("success", False)
    
    success = [d for d in data if d["_success"]]
    failure = [d for d in data if not d["_success"]]
    
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark_name}")
    print(f"总条数={len(data)}, 成功={len(success)}, 失败={len(failure)}, 成功率={len(success)/len(data)*100:.1f}%")
    print(f"{'='*60}")
    
    if failure:
        # 类型1: thought最长的失败轨迹
        by_thought = sorted(failure, key=lambda x: x.get("thought_length_mean", 0), reverse=True)
        print(f"\n【类型1: thought最长的失败轨迹 Top3】")
        for d in by_thought[:3]:
            print(f"  task_id={d['task_id']}, steps={d['total_steps']}, thought_mean={d['thought_length_mean']:.0f}")
        
        # 类型2: 步数最多的失败轨迹
        by_steps = sorted(failure, key=lambda x: x["total_steps"], reverse=True)
        print(f"\n【类型2: 步数最多的失败轨迹 Top3】")
        for d in by_steps[:3]:
            print(f"  task_id={d['task_id']}, steps={d['total_steps']}, thought_mean={d['thought_length_mean']:.0f}")
    
    if success:
        # 类型3: thought长但成功
        by_thought_succ = sorted(success, key=lambda x: x.get("thought_length_mean", 0), reverse=True)
        print(f"\n【类型3: thought长但成功 Top2】")
        for d in by_thought_succ[:2]:
            print(f"  task_id={d['task_id']}, steps={d['total_steps']}, thought_mean={d['thought_length_mean']:.0f}")
        
        # 类型4: 步数最少的成功
        by_steps_succ = sorted(success, key=lambda x: x["total_steps"])
        print(f"\n【类型4: 步数最少的成功 Top2】")
        for d in by_steps_succ[:2]:
            print(f"  task_id={d['task_id']}, steps={d['total_steps']}, thought_mean={d['thought_length_mean']:.0f}")

# ===== 改成你的文件路径 =====
files = {
    "TAU-bench Retail GPT-4.1": "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Retail/retail-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0330210746_triples.json",
    "TAU-bench Airline GPT-4.1": "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark1-TAU-bench Airline/airline-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0409224328_triples.json",
    "HotpotQA GPT-4.1": "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/hotpotqa-gpt-4.1-test.4.10,0.55.triples.json",
    "Bamboogle GPT-4.1": "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark3-Bamboogle/bamboogle-gpt-4.1-test.4.10,15.23.triples.json",
}

for name, path in files.items():
    try:
        analyze(path, name)
    except Exception as e:
        print(f"\n{name}: 读取失败 - {e}")