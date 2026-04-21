"""
从轨迹JSON文件中提取样本轨迹供分析
提取条件：
  1. 步数最多的 2 条失败轨迹
  2. 步数最多的 1 条成功轨迹
  3. 步数最少的 1 条成功轨迹
  4. 1 条中等步数的失败轨迹
"""

import json
import numpy as np
from typing import List, Dict

def load_trajectories(filepath: str) -> List[Dict]:
    """加载轨迹数据"""
    print(f"Loading from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "steps" in data:
            return [data]
        else:
            return list(data.values())
    return []

def extract_sample_trajectories(trajectories: List[Dict]) -> Dict[str, List[Dict]]:
    """
    按条件提取样本轨迹
    
    返回值：
        {
            "max_steps_failed_2": [...],  # 步数最多的2条失败
            "max_steps_success_1": [...],  # 步数最多的1条成功
            "min_steps_success_1": [...],  # 步数最少的1条成功
            "median_steps_failed_1": [...]  # 中等步数的1条失败
        }
    """
    
    # 按success/failure分组
    success_trajs = [t for t in trajectories if t.get("success", False)]
    failed_trajs = [t for t in trajectories if not t.get("success", False)]
    
    print(f"\n统计信息:")
    print(f"  成功轨迹: {len(success_trajs)} 条")
    print(f"  失败轨迹: {len(failed_trajs)} 条")
    
    # 按步数排序
    success_trajs.sort(key=lambda t: t.get("total_steps", 0), reverse=True)
    failed_trajs.sort(key=lambda t: t.get("total_steps", 0), reverse=True)
    
    # 1. 步数最多的2条失败轨迹
    max_steps_failed_2 = failed_trajs[:2]
    print(f"\n✓ 步数最多的2条失败轨迹:")
    for t in max_steps_failed_2:
        print(f"    task_id={t.get('task_id')}, steps={t.get('total_steps')}, "
              f"tokens={t.get('total_input_tokens', 0) + t.get('total_output_tokens', 0)}")
    
    # 2. 步数最多的1条成功轨迹
    max_steps_success_1 = [success_trajs[0]] if success_trajs else []
    if max_steps_success_1:
        t = max_steps_success_1[0]
        print(f"\n✓ 步数最多的1条成功轨迹:")
        print(f"    task_id={t.get('task_id')}, steps={t.get('total_steps')}, "
              f"tokens={t.get('total_input_tokens', 0) + t.get('total_output_tokens', 0)}")
    
    # 3. 步数最少的1条成功轨迹
    success_trajs_asc = sorted(success_trajs, key=lambda t: t.get("total_steps", 0))
    min_steps_success_1 = [success_trajs_asc[0]] if success_trajs_asc else []
    if min_steps_success_1:
        t = min_steps_success_1[0]
        print(f"\n✓ 步数最少的1条成功轨迹:")
        print(f"    task_id={t.get('task_id')}, steps={t.get('total_steps')}, "
              f"tokens={t.get('total_input_tokens', 0) + t.get('total_output_tokens', 0)}")
    
    # 4. 中等步数的1条失败轨迹
    median_steps_failed_1 = []
    if len(failed_trajs) > 0:
        # 找中位数步数的失败轨迹
        failed_steps = [t.get("total_steps", 0) for t in failed_trajs]
        median_step = int(np.median(failed_steps))
        
        # 找最接近中位数的轨迹
        closest_traj = min(failed_trajs, 
                          key=lambda t: abs(t.get("total_steps", 0) - median_step))
        median_steps_failed_1 = [closest_traj]
        
        t = median_steps_failed_1[0]
        print(f"\n✓ 中等步数的1条失败轨迹 (中位数={median_step}):")
        print(f"    task_id={t.get('task_id')}, steps={t.get('total_steps')}, "
              f"tokens={t.get('total_input_tokens', 0) + t.get('total_output_tokens', 0)}")
    
    return {
        "max_steps_failed_2": max_steps_failed_2,
        "max_steps_success_1": max_steps_success_1,
        "min_steps_success_1": min_steps_success_1,
        "median_steps_failed_1": median_steps_failed_1
    }

def save_samples(samples: Dict[str, List[Dict]], output_file: str):
    """保存样本轨迹到单独的JSON文件，按类别分别保存"""
    
    # 计算总轨迹数
    total_samples = sum(len(trajs) for trajs in samples.values())
    
    # 构造输出结构：按类别分别保存，保持每条轨迹的原始格式
    output_data = {
        "samples_summary": {
            "total_samples": total_samples,
            "categories": {
                "max_steps_failed_2": len(samples["max_steps_failed_2"]),
                "max_steps_success_1": len(samples["max_steps_success_1"]),
                "min_steps_success_1": len(samples["min_steps_success_1"]),
                "median_steps_failed_1": len(samples["median_steps_failed_1"])
            }
        },
        "max_steps_failed_2": samples["max_steps_failed_2"],
        "max_steps_success_1": samples["max_steps_success_1"],
        "min_steps_success_1": samples["min_steps_success_1"],
        "median_steps_failed_1": samples["median_steps_failed_1"]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVED] {output_file}")
    print(f"  总共提取 {total_samples} 条轨迹")
    for category, count in output_data["samples_summary"]["categories"].items():
        if count > 0:
            print(f"    - {category}: {count} 条")

if __name__ == "__main__":
    # 指定输入和输出文件路径
    INPUT_FILE = "/data/wanghy/agent_traj/data/retail-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0330210746_triples.json"
    OUTPUT_FILE = "./sample_trajectories_for_analysis.json"
    
    # 加载轨迹
    trajectories = load_trajectories(INPUT_FILE)
    
    # 提取样本
    samples = extract_sample_trajectories(trajectories)
    
    # 保存到文件
    save_samples(samples, OUTPUT_FILE)
    
    print(f"\n{'='*60}")
    print(f"提取完成！已保存到 {OUTPUT_FILE}")
    print(f"你可以用 VS Code 打开此文件进行详细阅读")
    print(f"{'='*60}")
