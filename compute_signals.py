"""
Signal Computation Script — Phase 1 信号验证
=============================================
在 qwen3-4b TAU-bench-retail 的 115 条轨迹上计算 semantic velocity 和 action repetition，
验证信号能否区分高效轨迹和低效轨迹。

使用方法:
  pip install sentence-transformers numpy matplotlib seaborn scipy scikit-learn
  python compute_signals.py

输出:
  - 终端：统计数据 + 显著性检验结果
  - analysis_results/ 目录下：4 张分析图 + 1 个 JSON 结果文件
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 配置（改成你的实际路径）
# ============================================================

TRAJ_FILE = "/data/wanghy/agent_traj/data/retail-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_0330210746_triples.json"
OUTPUT_DIR = "./sig2_analysis_results_gpt4.1/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Step 1: 加载数据
# ============================================================

def load_trajectories(filepath):
    """加载轨迹数据，支持 JSON 数组或单个对象"""
    print(f"Loading trajectories from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        trajectories = data
    elif isinstance(data, dict):
        if "steps" in data:
            trajectories = [data]
        else:
            trajectories = list(data.values())
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    print(f"Loaded {len(trajectories)} trajectories.")
    return trajectories

# ============================================================
# Step 2: 计算 Semantic Velocity
# ============================================================

def compute_semantic_velocity(steps, model):
    """
    计算每对相邻步骤的 thought 之间的语义距离。
    
    Semantic Velocity = 1 - cosine_similarity(thought_t, thought_{t-1})
    值大 = 推理在推进，值小(趋近0) = 推理原地打转
    """
    thoughts = []
    step_ids = []
    for step in steps:
        thought = step.get("thought") or ""
        if len(thought.strip()) > 20:
            thoughts.append(thought)
            step_ids.append(step["step_id"])
    
    if len(thoughts) < 2:
        return [], []
    
    embeddings = model.encode(thoughts, show_progress_bar=False)
    
    velocities = []
    velocity_step_ids = []
    for i in range(1, len(embeddings)):
        sim = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]
        velocity = 1.0 - sim
        velocities.append(velocity)
        velocity_step_ids.append(step_ids[i])
    
    return velocities, velocity_step_ids

# ============================================================
# Step 3: 计算 Action Repetition
# ============================================================

def compute_action_repetition(steps, window_size=3):
    """
    计算动作序列在滑动窗口内的重复率。
    
    Action Repetition = 1 - (unique_actions / window_size)
    值为 0 = 每个动作都不同，值高 = 大量重复
    """
    actions = []
    step_ids = []
    for step in steps:
        action = step.get("action_type") or ""
        if action and action not in ("respond", ""):
            actions.append(action)
            step_ids.append(step["step_id"])
    
    if len(actions) < window_size:
        return [], []
    
    repetition_rates = []
    rep_step_ids = []
    for i in range(window_size, len(actions) + 1):
        window = actions[i - window_size:i]
        unique_ratio = len(set(window)) / len(window)
        repetition_rate = 1.0 - unique_ratio
        repetition_rates.append(repetition_rate)
        rep_step_ids.append(step_ids[i - 1])
    
    return repetition_rates, rep_step_ids

# ============================================================
# Step 4: 批量计算
# ============================================================

def compute_all_signals(trajectories, model):
    """对所有轨迹计算信号"""
    results = []
    
    for i, traj in enumerate(trajectories):
        if (i + 1) % 20 == 0:
            print(f"  Processing trajectory {i+1}/{len(trajectories)}...")
        
        steps = traj.get("steps", [])
        success = traj.get("success", False)
        total_steps = traj.get("total_steps", len(steps))
        task_id = traj.get("task_id", i)
        
        velocities, v_steps = compute_semantic_velocity(steps, model)
        repetitions, r_steps = compute_action_repetition(steps)
        
        # 计算高级信号
        advanced_signals = compute_advanced_signals(steps, velocities)
        
        result_item = {
            "task_id": task_id,
            "success": success,
            "total_steps": total_steps,
            "total_tokens": traj.get("total_input_tokens", 0) + traj.get("total_output_tokens", 0),
            "avg_velocity": float(np.mean(velocities)) if velocities else None,
            "min_velocity": float(np.min(velocities)) if velocities else None,
            "max_velocity": float(np.max(velocities)) if velocities else None,
            "velocities": velocities,
            "velocity_step_ids": v_steps,
            "avg_repetition": float(np.mean(repetitions)) if repetitions else None,
            "max_repetition": float(np.max(repetitions)) if repetitions else None,
            "repetitions": repetitions,
            "repetition_step_ids": r_steps,
        }
        # 合并高级信号
        result_item.update(advanced_signals)
        results.append(result_item)
    
    return results

def compute_advanced_signals(steps, velocities):
    """基于第一轮分析的发现，计算改进版信号"""
    signals = {}
    
    # 信号 A: Velocity Variance（速度震荡度）
    # 依据：图2显示失败轨迹velocity剧烈震荡，成功轨迹平稳
    if len(velocities) >= 2:
        signals["velocity_variance"] = float(np.var(velocities))
        signals["velocity_std"] = float(np.std(velocities))
    
    # 信号 B: Token per Step（每步token消耗）
    # 依据：失败组总token多22%但步数只多11%，说明每步消耗更多
    total_tokens = sum(
        (s.get("tokens_input", 0) or 0) + (s.get("tokens_output", 0) or 0) 
        for s in steps
    )
    if len(steps) > 0:
        signals["tokens_per_step"] = total_tokens / len(steps)
    
    # 信号 C: Velocity Trend（速度趋势斜率）
    # 依据：健康推理应该velocity先高后低（收敛），overthinking则不收敛
    if len(velocities) >= 3:
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(range(len(velocities)), velocities)
        signals["velocity_slope"] = float(slope)
    
    # 信号 D: Late-stage Velocity（后半段平均速度）
    # 依据：差异可能主要在轨迹后半段
    if len(velocities) >= 4:
        mid = len(velocities) // 2
        signals["early_velocity"] = float(np.mean(velocities[:mid]))
        signals["late_velocity"] = float(np.mean(velocities[mid:]))
        signals["velocity_drop"] = signals["early_velocity"] - signals["late_velocity"]
    
    # 信号 E: Thought Length Variance（thought长度方差）
    # 依据：overthinking时某些步骤thought突然变长（犹豫不决）
    thought_lengths = [len(s.get("thought") or "") for s in steps 
                       if (s.get("thought") or "") and len(s.get("thought") or "") > 20]
    if len(thought_lengths) >= 2:
        signals["thought_length_var"] = float(np.var(thought_lengths))
        signals["thought_length_mean"] = float(np.mean(thought_lengths))
    
    return signals

# ============================================================
# Step 5: 分析 + 可视化
# ============================================================

def analyze(results):
    """分组对比 + 统计检验 + 生成图表"""
    
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    fail_count = total - success_count
    steps_list = [r["total_steps"] for r in results]
    tokens_list = [r["total_tokens"] for r in results]
    
    print(f"\n{'='*60}")
    print(f"基础统计")
    print(f"{'='*60}")
    print(f"总轨迹数: {total}")
    print(f"成功: {success_count} ({success_count/total:.1%})")
    print(f"失败: {fail_count} ({fail_count/total:.1%})")
    print(f"步数: min={np.min(steps_list)}, median={np.median(steps_list):.0f}, "
          f"mean={np.mean(steps_list):.1f}, max={np.max(steps_list)}")
    print(f"Token: min={np.min(tokens_list)}, median={np.median(tokens_list):.0f}, "
          f"mean={np.mean(tokens_list):.0f}, max={np.max(tokens_list)}")
    
    # ========== 分组 ==========
    valid_results = [r for r in results if r["avg_velocity"] is not None]
    successful = [r for r in valid_results if r["success"]]
    failed = [r for r in valid_results if not r["success"]]
    
    if len(failed) < 10 and len(successful) > 20:
        print(f"\n[INFO] 失败轨迹只有 {len(failed)} 条，改为按步数分组")
        median_steps = np.median([r["total_steps"] for r in successful])
        group_a = [r for r in successful if r["total_steps"] <= median_steps]
        group_b = [r for r in successful if r["total_steps"] > median_steps]
        group_names = (f"Efficient (steps <= {median_steps:.0f})", 
                       f"Inefficient (steps > {median_steps:.0f})")
    else:
        group_a = successful
        group_b = failed
        group_names = ("Successful", "Failed")
    
    print(f"\n分组:")
    print(f"  {group_names[0]}: {len(group_a)} 条")
    print(f"  {group_names[1]}: {len(group_b)} 条")
    
    # ========== 统计检验 ==========
    print(f"\n{'='*60}")
    print(f"信号区分度检验（含高级信号）")
    print(f"{'='*60}")
    
    signal_results = {}
    
    # 基础信号
    vel_a = [r["avg_velocity"] for r in group_a if r["avg_velocity"] is not None]
    vel_b = [r["avg_velocity"] for r in group_b if r["avg_velocity"] is not None]
    
    if vel_a and vel_b:
        stat, p_val = mannwhitneyu(vel_a, vel_b, alternative='two-sided')
        print(f"\nSemantic Velocity:")
        print(f"  {group_names[0]} mean: {np.mean(vel_a):.4f} (std={np.std(vel_a):.4f})")
        print(f"  {group_names[1]} mean: {np.mean(vel_b):.4f} (std={np.std(vel_b):.4f})")
        print(f"  Mann-Whitney U p-value: {p_val:.6f}")
        print(f"  {'*** SIGNIFICANT ***' if p_val < 0.05 else '(not significant)'}")
        signal_results["velocity"] = {"p": p_val, "a_mean": np.mean(vel_a), "b_mean": np.mean(vel_b)}
    
    rep_a = [r["avg_repetition"] for r in group_a if r["avg_repetition"] is not None]
    rep_b = [r["avg_repetition"] for r in group_b if r["avg_repetition"] is not None]
    
    if rep_a and rep_b:
        stat, p_val = mannwhitneyu(rep_a, rep_b, alternative='two-sided')
        print(f"\nAction Repetition:")
        print(f"  {group_names[0]} mean: {np.mean(rep_a):.4f} (std={np.std(rep_a):.4f})")
        print(f"  {group_names[1]} mean: {np.mean(rep_b):.4f} (std={np.std(rep_b):.4f})")
        print(f"  Mann-Whitney U p-value: {p_val:.6f}")
        print(f"  {'*** SIGNIFICANT ***' if p_val < 0.05 else '(not significant)'}")
        signal_results["repetition"] = {"p": p_val, "a_mean": np.mean(rep_a), "b_mean": np.mean(rep_b)}
    
    # 高级信号检验
    print(f"\n--- 高级信号 ---")
    advanced_signal_keys = [
        "velocity_variance", "velocity_std", "tokens_per_step", 
        "velocity_slope", "velocity_drop", "early_velocity", "late_velocity",
        "thought_length_var", "thought_length_mean"
    ]
    
    for signal_key in advanced_signal_keys:
        sig_a = [r[signal_key] for r in group_a if signal_key in r and r[signal_key] is not None]
        sig_b = [r[signal_key] for r in group_b if signal_key in r and r[signal_key] is not None]
        
        if sig_a and sig_b and len(sig_a) > 1 and len(sig_b) > 1:
            stat, p_val = mannwhitneyu(sig_a, sig_b, alternative='two-sided')
            sig_name = signal_key.replace("_", " ").title()
            print(f"\n{sig_name}:")
            print(f"  {group_names[0]} mean: {np.mean(sig_a):.4f} (std={np.std(sig_a):.4f})")
            print(f"  {group_names[1]} mean: {np.mean(sig_b):.4f} (std={np.std(sig_b):.4f})")
            print(f"  Mann-Whitney U p-value: {p_val:.6f}")
            print(f"  {'*** SIGNIFICANT ***' if p_val < 0.05 else '(not significant)'}")
            signal_results[signal_key] = {"p": p_val, "a_mean": np.mean(sig_a), "b_mean": np.mean(sig_b)}
    
    all_steps = [r["total_steps"] for r in results]
    all_success = [1 if r["success"] else 0 for r in results]
    if len(set(all_steps)) > 1:
        corr, p_corr = spearmanr(all_steps, all_success)
        print(f"\n步数 vs 成功率 (Spearman):")
        print(f"  correlation: {corr:.4f}, p-value: {p_corr:.6f}")
        if corr < 0 and p_corr < 0.05:
            print(f"  *** 步数越多成功率越低 → overthinking 有害 ***")
    
    # ========== 图 1: 信号分布箱线图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if vel_a and vel_b:
        bp = axes[0].boxplot([vel_a, vel_b], labels=list(group_names),
                             patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#2ecc71'); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#e74c3c'); bp['boxes'][1].set_alpha(0.7)
        axes[0].set_ylabel("Average Semantic Velocity", fontsize=12)
        axes[0].set_title("Semantic Velocity", fontsize=13)
        p = signal_results.get("velocity", {}).get("p", 1)
        axes[0].text(0.5, 0.97, f"p = {p:.4f} {'✓' if p < 0.05 else ''}",
                     transform=axes[0].transAxes, ha='center', va='top',
                     fontsize=11, color='red' if p < 0.05 else 'gray', fontweight='bold')
    
    if rep_a and rep_b:
        bp = axes[1].boxplot([rep_a, rep_b], labels=list(group_names),
                             patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#2ecc71'); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#e74c3c'); bp['boxes'][1].set_alpha(0.7)
        axes[1].set_ylabel("Average Action Repetition Rate", fontsize=12)
        axes[1].set_title("Action Repetition", fontsize=13)
        p = signal_results.get("repetition", {}).get("p", 1)
        axes[1].text(0.5, 0.97, f"p = {p:.4f} {'✓' if p < 0.05 else ''}",
                     transform=axes[1].transAxes, ha='center', va='top',
                     fontsize=11, color='red' if p < 0.05 else 'gray', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_signal_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVED] {OUTPUT_DIR}fig1_signal_distribution.png")
    
    # ========== 图 2: Velocity 时间序列 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    efficient_examples = sorted([r for r in group_a if r["velocities"]],
                                 key=lambda r: r["total_steps"])[:3]
    for r in efficient_examples:
        axes[0].plot(r["velocity_step_ids"], r["velocities"], '-o',
                     markersize=4, alpha=0.7, label=f'task_{r["task_id"]} ({r["total_steps"]}步)')
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Semantic Velocity")
    axes[0].set_title(f"{group_names[0]} Examples"); axes[0].legend(fontsize=8); axes[0].set_ylim(bottom=0)
    
    inefficient_examples = sorted([r for r in group_b if r["velocities"]],
                                   key=lambda r: r["total_steps"], reverse=True)[:3]
    for r in inefficient_examples:
        axes[1].plot(r["velocity_step_ids"], r["velocities"], '-o',
                     markersize=4, alpha=0.7, label=f'task_{r["task_id"]} ({r["total_steps"]}步)')
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Semantic Velocity")
    axes[1].set_title(f"{group_names[1]} Examples"); axes[1].legend(fontsize=8); axes[1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_velocity_timeline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {OUTPUT_DIR}fig2_velocity_timeline.png")
    
    # ========== 图 3: 步数分布 ==========
    fig, ax = plt.subplots(figsize=(10, 5))
    success_steps = [r["total_steps"] for r in results if r["success"]]
    fail_steps = [r["total_steps"] for r in results if not r["success"]]
    bins = range(0, max(steps_list) + 2)
    if success_steps:
        ax.hist(success_steps, bins=bins, alpha=0.6, label=f"Successful (n={len(success_steps)})",
                color='#2ecc71', edgecolor='white')
    if fail_steps:
        ax.hist(fail_steps, bins=bins, alpha=0.6, label=f"Failed (n={len(fail_steps)})",
                color='#e74c3c', edgecolor='white')
    ax.axvline(np.median(steps_list), color='black', linestyle='--', alpha=0.5,
               label=f'Median = {np.median(steps_list):.0f}')
    ax.set_xlabel("Total Steps"); ax.set_ylabel("Count")
    ax.set_title("Step Distribution: Successful vs Failed"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_step_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {OUTPUT_DIR}fig3_step_distribution.png")
    
    # ========== 图 4: Steps vs Velocity 散点图 ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in valid_results:
        color = '#2ecc71' if r["success"] else '#e74c3c'
        marker = 'o' if r["success"] else 'x'
        ax.scatter(r["total_steps"], r["avg_velocity"], c=color, marker=marker, alpha=0.6, s=50)
    ax.scatter([], [], c='#2ecc71', marker='o', label='Successful')
    ax.scatter([], [], c='#e74c3c', marker='x', label='Failed')
    ax.set_xlabel("Total Steps"); ax.set_ylabel("Average Semantic Velocity")
    ax.set_title("Steps vs Semantic Velocity"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_steps_vs_velocity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {OUTPUT_DIR}fig4_steps_vs_velocity.png")
    
    # ========== 保存结果 ==========
    output_data = [
        {
            "task_id": r["task_id"], "success": r["success"],
            "total_steps": r["total_steps"], "total_tokens": r["total_tokens"],
            "avg_velocity": r["avg_velocity"], "min_velocity": r["min_velocity"],
            "avg_repetition": r["avg_repetition"], "max_repetition": r["max_repetition"],
            **{k: r.get(k) for k in advanced_signal_keys if k in r}  # 包含所有高级信号
        } for r in results
    ]
    
    with open(os.path.join(OUTPUT_DIR, "signal_results.json"), 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {OUTPUT_DIR}signal_results.json")
    
    # ========== 汇总 ==========
    print(f"\n{'='*60}")
    print(f"Phase 2 信号验证汇总（含高级信号）")
    print(f"{'='*60}")
    print(f"数据: {total} 条 (成功 {success_count}, 失败 {fail_count})")
    print(f"分组: {group_names[0]} ({len(group_a)}) vs {group_names[1]} ({len(group_b)})")
    
    sig_count = 0
    sig_names = []
    
    print(f"\n【基础信号】")
    if "velocity" in signal_results:
        p = signal_results["velocity"]["p"]
        is_sig = p < 0.05
        print(f"  Semantic Velocity:  p={p:.6f}  {'✅' if is_sig else '❌'}")
        if is_sig:
            sig_count += 1
            sig_names.append("Velocity")
    
    if "repetition" in signal_results:
        p = signal_results["repetition"]["p"]
        is_sig = p < 0.05
        print(f"  Action Repetition:  p={p:.6f}  {'✅' if is_sig else '❌'}")
        if is_sig:
            sig_count += 1
            sig_names.append("Repetition")
    
    print(f"\n【高级信号】")
    advanced_signal_display = [
        ("velocity_variance", "Velocity Variance"),
        ("velocity_std", "Velocity Std Dev"),
        ("tokens_per_step", "Tokens Per Step"),
        ("velocity_slope", "Velocity Trend Slope"),
        ("velocity_drop", "Velocity Drop (Early-Late)"),
        ("early_velocity", "Early Stage Velocity"),
        ("late_velocity", "Late Stage Velocity"),
        ("thought_length_var", "Thought Length Variance"),
        ("thought_length_mean", "Thought Length Mean"),
    ]
    
    for signal_key, display_name in advanced_signal_display:
        if signal_key in signal_results:
            p = signal_results[signal_key]["p"]
            is_sig = p < 0.05
            print(f"  {display_name:30s} p={p:.6f}  {'✅' if is_sig else '❌'}")
            if is_sig:
                sig_count += 1
                sig_names.append(display_name)
    
    print(f"\n结论:")
    print(f"  显著信号数: {sig_count}/{len(signal_results)}")
    if sig_count >= 2:
        print(f"  ✅ {', '.join(sig_names[:3])} 等信号显著")
        print(f"  → 假设成立，可以进入实际应用阶段")
    elif sig_count == 1:
        if sig_names:
            print(f"  ⭐ {sig_names[0]} 信号显著")
        print(f"  → 部分成立，有效的保留，无效的需迭代")
    else:
        print(f"  ❌ 所有信号都不显著")
        print(f"  → 需要进一步调整信号设计")
    print(f"{'='*60}")

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("Loading sentence-transformers model...")
    print("(首次运行会下载约 80MB 模型)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.\n")
    
    trajectories = load_trajectories(TRAJ_FILE)
    
    if trajectories:
        t = trajectories[0]
        print(f"\n第一条轨迹检查:")
        print(f"  task_id: {t.get('task_id')}")
        print(f"  success: {t.get('success')}")
        print(f"  total_steps: {t.get('total_steps')}")
        print(f"  steps 数量: {len(t.get('steps', []))}")
        if t.get('steps'):
            s = t['steps'][0]
            print(f"  第一步 keys: {list(s.keys())}")
            thought = s.get('thought') or ''
            print(f"  第一步 thought 长度: {len(thought)} chars")
            print(f"  第一步 action_type: {s.get('action_type')}")
        print()
    
    print("Computing signals...")
    results = compute_all_signals(trajectories, model)
    print("Done.\n")
    
    analyze(results)