import json, numpy as np
from collections import Counter

# 读取标注结果（师弟核对完之后换成修正版文件名）
annotations = json.load(open("annotations_gpt4.json"))

# ===== 1. 基本分布统计 =====
total = len(annotations)
ot_yes = [a for a in annotations if a["is_overthinking"] == "Yes"]
ot_partial = [a for a in annotations if a["is_overthinking"] == "Partial"]
ot_no = [a for a in annotations if a["is_overthinking"] == "No"]

print("===== 标注分布 =====")
print(f"总标注: {total}")
print(f"Overthinking Yes: {len(ot_yes)} ({len(ot_yes)/total*100:.1f}%)")
print(f"Overthinking Partial: {len(ot_partial)} ({len(ot_partial)/total*100:.1f}%)")
print(f"Overthinking No: {len(ot_no)} ({len(ot_no)/total*100:.1f}%)")

# 模式分布
patterns = Counter(a["pattern"] for a in annotations if a["is_overthinking"] in ["Yes", "Partial"])
print(f"\n模式分布: {dict(patterns)}")

# ===== 2. 信号检测 vs 标注的覆盖率 =====
# 用每条轨迹自带的信号值，按百分位阈值判断
signals_all = [a["signals"] for a in annotations if a.get("signals")]

# 计算每个信号的中位数作为阈值
thresholds = {}
for sig_name in ["thought_length_mean", "thought_length_var", "tokens_per_step", 
                  "decision_oscillation", "consecutive_failure_count"]:
    values = [s[sig_name] for s in signals_all if s.get(sig_name) is not None]
    thresholds[sig_name] = np.percentile(values, 75) if values else 0

print(f"\n===== 全局信号阈值 (75th percentile) =====")
for k, v in thresholds.items():
    print(f"  {k}: {v:.2f}")

# ===== 按 source 分组计算阈值 (解决跨模型数据分布差异) =====
from collections import defaultdict
source_signals = defaultdict(list)
for a in annotations:
    if a.get("signals"):
        source_signals[a["source"]].append(a["signals"])

source_thresholds = {}
for src, sigs in source_signals.items():
    source_thresholds[src] = {}
    for sig_name in ["thought_length_mean", "thought_length_var", 
                      "tokens_per_step", "decision_oscillation", 
                      "consecutive_failure_count"]:
        values = [s[sig_name] for s in sigs if s.get(sig_name) is not None]
        source_thresholds[src][sig_name] = np.percentile(values, 75) if values else 0

print(f"\n===== 分模型信号阈值 (75th percentile) =====")
for src in sorted(source_thresholds.keys()):
    print(f"  {src}:")
    for k, v in source_thresholds[src].items():
        print(f"    {k}: {v:.2f}")

# 信号检测判定：用分模型阈值，任一 Tier 1 信号超过 75 百分位 OR consecutive_failure >= 3
for a in annotations:
    sig = a.get("signals", {})
    src = a["source"]
    # 优先用分模型阈值，如果没有则用全局阈值
    th = source_thresholds.get(src, thresholds)
    a["signal_detected"] = (
        sig.get("thought_length_var", 0) > th["thought_length_var"] or
        sig.get("thought_length_mean", 0) > th["thought_length_mean"] or
        sig.get("tokens_per_step", 0) > th["tokens_per_step"] or
        sig.get("consecutive_failure_count", 0) >= 3
    )

# 计算 Recall / Precision / F1
gt_positive = [a for a in annotations if a["is_overthinking"] in ["Yes", "Partial"]]
gt_negative = [a for a in annotations if a["is_overthinking"] == "No"]

TP = len([a for a in gt_positive if a["signal_detected"]])
FP = len([a for a in gt_negative if a["signal_detected"]])
FN = len([a for a in gt_positive if not a["signal_detected"]])
TN = len([a for a in gt_negative if not a["signal_detected"]])

recall = TP / (TP + FN) if (TP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n===== 信号检测覆盖率 =====")
print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
print(f"Recall:    {recall:.3f}")
print(f"Precision: {precision:.3f}")
print(f"F1:        {f1:.3f}")

# ===== 3. 按 benchmark 分组 =====
print(f"\n===== 按数据源分组 =====")
sources = sorted(set(a["source"] for a in annotations))
for src in sources:
    sub = [a for a in annotations if a["source"] == src]
    gt_pos = [a for a in sub if a["is_overthinking"] in ["Yes", "Partial"]]
    gt_neg = [a for a in sub if a["is_overthinking"] == "No"]
    tp = len([a for a in gt_pos if a["signal_detected"]])
    fp = len([a for a in gt_neg if a["signal_detected"]])
    r = tp / len(gt_pos) if gt_pos else 0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  {src:<25} 总{len(sub):>3}条, overthinking {len(gt_pos):>3}条, recall={r:.2f}, precision={p:.2f}")

# ===== 4. 按模式分组的检测率 =====
print(f"\n===== 按 overthinking 模式分组 =====")
for pattern in ["Type_A", "Type_B", "Type_C", "Type_D"]:
    sub = [a for a in annotations if a.get("pattern") == pattern]
    if not sub:
        continue
    detected = len([a for a in sub if a["signal_detected"]])
    print(f"  {pattern}: 总{len(sub)}条, 信号检测到{detected}条, 检测率={detected/len(sub):.2f}")