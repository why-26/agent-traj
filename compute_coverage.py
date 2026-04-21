import json
import numpy as np

annotations = json.load(open("annotations_gpt4.json"))

# ===== 基本统计 =====
total = len(annotations)
overthinking_yes = [a for a in annotations if a["is_overthinking"] == "Yes"]
overthinking_partial = [a for a in annotations if a["is_overthinking"] == "Partial"]
overthinking_no = [a for a in annotations if a["is_overthinking"] == "No"]

print(f"总标注: {total}")
print(f"Overthinking Yes: {len(overthinking_yes)} ({len(overthinking_yes)/total*100:.1f}%)")
print(f"Overthinking Partial: {len(overthinking_partial)} ({len(overthinking_partial)/total*100:.1f}%)")
print(f"Overthinking No: {len(overthinking_no)} ({len(overthinking_no)/total*100:.1f}%)")

# ===== 模式分布 =====
from collections import Counter
patterns = Counter(a["pattern"] for a in annotations if a["is_overthinking"] in ["Yes", "Partial"])
print(f"\n模式分布: {dict(patterns)}")

# ===== 信号阈值检测 vs 标注的覆盖率 =====
# 用 thought_length_var 做示例（你的最强信号）
# 阈值设为成功组均值 + 1 std（需要从之前的数据算）

# 这里用一个简单策略：信号值高于中位数就判为 overthinking
signals_list = [a["signals"] for a in annotations if a["signals"]]
median_thought_var = np.median([s["thought_length_var"] for s in signals_list])

# 信号判定
for a in annotations:
    sig = a.get("signals", {})
    # 简单规则：任意一个核心信号超过中位数就判为 overthinking
    a["signal_detected"] = (
        sig.get("thought_length_var", 0) > median_thought_var or
        sig.get("consecutive_failure_count", 0) >= 3
    )

# 算 Recall 和 Precision
ground_truth_positive = [a for a in annotations if a["is_overthinking"] in ["Yes", "Partial"]]
ground_truth_negative = [a for a in annotations if a["is_overthinking"] == "No"]
signal_positive = [a for a in annotations if a.get("signal_detected")]
signal_negative = [a for a in annotations if not a.get("signal_detected")]

TP = len([a for a in ground_truth_positive if a.get("signal_detected")])
FP = len([a for a in ground_truth_negative if a.get("signal_detected")])
FN = len([a for a in ground_truth_positive if not a.get("signal_detected")])
TN = len([a for a in ground_truth_negative if not a.get("signal_detected")])

recall = TP / (TP + FN) if (TP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n===== 信号检测覆盖率 =====")
print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
print(f"Recall: {recall:.3f}")
print(f"Precision: {precision:.3f}")
print(f"F1: {f1:.3f}")

# ===== 按 benchmark 分组看 =====
print(f"\n===== 按数据源分组 =====")
sources = set(a["source"] for a in annotations)
for src in sorted(sources):
    sub = [a for a in annotations if a["source"] == src]
    gt_pos = [a for a in sub if a["is_overthinking"] in ["Yes", "Partial"]]
    tp = len([a for a in gt_pos if a.get("signal_detected")])
    r = tp / len(gt_pos) if gt_pos else 0
    print(f"{src}: 总{len(sub)}条, overthinking {len(gt_pos)}条, 信号recall={r:.2f}")