"""
根据师弟的人工核对结果，更新 annotations_gpt4.json
"""
import json
import openpyxl

# ===== 路径配置 =====
EXCEL_PATH = "/data/wanghy/agent_traj/核对表格 1.xlsx"
ANNO_PATH = "/data/wanghy/agent_traj/annotations_gpt4.json"  # 改成你的实际路径
OUTPUT_PATH = "/data/wanghy/agent_traj/annotations_human_corrected.json"  # 修正后的输出

# ===== 读取师弟的核对表 =====
wb = openpyxl.load_workbook(EXCEL_PATH)
ws = wb['Sheet1']

corrections = {}  # task_id -> (human_label, is_modified, reason)
for row in ws.iter_rows(min_row=2, values_only=True):
    if row[0] is None:
        continue
    seq, task_id, gpt4_label, human_label, is_modified, reason = row
    # 用 task_id 作为 key，因为 annotations_gpt4.json 中也是按 task_id 组织的
    task_id_int = int(task_id)
    corrections[task_id_int] = {
        "human_label": str(human_label).strip(),
        "is_modified": str(is_modified).strip() == "是",
        "reason": str(reason).strip() if reason else ""
    }

print(f"读取核对记录: {len(corrections)} 条, 其中修改 {sum(1 for v in corrections.values() if v['is_modified'])} 条")

# ===== 解析师弟的标注格式 =====
def parse_human_label(label):
    """
    师弟的标注格式多样，统一解析为 (is_overthinking, pattern)
    例如: "Type_C" -> ("Yes", "Type_C")
          "No (Type_D)" -> ("No", "Type_D")
          "Type_D" -> ("No", "Type_D")
          "No" -> ("No", "Type_D")
    """
    label = label.strip()
    
    # "No (Type_D)" 或 "No"
    if label.startswith("No"):
        return "No", "Type_D"
    
    # "Type_D"
    if label == "Type_D":
        return "No", "Type_D"
    
    # "Yes (Type_B)" 等
    if label.startswith("Yes"):
        # 提取括号里的类型
        if "Type_A" in label:
            return "Yes", "Type_A"
        elif "Type_B" in label:
            return "Yes", "Type_B"
        elif "Type_C" in label:
            return "Yes", "Type_C"
        else:
            return "Yes", "Type_C"  # 默认
    
    # "Partial (Type_B)" 等
    if label.startswith("Partial"):
        if "Type_A" in label:
            return "Partial", "Type_A"
        elif "Type_B" in label:
            return "Partial", "Type_B"
        elif "Type_C" in label:
            return "Partial", "Type_C"
        else:
            return "Partial", "Type_C"
    
    # 直接是 "Type_A", "Type_B", "Type_C"
    if label in ["Type_A", "Type_B", "Type_C"]:
        return "Yes", label
    
    # 兜底
    print(f"  [WARN] 无法解析标注: '{label}', 默认为 No/Type_D")
    return "No", "Type_D"

# ===== 读取原始 annotations =====
annotations = json.load(open(ANNO_PATH))
print(f"读取原始标注: {len(annotations)} 条")

# ===== 更新标注 =====
# 按 annotations 的顺序遍历，用 task_id 匹配核对表
updated_count = 0
for idx, anno in enumerate(annotations):
    task_id = anno.get("task_id")
    
    # 用 task_id 直接匹配核对表
    if task_id not in corrections:
        continue
    
    correction = corrections[task_id]
    is_ot, pattern = parse_human_label(correction["human_label"])
    
    # 记录修改前后
    old_ot = anno.get("is_overthinking", "")
    old_pattern = anno.get("pattern", "")
    
    # 更新
    anno["is_overthinking"] = is_ot
    anno["pattern"] = pattern
    anno["human_verified"] = True
    anno["human_modified"] = correction["is_modified"]
    anno["human_reason"] = correction["reason"]
    
    if correction["is_modified"]:
        updated_count += 1
        print(f"  修正 task_id={task_id}: "
              f"({old_ot}, {old_pattern}) -> ({is_ot}, {pattern})")

print(f"\n共更新 {updated_count} 条标注")

# ===== 统计修正后的分布 =====
from collections import Counter

ot_dist = Counter(a["is_overthinking"] for a in annotations)
pattern_dist = Counter(a["pattern"] for a in annotations 
                       if a.get("is_overthinking") in ["Yes", "Partial"])
all_pattern_dist = Counter(a["pattern"] for a in annotations)

print(f"\n===== 修正后标注分布 =====")
print(f"Overthinking: {dict(ot_dist)}")
print(f"模式分布(含 Type_D): {dict(all_pattern_dist)}")
print(f"模式分布(仅 overthinking): {dict(pattern_dist)}")

# ===== 保存 =====
json.dump(annotations, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)
print(f"\n修正后标注已保存到: {OUTPUT_PATH}")