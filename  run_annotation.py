import json, os
from openai import OpenAI

ANNOTATION_PROMPT = """你是一个 agent 轨迹分析专家。请分析以下 agent 轨迹，判断该轨迹是否存在 overthinking（过度思考），以及属于哪种模式。

## 轨迹信息
- Benchmark: {source}
- Task ID: {task_id}
- 结果: 失败
- 总步数: {total_steps}
- 信号值: {signals}

## 每步详情
{steps_detail}

## 请判断以下三个问题，严格按 JSON 格式输出：

1. 该轨迹是否存在 overthinking？
   - "Yes": 明显存在冗余步骤，agent 在做无用功
   - "No": 每一步都有必要，失败是因为任务本身困难或外部约束
   - "Partial": 部分步骤冗余，但不严重

2. 如果存在 overthinking，属于哪种模式？
   - "Type_A": 对话循环型——respond→tool→respond 反复确认，本可以批量处理
   - "Type_B": 机械重试型——同一 tool 反复调用且返回空/相同结果，不调整策略
   - "Type_C": 推理冗余型——thought 中反复自我否定、过度验证、在多个候选间犹豫
   - "Type_D": 合理失败型——外部约束导致，agent 行为合理，无 overthinking
   - "Mixed": 混合多种模式
   
3. 从第几步开始出现 overthinking？（如果不存在则填 -1）

请严格按以下 JSON 格式输出，不要有其他内容：
{{"is_overthinking": "Yes/No/Partial", "pattern": "Type_A/Type_B/Type_C/Type_D/Mixed/None", "start_step": -1, "reason": "一句话解释"}}
"""

client = OpenAI(api_key="你的key")

sampled = json.load(open("sampled_for_annotation.json"))
results = []

for i, traj in enumerate(sampled):
    # 构造每步详情
    steps_detail = ""
    for s in traj["steps"]:
        thought = (s.get("thought") or "")[:200]
        action = s.get("action_type") or s.get("action", "") or ""
        obs = (s.get("observation") or "")[:100]
        step_id = s.get("step_id", "?")
        steps_detail += f"Step {step_id}: thought({len(s.get('thought') or '')}字符)={thought}... | action={action} | obs={obs}...\n"
    
    prompt = ANNOTATION_PROMPT.format(
        source=traj.get("_source", "unknown"),
        task_id=traj.get("task_id", "unknown"),
        total_steps=traj.get("total_steps", len(traj["steps"])),
        signals=json.dumps(traj.get("_signals", {})),
        steps_detail=steps_detail
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
        annotation = json.loads(answer)
    except Exception as e:
        annotation = {"is_overthinking": "Error", "pattern": "Error", 
                      "start_step": -1, "reason": str(e)}
    
    annotation["task_id"] = traj.get("task_id")
    annotation["source"] = traj.get("_source")
    annotation["total_steps"] = traj.get("total_steps", len(traj["steps"]))
    annotation["signals"] = traj.get("_signals", {})
    results.append(annotation)
    
    if (i+1) % 10 == 0:
        print(f"已标注 {i+1}/{len(sampled)} 条")

json.dump(results, open("annotations_gpt4.json", "w"), indent=2, ensure_ascii=False)
print(f"标注完成，共 {len(results)} 条")