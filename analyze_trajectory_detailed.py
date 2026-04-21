"""
详细轨迹分析工具
按步骤记录：
1. 这步有没有带来新进展（Progress）
2. Action是否重复（Repetition）
3. Agent是否在纠错（Error Recovery）
"""

import json
from collections import Counter
import os

# 创建输出目录
os.makedirs('/data/wanghy/agent_traj/traj_analysis', exist_ok=True)

# 设置输出文件
output_file = '/data/wanghy/agent_traj/traj_analysis/trajectory_analysis_report.txt'
output_f = open(output_file, 'w', encoding='utf-8')

def print_and_save(text=""):
    """同时打印到控制台和文件"""
    print(text)
    output_f.write(text + '\n')
    output_f.flush()

# 加载失败轨迹 - 处理不完整的JSON
with open('/data/wanghy/agent_traj/traj_analysis/fail_steps.json', 'r') as f:
    content = f.read()

# 查找JSON起点和终点
try:
    # 尝试直接解析
    data = json.loads(content)
except:
    # 如果失败，尝试找到完整的JSON对象
    # 从"max_steps_failed_2"开始到最后一个"}"
    start = content.find('"max_steps_failed_2"')
    if start != -1:
        # 找到这个键值对的开始
        start = content.rfind('{', 0, start)  # 向前找到最近的{
        # 从末尾向前找匹配的}
        bracket_count = 0
        end = len(content) - 1
        for i in range(len(content) - 1, start, -1):
            if content[i] == '}':
                bracket_count += 1
            elif content[i] == '{':
                bracket_count -= 1
                if bracket_count == 0:
                    end = i + 1
                    break
        
        json_str = content[start:end]
        data = json.loads(json_str)
    else:
        raise ValueError("无法找到有效的JSON数据")

traj = data['max_steps_failed_2'][0]  # 最长失败轨迹
steps = traj['steps']
ground_truth = traj['ground_truth']['actions']
question = traj['question']

print_and_save("=" * 80)
print_and_save(f"轨迹分析：Task {traj['task_id']} (失败，{len(steps)} 步)")
print_and_save("=" * 80)

print_and_save("\n📋 用户需求分析：")
print_and_save("-" * 80)
requirements = {
    "返回两个订单中的书架和拼图": "GT需要：return #W8660475书架 + return #W6239298拼图",
    "返回与吸尘器一起的背包": "GT需要：return #W9218746背包",
    "修改待处理订单为红色": "GT需要：modify_pending_order_items 为红色",
    "更改地址到芝加哥": "GT需要：modify_pending_order_address 到Chicago",
    "获取取消订单的跟踪号": "GT需要：查询 #W1154986 的跟踪号",
}
for req, gt in requirements.items():
    print_and_save(f"  • {req}")
    print_and_save(f"    → {gt}\n")

# ============================================================
# 逐步分析
# ============================================================

print_and_save("\n" + "=" * 80)
print_and_save("📊 逐步详细记录")
print_and_save("=" * 80)

# 记录所有action类型，用于检测重复
all_actions = []
action_sequence = []

for i, step in enumerate(steps):
    step_id = step['step_id']
    action_type = step['action_type']
    action_input = step.get('action_input', {})
    observation = step['observation']
    
    all_actions.append(action_type)
    
    # 简化action序列记录
    if action_type == 'respond':
        action_sequence.append(('respond', None))
    elif action_type == 'get_order_details':
        order_id = action_input.get('order_id', '?')
        action_sequence.append(('get_order', order_id))
    else:
        action_sequence.append((action_type, action_input.get('order_id', '?')))
    
    print_and_save(f"\n[Step {step_id}] {action_type}")
    print_and_save("-" * 80)
    
    # 判断进展
    progress = "无新进展"
    if action_type == 'respond':
        # respond步骤总是有进展（与用户沟通）
        thought = step.get('thought', '')
        if '确认' in thought or '请求' in thought or '说明' in thought:
            progress = "✓ 与用户沟通，收集信息"
        else:
            progress = "⚠ 简单回复"
    
    elif action_type.startswith('get_'):
        progress = "✓ 获取新信息"
    
    elif action_type.startswith('modify_') or action_type.startswith('return_'):
        progress = "✓ 执行操作"
        if "Error" in observation or "error" in observation:
            progress = "❌ 操作失败"
    
    print_and_save(f"进展: {progress}")
    
    # 检测重复
    prev_actions = [a for a in all_actions[:-1] if a == action_type]
    if prev_actions:
        print_and_save(f"重复: ⚠ 第 {len(prev_actions)+1} 次调用 {action_type}")
    
    # 检测是否在纠错
    observation_str = str(observation).lower()
    if "error" in observation_str or "cannot" in observation_str or "not found" in observation_str:
        print_and_save(f"纠错: ❌ 出现错误，需要恢复")
    elif step_id > 0:
        prev_obs = steps[step_id - 1]['observation']
        if action_type == steps[step_id - 1]['action_type']:
            print_and_save(f"纠错: ⚠ 重复执行同一操作（可能是前一步失败）")
    
    # 输出action细节
    if action_type != 'respond':
        print_and_save(f"动作: {action_type}")
        if 'order_id' in action_input:
            print_and_save(f"  - order_id: {action_input['order_id']}")
        if 'item_ids' in action_input:
            print_and_save(f"  - item_ids: {action_input['item_ids']}")
        if 'new_item_ids' in action_input:
            print_and_save(f"  - new_item_ids: {action_input['new_item_ids']}")
    
    # observation结果
    if len(str(observation)) > 200:
        obs_summary = str(observation)[:200] + "..."
    else:
        obs_summary = str(observation)
    print_and_save(f"结果: {obs_summary}")

# ============================================================
# 总体分析
# ============================================================

print_and_save("\n\n" + "=" * 80)
print_and_save("📈 轨迹失败原因分析")
print_and_save("=" * 80)

# 统计action频率
action_counts = Counter(all_actions)
print_and_save(f"\n1️⃣ Action 频率分布：")
print_and_save("-" * 80)
for action, count in action_counts.most_common():
    print_and_save(f"  {action:40s}: {count:3d} 次")

# 分析action序列
print_and_save(f"\n2️⃣ 关键问题诊断：")
print_and_save("-" * 80)

# 问题1: 获取订单太多次
get_order_count = sum(1 for a, _ in action_sequence if a == 'get_order')
unique_orders = len(set(order_id for a, order_id in action_sequence if a == 'get_order'))
print_and_save(f"  • 获取订单详情: {get_order_count} 次，覆盖 {unique_orders} 个订单")
if get_order_count > unique_orders + 2:
    print_and_save(f"    ❌ 浪费: 多次查询同一订单 ({get_order_count - unique_orders} 次重复查询)")

# 问题2: 执行的操作是否正确
print_and_save(f"\n  • 执行的修改/返回操作：")
for a, order_id in action_sequence:
    if 'return_' in a or 'modify_' in a:
        print_and_save(f"    - {a}: {order_id}")

# 问题3: 最后是否达成了GT需求
actual_returns = set()
actual_modifies = []
for a, order_id in action_sequence:
    if 'return_' in a:
        actual_returns.add(order_id)
    elif 'modify_' in a:
        actual_modifies.append(a)

gt_returns = set()
gt_modifies = set()
for gt_action in ground_truth:
    if 'return' in gt_action['name']:
        gt_returns.add(gt_action['kwargs'].get('order_id'))
    elif 'modify' in gt_action['name']:
        gt_modifies.add(gt_action['name'])

print_and_save(f"\n  • 目标 vs 实际:")
print_and_save(f"    GT返回订单: {sorted(gt_returns)}")
print_and_save(f"    实际返回订单: {sorted(actual_returns)}")
print_and_save(f"    GT修改操作: {sorted(gt_modifies)}")
print_and_save(f"    实际执行修改: {len(actual_modifies)} 次")

# 问题4: 理解偏差
print_and_save(f"\n  • 用户需求理解情况：")
if "#W6239298" not in actual_returns and "#W6239298" in gt_returns:
    print_and_save(f"    ❌ 遗漏: 没有返回 #W6239298 的拼图")
    print_and_save(f"       原因: Agent混淆了，以为 #W1154986 是活跃订单（实际已取消）")

if len(actual_returns) > len(gt_returns):
    print_and_save(f"    ⚠️ 过度: 返回了 {len(actual_returns) - len(gt_returns)} 个多余的订单")

# ============================================================
# 输出最终判断
# ============================================================

print_and_save("\n\n" + "=" * 80)
print_and_save("🎯 失败原因总结")
print_and_save("=" * 80)

print_and_save(f"""
根据轨迹分析，这条失败轨迹的主要浪费形式：

【问题类型1】: 信息重复查询
  - get_order_details 被调用 {get_order_count} 次，但只有 {unique_orders} 个不同订单
  - 原因: Agent在 step 4 一次性调用了所有5个订单，后来又重复查询
  - 指标: 重复查询率 = {get_order_count - unique_orders}/{unique_orders} = {(get_order_count - unique_orders) / unique_orders:.1%}

【问题类型2】: 理解偏差 → 执行错误
  - Agent 返回了 {len(actual_returns)} 个订单，但GT只需要 {len(gt_returns)} 个
  - 具体: 返回了 #W6239298 的拼图（错误）+ #W9218746 的背包（正确）+ #W8660475 的书架（正确）
  - 问题: 没有注意到 #W1154986 是 "cancelled" 订单，不能返回
  - 指标: 执行准确率 = {len(set([a for a, _ in action_sequence if 'return' in a]) & gt_returns)}/{len(gt_returns)}

【问题类型3】: 操作顺序错误 → 业务约束违反
  - Step 24: 先修改了pending order的item（变为 "item modified" 状态）
  - Step 27: 再尝试修改address，但返回 Error: "non-pending order cannot be modified"
  - 原因: Agent不知道修改item会锁定订单，导致后续address修改失败
  - 指标: 约束违反率 = 1（有一个操作因业务规则失败）

【问题类型4】: 目标混淆 → 优先级错误
  - Step 29: 最后回答用户"你的已取消订单 #W1154986 没有跟踪号"
  - 但用户要求的是"跟踪号"，应该查询的是成功返回的订单的跟踪号
  - 指标: 最终目标完成度 = 0（主要任务失败）

═══════════════════════════════════════════════════════════════

对应可量化的指标体系：

1. 信息查询效率 (Query Efficiency)
   ┌─ 重复查询率 = (重复查询次数) / (总查询次数)
   ├─ 信息冗余度 = (unique_info_points_found) / (total_queries)
   └─ 查询成本 = (total_tokens_for_queries) / (unique_orders_found)

2. 执行准确率 (Execution Accuracy)
   ┌─ 操作准确率 = (正确执行的操作数) / (应执行的操作数) 
   ├─ 参数匹配率 = (参数与GT一致的操作) / (总操作数)
   └─ 业务约束遵守率 = (不违反系统约束的操作) / (总操作数)

3. 理解偏差 (Understanding Gap)
   ┌─ 需求覆盖率 = (理解的需求项) / (明确的需求项)
   ├─ 误执行率 = (错误执行的操作) / (总操作数)
   └─ 越界度 = (超出需求的额外操作) / (应有操作数)

4. 恢复能力 (Error Recovery)
   ┌─ 错误检测率 = (Agent意识到的错误) / (实际出现的错误)
   ├─ 纠错成功率 = (成功恢复的错误) / (检测到的错误)
   └─ 重试有效性 = (重试后成功) / (重试次数)

5. 任务完成度 (Task Completion)
   ┌─ 最终成功率 = 成功完成核心需求 ? 1 : 0
   ├─ 子任务完成率 = (完成的子任务) / (总子任务数)
   └─ 步数效率 = (最优步数) / (实际步数)
""")

print_and_save("=" * 80)

# 关闭输出文件
output_f.close()
print(f"\n✅ 分析完成，结果已保存到: {output_file}")
