import json

# 读取JSON文件并抽取第一个样例
json_file = "/data/wanghy/agent_traj/agent 推理轨迹数据集/benchmark2-HotpotQA/qwen3-4b-thinking-hotpotqa-all.triples.json"

try:
    with open(json_file, 'r', encoding='utf-8') as f:
        # 如果是JSON数组
        data = json.load(f)
        
        if isinstance(data, list):
            print(f"总共有 {len(data)} 条数据\n")
            # 抽取第一个样例
            sample = data[0]
            print("第一个样例:")
            print(json.dumps(sample, indent=2, ensure_ascii=False))
        elif isinstance(data, dict):
            print(f"这是一个字典，包含以下键: {list(data.keys())}\n")
            # 如果字典中有数据列表，抽取第一条
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"从'{key}'字段抽取第一个样例:")
                    print(json.dumps(value[0], indent=2, ensure_ascii=False))
                    break
        
except json.JSONDecodeError as e:
    print(f"JSON解析错误: {e}")
except Exception as e:
    print(f"错误: {e}")
