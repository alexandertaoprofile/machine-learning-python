import json


# 加载JSON文件
def load_skill_mapping(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    skill_map = {}

    # 遍历所有顶层分类
    for category in data.values():
        # 遍历分类中的每个条目
        for key, value in category.items():
            # 检测技能ID特征：纯数字且长度>=3
            if key.isdigit() and len(key) >= 3:
                try:
                    skill_id = int(key)
                    # 过滤掉系统消息类ID（小于1000的ID）
                    if skill_id >= 1000:
                        skill_map[skill_id] = value.replace('\\n', ' ')  # 替换换行符
                except ValueError:
                    continue

    return skill_map


# 示例使用
if __name__ == "__main__":
    # 假设JSON文件名为 skills_translation.json
    skill_dict = load_skill_mapping('D:/揉大の烟雾弹/2100/text_data.json')

    # 打印前20个转换结果验证
    print("技能ID与名称对照表（前20项）：")
    for idx, (k, v) in enumerate(list(skill_dict.items())[:20]):
        print(f"{k}: {v}")
        if idx == 19: break

    # 在数据分析中的实际应用示例
    print("\n应用示例：")
    sample_skill_id = 202711
    print(f"技能ID {sample_skill_id} 对应的名称是：{skill_dict.get(sample_skill_id, '未知技能')}")
