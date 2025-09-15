import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# ----------------------
# 配置中文字体
# ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------
# 加载技能名称映射
# ----------------------
def load_skill_mapping(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    skill_map = {}
    for category in data.values():
        for key, value in category.items():
            if key.isdigit() and len(key) >= 4:
                skill_id = int(key)
                if 1000 <= skill_id <= 999999:
                    skill_map[skill_id] = value.replace('\\n', ' ')
    return skill_map


# ----------------------
# 数据加载与预处理（筛选特定马娘）
# ----------------------
def load_specific_uma_data(skill_map, target_card_id=100602, running_style=2):
    # 加载数据
    team = pd.read_csv("D:/揉大の烟雾弹/2100/Team_Member.csv")
    skills = pd.read_csv("D:/揉大の烟雾弹/2100/Skills.csv")

    # 筛选特定马娘和跑法
    target_uma = team[
        (team['card_id'] == target_card_id) &
        (team['running_style'] == running_style)
        ].copy()

    if target_uma.empty:
        raise ValueError(
            f"未找到card_id为{target_card_id}且跑法为{running_style}的马娘数据\n"
            f"可用跑法列表：{team['running_style'].unique().tolist()}"
        )

    # 获取对应的uma_id列表
    uma_ids = target_uma['uma_id'].unique()

    # 筛选技能数据（后续保持不变）
    target_skills = skills[skills['uma_id'].isin(uma_ids)]

    # 重构技能数据
    skill_cols = [col for col in target_skills.columns if col.startswith('skill_id_')]

    skills_long = (
        target_skills.melt(
            id_vars='uma_id',
            value_vars=skill_cols,
            value_name='skill_id'
        )
        .dropna(subset=['skill_id'])
        .drop('variable', axis=1)
        .assign(skill_id=lambda x: x['skill_id'].astype(int).astype(str))
    )

    # 创建二进制特征
    skills_encoded = pd.get_dummies(
        skills_long,
        columns=['skill_id'],
        prefix='skill',
        prefix_sep='_'
    ).groupby('uma_id').max()

    # 合并数据
    merged = pd.merge(target_uma, skills_encoded, on="uma_id", how='left')

    # 填充缺失技能为0
    skill_features = [col for col in merged.columns if col.startswith('skill_')]
    merged[skill_features] = merged[skill_features].fillna(0).astype(int)

    # 添加中文名称映射
    skill_names = [
        f"{skill_map.get(int(col.split('_')[1]), '未知技能')} ({col.split('_')[1]})"
        for col in skill_features
    ]

    # 提取目标变量
    target_cols = ['win_rate', 'paired_rate']
    X = merged[skill_features]
    y = merged[target_cols]

    return X, y, skill_names, merged


# ----------------------
# 特征重要性分析
# ----------------------
def analyze_skills(X, y, skill_names):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LassoCV(cv=5, random_state=42)
    model.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        'skill_id': X.columns,
        'skill_name': skill_names,
        'coefficient': model.coef_,
        'abs_impact': np.abs(model.coef_)
    })

    significant_skills = coef_df[coef_df['abs_impact'] > 0].sort_values('abs_impact', ascending=False)
    return significant_skills


# ----------------------
# 可视化（带中文名称）
# ----------------------
def plot_skill_impact(coef_df, target_name, uma_name):
    top_skills = coef_df.head(30).copy()

    plt.figure(figsize=(12, 6))
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in top_skills['coefficient']]

    bars = plt.barh(top_skills['skill_name'],
                    top_skills['abs_impact'],
                    color=colors,
                    height=0.7)

    # 添加数值标签
    for i, (val, color) in enumerate(zip(top_skills['coefficient'], colors)):
        label_pos = val if color == '#4CAF50' else 0
        ha = 'right' if color == '#4CAF50' else 'left'
        plt.text(label_pos, i,
                 f'{val:.2f}',
                 color='black' if abs(val) < 5 else 'white',
                 va='center',
                 ha=ha,
                 fontsize=9)

    plt.xlabel('标准化影响系数 (绝对值)')
    plt.title(f'{uma_name} 对{target_name}影响最大的前15个技能')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)

    # 添加图例
    plt.text(0.95, 0.15, '▲ 正向影响', color='#4CAF50',
             ha='right', va='center', transform=plt.gca().transAxes)
    plt.text(0.95, 0.10, '▼ 负向影响', color='#F44336',
             ha='right', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()


# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # 配置参数
    TARGET_CARD_ID = 100602  # 圣诞帽的card_id
    UMA_NAME = "圣诞帽"
    JSON_PATH = "D:/揉大の烟雾弹/2100/text_data_curate.json"

    try:
        # 加载映射表
        skill_map = load_skill_mapping(JSON_PATH)

        # 加载特定马娘数据
        X, y, skill_names, full_data = load_specific_uma_data(skill_map, TARGET_CARD_ID)

        # 显示基本信息
        print(f"分析对象：{UMA_NAME}（card_id={TARGET_CARD_ID}）")
        print(f"找到有效训练记录：{len(full_data)}条")
        print(f"涉及技能数量：{len(skill_names)}种")
        print("技能示例：", skill_names[:3])  # 打印前3个技能

        # 分析胜率
        print("\n=== 胜率分析 ===")
        win_skills = analyze_skills(X, y['win_rate'], skill_names)
        print(win_skills.head(30))
        plot_skill_impact(win_skills, "胜率", UMA_NAME)

        # 分析前二率
        print("\n=== 前二率分析 ===")
        paired_skills = analyze_skills(X, y['paired_rate'], skill_names)
        print(paired_skills.head(30))
        plot_skill_impact(paired_skills, "前二率", UMA_NAME)

        # 保存原始数据
        full_data.to_csv(f"{UMA_NAME}_分析数据.csv", index=False, encoding='utf_8_sig')
        print(f"\n原始数据已保存为：{UMA_NAME}_分析数据.csv")

    except Exception as e:
        print(f"发生错误：{str(e)}")
