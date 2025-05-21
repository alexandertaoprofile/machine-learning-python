import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

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
            if key.isdigit() and 1000 <= int(key) <= 999999:
                skill_map[int(key)] = value.replace('\\n', ' ')
    return skill_map


# ----------------------
# 数据加载与预处理
# ----------------------
def load_kobayashi_data(json_path):
    # 加载数据
    team = pd.read_csv("D:/揉大の烟雾弹/2100/Team_Member.csv")
    skills = pd.read_csv("D:/揉大の烟雾弹/2100/Skills.csv")

    # 筛选小林历奇数据
    kobayashi = team[team['card_id'] == 109801].copy()
    if kobayashi.empty:
        raise ValueError("未找到小林历奇数据")

    # 获取技能数据
    skill_cols = [col for col in skills.columns if col.startswith('skill_id_')]
    skills_long = (
        skills[skills['uma_id'].isin(kobayashi['uma_id'])]
        .melt(id_vars='uma_id', value_vars=skill_cols, value_name='skill_id')
        .dropna(subset=['skill_id'])
        .drop('variable', axis=1)
        .assign(skill_id=lambda x: x['skill_id'].astype(int))
    )

    # 创建二进制特征
    skills_encoded = pd.get_dummies(
        skills_long,
        columns=['skill_id'],
        prefix='skill',
        prefix_sep='_'
    ).groupby('uma_id').max()

    # 合并数据
    merged = pd.merge(kobayashi, skills_encoded, on="uma_id", how='left')

    # 处理缺失值
    skill_features = [col for col in merged.columns if col.startswith('skill_')]
    merged[skill_features] = merged[skill_features].fillna(0).astype(int)

    # 加载技能名称映射
    skill_map = load_skill_mapping(json_path)
    skill_names = {
        col: f"{skill_map.get(int(col.split('_')[1]), '未知技能')}({col.split('_')[1]})"
        for col in skill_features
    }


    return merged[skill_features], merged[['win_rate', 'paired_rate']], skill_names


# ----------------------
# XGBoost分析及可视化
# ----------------------
def xgb_analysis(X, y, target_name, skill_names):
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    model = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        importance_type='gain',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # 创建结果DataFrame
    results = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_,
        'shap_mean': np.abs(shap_values).mean(axis=0),
        'shap_direction': np.sign(shap_values.mean(axis=0))
    }).sort_values('shap_mean', ascending=False)

    # 映射技能名称
    results['skill_name'] = results['feature'].map(skill_names)

    # 可视化
    plt.figure(figsize=(12, 8))
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in results['shap_direction'].head(15)]

    bars = plt.barh(
        results['skill_name'].head(15)[::-1],  # 反转顺序使最重要项在上方
        results['shap_mean'].head(15)[::-1],
        color=colors[::-1],
        height=0.7
    )

    # 添加数值标签
    for i, (mean_val, dir_val) in enumerate(zip(
            results['shap_mean'].head(15)[::-1],
            results['shap_direction'].head(15)[::-1]
    )):
        label = f"{dir_val:+.2f}"
        plt.text(
            mean_val / 2 if dir_val > 0 else mean_val,
            i,
            label,
            color='white',
            va='center',
            ha='center',
            fontsize=10
        )

    plt.xlabel('特征影响力（SHAP绝对值均值）')
    plt.title(f'小林历奇技能对{target_name}的影响分析\n（颜色表示影响方向，数值为平均SHAP值）')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)

    # 添加图例
    plt.text(0.95, 0.15, '▲ 正向影响', color='#4CAF50',
             ha='right', va='center', transform=plt.gca().transAxes)
    plt.text(0.95, 0.10, '▼ 负向影响', color='#F44336',
             ha='right', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

    return results


# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # 配置参数
    JSON_PATH = "D:/揉大の烟雾弹/2100/text_data_curate.json"

    try:
        # 加载数据
        X, y, skill_names = load_kobayashi_data(JSON_PATH)

        print("=== 小林历奇技能对胜率的影响 ===")
        win_results = xgb_analysis(X, y['win_rate'], '胜率', skill_names)
        print(win_results.head(15))

        print("\n=== 小林历奇技能对前二率的影响 ===")
        paired_results = xgb_analysis(X, y['paired_rate'], '前二率', skill_names)
        print(paired_results.head(15))

    except Exception as e:
        print(f"发生错误：{str(e)}")