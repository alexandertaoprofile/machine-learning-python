import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import xgboost as xgb  
import re

# ----------------------
# 配置中文字体
# ----------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
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
            if key.isdigit() and len(key) >= 4:  # 只保留 4 位数以上的 ID
                skill_id = int(key)
                if 1000 <= skill_id <= 999999999:
                    skill_map[skill_id] = value.replace('\\n', ' ')
    return skill_map

# ----------------------
# 数据加载与预处理
# ----------------------
def load_data(skill_map):
    # 加载数据
    team = pd.read_csv("D:/loh_uaf_3200/Team_Member.csv")
    skills = pd.read_csv("D:/loh_uaf_3200/Skills.csv")

    # 重构技能数据
    skill_cols = [col for col in skills.columns if col.startswith('skill_id_') and col != 'skill_id_1']

    skills_long = (
        skills.melt(
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
    merged = pd.merge(team, skills_encoded, on="uma_id", how='left')

    # 填充缺失技能为0
    skill_features = [col for col in merged.columns if col.startswith('skill_')]
    merged[skill_features] = merged[skill_features].fillna(0).astype(int)

    # 添加中文名称映射
    merged_skills = merged[skill_features].copy()
    merged_skills.columns = [
        f"{skill_map.get(int(col.split('_')[1]), '未知技能')} ({col.split('_')[1]})"
        for col in merged_skills.columns
    ]

    # 提取目标变量
    target_cols = ['win_rate', 'paired_rate']
    X = merged[skill_features]
    y = merged[target_cols]

    return X, y, merged_skills.columns.tolist()

# ----------------------
# XGBoost模型训练与评估
# ----------------------
def train_xgboost(X, y, target_name):
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost模型
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
    
    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 性能评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{target_name} MSE: {mse:.4f}")
    print(f"{target_name} R²: {r2:.4f}")

    return model

def plot_shap(X, model, skill_map, top_n=20):
    # 使用 TreeExplainer 来解决 SHAP 的问题
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 获取特征名称
    feature_names = X.columns

    # SHAP值是一个数组，获取每个特征的 SHAP 值
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)

    # 将技能ID转换为实际技能名称
    shap_values_df.columns = [
        skill_map.get(int(col.split('_')[1]), col) if 'skill_' in col else col
        for col in shap_values_df.columns
    ]

    # 打印出技能ID与对应的SHAP值
    print("\n技能ID与名称和对应的SHAP值：")
    for col in shap_values_df.columns:
        if 'skill_' in col:
            skill_name = skill_map.get(int(col.split('_')[1]), col)  # 获取中文技能名称
            print(f"{col}: {skill_name}, SHAP值: {shap_values_df[col].mean()}")
        else:
            print(f"{col}: 无技能ID")

    # 绘制SHAP值摘要图

    shap.summary_plot(shap_values, X, feature_names=shap_values_df.columns)

    # 获取重要的前top_n特征
    top_features = shap_values_df.mean(axis=0).abs().sort_values(ascending=False).head(top_n).index

    # 仅绘制每个特征与目标的关系图，跳过特征与自己的对比
    for feat in top_features:
        if feat != feat:  # Skip if feature is itself
            shap.dependence_plot(feat, shap_values, X, feature_names=shap_values_df.columns, interaction_index=None)

# --------------------------
# 特征重要性分析 (使用 gain)
# --------------------------
def plot_feature_importance(model, skill_map, importance_type="gain", top_n=20):
    # 通过 XGBoost 的内置函数来绘制特征重要性
    importance = model.get_booster().get_score(importance_type=importance_type)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # 替换技能ID为中文技能名称
    sorted_importance_with_name = [
        (skill_map.get(int(key.split('_')[1]), key), value)
        for key, value in sorted_importance
    ]

    # 转换成 DataFrame 便于绘图
    importance_df = pd.DataFrame(sorted_importance_with_name, columns=['Feature', 'Importance'])
    
    # 可视化前top_n个特征
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance (Gain)')
    plt.title(f'Feature Importance - Top {top_n} by Gain')
    plt.gca().invert_yaxis()
    plt.show()

# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    # 加载技能映射
    skill_map = load_skill_mapping(r'D:/loh_uaf_3200/text_data_curate.json')

    # 加载数据
    X, y, skill_names = load_data(skill_map)

    # 训练并评估模型
    print("\n=== 胜率分析 ===")
    win_model = train_xgboost(X, y['win_rate'], "胜率")

    print("\n=== 前二率分析 ===")
    paired_model = train_xgboost(X, y['paired_rate'], "前二率")

    # 绘制特征重要性图
    print("\n=== 特征重要性 ===")
    plot_feature_importance(win_model, skill_map, importance_type="gain", top_n=20)  # 使用 gain 来分析贡献
    plot_feature_importance(paired_model, skill_map, importance_type="gain", top_n=20)

    # SHAP分析
    print("\n=== SHAP分析 ===")
    plot_shap(X, win_model, skill_map)
    plot_shap(X, paired_model, skill_map)
