import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import shap

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

    # 添加数值特征（speed, stamina, power, guts, wiz）
    merged['speed'] = merged['speed'].fillna(0)  # 填充缺失值
    merged['stamina'] = merged['stamina'].fillna(0)
    merged['power'] = merged['power'].fillna(0)
    merged['guts'] = merged['guts'].fillna(0)
    merged['wiz'] = merged['wiz'].fillna(0)

    # 添加card_id和running_style进行独热编码
    merged['card_id'] = merged['card_id'].astype(str)  # 确保card_id是字符串类型
    merged['running_style'] = merged['running_style'].astype(int)  # 确保running_style是整数类型

    # 对card_id和running_style进行独热编码
    merged = pd.get_dummies(merged, columns=['card_id', 'running_style'], drop_first=True)

    # 提取目标变量
    target_cols = ['win_rate', 'paired_rate']
    X = merged[skill_features + ['speed', 'stamina', 'power', 'guts', 'wiz'] + [col for col in merged.columns if 'card_id_' in col or 'running_style_' in col]]
    y = merged[target_cols]

    return X, y, merged['uma_id']  # 返还uma_id用于对比

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

    # 返回模型对象以及预测结果
    return model, y_test, y_pred

# ----------------------
# 绘制实际 vs 预测图
# ----------------------
def plot_actual_vs_predicted(y_test, y_pred, target_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel(f'Actual {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title(f'{target_name} - Actual vs Predicted')
    plt.show()

# ----------------------
# 打印部分实际 vs 预测值
# ----------------------
def print_actual_vs_predicted(y_test, y_pred, uma_ids):
    print("\n--- 一些样本的实际与预测胜率对比 ---")
    # 只选择测试集中的 uma_ids
    test_uma_ids = uma_ids.iloc[y_test.index]  # 使用 y_test 的索引来筛选对应的 uma_id
    comparison = pd.DataFrame({
        'uma_id': test_uma_ids,
        'Actual Win Rate': y_test,
        'Predicted Win Rate': y_pred
    })
    print(comparison.head(10))  # 打印前10个样本对比

# ----------------------
# SHAP分析
# ----------------------
def plot_shap(X, model, skill_map):
    # 使用SHAP的TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 获取特征名称并替换成中文
    feature_names = X.columns
    feature_names_chinese = [skill_map.get(int(col.split('_')[1]), col) if 'skill_' in col else col for col in feature_names]

    # 绘制SHAP的特征重要性图，显示正负影响
    shap.summary_plot(shap_values, X, feature_names=feature_names_chinese)

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # 加载技能映射
    skill_map = load_skill_mapping(r'D:/loh_uaf_3200/text_data_curate.json')

    # 加载数据
    X, y, uma_ids = load_data(skill_map)

    # 训练并评估胜率模型
    print("\n=== 胜率分析 ===")
    win_model, y_test, y_pred = train_xgboost(X, y['win_rate'], "胜率")

    # 打印实际 vs 预测值
    print_actual_vs_predicted(y_test, y_pred, uma_ids)

    # 绘制实际 vs 预测图
    plot_actual_vs_predicted(y_test, y_pred, "胜率")

    # SHAP分析
    print("\n=== SHAP分析 ===")
    plot_shap(X, win_model, skill_map)

    # 训练并评估前二率模型
    print("\n=== 前二率分析 ===")
    paired_model, y_test, y_pred = train_xgboost(X, y['paired_rate'], "前二率")

    # 打印实际 vs 预测值
    print_actual_vs_predicted(y_test, y_pred, uma_ids)

    # 绘制实际 vs 预测图
    plot_actual_vs_predicted(y_test, y_pred, "前二率")

    # SHAP分析
    print("\n=== SHAP分析 ===")
    plot_shap(X, paired_model, skill_map)
