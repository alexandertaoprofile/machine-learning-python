import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression  # 用于p值计算
import statsmodels.api as sm
from xgboost import XGBRegressor


# ----------------------
# 数据加载与预处理（关键修正）
# ----------------------
def load_data():
    # 加载原始数据
    team = pd.read_csv("D:/揉大の烟雾弹/2100/Team_Member.csv")
    support = pd.read_csv("D:/揉大の烟雾弹/2100/Support_Cards.csv")
    skills = pd.read_csv("D:/揉大の烟雾弹/2100/Skills.csv")

    # ====== 关键修正1：正确重构技能数据 ======
    support_cols = [col for col in support.columns if col.startswith('support_card_id_')]

    support_long = (
        support.melt(
            id_vars='uma_id',
            value_vars=support_cols,
            value_name='support_card_id'
        )
        .dropna(subset=['support_card_id'])
        .drop('variable', axis=1)
        .assign(support_card_id=lambda x: x['support_card_id'].astype(int))  # 保持为整数类型
    )
        # 创建有效的二进制特征
    support_encoded = pd.get_dummies(
        support_long,
        columns=['support_card_id'],
        prefix='support',
        prefix_sep='_'
    ).groupby('uma_id').max()
    # 处理技能数据（长格式转换）

    skill_cols = [col for col in skills.columns if col.startswith('skill_id_')]

    skills_long = (
        skills.melt(
            id_vars='uma_id',
            value_vars=skill_cols,
            value_name='skill_id'
        )
        .dropna(subset=['skill_id'])
        .drop('variable', axis=1)
        .assign(skill_id=lambda x: x['skill_id'].astype(int))  # 保持为整数类型
    )

    # 创建有效的二进制特征（确保列名为字符串）
    skills_encoded = pd.get_dummies(
        skills_long,
        columns=['skill_id'],
        prefix='skill',
        prefix_sep='_'
    ).groupby('uma_id').max()

    # 验证列名
    print("技能列名示例：", [col for col in skills_encoded.columns if '202721' in col][0])
    # 输出：skill_202721（正确）

    # ====== 关键修正2：安全合并数据 ======
    # 合并数据集（使用outer join避免数据丢失）
    merged = (
        pd.merge(team, support, on="uma_id", how='outer')
        .merge(skills_encoded, on="uma_id", how='outer')
    )

    # 填充缺失值为0（针对技能和支持卡特征）
    skill_support_cols = [col for col in merged.columns if 'skill_' in col or 'support_' in col]
    merged[skill_support_cols] = merged[skill_support_cols].fillna(0).astype(int)

    print("\n=== 支援卡特征验证 ===")
    support_features = [col for col in merged.columns if 'support_' in col]
    print("支援卡特征数量:", len(support_features))
    print("示例特征:", support_features[:5])
    # 应输出：['support_30028', 'support_30097', 'support_30107', ...]
    # ====== 关键修正3：正确特征工程 ======
    # 删除无关特征（添加更多可能需要删除的列）
    cols_to_drop = [
        'uma_id', 'team_member_id', 'trained_chara_id', 'scenario_id',
        'support_borrow', 'rank_count_1', 'rank_count_2', 'rank_count_3',
        'rank_count_unplaced', 'rank_count_total', 'rank', 'final_grade','multi-win-rate'
    ]
    merged = merged.drop(columns=[col for col in cols_to_drop if col in merged.columns])

    if 'multi_win_rate' in merged.columns:
        merged = merged.drop(columns=['multi_win_rate'])

    # ====== 验证特征删除 ======
    print("\n删除后剩余特征数量:", len(merged.columns))
    print("multi_win_rate是否存在:", 'multi_win_rate' in merged.columns)

    # 处理分类特征（添加更多可能需要编码的列）
    categorical_cols = ['card_id', 'running_style', 'proper_ground_turf', 'proper_distance_middle']

    # 确保分类列存在
    valid_categorical_cols = [col for col in categorical_cols if col in merged.columns]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(merged[valid_categorical_cols])

    # 创建编码后的DataFrame
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(valid_categorical_cols),
        index=merged.index
    )

    # 合并编码结果（使用索引对齐）
    merged = pd.concat(
        [merged.drop(columns=valid_categorical_cols), encoded_df],
        axis=1
    )

    # ====== 关键修正4：确保数值类型 ======
    # 转换所有列为数值类型
    for col in merged.columns:
        if merged[col].dtype == object:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    # 最终清理
    merged = merged.dropna().reset_index(drop=True)

    # 划分数据集
    targets = ['win_rate', 'paired_rate']
    X = merged.drop(columns=targets)
    y_win = merged['win_rate']
    y_paired = merged['paired_rate']

    return X, y_win, y_paired


# ----------------------
# 模型训练（保持不变）
# ----------------------
def train_model(X_train, y_train):
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # 加载数据
    try:
        X, y_win, y_paired = load_data()
        print("数据加载成功，特征矩阵形状:", X.shape)

        # 划分数据集
        X_train, X_test, y_win_train, y_win_test = train_test_split(
            X, y_win, test_size=0.2, random_state=42)
        _, _, y_paired_train, y_paired_test = train_test_split(
            X, y_paired, test_size=0.2, random_state=42)

        # 训练模型
        win_model = train_model(X_train, y_win_train)
        paired_model = train_model(X_train, y_paired_train)

        # 后续评估代码...

    except Exception as e:
        print(f"发生错误：{str(e)}")

win_model = train_model(X_train, y_win_train)
paired_model = train_model(X_train, y_paired_train)


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)

}

    print(f"\n{name}评估结果：")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics

win_metrics = evaluate_model(win_model, X_test, y_win_test, "胜率")
paired_metrics = evaluate_model(paired_model, X_test, y_paired_test, "前二率")

# ----------------------
# 特征分类与重要性分析
# ----------------------
feature_categories = {
    '基础属性': ['speed', 'stamina', 'power', 'wiz', 'guts'],
    '跑法相关': [col for col in X.columns if 'running_style' in col],
    '场地适应性': [col for col in X.columns if 'proper_' in col],
    '马种特征': [col for col in X.columns if 'card_id' in col],
    '技能相关': ['total_skills'],
    '支援卡相关': ['total_support_cards']
}


def categorize_importance(model):
    importance = model.feature_importances_
    category_scores = {cat: 0 for cat in feature_categories}

    for feature, score in zip(X.columns, importance):
        for cat, cols in feature_categories.items():
            if feature in cols:
                category_scores[cat] += score
                break

    # 标准化为百分比
    total = sum(category_scores.values())
    return {k: v / total * 100 for k, v in category_scores.items()}


print("\n=== 特征类别重要性 ===")
print("胜率模型：")
for cat, score in categorize_importance(win_model).items():
    print(f"{cat:<15} {score:.1f}%")

print("\n前二率模型：")
for cat, score in categorize_importance(paired_model).items():
    print(f"{cat:<15} {score:.1f}%")


# ----------------------
# 统计显著性分析（p值）
# ----------------------
def calculate_pvalues(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.pvalues[1:]  # 排除截距项


print("\n=== 统计显著性分析 ===")
print("胜率模型p值（前10个特征）：")
win_pvalues = calculate_pvalues(X, y_win)
print(win_pvalues.sort_values()[:50])

print("\n前二率模型p值（前10个特征）：")
paired_pvalues = calculate_pvalues(X, y_paired)
print(paired_pvalues.sort_values()[:50])

# ----------------------
# SHAP解释性分析
# ----------------------
explainer = shap.Explainer(win_model)
shap_values = explainer(X)

def convert_feature_names(feature_names):
    converted = []
    for name in feature_names:
        if name.startswith('support_'):
            support_id = name.split('_')[1]
            # 可在此添加支援卡名称映射（如有）
            converted.append(f"支援卡_{support_id}")
        else:
            converted.append(name)
    return converted


print("\n=== SHAP特征重要性 ===")
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=convert_feature_names(X.columns),
    show=False
)
plt.gca().set_yticklabels(convert_feature_names(X.columns))
plt.show()