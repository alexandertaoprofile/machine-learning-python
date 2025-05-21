import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression  # 用于p值计算
import statsmodels.api as sm
from xgboost import XGBRegressor
#import shap

# ----------------------
# 数据预处理
# ----------------------
# 加载数据
team = pd.read_csv("D:/揉大の烟雾弹/2100/Team_Member.csv")
support = pd.read_csv("D:/揉大の烟雾弹/2100/Support_Cards.csv")
skills = pd.read_csv("D:/揉大の烟雾弹/2100/Skills.csv")

# 重构技能数据
skill_cols = [col for col in skills.columns if col.startswith('skill_id_')]

# 转换长格式时修复技能ID类型
skills_long = (
    skills.melt(
        id_vars='uma_id',
        value_vars=skill_cols,
        value_name='skill_id'
    )
    .dropna(subset=['skill_id'])
    .drop('variable', axis=1)
    .assign(skill_id=lambda x: x['skill_id'].astype(int).astype(str))
)  # 此处正确闭合括号

# 创建二进制特征
skills_encoded = pd.get_dummies(
    skills_long,
    columns=['skill_id'],
    prefix='skill',
    prefix_sep='_'
).groupby('uma_id').max()

# 合并数据集
merged = pd.merge(pd.merge(team, support, on="uma_id"), skills_encoded, on="uma_id")

print(merged)
# 删除无关特征
cols_to_drop = [
    'uma_id', 'team_member_id', 'trained_chara_id', 'scenario_id',
    'support_borrow', 'rank_count_1', 'rank_count_2', 'rank_count_3',
    'rank_count_unplaced', 'rank_count_total', 'rank', 'final_grade'
]
merged = merged.drop(columns=cols_to_drop)

# 处理分类特征
categorical_cols = ['card_id', 'running_style', 'proper_ground_turf', 'proper_distance_middle']
merged[categorical_cols] = merged[categorical_cols].astype('category')

# 独热编码
# ----------------------
# 修正后的编码处理部分
# ----------------------
# 创建编码器
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 执行编码
encoded_features = encoder.fit_transform(merged[categorical_cols])

# 正确创建encoded_df（注意括号闭合）
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder.get_feature_names_out(categorical_cols)  # <- 这里闭合括号
)  # <- 这里再次闭合

# 合并数据（确保在相同代码块中）
merged = pd.concat(
    [merged.drop(columns=categorical_cols), encoded_df],
    axis=1
)



# 构造衍生特征
skill_cols = [col for col in merged.columns if col.startswith('skill_id_')]
support_cols = [col for col in merged.columns if col.startswith('support_card_id_')]

merged['total_skills'] = merged[skill_cols].notna().sum(axis=1)
merged['total_support_cards'] = merged[support_cols].notna().sum(axis=1)
merged = merged.drop(columns=skill_cols + support_cols)

# 划分数据集
targets = ['win_rate', 'paired_rate']
X = merged.drop(columns=targets)
y_win = merged['win_rate']
y_paired = merged['paired_rate']

X_train, X_test, y_win_train, y_win_test = train_test_split(X, y_win, test_size=0.2, random_state=42)
_, _, y_paired_train, y_paired_test = train_test_split(X, y_paired, test_size=0.2, random_state=42)

# ----------------------
# 模型训练与评估
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
print(win_pvalues.sort_values()[:10])

print("\n前二率模型p值（前10个特征）：")
paired_pvalues = calculate_pvalues(X, y_paired)
print(paired_pvalues.sort_values()[:10])

# ----------------------
# SHAP解释性分析
# ----------------------
explainer = shap.Explainer(win_model)
shap_values = explainer(X)

print("\n=== SHAP特征重要性 ===")
shap.summary_plot(shap_values, X)

