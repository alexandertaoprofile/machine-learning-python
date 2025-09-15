import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap

# ----------------------
# 配置中文字体
# ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------
# 加载名称映射
# ----------------------
def load_name_mappings(json_path):
    """加载并展平多层嵌套的JSON映射文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_mapping = json.load(f)

    name_map = {}
    for category in raw_mapping.values():
        if isinstance(category, dict):
            for key, value in category.items():
                if key.isdigit():
                    # 移除换行符并清理特殊字符
                    cleaned_value = value.replace('\\n', ' ').replace('(', '_').replace(')', '_')
                    name_map[int(key)] = cleaned_value
    return name_map


# ----------------------
# 数据预处理
# ----------------------
def preprocess_data(team_path, support_path, skills_path, name_map):
    # 加载原始数据
    team = pd.read_csv(team_path)
    support = pd.read_csv(support_path)
    skills = pd.read_csv(skills_path)

    # ====== 处理支援卡数据 ======
    support_cols = [col for col in support.columns if col.startswith('support_card_id_')]

    # 转换长格式并映射中文名称
    support_long = (
        support.melt(
            id_vars='uma_id',
            value_vars=support_cols,
            value_name='support_id'
        )
        .dropna(subset=['support_id'])
        .assign(
            support_id=lambda x: x['support_id'].astype(int),
            support_name=lambda x: x['support_id'].map(
                lambda i: f"支援_{name_map.get(i, '未知')}_{i}"
            )
        )
        .drop(columns=['variable'])
    )

    # 创建二进制特征并分组
    support_encoded = pd.get_dummies(
        support_long,
        columns=['support_name'],
        prefix='',
        prefix_sep=''
    ).groupby('uma_id').max()

    # ====== 处理技能数据 ======
    skill_cols = [col for col in skills.columns if col.startswith('skill_id_')]

    skills_long = (
        skills.melt(
            id_vars='uma_id',
            value_vars=skill_cols,
            value_name='skill_id'
        )
        .dropna(subset=['skill_id'])
        .assign(
            skill_id=lambda x: x['skill_id'].astype(int),
            skill_name=lambda x: x['skill_id'].map(
                lambda i: f"技能_{name_map.get(i, '未知')}_{i}"
            )
        )
        .drop(columns=['variable'])
    )

    skills_encoded = pd.get_dummies(
        skills_long,
        columns=['skill_name'],
        prefix='',
        prefix_sep=''
    ).groupby('uma_id').max()

    # ====== 合并数据集 ======
    merged = (
        pd.merge(team, support_encoded, on="uma_id", how='left')
        .merge(skills_encoded, on="uma_id", how='left')
        .fillna(0)
    )

    # ====== 数据清洗 ======
    # 删除原始ID和无关列
    cols_to_drop = [
        'uma_id', 'team_member_id', 'trained_chara_id', 'scenario_id',
        'support_borrow', 'multi_win_rate', 'final_grade', 'rank',
        'rank_count_1', 'rank_count_2', 'rank_count_3', 'rank_count_unplaced','card_id','skill_id','support_id',
    ]
    merged = merged.drop(columns=[col for col in cols_to_drop if col in merged.columns])

    # 处理分类特征
    categorical_cols = ['running_style', 'proper_ground_turf', 'proper_distance_middle']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(merged[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=merged.index
    )

    # 最终合并
    merged = pd.concat([merged.drop(columns=categorical_cols), encoded_df], axis=1)

    # ====== 特征名称清洗 ======
    def clean_feature_names(df):
        """清洗特征名称中的非法字符"""
        df.columns = [
            col.translate(str.maketrans('', '', '[]<>()'))
            .replace(' ', '_')
            .replace('__', '_')
            for col in df.columns
        ]
        return df

    return clean_feature_names(merged)


# ----------------------
# 模型训练与评估
# ----------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, name):
    # 模型配置
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 评估指标
    y_pred = model.predict(X_test)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    }

    print(f"\n=== {name}模型评估 ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return model, metrics


# ----------------------
# 特征分析
# ----------------------
def analyze_features(model, feature_names, title):
    """生成特征重要性分析"""
    # 获取重要性排名
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    # 取前15个特征
    top_features = [(feature_names[i], importance[i]) for i in sorted_idx[:15]]

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.barh(
        [name for name, _ in reversed(top_features)],
        [score for _, score in reversed(top_features)]
    )
    plt.xlabel('特征重要性')
    plt.title(f"{title} - Top15特征")
    plt.tight_layout()
    plt.show()

    return top_features


# ----------------------
# SHAP分析
# ----------------------
def shap_analysis(model, X, feature_names, title):
    """生成SHAP特征影响图"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f"{title} SHAP分析")
    plt.tight_layout()
    plt.show()


# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # ====== 配置路径 ======
    JSON_PATH = "D:/揉大の烟雾弹/2100/text_data_curate.json"
    TEAM_PATH = "D:/揉大の烟雾弹/2100/Team_Member.csv"
    SUPPORT_PATH = "D:/揉大の烟雾弹/2100/Support_Cards.csv"
    SKILLS_PATH = "D:/揉大の烟雾弹/2100/Skills.csv"

    # ====== 数据预处理 ======
    print("正在加载名称映射...")
    name_map = load_name_mappings(JSON_PATH)

    print("\n正在预处理数据...")
    merged = preprocess_data(TEAM_PATH, SUPPORT_PATH, SKILLS_PATH, name_map)

    # ====== 准备数据集 ======
    targets = ['win_rate', 'paired_rate']
    feature_names = merged.columns.drop(targets).tolist()

    X = merged.drop(columns=targets)
    y_win = merged['win_rate']
    y_paired = merged['paired_rate']

    # 划分数据集
    X_train, X_test, y_win_train, y_win_test = train_test_split(
        X, y_win, test_size=0.2, random_state=42)
    _, _, y_paired_train, y_paired_test = train_test_split(
        X, y_paired, test_size=0.2, random_state=42)

    # ====== 训练模型 ======
    print("\n正在训练胜率模型...")
    win_model, win_metrics = train_and_evaluate(X_train, X_test, y_win_train, y_win_test, "胜率")

    print("\n正在训练前二率模型...")
    paired_model, paired_metrics = train_and_evaluate(X_train, X_test, y_paired_train, y_paired_test, "前二率")

    # ====== 特征分析 ======
    print("\n正在进行特征重要性分析...")
    win_top = analyze_features(win_model, feature_names, "胜率模型")
    paired_top = analyze_features(paired_model, feature_names, "前二率模型")

    # ====== SHAP分析 ======
    print("\n正在进行SHAP分析...")
    shap_analysis(win_model, X_test, feature_names, "胜率")
    shap_analysis(paired_model, X_test, feature_names, "前二率")

    # ====== 输出结果 ======
    print("\n=== 最终结果汇总 ===")
    print("胜率模型Top5特征：")
    for name, score in win_top[:5]:
        print(f"{name:<40} {score:.4f}")

    print("\n前二率模型Top5特征：")
    for name, score in paired_top[:5]:
        print(f"{name:<40} {score:.4f}")
