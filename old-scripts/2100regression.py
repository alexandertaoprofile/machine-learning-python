import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy import stats

# ----------------------
# 配置中文字体
# ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------
# 数据加载与预处理
# ----------------------
def load_data():
    # 加载数据（请替换为实际路径）
    df = pd.read_csv("D:/揉大の烟雾弹/2100/Team_Member.csv")
    df = df[df['card_id'] == 100602].copy()
    df = df[df['running_style'] == 2].copy()
    # 选择目标特征
    target_features = ['speed', 'stamina', 'power', 'guts', 'wiz']
    targets = ['win_rate', 'paired_rate']

    # 数据清洗
    df = df[target_features + targets].dropna()

    # 标准化处理（便于比较系数）
    df[target_features] = (df[target_features] - df[target_features].mean()) / df[target_features].std()

    return df


# ----------------------
# 分析主程序
# ----------------------
def analyze_attributes(df):
    # ====== 数据探索 ======
    print("数据概览：")
    print(df.describe().T)

    # ====== 相关性分析 ======
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("属性相关性矩阵")
    plt.show()

    # ====== 线性关系验证 ======
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    for i, feat in enumerate(['speed', 'stamina', 'power', 'guts', 'wiz']):
        # 胜率关系
        sns.regplot(
            x=df[feat], y=df['win_rate'],
            ax=axes[i][0], line_kws={'color': 'red'},
            scatter_kws={'alpha': 0.3}
        )
        axes[i][0].set_title(f"{feat} vs 胜率")

        # 前二率关系
        sns.regplot(
            x=df[feat], y=df['paired_rate'],
            ax=axes[i][1], line_kws={'color': 'blue'},
            scatter_kws={'alpha': 0.3}
        )
        axes[i][1].set_title(f"{feat} vs 前二率")
    plt.tight_layout()
    plt.show()

    # ====== 多元线性回归 ======
    def run_regression(target):
        X = df[['speed', 'stamina', 'power', 'guts', 'wiz']]
        X = sm.add_constant(X)
        y = df[target]

        model = sm.OLS(y, X).fit()
        print(f"\n=== {target}回归分析 ===")
        print(model.summary())

        # 绘制系数图
        coef_df = pd.DataFrame({
            '特征': model.params.index[1:],
            '系数': model.params[1:],
            'CI下限': model.conf_int().iloc[1:, 0],
            'CI上限': model.conf_int().iloc[1:, 1]
        })

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            coef_df['系数'], coef_df['特征'],
            xerr=[coef_df['系数'] - coef_df['CI下限'], coef_df['CI上限'] - coef_df['系数']],
            fmt='o', color='darkorange'
        )
        plt.axvline(0, linestyle='--', color='grey')
        plt.title(f"{target}属性影响系数 (95%置信区间)")
        plt.xlabel("回归系数")
        plt.grid(True)
        plt.show()

        return model

    win_model = run_regression('win_rate')
    paired_model = run_regression('paired_rate')

    # ====== 非线性分析 ======
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[['speed', 'stamina', 'power', 'guts', 'wiz']]

    fig, ax = plt.subplots(5, 2, figsize=(20, 25))
    for i, feat in enumerate(X.columns):
        # 胜率PDP分析
        PartialDependenceDisplay.from_estimator(
            rf.fit(X, df['win_rate']),
            X, [feat], ax=ax[i][0], line_kw={'color': 'red'}
        )
        ax[i][0].set_title(f"胜率部分依赖图 - {feat}")

        # 前二率PDP分析
        PartialDependenceDisplay.from_estimator(
            rf.fit(X, df['paired_rate']),
            X, [feat], ax=ax[i][1], line_kw={'color': 'blue'}
        )
        ax[i][1].set_title(f"前二率部分依赖图 - {feat}")
    plt.tight_layout()
    plt.show()

    # ====== 实际提升计算 ======
    def calculate_effect(model, feature, unit=1):
        """计算指定属性每提升1单位的标准差带来的实际胜率提升"""
        coef = model.params[feature]
        std = df[feature].std()
        return (coef / std) * unit

    print("\n=== 属性实际提升效果 ===")
    print("（基于标准化后的回归系数，显示每提升1个原始单位的预期变化）")

    print("\n胜率提升：")
    for feat in X.columns:
        effect = calculate_effect(win_model, feat)
        print(f"{feat:<8}: {effect:.4f}%")

    print("\n前二率提升：")
    for feat in X.columns:
        effect = calculate_effect(paired_model, feat)
        print(f"{feat:<8}: {effect:.4f}%")

    return {
        'win_model': win_model,
        'paired_model': paired_model,
        'effect_sizes': {
            'win_rate': {feat: calculate_effect(win_model, feat) for feat in X.columns},
            'paired_rate': {feat: calculate_effect(paired_model, feat) for feat in X.columns}
        }
    }


# ----------------------
# 执行主程序
# ----------------------
if __name__ == "__main__":
    df = load_data()
    results = analyze_attributes(df)
