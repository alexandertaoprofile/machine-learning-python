import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import shap
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data(path):
    print("▶ 当前工作目录:", os.getcwd())
    print(f"▶ 尝试读取文件：{path}", end=" ... ")
    if not os.path.exists(path):
        print("❌ 文件不存在！")
        raise FileNotFoundError(path)
    print("✅")
    df = pd.read_csv(path)
    print("▶ 原始数据形状:", df.shape)
    features = ['speed', 'stamina', 'power', 'guts', 'wiz']
    targets = ['win_rate', 'paired_rate']
    df = df[features + targets].dropna()
    print("▶ 清洗后形状:", df.shape)
    # 记录原始均值与标准差
    stats = {f: (df[f].mean(), df[f].std()) for f in features}
    df_std = df.copy()
    df_std[features] = (df_std[features] - df_std[features].mean()) / df_std[features].std()
    print("▶ 标准化完成")
    return df, df_std, stats


def plot_feature_importance_and_pdp(model, X, features, stats):
    """
    显示特征重要性和偏依赖图（PDP）
    """
    # === 1. GAM 分析（偏依赖图）===
    print("\n=== GAM 偏依赖分析 ===")
    gam = model
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        ax = axes.flat[i]
        # 生成标准化网格并反归一化到实际值
        XX = gam.generate_X_grid(term=i)
        std_vals = XX[:, term.feature]
        mean, std = stats[features[term.feature]]
        real_vals = std_vals * std + mean
        pdep, conf = gam.partial_dependence(term=i, X=XX, width=0.95)
        ax.plot(real_vals, pdep)
        ax.fill_between(real_vals, conf[:, 0], conf[:, 1], alpha=0.3)
        ax.set_title(f"{features[term.feature]} 偏依赖图")
        ax.set_xlabel(f"{features[term.feature]} 实际数值")
        ax.set_ylabel("对 win_rate 的影响")
    plt.tight_layout()
    plt.show()

    return None

def run_models_and_visualize(X_std, y, features):
    # 使用随机森林进行特征重要性分析
    print("\n=== 随机森林 + SHAP 特征重要性分析 ===")
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_std, y)

    # 使用SHAP解释模型
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_std)

    # 绘制 SHAP 特征重要性摘要图
    shap.summary_plot(shap_values, X_std, feature_names=features, show=True)

    # 绘制 SHAP 依赖图
    top_idx = np.argmax(np.abs(shap_values).mean(0))
    print(f"最重要特征：{features[top_idx]}")
    shap.dependence_plot(top_idx, shap_values, X_std, feature_names=features, show=True)

    return rf

def main():
    # 这里加载你的数据（假设数据已加载到 df 中）
    csv_path = r"D:\loh_uaf_3200\Team_Member.csv"
    df, df_std, stats = load_data(csv_path)
    features = ['speed', 'stamina', 'power', 'guts', 'wiz']
    X_std = df_std[features]
    y = df['win_rate']

    # === 1. 使用 GAM 进行偏依赖分析 ===
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4)).fit(X_std, y)
    print(gam.summary())
    plot_feature_importance_and_pdp(gam, X_std, features, stats)

    # === 2. 使用随机森林和 SHAP 进行特征重要性分析 ===
    rf = run_models_and_visualize(X_std, y, features)

    # === 3. 线性回归和二次多项式回归分析 ===
    print("\n=== 线性回归模型 ===")
    X_lin = sm.add_constant(df[features])
    ols = sm.OLS(df['win_rate'], X_lin).fit()
    print(ols.summary())

    print("\n=== 二次多项式回归模型 ===")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df[features])
    Xp = sm.add_constant(X_poly)
    ols2 = sm.OLS(df['win_rate'], Xp).fit()
    print(ols2.summary())

    # === 4. 模型验证（交叉验证）===
    print("\n=== 交叉验证分析 ===")
    scores = cross_val_score(rf, X_std, y, cv=5, scoring='neg_mean_squared_error')
    print(f"交叉验证 MSE: {scores.mean()} ± {scores.std()}")

if __name__ == "__main__":
    main()

