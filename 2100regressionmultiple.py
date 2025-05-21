import pandas as pd
import numpy as np
np.int = int
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
# ---- Monkey Patch：让 csr_matrix.A 等价于 toarray() ----
# sp.csr_matrix.A = property(lambda self: self.toarray())
from pygam import LinearGAM, s

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
import shap

# ----------------------
# 配置中文字体
# ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载并预处理数据"""
    df = pd.read_csv("Team_Member.csv")
    features = ['speed', 'stamina', 'power', 'guts', 'wiz']
    targets = ['win_rate', 'paired_rate']

    # 数据清洗
    df = df[features + targets].dropna()

    # 标准化处理（保留原始值用于解释）
    df_std = df.copy()
    df_std[features] = (df_std[features] - df_std[features].mean()) / df_std[features].std()

    return df, df_std


def nonlinear_analysis(df, df_std):
    """非线性多元分析主流程"""

    # ====== 广义加性模型分析 ======
    def gam_analysis(target):
        """GAM模型拟合与可视化"""
        X = df_std[['speed', 'stamina', 'power', 'guts', 'wiz']]
        y = df_std[target]

        # 构建GAM模型（每个特征使用样条函数）
        gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4)).fit(X, y)

        # 输出模型摘要
        print(f"\n=== {target} GAM模型摘要 ===")
        print(gam.summary())

        # 绘制偏依赖图
        plt.figure(figsize=(15, 10))
        titles = ['速度', '耐力', '力量', '毅力', '智慧']
        for i, term in enumerate(gam.terms):
            if term.isintercept:
                continue
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

            plt.subplot(2, 3, i + 1)
            plt.plot(XX[:, term.feature], pdep, color='darkred')
            plt.fill_between(XX[:, term.feature], confi[:, 0], confi[:, 1], alpha=0.2)
            plt.title(f"{titles[i]} 偏效应")
            plt.xlabel("标准化值")
            plt.ylabel("对" + target + "的影响")
        plt.tight_layout()
        plt.show()
        return gam

    print("\n正在进行GAM分析...")
    gam_win = gam_analysis('win_rate')
    gam_paired = gam_analysis('paired_rate')

    # ====== 随机森林+SHAP分析 ======
    def shap_analysis(target):
        """SHAP非线性影响分析"""
        X_train, X_test, y_train, y_test = train_test_split(
            df_std[['speed', 'stamina', 'power', 'guts', 'wiz']],
            df_std[target],
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # SHAP值分析
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 特征级摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=['速度', '耐力', '力量', '毅力', '智慧'], show=False)
        plt.title(f"{target} SHAP特征影响")
        plt.tight_layout()
        plt.show()

        # 特征交互分析
        interaction_idx = np.argmax(np.abs(shap_values).mean(0))
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(
            interaction_idx, shap_values, X_test,
            feature_names=['速度', '耐力', '力量', '毅力', '智慧'],
            interaction_index='auto'
        )
        plt.title(f"{target} 主要交互效应")
        plt.show()

        return model

    print("\n正在进行SHAP分析...")
    rf_win = shap_analysis('win_rate')
    rf_paired = shap_analysis('paired_rate')

    # ====== 实际影响量化 ======
    def calculate_effect(feature, target='win_rate', unit=1):
        """计算原始尺度下属性提升的实际影响"""
        # 基于GAM模型预测
        base = df[feature].mean()
        x_values = np.linspace(base, base + unit, 100)
        preds = []

        for val in x_values:
            temp_df = df_std.copy()
            temp_df[feature] = (val - df[feature].mean()) / df[feature].std()
            pred = gam_win.predict(temp_df) if target == 'win_rate' else gam_paired.predict(temp_df)
            preds.append(pred.mean())

        return np.mean(np.diff(preds))

    print("\n=== 属性实际提升效果（非线性估计）===")
    print("（显示每提升1个原始单位属性的平均预期变化）")
    features_ch = {'speed': '速度', 'stamina': '耐力', 'power': '力量', 'guts': '毅力', 'wiz': '智慧'}

    print("\n胜率提升：")
    for feat in features_ch:
        effect = calculate_effect(feat, 'win_rate')
        print(f"{features_ch[feat]:<4}: {effect:.4f}%")

    print("\n前二率提升：")
    for feat in features_ch:
        effect = calculate_effect(feat, 'paired_rate')
        print(f"{features_ch[feat]:<4}: {effect:.4f}%")


if __name__ == "__main__":
    df_raw, df_std = load_data()
    nonlinear_analysis(df_raw, df_std)