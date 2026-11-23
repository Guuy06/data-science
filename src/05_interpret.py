# -*- coding: utf-8 -*-
"""
阶段 05：SHAP 可解释性与业务洞察
================================
使用 XGBoost 内置的精确 TreeSHAP（pred_contribs=True）计算各样本的
Shapley 值，与 shap 库结果等价但不受版本兼容性影响。

输出：fig14 SHAP 蜂群图（全局特征贡献）
      fig15 SHAP 依赖图（小时 / 温度 / 城市 / 通勤交互）
      fig16 分城市 SHAP 对比（解释"哪些特征可迁移、哪些不可迁移"）
      tbl_shap_importance.csv
运行：python src/05_interpret.py
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (FEATS_FULL, MOD, PALETTE, set_plot_style,
                    savefig, savetable)
from utils import get_xy, load_data, predict

warnings.filterwarnings("ignore")
plt = set_plot_style()
df, split = load_data()
te = df.loc[split["test_c0"] + split["test_c1"]].reset_index(drop=True)
Xte, yte = get_xy(te)

booster = xgb.Booster()
booster.load_model(str(MOD / "xgb_global.json"))
print(f"载入全局模型，测试集 {Xte.shape}")

# 精确 TreeSHAP：最后一列为 base_score
contrib = booster.predict(xgb.DMatrix(Xte), pred_contribs=True)
shap_v = contrib[:, :-1]
base = contrib[:, -1]
print(f"SHAP 矩阵 {shap_v.shape}，base_score 均值 = {base.mean():.4f}")
print(f"可加性校验：max|ΣSHAP + base - ŷ| = "
      f"{np.abs(shap_v.sum(1) + base - predict(booster, Xte)).max():.2e}")

shap_df = pd.DataFrame(shap_v, columns=FEATS_FULL)
city = te["city"].values

# ----------------------------------------------------------------------
print("\n=== 全局 SHAP 重要性（mean|SHAP|）===")
imp = (shap_df.abs().mean().sort_values(ascending=False)
       .rename("mean_abs_shap").to_frame())
imp["占比%"] = (imp["mean_abs_shap"] / imp["mean_abs_shap"].sum() * 100).round(2)
imp["方向"] = [("正" if shap_df[c].corr(te[c]) > 0 else "负") for c in imp.index]
imp = imp.round(4)
print(imp.to_string())
imp.reset_index().rename(columns={"index": "特征"}).pipe(
    lambda d: savetable(d, "tbl_shap_importance"))

# ----------------------------------------------------------------------
print("\n=== 出图 ===")
C0, C1 = PALETTE["src"], PALETTE["tgt"]
TOPN = 12
top = imp.index[:TOPN].tolist()


def beeswarm(ax, feats):
    """手工实现 SHAP 蜂群图：按特征值排序后在每个分箱内均匀散布。"""
    n = len(feats)
    for i, f in enumerate(feats):
        y = n - 1 - i
        sv = shap_df[f].values
        fv = Xte[f].values.astype(float)
        rng = fv.max() - fv.min()
        norm = (fv - fv.min()) / rng if rng > 1e-9 else np.zeros_like(fv)
        order = np.argsort(fv)
        nb = 60
        bins = np.array_split(order, nb)
        for b in bins:
            if len(b) == 0:
                continue
            off = np.linspace(-0.36, 0.36, len(b)) if len(b) > 1 else [0.0]
            ax.scatter(sv[b], y + np.array(off), c=norm[b], cmap="coolwarm",
                       s=7, alpha=0.75, edgecolors="none", vmin=0, vmax=1,
                       rasterized=True)
    ax.axvline(0, color="#666", lw=1, ls="--")
    ax.set_yticks(range(n)[::-1])
    ax.set_yticklabels(feats)
    ax.set_ylim(-0.8, n - 0.2)


fig, ax = plt.subplots(figsize=(8.2, 6.0))
beeswarm(ax, top)
ax.set_xlabel("SHAP 值（对预测租借量的贡献，辆/小时）")
ax.set_title(f"全局模型 SHAP 蜂群图（前 {TOPN} 个特征）")
sm = plt.cm.ScalarMappable(cmap="coolwarm")
cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=34)
cb.set_label("特征取值（归一化，蓝=低 / 红=高）")
savefig(fig, "fig14_shap_beeswarm")
plt.close(fig)

# ----------------------------------------------------------------------
# 图15 SHAP 依赖图：连续特征用散点，离散特征用分组箱线图
fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))

# (a) 小时：小时类特征 SHAP 合计，按 24 个小时做箱线图
hour_feats = [f for f in ["hour", "hour_sin", "hour_cos", "hour_workday_cos"]
              if f in shap_df.columns]
shap_hour = shap_df[hour_feats].sum(1)
data = [shap_hour[te.hour == h].values for h in range(24)]
bp = axes[0, 0].boxplot(data, positions=range(24), widths=0.62,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#222", lw=1.4))
for i, p in enumerate(bp["boxes"]):
    p.set_facecolor(plt.cm.viridis(i / 23))
    p.set_alpha(0.85)
axes[0, 0].axhline(0, color="#666", lw=1, ls="--")
axes[0, 0].set_xlim(-0.8, 23.8)
axes[0, 0].set_xticks(list(range(0, 24, 3)))
axes[0, 0].set_xticklabels([str(h) for h in range(0, 24, 3)])
axes[0, 0].set_xlabel("小时")
axes[0, 0].set_ylabel("小时类特征 SHAP 合计（辆/小时）")
axes[0, 0].set_title("(a) 小时类特征的边际贡献（分 24 个小时）")

# (b) 温度：连续散点 + 分箱均值
sc = axes[0, 1].scatter(Xte["temp_mean"], shap_df["temp_mean"],
                        c=Xte["weather"], cmap="viridis", s=9, alpha=0.6,
                        edgecolors="none", rasterized=True)
q = pd.qcut(Xte["temp_mean"], 12, duplicates="drop")
tr = pd.DataFrame({"t": Xte["temp_mean"], "s": shap_df["temp_mean"]}) \
    .groupby(q, observed=True).mean()
axes[0, 1].plot(tr.t, tr.s, color=PALETTE["tgt"], lw=2.4, marker="o", ms=5,
                label="分箱均值趋势")
axes[0, 1].axhline(0, color="#666", lw=1, ls="--")
axes[0, 1].set_xlabel("平均温度 temp_mean（℃）")
axes[0, 1].set_ylabel("SHAP(temp_mean)")
axes[0, 1].set_title("(b) 温度的非线性边际效应")
axes[0, 1].legend(frameon=False, fontsize=9)
cb = fig.colorbar(sc, ax=axes[0, 1], pad=0.015, aspect=26)
cb.set_label("天气等级", fontsize=8.5)

# (c) 城市：分组箱线图
bp = axes[1, 0].boxplot([shap_df.loc[city == 0, "city"],
                         shap_df.loc[city == 1, "city"]],
                        labels=["城市 0（源域）", "城市 1（目标域）"],
                        patch_artist=True, widths=0.45, showfliers=False,
                        medianprops=dict(color="#222", lw=1.6))
for p, col in zip(bp["boxes"], [C0, C1]):
    p.set_facecolor(col)
    p.set_alpha(0.85)
axes[1, 0].axhline(0, color="#666", lw=1, ls="--")
axes[1, 0].set_ylabel("SHAP(city)（辆/小时）")
axes[1, 0].set_title("(c) 城市标识的定向贡献：即『水平偏移』的量化")

# (d) 通勤交互：4 组箱线图
groups, labels = [], []
for wd, wdn in [(0, "非工作日"), (1, "工作日")]:
    for r, rn in [(0, "平峰"), (1, "高峰")]:
        m = (te.is_workday == wd) & (te.is_rush == r)
        groups.append(shap_df.loc[m.values, "rush_x_workday"])
        labels.append(f"{wdn}\n{rn}")
bp = axes[1, 1].boxplot(groups, labels=labels, patch_artist=True, widths=0.5,
                        showfliers=False,
                        medianprops=dict(color="#222", lw=1.6))
for p, col in zip(bp["boxes"], ["#A8C4E0", "#7FA8D0", "#E8A0A0", "#D97B7B"]):
    p.set_facecolor(col)
    p.set_alpha(0.9)
axes[1, 1].axhline(0, color="#666", lw=1, ls="--")
axes[1, 1].set_ylabel("SHAP(rush_x_workday)")
axes[1, 1].set_title("(d) 通勤高峰×工作日交互的贡献")

fig.suptitle("SHAP 依赖图：关键特征的边际效应与交互",
             y=0.995, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
savefig(fig, "fig15_shap_dependence")
plt.close(fig)

# ----------------------------------------------------------------------
# 图16 分城市 SHAP 对比
by_city = shap_df.groupby(city).mean()
by_city_abs = shap_df.abs().groupby(city).mean()
order = by_city_abs.mean().sort_values(ascending=False).index[:10]
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

x = np.arange(len(order))
w = 0.38
axes[0].barh(x - w / 2, by_city_abs.loc[0, order], w, color=C0,
             label="城市 0（源域）")
axes[0].barh(x + w / 2, by_city_abs.loc[1, order], w, color=C1,
             label="城市 1（目标域）")
axes[0].set_yticks(x)
axes[0].set_yticklabels(order)
axes[0].invert_yaxis()
axes[0].set_xlabel("mean|SHAP|（辆/小时）")
axes[0].set_title("(a) 特征贡献强度的城市差异")
axes[0].legend(frameon=False, fontsize=9)

delta = (by_city.loc[1, order] - by_city.loc[0, order])
cols = [C1 if v > 0 else C0 for v in delta]
axes[1].barh(x, delta.values, color=cols, alpha=0.9)
axes[1].axvline(0, color="#333", lw=1)
axes[1].set_yticks(x)
axes[1].set_yticklabels(order)
axes[1].invert_yaxis()
axes[1].set_xlabel("城市 1 平均 SHAP − 城市 0 平均 SHAP")
axes[1].set_title("(b) 定向贡献差异（红=目标域更受益，蓝=更受损）")
for i, v in enumerate(delta.values):
    axes[1].text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=8)
_lo = min(delta.min() * 1.30, -1)
_hi = max(delta.max() * 1.30, 1)
axes[1].set_xlim(_lo, _hi)
fig.suptitle("分城市 SHAP 贡献对比：水平偏移集中在哪些特征上",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig16_shap_by_city")
plt.close(fig)

savetable(by_city_abs.loc[:, order].T.reset_index()
          .rename(columns={"index": "特征", 0: "city0_mean_abs_shap",
                           1: "city1_mean_abs_shap"}).round(4),
          "tbl_shap_by_city")

# ----------------------------------------------------------------------
print("\n=== 城市间 SHAP 贡献差异（定向）===")
print(delta.round(3).to_string())
print("\n=== 关键结论支撑数据 ===")
print(f"hour 类特征合计贡献占比："
      f"{imp.loc[[i for i in imp.index if 'hour' in i], '占比%'].sum():.2f}%")
print(f"温度类特征合计贡献占比："
      f"{imp.loc[[i for i in imp.index if 'temp' in i], '占比%'].sum():.2f}%")
print(f"city 自身贡献占比：{imp.loc['city', '占比%']:.2f}%")
print(f"城市 0 平均预测 {predict(booster, Xte[city==0]).mean():.2f}，"
      f"城市 1 平均预测 {predict(booster, Xte[city==1]).mean():.2f}")

# 峰值时段贡献拆解：解释 17-18 点的系统性偏差
te2 = te.copy()
te2["shap_hour"] = shap_df["hour_cos"] + shap_df["hour_sin"] + shap_df["hour"]
peak = te2.hour.isin([17, 18])
print(f"\n17-18 点样本 {int(peak.sum())} 条，"
      f"小时类特征平均 SHAP：城市0="
      f"{te2[peak & (te2.city==0)].shap_hour.mean():.2f}，"
      f"城市1={te2[peak & (te2.city==1)].shap_hour.mean():.2f}")

# ----------------------------------------------------------------------
# 表 7-2 分阶段部署策略：把 7.3 节的口头建议固化成可核对的矩阵
advice = pd.DataFrame([
    ["冷启动期", "50~500 条（≤3 周）",
     "T7 融合：源域预训练 + 乘性校正 + 样本加权",
     "以目标域样本估计 k；k 周度漂移 >5% 即提前转独立建模"],
    ["成长期", "500~1000 条（3 周~3 个月）",
     "T3 乘性校正为主，逐步下调源域权重",
     "源域权重由 0.15 上调至 0.5；每周重估一次 k"],
    ["成熟期", ">1000 条",
     "目标域独立建模或全局多城市模型",
     "迁移增益已降至 5.55% 以下，继续用源域易引入偏差"],
    ["全周期", "—", "城市 × 时段平均残差监控",
     "高峰残差绝对值持续 >10 辆/小时即触发重校准"],
], columns=["运营阶段", "目标域样本量", "推荐方案", "关键动作与失效信号"])
savetable(advice, "tbl_deploy_advice")

print("\n阶段 05 完成。")
