# -*- coding: utf-8 -*-
"""
阶段 02：探索性分析（EDA）
==========================
输出：outputs/figures/fig01~fig05
      outputs/tables/tbl_describe.csv        分城市描述性统计
      outputs/tables/tbl_city_diff_test.csv  城市间分布差异检验
运行：python src/02_eda.py
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import CACHE, TARGET, PALETTE, set_plot_style, savefig, savetable

plt = set_plot_style()
C0, C1 = PALETTE["src"], PALETTE["tgt"]

df = pd.read_csv(CACHE / "clean.csv", encoding="utf-8-sig")
print(f"读入 {df.shape}，城市分布 {df['city'].value_counts().to_dict()}")


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ======================================================================
# 图1 目标变量分布（分城市）
# ======================================================================
hr("图1 目标变量 y 的分布（分城市）")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
bins = np.arange(0, 260, 10)
for c, col, name in [(0, C0, "城市 0（源域）"), (1, C1, "城市 1（目标域）")]:
    s = df.loc[df.city == c, TARGET]
    axes[0].hist(s, bins=bins, alpha=0.62, color=col, edgecolor="white",
                 linewidth=0.4, label=f"{name}  n={len(s)}")
axes[0].set_xlabel("小时租借量 y（辆/小时）")
axes[0].set_ylabel("频数")
axes[0].set_title("(a) 需求分布直方图")
axes[0].legend(frameon=False, fontsize=9)

for c, col, name in [(0, C0, "城市 0（源域）"), (1, C1, "城市 1（目标域）")]:
    s = np.sort(df.loc[df.city == c, TARGET].values)
    axes[1].plot(s, np.linspace(0, 1, len(s)), color=col, lw=2, label=name)
axes[1].set_xlabel("小时租借量 y（辆/小时）")
axes[1].set_ylabel("累积概率")
axes[1].set_title("(b) 经验累积分布（ECDF）")
axes[1].legend(frameon=False, fontsize=9)
fig.suptitle("目标变量分布与城市间水平差异", y=1.02, fontsize=13,
             fontweight="bold")
savefig(fig, "fig01_y_distribution")
plt.close(fig)

# 定量：城市间分布差异
s0 = df.loc[df.city == 0, TARGET].values
s1 = df.loc[df.city == 1, TARGET].values
ks = stats.ks_2samp(s0, s1)
mw = stats.mannwhitneyu(s0, s1, alternative="two-sided")
ratio = s1.mean() / s0.mean()
print(f"KS 检验 D={ks.statistic:.4f}, p={ks.pvalue:.3e}")
print(f"Mann-Whitney U p={mw.pvalue:.3e}")
print(f"目标域/源域 均值比 = {ratio:.4f}（水平偏移 {100*(1-ratio):.1f}%）")
pd.DataFrame([
    dict(检验="Kolmogorov-Smirnov", 统计量=round(ks.statistic, 4),
          p值=f"{ks.pvalue:.3e}",
         结论="两城市需求分布存在极显著差异" if ks.pvalue < 0.01 else "无显著差异"),
    dict(检验="Mann-Whitney U", 统计量=float(mw.statistic),
         p值=f"{mw.pvalue:.3e}",
         结论="两城市需求水平存在极显著差异" if mw.pvalue < 0.01 else "无显著差异"),
    dict(检验="均值比（目标/源）", 统计量=round(ratio, 4), p值="-",
         结论=f"目标域需求水平仅为源域的 {ratio*100:.1f}%，存在明显标签偏移"),
]).pipe(lambda d: savetable(d, "tbl_city_diff_test"))

# ======================================================================
# 图2 小时需求曲线（城市 × 工作日）
# ======================================================================
hr("图2 小时需求曲线")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
for ax, wd, title in [(axes[0], 1, "(a) 工作日"), (axes[1], 0, "(b) 非工作日")]:
    for c, col, name in [(0, C0, "城市 0（源域）"), (1, C1, "城市 1（目标域）")]:
        g = (df[(df.city == c) & (df.is_workday == wd)]
             .groupby("hour")[TARGET].agg(["mean", "sem"]))
        ax.fill_between(g.index, g["mean"] - 1.96 * g["sem"],
                        g["mean"] + 1.96 * g["sem"], color=col, alpha=0.16)
        ax.plot(g.index, g["mean"], color=col, lw=2, marker="o", ms=3,
                label=name)
    ax.set_xlabel("小时")
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=9)
axes[0].set_ylabel("平均租借量（辆/小时）")
fig.suptitle("分城市、分工作日类型的小时需求曲线（阴影为 95% 置信带）",
             y=1.02, fontsize=13, fontweight="bold")
savefig(fig, "fig02_hourly_curve")
plt.close(fig)

# 曲线形状相似度：两城小时均值的相关系数
h0 = df[df.city == 0].groupby("hour")[TARGET].mean()
h1 = df[df.city == 1].groupby("hour")[TARGET].mean()
r_shape = np.corrcoef(h0, h1)[0, 1]
print(f"两城 24 小时均值曲线 Pearson 相关 = {r_shape:.4f}"
      f"（形状高度一致 -> 支持『形态可迁移』假设）")

# ======================================================================
# 图3 相关性热力图
# ======================================================================
hr("图3 特征相关性")
num_cols = ["y", "hour", "is_workday", "weather", "temp_1", "temp_2", "wind"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(6.4, 5.2))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha="right")
ax.set_yticks(range(len(num_cols)))
ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=8.5,
                color="white" if abs(corr.iloc[i, j]) > 0.55 else "#333")
ax.set_title("数值特征与目标变量的 Pearson 相关系数")
fig.colorbar(im, ax=ax, shrink=0.82, label="相关系数")
ax.grid(False)
savefig(fig, "fig03_corr_heatmap")
plt.close(fig)
print(corr["y"].drop("y").sort_values(key=abs, ascending=False).to_string())

# ======================================================================
# 图4 天气 / 温度 / 风力 效应
# ======================================================================
hr("图4 环境因素效应")
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))

# (a) 天气等级
data = [df.loc[df.weather == w, TARGET].values for w in [1, 2, 3]]
bp = axes[0, 0].boxplot(data, labels=["1 晴好", "2 一般", "3 恶劣"],
                        patch_artist=True, widths=0.55, showfliers=False)
for p, col in zip(bp["boxes"], ["#8EC6E6", "#F2C57C", "#E58C8C"]):
    p.set_facecolor(col)
axes[0, 0].set_xlabel("天气等级")
axes[0, 0].set_ylabel("租借量 y")
axes[0, 0].set_title("(a) 天气等级对需求的影响")

# (b) 温度
for c, col, name in [(0, C0, "城市 0"), (1, C1, "城市 1")]:
    d = df[df.city == c]
    axes[0, 1].scatter(d.temp_mean, d[TARGET], s=5, alpha=0.18, color=col,
                       label=name, edgecolors="none")
    # 分箱均值趋势
    q = pd.qcut(d.temp_mean, 14, duplicates="drop")
    tr = d.groupby(q, observed=True).agg(t=("temp_mean", "mean"),
                                         y=(TARGET, "mean"))
    axes[0, 1].plot(tr.t, tr.y, color=col, lw=2.2, marker="s", ms=4)
axes[0, 1].set_xlabel("平均温度（℃）")
axes[0, 1].set_ylabel("租借量 y")
axes[0, 1].set_title("(b) 温度与需求（折线为分箱均值）")
axes[0, 1].legend(frameon=False, fontsize=9)

# (c) 风力分箱
g = df.groupby(["wind_bin", "city"])[TARGET].mean().unstack()
x = np.arange(len(g))
axes[1, 0].bar(x - 0.19, g[0], 0.38, color=C0, label="城市 0（源域）")
axes[1, 0].bar(x + 0.19, g[1], 0.38, color=C1, label="城市 1（目标域）")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(["0-2 级", "3-4 级", "5+ 级"])
axes[1, 0].set_xlabel("风力分箱")
axes[1, 0].set_ylabel("平均租借量")
axes[1, 0].set_title("(c) 风力分箱下的需求均值")
axes[1, 0].legend(frameon=False, fontsize=9)

# (d) 工作日 × 城市
g = df.groupby(["is_workday", "city"])[TARGET].mean().unstack()
x = np.arange(2)
axes[1, 1].bar(x - 0.19, g[0], 0.38, color=C0, label="城市 0（源域）")
axes[1, 1].bar(x + 0.19, g[1], 0.38, color=C1, label="城市 1（目标域）")
for i, c in enumerate([0, 1]):
    for j, wd in enumerate([0, 1]):
        axes[1, 1].text(j + (-0.19 if i == 0 else 0.19), g.loc[wd, c] + 0.8,
                        f"{g.loc[wd, c]:.1f}", ha="center", fontsize=9)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(["非工作日", "工作日"])
axes[1, 1].set_ylabel("平均租借量")
axes[1, 1].set_ylim(0, 78)          # 留出空间，避免图例压住柱顶数值标签
axes[1, 1].set_title("(d) 工作日类型下的需求均值")
axes[1, 1].legend(frameon=False, fontsize=9, loc="upper center", ncol=2)

fig.suptitle("环境与时间因素对骑行需求的效应", y=1.0, fontsize=13,
             fontweight="bold")
fig.tight_layout()
savefig(fig, "fig04_env_effects")
plt.close(fig)

# ======================================================================
# 图5 城市间"形态同、水平异"的证据图
# ======================================================================
hr("图5 形态-水平分解")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
# (a) 原始曲线 vs 缩放后曲线
axes[0].plot(h0.index, h0.values, color=C0, lw=2.4, marker="o", ms=4,
             label="城市 0（源域）")
axes[0].plot(h1.index, h1.values, color=C1, lw=2.4, marker="s", ms=4,
             label="城市 1（目标域）")
axes[0].set_xlabel("小时")
axes[0].set_ylabel("平均租借量（辆/小时）")
axes[0].set_title(f"(a) 原始小时曲线（Pearson r={r_shape:.3f}）")
axes[0].legend(frameon=False, fontsize=9)

axes[1].plot(h0.index, h0.values / h0.mean(), color=C0, lw=2.4, marker="o",
             ms=4, label="城市 0（归一化）")
axes[1].plot(h1.index, h1.values / h1.mean(), color=C1, lw=2.4, marker="s",
             ms=4, label="城市 1（归一化）")
axes[1].axhline(1.0, color="#888", ls=":", lw=1)
axes[1].set_xlabel("小时")
axes[1].set_ylabel("相对需求水平（均值=1）")
r_norm = np.corrcoef(h0 / h0.mean(), h1 / h1.mean())[0, 1]
axes[1].set_title(f"(b) 水平归一化后曲线几乎重合（r={r_norm:.3f}）")
axes[1].legend(frameon=False, fontsize=9)
fig.suptitle("“形态同、水平异”：迁移学习可行性的直接证据",
             y=1.02, fontsize=13, fontweight="bold")
savefig(fig, "fig05_shape_level")
plt.close(fig)
print(f"归一化后两城曲线相关 = {r_norm:.4f}；水平缩放系数 = {ratio:.4f}")

# ======================================================================
# 描述性统计表
# ======================================================================
hr("描述性统计表")
rows = []
for c in [0, 1]:
    d = df[df.city == c]
    for col in ["y", "hour", "temp_1", "temp_2", "wind"]:
        rows.append(dict(城市=f"city={c}", 变量=col, 样本量=len(d),
                         均值=round(d[col].mean(), 3),
                         标准差=round(d[col].std(), 3),
                         最小值=round(d[col].min(), 3),
                         P25=round(d[col].quantile(.25), 3),
                         中位数=round(d[col].median(), 3),
                         P75=round(d[col].quantile(.75), 3),
                         最大值=round(d[col].max(), 3)))
savetable(pd.DataFrame(rows), "tbl_describe")

print("\n阶段 02 完成。")
