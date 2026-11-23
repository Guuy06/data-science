# -*- coding: utf-8 -*-
"""
阶段 03b：稀疏场景下派生特征的边际价值（表 5-3 / 图 5-2）
=========================================================
在同一目标域样本量 n 下，比较"仅原始 7 特征"与"全部 17 特征"的 RMSE，
回答 5.1 节提出的问题：派生特征的价值是否随样本量变化。

早停口径（关键）：验证集从**目标域训练样本内部**再切 20%，目标域测试集
全程不参与训练、早停与调参（正文约束 C1）。轮数确定后用全部 n 条重训。
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from config import (FEATS_BASE, FEATS_FULL, PALETTE, SEED, TARGET_CITY,
                    XGB_GLOBAL, XGB_ROUNDS, XGB_SMALL, XGB_SMALL_ROUNDS,
                    TAB, savefig, set_plot_style)
from utils import get_xy, load_data, metrics, predict

plt = set_plot_style()

df, split = load_data()
tgt_tr = df.loc[split[f"train_c{TARGET_CITY}"]].reset_index(drop=True)
tgt_te = df.loc[split[f"test_c{TARGET_CITY}"]].reset_index(drop=True)

GRID = [50, 100, 200, 500, 1000, 2000, len(tgt_tr)]
REP = 5
VAL_FRAC = 0.2          # 内部验证集比例（仅用于早停定轮数）
SETS = [("仅原始 7 特征", FEATS_BASE), ("全部 17 特征", FEATS_FULL)]


def fit_es(Xn, yn, tr_idx, va_idx, params, rounds, early):
    """内部验证集早停定轮数，再用该轮数在全部 n 条上重训。

    返回的模型只见过目标域训练样本，未接触测试集。
    """
    p = dict(params)
    p.setdefault("verbosity", 0)
    p["seed"] = SEED
    d_tr = xgb.DMatrix(Xn.iloc[tr_idx], label=yn[tr_idx])
    d_va = xgb.DMatrix(Xn.iloc[va_idx], label=yn[va_idx])
    tmp = xgb.train(p, d_tr, num_boost_round=rounds, evals=[(d_va, "val")],
                    early_stopping_rounds=early, verbose_eval=False)
    best = int(tmp.best_iteration) + 1
    return xgb.train(p, xgb.DMatrix(Xn, label=yn), num_boost_round=best), best


rows = []
for n in GRID:
    small = n < 1000
    for rep in range(REP):
        rng = np.random.RandomState(SEED + 1000 * n + rep)
        idx = rng.choice(len(tgt_tr), size=min(n, len(tgt_tr)), replace=False)
        sub = tgt_tr.iloc[idx].reset_index(drop=True)
        perm = rng.permutation(len(sub))
        n_val = max(10, int(round(len(sub) * VAL_FRAC)))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        for name, feats in SETS:
            Xn, yn = get_xy(sub, feats)
            Xte, yte = get_xy(tgt_te, feats)
            m, _ = fit_es(Xn, yn, tr_idx, va_idx,
                          XGB_SMALL if small else XGB_GLOBAL,
                          XGB_SMALL_ROUNDS if small else XGB_ROUNDS,
                          40 if small else 60)
            mt = metrics(yte, predict(m, Xte))
            rows.append(dict(n=n, rep=rep, 特征方案=name, **mt))

r = pd.DataFrame(rows)
piv = r.groupby(["n", "特征方案"])[["RMSE", "R2"]].mean().reset_index()
wide = piv.pivot(index="n", columns="特征方案", values="RMSE")
wide["RMSE降幅%"] = (wide["仅原始 7 特征"] - wide["全部 17 特征"]) / \
    wide["仅原始 7 特征"] * 100

# 逐次重复的降幅，用来看稳定性
pairs = r.pivot_table(index=["n", "rep"], columns="特征方案", values="RMSE")
pairs["降幅%"] = (pairs["仅原始 7 特征"] - pairs["全部 17 特征"]) / \
    pairs["仅原始 7 特征"] * 100
stat = pairs.groupby("n")["降幅%"].agg(["mean", "std", "min", "max"])

print("\n" + "=" * 74)
print("派生特征边际价值：同一目标域样本量下，17 特征 vs 7 特征（重复 5 次）")
print("=" * 74)
out = wide.join(stat, on="n")
print(out.round(3).to_string())
print("\n降幅跨重复的波动（5 次重复，样本量小时波动极大）：")
print(stat.round(2).to_string())

# 报告用表：列名与顺序按正文口径整理
rep_df = pd.DataFrame({
    "n": wide.index,
    "7 特征 RMSE": [f"{x:.2f}" for x in wide["仅原始 7 特征"].values],
    "17 特征 RMSE": [f"{x:.2f}" for x in wide["全部 17 特征"].values],
    "降幅(%)": [f"{x:.2f}" for x in wide["RMSE降幅%"].values],
    "降幅标准差": [f"{x:.2f}" for x in stat["std"].values],
    "降幅区间": [f"{a:.1f}~{b:.1f}" for a, b in zip(stat["min"], stat["max"])],
})
rep_df.to_csv(TAB / "tbl_feat_ablation_small.csv", index=False,
              encoding="utf-8-sig")
print("\n[表] tbl_feat_ablation_small.csv")
print(rep_df.to_string(index=False))

# ----------------------------------------------------------------------
# 图 5-2：增益随样本量的变化
fs_ = rep_df.copy()
for c in ["n", "降幅(%)", "降幅标准差"]:
    fs_[c] = fs_[c].astype(float)

fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.axvspan(fs_["n"].min() * 0.8, 200 * 1.2, color="#FFF3CD", alpha=0.55,
           zorder=0, label="数据稀疏区（n ≤ 200）")
ax.errorbar(fs_["n"], fs_["降幅(%)"], yerr=fs_["降幅标准差"],
            color=PALETTE["pur"], lw=2.2, marker="o", ms=7,
            capsize=5, capthick=1.2, ecolor=PALETTE["pur"],
            elinewidth=1.2, zorder=3, label="增益均值 ± 标准差（5 次重复）")
ax.axhline(0, color="#666", ls="--", lw=1.1)
pk = fs_.loc[fs_["降幅(%)"].idxmax()]
ax.annotate(f"峰值 n={int(pk['n'])}：+{pk['降幅(%)']:.2f}%",
            xy=(pk["n"], pk["降幅(%)"]),
            xytext=(pk["n"] * 1.6, pk["降幅(%)"] + 7.0),
            fontsize=9.5, color="#333",
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
last = fs_["n"].max()
pk_n = pk["n"]
for _, row in fs_.iterrows():
    if row["n"] == pk_n:            # 峰值已有专门标注，不重复
        continue
    right = row["n"] < last
    ax.annotate(f"{row['降幅(%)']:.1f}",
                xy=(row["n"], row["降幅(%)"]),
                xytext=(11 if right else -11, 4),
                textcoords="offset points",
                ha="left" if right else "right", va="center",
                fontsize=8.5, color="#444")
ax.set_xscale("log")
ax.set_xlim(fs_["n"].min() * 0.55, fs_["n"].max() * 1.30)
ax.set_xticks(fs_["n"].tolist())
ax.set_xticklabels([f"{int(v)}" for v in fs_["n"]], fontsize=8.5)
ax.minorticks_off()
ax.set_xlabel("目标域可用样本量 n（对数刻度）")
ax.set_ylabel("派生特征带来的 RMSE 降幅（%）")
ax.set_ylim(min(-14, fs_["降幅(%)"].min() - 6),
            fs_["降幅(%)"].max() + fs_["降幅标准差"].max() + 13)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("派生特征的增益随样本量变化（仅用目标域数据独立训练）")
savefig(fig, "fig05b_feat_sparse")
plt.close(fig)

print("\n阶段 03b 完成。")
