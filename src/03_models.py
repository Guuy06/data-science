# -*- coding: utf-8 -*-
"""
阶段 03：基线 / 全局 / 分层建模与性能对比
==========================================
对照任务书 T04 的核心方法"分层建模 vs 全局建模"，并给出多模型横向对比。

模型清单
  B1 城市均值基线      —— 最弱参照，证明特征确实有效
  B2 Ridge 线性回归    —— 线性可加参照
  M1 RandomForest      —— Bagging 树集成
  M2 XGBoost(全局)     —— 含 city 标识的单一全局模型（达标线考察对象）
  M3 XGBoost(分层)     —— 每城独立建模，对照"分层 vs 全局"

另做三组消融：去 city 标识 / 仅原始特征 / 去掉温度类特征。

输出：outputs/tables/tbl_model_compare.csv   附表4 模型性能对比
      outputs/tables/tbl_model_per_city.csv  分城市性能对比表（必交）
      outputs/tables/tbl_ablation.csv        消融实验
      outputs/tables/tbl_cv.csv              5 折交叉验证
      outputs/figures/fig06~fig09
运行：python src/03_models.py
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from config import (FEATS_BASE, FEATS_FULL, FEATS_NO_CITY, MOD, N_FOLDS,
                    PALETTE, SEED, XGB_EARLY, XGB_GLOBAL, XGB_ROUNDS,
                    XGB_SMALL, XGB_SMALL_ROUNDS, set_plot_style, savefig,
                    savetable)
from utils import (bootstrap_ci, fit_xgb, gain_importance, get_xy,
                   load_data, metrics, predict)

warnings.filterwarnings("ignore")
plt = set_plot_style()
df, split = load_data()
tr_idx = split["train_c0"] + split["train_c1"]
te_idx = split["test_c0"] + split["test_c1"]
dtr, dte = df.loc[tr_idx], df.loc[te_idx]
Xtr, ytr = get_xy(dtr)
Xte, yte = get_xy(dte)
city_te = dte["city"].values
print(f"训练集 {Xtr.shape}，测试集 {Xte.shape}")


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


results, preds, timings = {}, {}, {}


def record(name, p, note, secs=0.0):
    m = metrics(yte, p)
    results[name] = dict(模型=name, 说明=note,
                         **{k: round(v, 4) for k, v in m.items()},
                         训练耗时秒=round(secs, 2))
    preds[name] = p
    print(f"  {name:24s} RMSE={m['RMSE']:7.3f}  MAE={m['MAE']:7.3f}  "
          f"R2={m['R2']:.4f}  ({secs:.2f}s)")


# ======================================================================
hr("B1 城市均值基线")
mu = dtr.groupby("city")["y"].mean()
p = dte["city"].map(mu).values
record("B1 城市均值基线", p, "仅用训练集各城市 y 均值，无任何特征")

# ======================================================================
hr("B2 Ridge 线性回归")
t0 = time.time()
Xtr_s = (Xtr - Xtr.mean()) / Xtr.std()
Xte_s = (Xte - Xtr.mean()) / Xtr.std()
ridge = Ridge(alpha=1.0).fit(Xtr_s, ytr)
record("B2 Ridge 回归", ridge.predict(Xte_s), "L2 正则线性模型，特征标准化", time.time() - t0)

# ======================================================================
hr("M1 RandomForest")
t0 = time.time()
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                           max_features=0.8, random_state=SEED, n_jobs=4)
rf.fit(Xtr, ytr)
record("M1 RandomForest", rf.predict(Xte), "500 棵树，min_samples_leaf=3", time.time() - t0)

# ======================================================================
hr("M2 XGBoost 全局模型（含 city 标识）")
t0 = time.time()
rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(Xtr), int(len(Xtr) * 0.15), replace=False)
mask = np.ones(len(Xtr), bool)
mask[val_idx] = False
gbm, best_round = fit_xgb(Xtr.iloc[mask], ytr[mask], Xtr.iloc[val_idx],
                          ytr[val_idx], XGB_GLOBAL, XGB_ROUNDS, XGB_EARLY,
                          verbose=True)
secs = time.time() - t0
record("M2 XGBoost 全局", predict(gbm, Xte), f"单一全局模型，早停最优 {best_round} 轮", secs)
gbm.save_model(str(MOD / "xgb_global.json"))

# ======================================================================
hr("M3 XGBoost 分层模型（每城独立）")
t0 = time.time()
p_sep = np.zeros(len(Xte))
sep_rounds = {}
for c in [0, 1]:
    m_tr = dtr["city"].values == c
    m_te = city_te == c
    b, br = fit_xgb(Xtr[m_tr], ytr[m_tr], params=XGB_SMALL,
                    rounds=XGB_SMALL_ROUNDS, seed=SEED)
    sep_rounds[c] = br
    p_sep[m_te] = predict(b, Xte[m_te])
    print(f"    city={c}: 训练 {m_tr.sum()} 条，{br} 轮")
record("M3 XGBoost 分层", p_sep, "每城独立建模，小样本浅树(depth=4)", time.time() - t0)

# ======================================================================
hr("消融实验")
abl = []


def ablate(name, feats, params=None, rounds=None):
    Xa, ya = get_xy(dtr, feats)
    Xb = dte[feats]
    b, br = fit_xgb(Xa.iloc[mask], ya[mask], Xa.iloc[val_idx], ya[val_idx],
                    params or XGB_GLOBAL, rounds or XGB_ROUNDS, XGB_EARLY)
    p = predict(b, Xb)
    m = metrics(yte, p)
    abl.append(dict(特征方案=name, 特征数=len(feats),
                    RMSE=round(m["RMSE"], 4), MAE=round(m["MAE"], 4),
                    R2=round(m["R2"], 4), 最优轮数=br))
    print(f"  {name:28s} RMSE={m['RMSE']:7.3f}  R2={m['R2']:.4f}")
    return m


full = ablate("全部特征（17 个）", FEATS_FULL)
ablate("去掉 city 标识", FEATS_NO_CITY)
ablate("仅原始特征（7 个）", FEATS_BASE)
no_temp = [f for f in FEATS_FULL if f not in ("temp_1", "temp_2", "temp_mean", "temp_diff")]
ablate("去掉温度类特征", no_temp)
no_hour = [f for f in FEATS_FULL if f not in ("hour", "hour_sin", "hour_cos",
                                              "hour_workday_cos")]
ablate("去掉小时类特征", no_hour)
savetable(pd.DataFrame(abl), "tbl_ablation")

# ----------------------------------------------------------------------
# 图 5-1 特征消融的边际贡献：去掉某组特征后 RMSE 相对全特征基线的上升量
ab = pd.DataFrame(abl).copy()
base_rmse = ab.loc[ab["特征方案"] == "全部特征（17 个）", "RMSE"].iloc[0]
ab["dRMSE"] = ab["RMSE"] - base_rmse
ab = ab.sort_values("dRMSE").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.0, 3.7))
cols = [PALETTE["ours"] if v <= 0.05 else
        (PALETTE["acc"] if v < 3 else PALETTE["tgt"]) for v in ab["dRMSE"]]
ax.barh(range(len(ab)), ab["dRMSE"], color=cols, height=0.62)
ax.set_yticks(range(len(ab)))
ax.set_yticklabels(ab["特征方案"], fontsize=9.5)
ax.axvline(0, color="#333", lw=1.0)
ax.set_xlabel("相对全特征基线的 RMSE 上升量（辆/小时，越大说明该组越不可替代）")
ax.set_xlim(0, ab["dRMSE"].max() * 1.42)
for i, r in ab.iterrows():
    txt = f"+{r['dRMSE']:.2f}（R² {r['R2']:.4f}）" if r["dRMSE"] > 0.05 \
        else f"基线 {r['RMSE']:.2f}（R² {r['R2']:.4f}）"
    ax.text(r["dRMSE"] + ab["dRMSE"].max() * 0.02, i, txt, va="center",
            fontsize=9)
ax.set_title("特征组的边际贡献：去掉该组后的性能损失")
savefig(fig, "fig05a_feat_ablation")
plt.close(fig)

# ----------------------------------------------------------------------
# 图 5-2（派生特征增益随样本量的变化）与表 5-3 由 03b_feat_ablation.py 生成。
# 该实验的早停验证集从目标域训练样本内部再切 20%，不使用测试集（约束 C1）。

# ======================================================================
hr("5 折交叉验证（全局模型稳定性）")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_rows = []
for i, (ti, vi) in enumerate(kf.split(Xtr), 1):
    b, br = fit_xgb(Xtr.iloc[ti], ytr[ti], Xtr.iloc[vi], ytr[vi],
                    XGB_GLOBAL, XGB_ROUNDS, XGB_EARLY)
    m = metrics(ytr[vi], predict(b, Xtr.iloc[vi]))
    cv_rows.append(dict(折=i, 训练量=len(ti), 验证量=len(vi),
                        RMSE=round(m["RMSE"], 4), MAE=round(m["MAE"], 4),
                        R2=round(m["R2"], 4), 最优轮数=br))
    print(f"  Fold {i}: RMSE={m['RMSE']:7.3f}  R2={m['R2']:.4f}")
cv = pd.DataFrame(cv_rows)
cv.loc[len(cv)] = dict(折="均值±标准差", 训练量="-", 验证量="-",
                       RMSE=f"{cv.RMSE.mean():.3f} ± {cv.RMSE.std():.3f}",
                       MAE=f"{cv.MAE.mean():.3f} ± {cv.MAE.std():.3f}",
                       R2=f"{cv.R2.mean():.4f} ± {cv.R2.std():.4f}",
                       最优轮数="-")
savetable(cv, "tbl_cv")

# ======================================================================
hr("分城市性能对比表（必交交付物 1）")
per_city = []
for name, p in preds.items():
    for c in [0, 1]:
        m = city_te == c
        mt = metrics(yte[m], p[m])
        per_city.append(dict(模型=name, 城市=f"city={c}",
                             样本量=int(m.sum()),
                             RMSE=round(mt["RMSE"], 4),
                             MAE=round(mt["MAE"], 4),
                             R2=round(mt["R2"], 4),
                             sMAPE=round(mt["sMAPE"], 2)))
per_city = pd.DataFrame(per_city)
savetable(per_city, "tbl_model_per_city")
print(per_city.pivot(index="模型", columns="城市", values="R2").to_string())

# 主表（附表4）
main = pd.DataFrame(results.values())
# 补充 Bootstrap 置信区间
ci = bootstrap_ci(yte, preds["M2 XGBoost 全局"])
main["R2_95CI"] = ""
main.loc[main.模型 == "M2 XGBoost 全局", "R2_95CI"] = \
    f"[{ci['R2_lo']:.3f}, {ci['R2_hi']:.3f}]"
main.loc[main.模型 == "M2 XGBoost 全局", "RMSE_95CI"] = \
    f"[{ci['RMSE_lo']:.2f}, {ci['RMSE_hi']:.2f}]"
main = main.rename(columns={"训练耗时秒": "训练耗时(s)"})
savetable(main, "tbl_model_compare")
print(main.to_string(index=False))
print(f"\n【达标线校验】全局模型 R² = {results['M2 XGBoost 全局']['R2']:.4f} "
      f">= 0.80 -> {'通过' if results['M2 XGBoost 全局']['R2'] >= 0.8 else '未通过'}")

# ======================================================================
hr("出图")
C0, C1 = PALETTE["src"], PALETTE["tgt"]
pg = preds["M2 XGBoost 全局"]

# 图6 预测-实际散点（分城市）
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True, sharey=True)
for ax, c, col, nm in [(axes[0], 0, C0, "城市 0（源域）"),
                       (axes[1], 1, C1, "城市 1（目标域）")]:
    m = city_te == c
    ax.scatter(yte[m], pg[m], s=9, alpha=0.35, color=col, edgecolors="none")
    lim = [0, 255]
    ax.plot(lim, lim, "--", color="#555", lw=1.2, label="理想 y=x")
    mt = metrics(yte[m], pg[m])
    ax.text(0.04, 0.95, f"R²={mt['R2']:.4f}\nRMSE={mt['RMSE']:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="white", ec="#ccc", alpha=0.9))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("真实租借量 y")
    ax.set_title(f"{nm}  n={m.sum()}")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
axes[0].set_ylabel("预测租借量 ŷ")
fig.suptitle("全局 XGBoost 模型的预测-实际对比（分城市）", y=1.02,
             fontsize=13, fontweight="bold")
savefig(fig, "fig06_pred_vs_true")
plt.close(fig)

# 图7 残差分析
fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.9))
res = yte - pg
for c, col, nm in [(0, C0, "城市 0"), (1, C1, "城市 1")]:
    m = city_te == c
    axes[0].hist(res[m], bins=45, alpha=0.6, color=col, label=nm,
                 edgecolor="white", linewidth=0.3)
axes[0].axvline(0, color="#333", ls="--", lw=1)
axes[0].set_xlabel("残差（真实 - 预测）")
axes[0].set_ylabel("频数")
axes[0].set_title("(a) 残差分布")
axes[0].legend(frameon=False, fontsize=9)

axes[1].scatter(pg, res, s=7, alpha=0.3, color=PALETTE["pur"],
                edgecolors="none")
axes[1].axhline(0, color="#333", ls="--", lw=1)
axes[1].set_xlabel("预测值 ŷ")
axes[1].set_ylabel("残差")
axes[1].set_title("(b) 残差 vs 预测值")

ev = pd.DataFrame({"y": yte, "p": pg, "hour": dte["hour"].values,
                   "city": city_te, "res": res})
for c, col, nm in [(0, C0, "城市 0"), (1, C1, "城市 1")]:
    hb = ev[ev.city == c].groupby("hour")["res"].mean()
    axes[2].plot(hb.index, hb.values, color=col, lw=2, marker="o", ms=3,
                 label=nm)
axes[2].axhline(0, color="#333", ls="--", lw=1)
axes[2].set_xlabel("小时")
axes[2].set_ylabel("平均残差")
axes[2].set_title("(c) 分小时平均残差（系统性偏差诊断）")
axes[2].legend(frameon=False, fontsize=9)
fig.suptitle("全局模型残差诊断", y=1.03, fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig07_residual")
plt.close(fig)

# 图8 特征重要性
imp = gain_importance(gbm)
fig, ax = plt.subplots(figsize=(7.4, 5.4))
cols = [PALETTE["acc"] if i < 4 else PALETTE["base"] for i in range(len(imp))]
ax.barh(range(len(imp))[::-1], imp.values, color=cols)
ax.set_yticks(range(len(imp))[::-1])
ax.set_yticklabels(imp.index)
ax.set_xlabel("重要性占比（total_gain, %）")
ax.set_title("全局模型特征重要性（前 4 名高亮）")
for i, (k, v) in enumerate(imp.items()):
    ax.text(v + 0.4, len(imp) - 1 - i, f"{v:.1f}%", va="center", fontsize=8.5)
ax.set_xlim(0, imp.max() * 1.16)
savefig(fig, "fig08_feature_importance")
plt.close(fig)
imp.rename("重要性占比%").to_csv(MOD / "feature_importance.csv",
                              encoding="utf-8-sig")

# 图9 模型对比
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
names = main["模型"].tolist()
x = np.arange(len(names))
axes[0].bar(x, main["RMSE"], color=[PALETTE["ours"] if "M2" in n else PALETTE["base"]
                                    for n in names])
axes[0].set_xticks(x)
axes[0].set_xticklabels([n.split(" ")[0] for n in names], fontsize=9)
axes[0].set_ylabel("测试集 RMSE（越小越好）")
axes[0].set_title("(a) RMSE 对比")
for i, v in enumerate(main["RMSE"]):
    axes[0].text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=8.5)

axes[1].bar(x, main["R2"], color=[PALETTE["ours"] if "M2" in n else PALETTE["base"]
                                  for n in names])
axes[1].axhline(0.80, color=PALETTE["tgt"], ls="--", lw=1.6,
                label="达标线 R²=0.80")
axes[1].set_xticks(x)
axes[1].set_xticklabels([n.split(" ")[0] for n in names], fontsize=9)
axes[1].set_ylabel("测试集 R²")
axes[1].set_title("(b) R² 对比")
axes[1].legend(frameon=False, fontsize=9)
for i, v in enumerate(main["R2"]):
    axes[1].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
axes[1].set_ylim(0, 1.0)
fig.suptitle("五类模型的整体性能对比", y=1.02, fontsize=13,
             fontweight="bold")
fig.tight_layout()
savefig(fig, "fig09_model_compare")
plt.close(fig)

np.save(MOD / "pred_global_test.npy", pg)
print("\n阶段 03 完成。")

# ======================================================================
hr("图 6-5 学习曲线（训练样本量 vs 误差）")
# 从训练集固定切出 15% 作验证集（与早停口径一致），其余作训练池；
# 验证集全程不变，只改变训练池的抽样比例，曲线才可比。
from sklearn.model_selection import train_test_split

tr_pool_idx, val_idx = train_test_split(
    np.arange(len(Xtr)), test_size=0.15, random_state=SEED,
    stratify=dtr["city"].values)
Xp, yp = Xtr.iloc[tr_pool_idx], ytr[tr_pool_idx]
Xv, yv = Xtr.iloc[val_idx], ytr[val_idx]
print(f"  训练池 {Xp.shape[0]} 条，固定验证集 {Xv.shape[0]} 条")

LC_RATIOS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
LC_REPEAT = 5
lc_rows = []
for r in LC_RATIOS:
    n = max(int(len(Xp) * r), 50)
    tr_r, va_r = [], []
    for rep in range(LC_REPEAT):
        rs = np.random.RandomState(SEED + 100 * int(r * 100) + rep)
        sub = rs.choice(len(Xp), size=n, replace=False)
        # 轮数固定，避免早停使各档位的模型容量不同、曲线不可比
        bst, _ = fit_xgb(Xp.iloc[sub], yp[sub],
                          params=XGB_GLOBAL, rounds=200)
        tr_r.append(metrics(yp[sub], predict(bst, Xp.iloc[sub]))["RMSE"])
        va_r.append(metrics(yv, predict(bst, Xv))["RMSE"])
    lc_rows.append(dict(样本量=n, 训练RMSE=np.mean(tr_r),
                        验证RMSE=np.mean(va_r),
                        训练std=np.std(tr_r), 验证std=np.std(va_r)))
    print(f"  n={n:5d}  train={np.mean(tr_r):6.2f}  val={np.mean(va_r):6.2f}")
lc = pd.DataFrame(lc_rows)
savetable(lc, "tbl_learning_curve")

gap0 = lc["验证RMSE"].iloc[0] - lc["训练RMSE"].iloc[0]
gap1 = lc["验证RMSE"].iloc[-1] - lc["训练RMSE"].iloc[-1]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.fill_between(lc["样本量"], lc["训练RMSE"] - lc["训练std"],
                lc["训练RMSE"] + lc["训练std"], color=PALETTE["src"],
                alpha=0.16, lw=0)
ax.fill_between(lc["样本量"], lc["验证RMSE"] - lc["验证std"],
                lc["验证RMSE"] + lc["验证std"], color=PALETTE["tgt"],
                alpha=0.16, lw=0)
ax.plot(lc["样本量"], lc["训练RMSE"], "-o", color=PALETTE["src"], lw=2,
        ms=5, label="训练集 RMSE", zorder=3)
ax.plot(lc["样本量"], lc["验证RMSE"], "-s", color=PALETTE["tgt"], lw=2,
        ms=5, label="验证集 RMSE", zorder=3)

# 泛化间隙：首末两档各画一条双向箭头，文字横向摆在箭头外侧，避免压线
xs = lc["样本量"].values
for x_, gap_, ha, dx in ((xs[0], gap0, "left", 1.22), (xs[-1], gap1, "right", 0.86)):
    y_tr = np.interp(x_, xs, lc["训练RMSE"].values)
    y_va = np.interp(x_, xs, lc["验证RMSE"].values)
    ax.annotate("", xy=(x_, y_va - 0.35), xytext=(x_, y_tr + 0.35),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["acc"],
                                lw=1.2, shrinkA=0, shrinkB=0), zorder=2)
    ax.text(x_ * dx, (y_tr + y_va) / 2, f"间隙 {gap_:.2f}", color=PALETTE["acc"],
            fontsize=9.5, ha=ha, va="center", zorder=4)

ax.set_xlabel("训练样本量（条）")
ax.set_ylabel("RMSE（辆/小时）")
ax.set_title("学习曲线：泛化间隙随样本量收窄", fontsize=12, fontweight="bold")
ax.set_xlim(xs[0] * 0.72, xs[-1] * 1.08)
ax.set_ylim(2, lc["验证RMSE"].max() + 2.2)
ax.legend(frameon=False, fontsize=9.5, loc="center right")
ax.grid(alpha=0.25, ls=":")
fig.tight_layout()
savefig(fig, "fig18_learning_curve")
plt.close(fig)

# ======================================================================
hr("表 6-7 误差案例：测试集绝对误差最大的 6 条")
err = np.abs(pg - yte)
order = np.argsort(-err)[:6]
dte_r = dte.reset_index(drop=True)
case = pd.DataFrame({
    "城市": [f"city{int(c)}" for c in dte_r.loc[order, "city"]],
    "hour": dte_r.loc[order, "hour"].astype(int).values,
    "是否工作日": ["是" if v else "否" for v in dte_r.loc[order, "is_workday"]],
    "天气": dte_r.loc[order, "weather"].astype(int).values,
    "temp_1": dte_r.loc[order, "temp_1"].round(1).values,
    "wind": dte_r.loc[order, "wind"].astype(int).values,
    "真实值": yte[order].astype(int),
    "预测值": pg[order].round(1),
    "绝对误差": err[order].round(1),
})
savetable(case, "tbl_error_cases")
print(case.to_string(index=False))

# 误差最大的 10% 样本 vs 其余样本的特征均值对比，用于定位失败模式
thr = np.quantile(err, 0.9)
bad = err >= thr
cmp_rows = []
for c in ("hour", "temp_1", "wind", "is_workday", "y"):
    cmp_rows.append(dict(特征=c,
                         高误差组=round(float(dte_r.loc[bad, c].mean()), 2),
                         其余样本=round(float(dte_r.loc[~bad, c].mean()), 2)))
cmp_df = pd.DataFrame(cmp_rows)
savetable(cmp_df, "tbl_error_profile")
print(cmp_df.to_string(index=False))
print(f"  高误差组阈值 |e| >= {thr:.2f}，占测试集 {bad.mean() * 100:.0f}%")
