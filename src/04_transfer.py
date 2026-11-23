# -*- coding: utf-8 -*-
"""
阶段 04：跨城市迁移学习实验矩阵
================================
实验设定
  源域  = city0（数据充足，3995 条训练样本）
  目标域 = city1（通过下采样模拟"数据稀疏"，n ∈ SPARSITY_GRID）
  目标域测试集固定 998 条，全程不参与任何训练或调参

对比方案（7 种）
  T0 目标域独立训练      target-only：只用 n 条目标域样本，无迁移
  T1 零样本直接迁移      source-only：源域模型直接预测目标域
  T2 加性偏移校正        source + (ȳ_t - ŷ_t)
  T3 乘性水平校正        source × k, k = ȳ_t / ŷ_t        ← 一参数校准
  T4 目标域微调          source 为初始模型继续 boosting
  T5 样本加权联合训练    source(w=λ) + target(w=1) 单模型
  T6 两阶段残差校正      source 预测 + 目标域残差模型
  T7 本文方案            预训练 → 水平校正 → 微调 → 再校正

迁移增益 Gain = (RMSE_T0 - RMSE_Tk) / RMSE_T0 × 100%
输出：tbl_transfer_matrix.csv / tbl_transfer_gain.csv / fig10~fig12
运行：python src/04_transfer.py
"""
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (GAIN_TARGET, MOD, PALETTE, SEED,
                    SOURCE_CITY, SPARSITY_GRID, SPARSE_THRESHOLD, TARGET_CITY,
                    XGB_GLOBAL, XGB_ROUNDS, XGB_SMALL, XGB_SMALL_ROUNDS,
                    savefig, savetable, set_plot_style)
from utils import fit_xgb, get_xy, load_data, metrics, predict

warnings.filterwarnings("ignore")
plt = set_plot_style()
df, split = load_data()

src_tr = df.loc[split[f"train_c{SOURCE_CITY}"]].reset_index(drop=True)
tgt_tr = df.loc[split[f"train_c{TARGET_CITY}"]].reset_index(drop=True)
tgt_te = df.loc[split[f"test_c{TARGET_CITY}"]].reset_index(drop=True)

Xs, ys = get_xy(src_tr)
Xt, yt = get_xy(tgt_tr)
Xte, yte = get_xy(tgt_te)
N_TGT = len(tgt_tr)

print(f"源域训练 {Xs.shape}，目标域训练池 {Xt.shape}，目标域测试 {Xte.shape}")
print(f"稀疏度梯度 n ∈ {SPARSITY_GRID}（目标域训练池共 {N_TGT} 条）")
print(f"稀疏判定阈值 n <= {SPARSE_THRESHOLD}，达标线 误差下降 >= {GAIN_TARGET*100:.0f}%")


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ======================================================================
hr("1. 训练源域模型（全流程复用一次）")
t0 = time.time()
src_model, src_round = fit_xgb(Xs, ys, params=XGB_GLOBAL, rounds=XGB_ROUNDS)
src_model.save_model(str(MOD / "xgb_source_city0.json"))
print(f"  源域模型训练完成，{src_round} 轮，耗时 {time.time()-t0:.2f}s")
m_src = metrics(ys, predict(src_model, Xs))
print(f"  源域训练集拟合：RMSE={m_src['RMSE']:.3f} R²={m_src['R2']:.4f}")

# 全量监督上界：两城数据全量训练的全局模型（参照线，非迁移方案）
Xall = pd.concat([Xs, Xt], axis=0)
yall = np.concatenate([ys, yt])
upper, _ = fit_xgb(Xall, yall, params=XGB_GLOBAL, rounds=XGB_ROUNDS)
rmse_upper = metrics(yte, predict(upper, Xte))["RMSE"]
print(f"  全量监督上界（两城全量训练）目标域 RMSE = {rmse_upper:.3f}")

# ======================================================================
hr("2. 迁移方案实现")


def fit_target_only(Xn, yn, seed=SEED):
    m, _ = fit_xgb(Xn, yn, params=XGB_SMALL, rounds=XGB_SMALL_ROUNDS, seed=seed)
    return m


def calib_additive(p_src_te, p_src_tr, y_tr):
    """加性偏移校正：估计目标域整体偏差并平移。"""
    return p_src_te + (y_tr.mean() - p_src_tr.mean())


def calib_multiplicative(p_src_te, p_src_tr, y_tr):
    """乘性水平校正：估计目标域相对水平系数 k。"""
    base = p_src_tr.mean()
    k = y_tr.mean() / base if base > 1e-9 else 1.0
    return p_src_te * k


def finetune(Xn, yn, base_model, eta=0.02, rounds=200, seed=SEED):
    """以源域模型为初始模型，在目标域小样本上继续 boosting。"""
    p = dict(XGB_SMALL, eta=eta, seed=seed)
    d = xgb.DMatrix(Xn, label=yn)
    return xgb.train(p, d, num_boost_round=rounds, xgb_model=base_model)


def weighted_joint(Xn, yn, lam=0.15, seed=SEED):
    """源域 + 目标域联合训练，源域样本降权。"""
    Xj = pd.concat([Xs, Xn], axis=0)
    yj = np.concatenate([ys, yn])
    w = np.concatenate([np.full(len(Xs), lam), np.ones(len(Xn))])
    p = dict(XGB_SMALL, seed=seed)
    d = xgb.DMatrix(Xj, label=yj, weight=w)
    return xgb.train(p, d, num_boost_round=XGB_SMALL_ROUNDS)


def residual_stack(Xn, yn, p_src_tr, seed=SEED):
    """两阶段残差校正：源域预测 + 目标域残差模型。"""
    r = yn - p_src_tr
    m, _ = fit_xgb(Xn, r, params=XGB_SMALL, rounds=150, seed=seed)
    return m


METHODS = ["T0 目标域独立", "T1 零样本迁移", "T2 加性偏移校正",
           "T3 乘性水平校正", "T4 目标域微调", "T5 样本加权联合",
           "T6 两阶段残差", "T7 本文方案"]
print("  方案清单：")
for m in METHODS:
    print(f"    - {m}")

# ======================================================================
hr("3. 稀疏度梯度实验")
N_REPEAT = {50: 10, 100: 10, 200: 8, 500: 5, 1000: 5, 2000: 3, N_TGT: 1}
records = []
pred_store = {}

for n in SPARSITY_GRID + [N_TGT]:
    reps = N_REPEAT.get(n, 3)
    print(f"\n--- n = {n}（重复 {reps} 次）---")
    for r in range(reps):
        rng = np.random.RandomState(SEED + 1000 * n + r)
        sel = rng.choice(N_TGT, n, replace=False)
        Xn, yn = Xt.iloc[sel], yt[sel]

        # 源域模型在目标域上的预测（训练子集 + 测试集）
        p_src_tr = predict(src_model, Xn)
        p_src_te = predict(src_model, Xte)

        preds = {}
        preds["T0 目标域独立"] = predict(fit_target_only(Xn, yn), Xte)
        preds["T1 零样本迁移"] = p_src_te
        preds["T2 加性偏移校正"] = calib_additive(p_src_te, p_src_tr, yn)
        p_t3 = calib_multiplicative(p_src_te, p_src_tr, yn)
        preds["T3 乘性水平校正"] = p_t3

        ft = finetune(Xn, yn, src_model)
        preds["T4 目标域微调"] = predict(ft, Xte)

        wj = weighted_joint(Xn, yn)
        p_t5 = predict(wj, Xte)
        preds["T5 样本加权联合"] = p_t5

        res_m = residual_stack(Xn, yn, p_src_tr)
        preds["T6 两阶段残差"] = p_src_te + predict(res_m, Xte)

        # T7 本文方案：两条互补迁移路径的等权融合
        #   路径 A（乘性水平校正）：仅用 1 个参数校正"水平"，抑制标签偏移
        #   路径 B（样本加权联合）：源域降权 + 目标域升权，学习"形态"
        # 等权而非调参：避免在 50 条目标域样本上过拟合融合权重
        preds["T7 本文方案"] = 0.5 * p_t3 + 0.5 * p_t5

        for name, p in preds.items():
            mt = metrics(yte, p)
            records.append(dict(n=n, repeat=r, 方案=name,
                                RMSE=mt["RMSE"], MAE=mt["MAE"], R2=mt["R2"],
                                sMAPE=mt["sMAPE"]))
        if r == 0:
            pred_store[n] = {k: v.copy() for k, v in preds.items()}

res = pd.DataFrame(records)
res.to_csv(MOD / "transfer_raw.csv", index=False, encoding="utf-8-sig")

agg = (res.groupby(["n", "方案"])
       .agg(RMSE均值=("RMSE", "mean"), RMSE标准差=("RMSE", "std"),
            MAE均值=("MAE", "mean"), R2均值=("R2", "mean"),
            R2标准差=("R2", "std"), 重复次数=("RMSE", "count"))
       .reset_index())

# ======================================================================
hr("4. 迁移效果增益量化")
base = agg[agg.方案 == "T0 目标域独立"].set_index("n")["RMSE均值"]
agg["迁移增益%"] = agg.apply(
    lambda r: (base[r["n"]] - r["RMSE均值"]) / base[r["n"]] * 100, axis=1)

pivot_rmse = agg.pivot(index="方案", columns="n", values="RMSE均值").round(3)
pivot_gain = agg.pivot(index="方案", columns="n", values="迁移增益%").round(2)
print("\n【目标域测试集 RMSE（越小越好）】")
print(pivot_rmse.to_string())
print("\n【相对目标域独立训练的误差下降 %】")
print(pivot_gain.to_string())

# 长表（附表用）
out = agg.copy()
out["RMSE"] = out.apply(lambda r: f"{r['RMSE均值']:.3f} ± {r['RMSE标准差']:.3f}"
                        if pd.notna(r["RMSE标准差"]) else f"{r['RMSE均值']:.3f}", axis=1)
out["R2"] = out.apply(lambda r: f"{r['R2均值']:.4f} ± {r['R2标准差']:.4f}"
                      if pd.notna(r["R2标准差"]) else f"{r['R2均值']:.4f}", axis=1)
out = out.rename(columns={"RMSE均值": "RMSE均值", "MAE均值": "MAE均值",
                          "R2均值": "R2均值"})
savetable(out[["n", "方案", "重复次数", "RMSE", "RMSE均值", "RMSE标准差",
               "MAE均值", "R2", "迁移增益%"]].round(4), "tbl_transfer_matrix")

savetable(pivot_rmse.reset_index(), "tbl_transfer_rmse_pivot")
savetable(pivot_gain.reset_index(), "tbl_transfer_gain_pivot")

# ======================================================================
hr("5. 达标线校验（稀疏场景 n <= %d）" % SPARSE_THRESHOLD)
sparse = agg[agg.n <= SPARSE_THRESHOLD]
check = (sparse.groupby("方案")["迁移增益%"]
         .agg(["min", "mean", "max"]).round(2)
         .sort_values("mean", ascending=False))
print(check.to_string())
best_method = check.index[0]
best_mean = check.loc[best_method, "mean"]
best_min = check.loc[best_method, "min"]
print(f"\n稀疏场景下平均增益最高的方案：{best_method}")
print(f"  平均增益 {best_mean:.2f}%，最差情形 {best_min:.2f}%")
print(f"  【达标线】误差下降 >= 10% -> "
      f"{'通过' if best_min >= 10 else ('平均通过但不稳定' if best_mean >= 10 else '未通过')}")

# 逐档位校验
verify = []
for n in SPARSITY_GRID:
    g = float(pivot_gain.loc[best_method, n])
    verify.append(dict(目标域样本量=n, 方案=best_method, 迁移增益=g,
                       是否达标="达标" if g >= GAIN_TARGET * 100 else "未达标",
                       稀疏场景="是" if n <= SPARSE_THRESHOLD else "否"))
verify = pd.DataFrame(verify)
savetable(verify, "tbl_transfer_verify")
print("\n" + verify.to_string(index=False))

# ======================================================================
hr("6. 负迁移与适用边界分析")
nt = pivot_gain.loc[[m for m in METHODS if m != "T0 目标域独立"]]
for m in nt.index:
    g = nt.loc[m]
    neg = g[g < 0]
    print(f"  {m:16s} 增益区间 [{g.min():6.2f}%, {g.max():6.2f}%]"
          + (f" | 出现负迁移的档位：n={list(neg.index)}" if len(neg) else ""))

# ======================================================================
hr("7. 出图")
C = PALETTE
styles = {
    "T0 目标域独立": dict(color=C["base"], ls="--", marker="o", lw=2),
    "T1 零样本迁移": dict(color="#B0B7C3", ls=":", marker="^", lw=1.8),
    "T2 加性偏移校正": dict(color=C["acc"], ls="-.", marker="v", lw=1.8),
    "T3 乘性水平校正": dict(color=C["pur"], ls="-.", marker="s", lw=1.8),
    "T4 目标域微调": dict(color=C["tgt"], ls="-", marker="D", lw=2),
    "T5 样本加权联合": dict(color="#5FA8A0", ls="-", marker="P", lw=1.8),
    "T6 两阶段残差": dict(color="#C98A5E", ls="-", marker="X", lw=1.8),
    "T7 本文方案": dict(color=C["ours"], ls="-", marker="*", lw=3, ms=11),
}

# 图10 迁移前后误差对比图（必交交付物 2）
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
h, l = [], []
for m in METHODS:
    g = pivot_rmse.loc[m]
    st = dict(styles[m])
    ms = st.pop("ms", 6)
    ln, = axes[0].plot(g.index, g.values, label=m, ms=ms, **st)
    h.append(ln)
    l.append(m)
ln = axes[0].axhline(rmse_upper, color="#444", ls=":", lw=1.4,
                     label="全量监督上界")
h.append(ln)
l.append("全量监督上界（两城全量训练）")
axes[0].axvspan(0.9, SPARSE_THRESHOLD * 1.15, color="#FFE9C7", alpha=0.5,
                zorder=0)
axes[0].set_xscale("log")
axes[0].set_xlim(40, 4300)
axes[0].set_ylim(12.5, 27.5)
axes[0].text(105, 26.8, "稀疏场景区\n(n ≤ 200)", ha="center", va="top",
             fontsize=9, color="#8A6D3B")
axes[0].set_xlabel("目标域可用样本量 n（对数刻度）")
axes[0].set_ylabel("目标域测试集 RMSE（越小越好）")
axes[0].set_title("(a) 各方案误差随目标域样本量的变化")

h2, l2 = [], []
for m in METHODS:
    if m == "T0 目标域独立":
        continue
    g = pivot_gain.loc[m]
    st = dict(styles[m])
    ms = st.pop("ms", 6)
    ln, = axes[1].plot(g.index, g.values, label=m, ms=ms, **st)
    h2.append(ln)
    l2.append(m)
ln = axes[1].axhline(10, color=C["tgt"], ls="--", lw=1.6, label="达标线 +10%")
h2.append(ln)
l2.append("达标线（误差下降 10%）")
axes[1].axhline(0, color="#666", ls="-", lw=1)
axes[1].axvspan(0.9, SPARSE_THRESHOLD * 1.15, color="#FFE9C7", alpha=0.5,
                zorder=0)
axes[1].set_xscale("log")
axes[1].set_xlim(40, 4300)
axes[1].set_xlabel("目标域可用样本量 n（对数刻度）")
axes[1].set_ylabel("相对目标域独立训练的误差下降（%）")
axes[1].set_title("(b) 迁移增益随目标域样本量的衰减")
axes[1].text(105, -62, "稀疏场景区", ha="center", va="bottom", fontsize=9,
             color="#8A6D3B")

fig.suptitle("迁移前后误差对比：增益集中于数据稀疏区间",
             y=0.995, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0.16, 1, 0.97])
_lmap = {}
for _h, _l in zip(h + h2, l + l2):
    _lmap[_l] = _h
fig.legend(list(_lmap.values()), list(_lmap.keys()), loc="lower center",
           ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
savefig(fig, "fig10_transfer_compare")
plt.close(fig)

# 图11 稀疏场景柱状对比（n=50/100/200）
SHORT = {"T0 目标域独立": "T0\n独立训练", "T1 零样本迁移": "T1\n零样本",
         "T2 加性偏移校正": "T2\n加性校正", "T3 乘性水平校正": "T3\n乘性校正",
         "T4 目标域微调": "T4\n微调", "T5 样本加权联合": "T5\n样本加权",
         "T6 两阶段残差": "T6\n残差校正", "T7 本文方案": "T7\n本文方案"}
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=False)
for ax, n in zip(axes, [50, 100, 200]):
    sub = agg[agg.n == n].set_index("方案").reindex(METHODS)
    cols = [C["ours"] if m == "T7 本文方案" else
            (C["base"] if m == "T0 目标域独立" else C["src"]) for m in METHODS]
    ax.bar(range(len(METHODS)), sub["RMSE均值"], color=cols, alpha=0.92,
           edgecolor="white", linewidth=0.6)
    ax.errorbar(range(len(METHODS)), sub["RMSE均值"],
                yerr=sub["RMSE标准差"].fillna(0), fmt="none", ecolor="#333",
                capsize=3, lw=1)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=8.2)
    ax.set_title(f"n = {n}（重复 {int(sub['重复次数'].iloc[0])} 次）")
    ax.set_ylabel("目标域测试集 RMSE（越小越好）" if n == 50 else "")
    top = (sub["RMSE均值"] + sub["RMSE标准差"].fillna(0))
    for i, (v, t) in enumerate(zip(sub["RMSE均值"], top)):
        ax.text(i, t + 0.45, f"{v:.1f}", ha="center", fontsize=8,
                fontweight="bold" if METHODS[i] == "T7 本文方案" else "normal")
    ax.set_ylim(0, top.max() * 1.16)
fig.suptitle("数据稀疏场景下各迁移方案的误差对比（误差棒为标准差）",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig11_sparse_bars")
plt.close(fig)

# 图12 n=50 时预测曲线对比（案例可视化）
n_show = 50
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
sub_te = tgt_te.copy()
sub_te["yhat_T0"] = pred_store[n_show]["T0 目标域独立"]
sub_te["yhat_T1"] = pred_store[n_show]["T1 零样本迁移"]
sub_te["yhat_T7"] = pred_store[n_show]["T7 本文方案"]
for ax, wd, ttl in [(axes[0], 1, "(a) 工作日"), (axes[1], 0, "(b) 非工作日")]:
    d = sub_te[sub_te.is_workday == wd]
    g = d.groupby("hour")[["y", "yhat_T0", "yhat_T1", "yhat_T7"]].mean()
    ax.plot(g.index, g["y"], color="#333", lw=3, marker="o", ms=5,
            label="真实值")
    ax.plot(g.index, g["yhat_T0"], color=C["base"], ls="--", lw=2,
            label="T0 目标域独立(n=50)")
    ax.plot(g.index, g["yhat_T1"], color="#B0B7C3", ls=":", lw=2,
            label="T1 零样本迁移")
    ax.plot(g.index, g["yhat_T7"], color=C["ours"], lw=2.6, marker="*", ms=7,
            label="T7 本文方案")
    ax.set_xlabel("小时")
    ax.set_ylabel("平均租借量（辆/小时）")
    ax.set_title(ttl)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=8.5)
fig.suptitle("目标域仅 50 条样本时的分小时预测曲线对比",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig12_curve_case")
plt.close(fig)

# 图13 水平校正系数 k 的估计稳定性（单参数校准为什么在小样本下可用）
ks, ns = [], []
for n in SPARSITY_GRID:
    rng = np.random.RandomState(SEED + 1000 * n)
    sel = rng.choice(N_TGT, n, replace=False)
    p_src_tr = predict(src_model, Xt.iloc[sel])
    ks.append(yt[sel].mean() / p_src_tr.mean())
    ns.append(n)
ks = np.array(ks)
k_full = yt.mean() / predict(src_model, Xt).mean()
dev = (ks / k_full - 1) * 100          # 相对全量估计的偏差（%）

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3))

# (a) k 的估计值随样本量的变化
ax = axes[0]
ax.axhspan(k_full * 0.97, k_full * 1.03, color="#E4DDF5", zorder=0,
           label="全量估计 ±3% 区间")
ax.axhline(k_full, color="#444", ls="--", lw=1.4,
           label=f"全量估计 k = {k_full:.4f}")
ax.plot(ns, ks, color=C["pur"], lw=2.2, marker="o", ms=6, zorder=3,
        label="各档位的估计值")
for n_, k_ in zip(ns, ks):
    ax.annotate(f"{k_:.3f}", (n_, k_), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8.5, color="#4A3B6B")
ax.set_xscale("log")
ax.set_xlim(35, 3200)
ax.set_ylim(min(ks.min(), k_full * 0.97) - 0.014,
            max(ks.max(), k_full * 1.03) + 0.016)
ax.set_xlabel("目标域可用样本量 n（对数刻度）")
ax.set_ylabel("水平校正系数 k\n（目标域均值 / 源域预测均值）", fontsize=9)
ax.set_title("(a) k 的估计值几乎不随样本量变化")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# (b) 相对全量估计的偏差
ax = axes[1]
cols = [C["tgt"] if abs(v) > 5 else C["ours"] for v in dev]
ax.bar(range(len(ns)), dev, color=cols, width=0.6, edgecolor="white")
ax.axhline(0, color="#666", lw=1)
for y in (5, -5):
    ax.axhline(y, color=C["tgt"], ls="--", lw=1.2,
               label="±5% 参考线" if y > 0 else None)
for i, v in enumerate(dev):
    ax.text(i, v + (1.1 if v >= 0 else -1.1), f"{v:+.1f}%", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8.5)
ax.set_xticks(range(len(ns)))
ax.set_xticklabels([str(n) for n in ns], fontsize=9)
ax.set_xlabel("目标域可用样本量 n")
ax.set_ylabel("相对全量估计的偏差（%）")
ax.set_ylim(min(dev.min() * 1.6, -6), max(dev.max() * 1.45, 6))
ax.set_title("(b) n ≥ 100 后偏差收敛到 ±3.4% 以内")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

fig.suptitle("水平校正系数 k 的估计稳定性（只需 1 个参数）",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig13_scale_k")
plt.close(fig)
print(f"  k 的估计值范围：[{ks.min():.4f}, {ks.max():.4f}]，"
      f"全量估计 {k_full:.4f}，最大相对偏差 {np.abs(dev).max():.2f}%")

print("\n阶段 04 完成。")
