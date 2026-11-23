# -*- coding: utf-8 -*-
"""
阶段 06：技术路线图（报告 6.1 节配图）
=====================================
输出：outputs/figures/fig17_pipeline.png
运行：python src/06_pipeline_fig.py

说明：图中出现的数值（样本量、特征数、R²）全部从 outputs/tables 下的
结果表与 config 读取，不写死字面量，保证模型重跑后配图自动同步。
"""
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import (DERIVED_FEATS, FEATS_FULL, PALETTE, RAW_FEATS,
                    SPARSITY_GRID, TAB, savefig, set_plot_style)

plt = set_plot_style()

# ---------- 从结果表读取数值 ----------
_cmp = pd.read_csv(TAB / "tbl_model_compare.csv", encoding="utf-8-sig")
_r2 = _cmp.set_index("模型")["R2"]
R2_GLOBAL = _r2["M2 XGBoost 全局"]
R2_STRAT = _r2["M3 XGBoost 分层"]

_log = pd.read_csv(TAB / "tbl_clean_log.csv", encoding="utf-8-sig")
_n_raw, _n_clean = [int(s) for s in _log.loc[0, "样本量变化"].split("->")]

N_RAW, N_CLEAN = f"{_n_raw:,}", f"{_n_clean:,}"
N_DROP = _n_raw - _n_clean
N_RAW_F, N_DER_F, N_ALL_F = len(RAW_FEATS), len(DERIVED_FEATS), len(FEATS_FULL)
GRID = ", ".join(str(n) for n in SPARSITY_GRID)

fig, ax = plt.subplots(figsize=(13.0, 6.9))
ax.set_xlim(-6, 138)
ax.set_ylim(-5, 103)
ax.axis("off")

C = PALETTE


def box(x, y, w, h, text, fc, ec="#3F4A5A", fs=10.5, bold=False, tc="#1A1A1A"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                       linewidth=1.3, edgecolor=ec, facecolor=fc, alpha=0.95,
                       zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3,
            linespacing=1.5)


def arrow(x1, y1, x2, y2, style="-|>", color="#4A5568", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=15, linewidth=lw,
                                 color=color, linestyle=ls, zorder=1,
                                 shrinkA=1, shrinkB=1))


# 数据层（5 格，含探索性分析）
box(3, 79.5, 20, 13, f"原始数据 bike.csv\n{N_RAW} 条 × 9 字段", "#E8EEF6",
    bold=True, fs=10)
box(27.5, 79.5, 20, 13, "数据质量诊断\n缺失/重复/异常/口径", "#E8EEF6", fs=10)
box(52, 79.5, 20, 13, f"清洗与归并\n去重 {N_DROP} 条 → {N_CLEAN} 条",
    "#E8EEF6", fs=10)
box(76.5, 79.5, 20, 13, "探索性分析 EDA\n分布/小时曲线/相关性", "#E8EEF6", fs=10)
box(101, 79.5, 20, 13, f"特征工程\n{N_RAW_F} 原始 + {N_DER_F} 派生\n"
    f"共 {N_ALL_F} 个特征", "#E8EEF6", fs=9.5)

# 实验层
box(3, 58, 26, 13, f"全局建模\nXGBoost + city 编码\nR²={R2_GLOBAL:.3f}",
    "#DFF0E4", bold=True)
box(33.67, 58, 26, 13, f"分层建模\n每城独立训练\nR²={R2_STRAT:.3f}", "#DFF0E4")
box(64.33, 58, 26, 13, "消融实验\n特征组贡献量化", "#DFF0E4")
box(95, 58, 26, 13, "交叉验证\n5 折稳定性", "#DFF0E4")

# 迁移层
box(3, 34, 36, 14, "源域预训练\n城市 0（充足）→ f_S", "#FDEBD3", bold=True)
box(44, 34, 36, 14, f"目标域稀疏模拟\nn ∈ {{{GRID}}}", "#FDEBD3", fs=9.5)
box(85, 34, 36, 14, "迁移方案矩阵\nT0–T7 八种方案对比", "#FDEBD3", bold=True)

# 解释层
box(3, 12, 36, 13, "SHAP 可解释性\n蜂群图/依赖图/分城市差异", "#E9E4F4")
box(44, 12, 36, 13, "迁移增益量化\n达标线校验 + 负迁移边界", "#E9E4F4")
box(85, 12, 36, 13, "业务洞察\n4 条可落地结论", "#E9E4F4", bold=True)

# 横向箭头（数据层）
for x in [23, 47.5, 72, 96.5]:
    arrow(x, 86, x + 4.5, 86)

# 数据层 → 实验层
for x in [16, 46.67, 77.33, 108]:
    arrow(x, 79.5, x, 71)

# 实验层内部
for x in [29, 59.67, 90.33]:
    arrow(x, 64.5, x + 4.67, 64.5)

# 实验层 → 迁移层
arrow(16, 58, 16, 48)
arrow(108, 58, 108, 48)

# 迁移层内部
arrow(39, 41, 44, 41)
arrow(80, 41, 85, 41)

# 迁移层 → 解释层
for x in [21, 62, 103]:
    arrow(x, 34, x, 25)

# 分层标签
for y, t, col in [(86, "① 数据理解与准备", "#5B7BA6"),
                  (64.5, "② 建模与对照", "#4F8A62"),
                  (41.0, "③ 跨城市迁移", "#C9843C"),
                  (18.5, "④ 评估与解释", "#6E5FA3")]:
    ax.text(-4.2, y, t, fontsize=10.5, fontweight="bold", color=col,
            rotation=90, va="center", ha="center")

# 反馈回路：业务洞察 → 右侧 → 特征工程（真回路，两端都接框）
FB = "#B03A3A"
ax.plot([121, 127, 127], [18.5, 18.5, 86], color=FB, ls="--", lw=1.3,
        zorder=1, solid_capstyle="butt")
arrow(127, 86, 121.6, 86, color=FB, ls="--", lw=1.3)
ax.text(131.8, 52, "误差反馈\n迭代调优", fontsize=8.5, color=FB,
        va="center", ha="center", rotation=90, linespacing=1.6)

ax.set_title("技术路线：从数据理解到迁移解释的完整链路",
             fontsize=13.5, fontweight="bold", pad=14)
savefig(fig, "fig17_pipeline")
plt.close(fig)
print(f"阶段 06 完成。图中数值：样本 {N_RAW}→{N_CLEAN}，特征 {N_ALL_F}，"
      f"R² {R2_GLOBAL:.3f}/{R2_STRAT:.3f}")
