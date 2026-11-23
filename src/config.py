# -*- coding: utf-8 -*-
"""
T04 共享单车跨城市需求预测与迁移学习 —— 全局配置
=================================================
统一管理路径、随机种子、特征清单、模型超参与绘图风格，
保证全流程可复现（所有随机性均受 SEED 控制）。
"""
import os
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent          # bike_transfer/


def _locate_csv():
    """按优先级定位原始数据，不写死绝对路径，便于换机复现。

    查找顺序：
      1) 环境变量 BIKE_CSV（绝对路径）
      2) bike_transfer/data/bike.csv
      3) bike_transfer 同级目录 data/bike.csv
      4) 当前工作目录 data/bike.csv 或 bike.csv
    """
    cands = []
    if os.environ.get("BIKE_CSV"):
        cands.append(Path(os.environ["BIKE_CSV"]))
    cands += [
        ROOT / "data" / "bike.csv",
        ROOT.parent / "data" / "bike.csv",
        Path.cwd() / "data" / "bike.csv",
        Path.cwd() / "bike.csv",
    ]
    for c in cands:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "未找到 bike.csv。请任选一种方式提供：\n"
        "  1) 设置环境变量 BIKE_CSV=<绝对路径>\n"
        "  2) 放到 bike_transfer/data/bike.csv\n"
        "  3) 放到 bike_transfer 同级目录的 data/ 下\n"
        "已尝试以下位置：\n" + "\n".join(f"  - {c}" for c in cands))


RAW_CSV = _locate_csv()      # 原始数据（只读，禁止修改）
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
MOD = OUT / "models"
CACHE = OUT / "cache"
for _d in (OUT, FIG, TAB, MOD, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 复现性 ----------
SEED = 42
TEST_SIZE = 0.2          # 测试集比例（按 city 分层）
N_FOLDS = 5              # 交叉验证折数
N_BOOT = 1000            # Bootstrap 次数（置信区间）

# ---------- 特征 ----------
RAW_FEATS = ["city", "hour", "is_workday", "weather",
             "temp_1", "temp_2", "wind"]
TARGET = "y"

# 派生特征（构造逻辑见 02_features 阶段与附表3）
DERIVED_FEATS = [
    "hour_sin", "hour_cos",          # 小时周期编码
    "is_rush", "is_night",           # 通勤高峰 / 夜间
    "rush_x_workday",                # 高峰×工作日交互
    "temp_mean", "temp_diff",        # 温度均值 / 温差
    "weather_sev",                   # 天气恶劣度序数编码
    "wind_bin",                      # 风力分箱
    "hour_workday_cos",              # 小时周期×工作日交互
]
FEATS_FULL = RAW_FEATS + DERIVED_FEATS
FEATS_BASE = RAW_FEATS                      # 消融用：仅原始特征
FEATS_NO_CITY = [f for f in FEATS_FULL if f != "city"]   # 消融用：去掉城市标识

# ---------- XGBoost 超参 ----------
XGB_GLOBAL = dict(
    objective="reg:squarederror", max_depth=7, eta=0.05,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
    reg_lambda=1.0, verbosity=0, nthread=4, seed=SEED,
)
XGB_ROUNDS = 600
XGB_EARLY = 60

# 小样本（迁移 / 分城市）场景：更浅的树以抑制过拟合
XGB_SMALL = dict(XGB_GLOBAL, max_depth=4, min_child_weight=3)
XGB_SMALL_ROUNDS = 300

# ---------- 迁移实验 ----------
SOURCE_CITY = 0                     # 源域：数据充足城市
TARGET_CITY = 1                     # 目标域：待模拟稀疏的城市
SPARSITY_GRID = [50, 100, 200, 500, 1000, 2000]   # 目标域可用样本量梯度
SPARSE_THRESHOLD = 200              # "数据稀疏"判定阈值（占目标域训练池 5%）
GAIN_TARGET = 0.10                  # 达标线：误差下降 ≥10%

# ---------- 绘图 ----------
FIG_DPI = 200
PALETTE = {
    "src": "#4C78A8",        # 源域 蓝
    "tgt": "#E45756",        # 目标域 红
    "base": "#9DA9B8",       # 基线 灰
    "ours": "#2E8B57",       # 本文方案 绿
    "acc": "#F2A93B",        # 强调 橙
    "pur": "#7B6CB5",
}


def set_plot_style():
    """统一中文字体与绘图风格（Windows 下优先微软雅黑）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
        if cand in installed:
            plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    })
    return plt


def savefig(fig, name):
    """统一保存到 outputs/figures。"""
    p = FIG / f"{name}.png"
    fig.savefig(p)
    print(f"  [图] {p.name}")
    return p


def savetable(df, name):
    """统一保存表格到 outputs/tables（同时存 CSV 便于核对）。"""
    p = TAB / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  [表] {p.name}  shape={df.shape}")
    return p
