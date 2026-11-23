# -*- coding: utf-8 -*-
"""
公共工具：数据读取、评价指标、Bootstrap 置信区间、XGBoost 早停训练
==================================================================
被 03_models.py / 04_transfer.py / 05_interpret.py 复用，
确保三处实验使用完全一致的指标口径与训练流程。
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import CACHE, FEATS_FULL, SEED, TARGET


# ----------------------------------------------------------------------
def load_data():
    """读取清洗后宽表 + 固定划分索引。"""
    df = pd.read_csv(CACHE / "clean.csv", encoding="utf-8-sig")
    with open(CACHE / "split.json", encoding="utf-8") as f:
        split = json.load(f)
    return df, split


def get_xy(df, feats=None):
    feats = feats or FEATS_FULL
    return df[feats], df[TARGET].values.astype(np.float64)


# ----------------------------------------------------------------------
def metrics(y_true, y_pred):
    """
    统一评价口径：RMSE 为主指标，MAE / R² 为辅助指标。

    注：不采用 MAPE。本数据夜间租借量可低至 0-2 辆/小时，MAPE 的分母
    接近 0 会导致指标爆炸（实测基线 MAPE 达 452%），失去可比性，
    故改用有界的 sMAPE（对称平均绝对百分比误差，取值 [0, 200%]）。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom < 1e-9, 0.0,
                                   2 * np.abs(y_true - y_pred) / denom)) * 100)
    return dict(RMSE=rmse, MAE=mae, R2=r2, sMAPE=smape)


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=SEED, alpha=0.05):
    """对测试集做有放回重采样，给出 RMSE / R² 的 95% 置信区间。"""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    rmses, r2s = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        rmses.append(np.sqrt(np.mean((yt - yp) ** 2)))
        if yt.std() > 1e-9:
            r2s.append(1 - np.sum((yt - yp) ** 2) /
                       np.sum((yt - yt.mean()) ** 2))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return dict(
        RMSE_lo=float(np.percentile(rmses, lo)),
        RMSE_hi=float(np.percentile(rmses, hi)),
        R2_lo=float(np.percentile(r2s, lo)) if r2s else np.nan,
        R2_hi=float(np.percentile(r2s, hi)) if r2s else np.nan,
    )


# ----------------------------------------------------------------------
def fit_xgb(Xtr, ytr, Xval=None, yval=None, params=None, rounds=600,
            early=60, seed=SEED, verbose=False):
    """
    训练 XGBoost。若给验证集则早停，并用最优轮数在**全量训练集**上重训，
    避免早停验证集的信息泄漏进最终模型。
    返回 (booster, best_round)。
    """
    params = dict(params or {})
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("verbosity", 0)
    params.setdefault("seed", seed)
    params["seed"] = seed

    dtr = xgb.DMatrix(Xtr, label=ytr)
    if Xval is not None:
        dval = xgb.DMatrix(Xval, label=yval)
        tmp = xgb.train(params, dtr, num_boost_round=rounds,
                        evals=[(dval, "val")],
                        early_stopping_rounds=early,
                        verbose_eval=False)
        best = int(tmp.best_iteration) + 1
        final = xgb.train(params, dtr, num_boost_round=best)
    else:
        best = rounds
        final = xgb.train(params, dtr, num_boost_round=rounds)
    if verbose:
        print(f"    最优迭代轮数 = {best}")
    return final, best


def predict(booster, X):
    return booster.predict(xgb.DMatrix(X))


def gain_importance(booster):
    """把 total_gain 归一化为百分比，返回降序 Series（key 为特征名）。"""
    sc = booster.get_score(importance_type="total_gain")
    s = pd.Series(sc, dtype=float).sort_values(ascending=False)
    return (s / s.sum() * 100).round(3)
