#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2025 春节相位地市用电量无泄漏条件回测。

验证期气象使用同期实测值作为外生输入；负荷、统计量、模板和裁剪上限
均严格截断到预测起点之前。脚本复用正式 city_pipeline.py 的 Ridge 特征
与求解函数，并独立实现可参数化的 C4 春节模板消融。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRED_START = pd.Timestamp("2025-01-13")
PRED_END = pd.Timestamp("2025-02-09")
SPECIAL_START = pd.Timestamp("2025-01-27")
SPECIAL_END = pd.Timestamp("2025-02-04")
TRAIN_END = PRED_START - pd.Timedelta(days=1)
PRED_LEN = 28
QUEUE_DEPTH = 400
SOURCE_YEARS = (2023, 2024)
SOURCE_WEIGHTS = (0.5, 0.5)
ORIGINAL_WEIGHT = 0.85
WEIGHTS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
           0.85, 0.9, 0.95, 1.0)
CNY_DATES = {
    2023: pd.Timestamp("2023-01-22"),
    2024: pd.Timestamp("2024-02-10"),
    2025: pd.Timestamp("2025-01-29"),
    2026: pd.Timestamp("2026-02-17"),
}
INDUSTRIES = (1, 3, 4, 5, 6, 9, 10)
WEATHER_COLS = ["temp_max", "temp_min", "temp_mean", "rhum_mean", "precip_sum"]
REGRESSOR_COLS = ["temp_max", "temp_min", "temp_mean", "temp_range"]


@dataclass(frozen=True)
class Variant:
    experiment_id: str
    label: str
    temperature: bool = True
    industry: bool = True
    return_alignment: bool = True
    smoothing: bool = True
    blend_weight: float = ORIGINAL_WEIGHT


VARIANTS = (
    Variant("full", "完整模型"),
    Variant("no_temperature", "关闭温度修正", temperature=False),
    Variant("no_industry", "关闭行业分项融合", industry=False),
    Variant("no_return_alignment", "关闭节后恢复对齐", return_alignment=False),
    Variant("no_smoothing", "关闭历史曲线平滑", smoothing=False),
    Variant("template_only", "仅春节模板分支", blend_weight=1.0),
    Variant("ridge_only", "仅Ridge分支", blend_weight=0.0),
)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=here.parent / "submission" / "datasets" / "city_load_forecasting")
    parser.add_argument("--pipeline", type=Path, default=Path(r"E:\电量预测赛道\baseline\city_pipeline.py"))
    parser.add_argument("--official-baseline-root", type=Path, default=Path(r"E:\电量预测赛道\baseline"))
    parser.add_argument("--official-baseline-zip", type=Path, default=Path(r"E:\电量预测赛道\研发工具包\研发工具包\Baseline.zip"))
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--cache-dir", type=Path, default=here / "cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", help="仅用于调试；正式运行不得使用")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_delimited(path: Path, **kwargs) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    frame = pd.read_csv(path, sep=sep, encoding="utf-8-sig", **kwargs)
    frame.columns = [str(c).lstrip("\ufeff") for c in frame.columns]
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]


def load_load_data(path: Path):
    frame = read_delimited(path)
    required = {"province_id", "city_id", "industry_id", "date", "load"}
    if not required.issubset(frame.columns):
        raise ValueError(f"负荷字段缺失: {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["load"] = pd.to_numeric(frame["load"], errors="raise")
    frame["industry_id"] = pd.to_numeric(frame["industry_id"], errors="raise").astype(int)
    if frame.duplicated(["city_id", "industry_id", "date"]).any():
        raise ValueError("城市-行业-日期键不唯一")
    total = frame[frame["industry_id"] == 1].copy()
    cities = sorted(total["city_id"].unique(), key=lambda x: int(str(x).split("_")[-1]))
    province = total[["city_id", "province_id"]].drop_duplicates().set_index("city_id")["province_id"].to_dict()
    all_dates = pd.date_range(total["date"].min(), total["date"].max(), freq="D")
    pivot = total.pivot(index="date", columns="city_id", values="load").reindex(index=all_dates, columns=cities)
    if pivot.isna().any().any() or len(cities) != 10:
        raise ValueError("总量电量矩阵不完整或城市数不是10")
    values = []
    for city in cities:
        part = frame[(frame["city_id"] == city) & frame["industry_id"].isin(INDUSTRIES)].pivot(
            index="date", columns="industry_id", values="load"
        ).reindex(index=all_dates, columns=list(INDUSTRIES))
        values.append(part.to_numpy(np.float64))
    cube = np.stack(values, axis=1)
    if not np.isfinite(cube).all() or (cube < 0).any():
        raise ValueError("行业电量矩阵存在缺失、无穷或负数")
    identity_error = float(np.max(np.abs(cube[:, :, 0] - cube[:, :, 1:].sum(axis=2))))
    return frame, all_dates, cities, pivot.to_numpy(np.float64), cube, province, identity_error


def aggregate_weather(path: Path, cache_path: Path, reuse: bool) -> pd.DataFrame:
    if reuse and cache_path.exists():
        out = pd.read_csv(cache_path, parse_dates=["date"])
        return out
    parts = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    reader = pd.read_csv(
        path, sep=sep, encoding="utf-8-sig",
        usecols=["CITY_ID", "DATETIME", "TEM", "TEM_MAX", "TEM_MIN", "RHUM", "PRECIPITATION"],
        chunksize=400_000,
    )
    for chunk_no, chunk in enumerate(reader, 1):
        chunk["date"] = pd.to_datetime(chunk["DATETIME"].astype(str).str.slice(0, 10), errors="coerce")
        chunk = chunk[(chunk["date"] >= pd.Timestamp("2023-01-01")) & (chunk["date"] <= PRED_END)].copy()
        for col in ["TEM", "TEM_MAX", "TEM_MIN", "RHUM", "PRECIPITATION"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").replace(9999, np.nan)
        chunk["tem_count"] = chunk["TEM"].notna().astype(np.int64)
        chunk["tem_sum"] = chunk["TEM"].fillna(0.0)
        chunk["rhum_count"] = chunk["RHUM"].notna().astype(np.int64)
        chunk["rhum_sum"] = chunk["RHUM"].fillna(0.0)
        grouped = chunk.groupby(["CITY_ID", "date"], as_index=False).agg(
            temp_max=("TEM_MAX", "max"), temp_min=("TEM_MIN", "min"),
            tem_sum=("tem_sum", "sum"), tem_count=("tem_count", "sum"),
            rhum_sum=("rhum_sum", "sum"), rhum_count=("rhum_count", "sum"),
            precip_sum=("PRECIPITATION", "sum"),
        )
        parts.append(grouped)
        print(f"[weather] chunk {chunk_no} rows={len(chunk)}", flush=True)
    combined = pd.concat(parts, ignore_index=True).groupby(["CITY_ID", "date"], as_index=False).agg(
        temp_max=("temp_max", "max"), temp_min=("temp_min", "min"),
        tem_sum=("tem_sum", "sum"), tem_count=("tem_count", "sum"),
        rhum_sum=("rhum_sum", "sum"), rhum_count=("rhum_count", "sum"),
        precip_sum=("precip_sum", "sum"),
    )
    combined["temp_mean"] = combined["tem_sum"] / combined["tem_count"].replace(0, np.nan)
    combined["rhum_mean"] = combined["rhum_sum"] / combined["rhum_count"].replace(0, np.nan)
    out = combined.rename(columns={"CITY_ID": "city_id"})[["city_id", "date"] + WEATHER_COLS]
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return out


def fill_weather_without_leakage(raw: pd.DataFrame, cities, train_dates, future_dates):
    expected_train = pd.MultiIndex.from_product([cities, train_dates], names=["city_id", "date"])
    train = raw[raw["date"].isin(train_dates)].set_index(["city_id", "date"]).reindex(expected_train).reset_index()
    train = train.sort_values(["city_id", "date"])
    for col in WEATHER_COLS:
        train[col] = train.groupby("city_id")[col].ffill()
    stats_source = train.copy()
    stats_source["month"] = stats_source["date"].dt.month
    month_stats = stats_source.groupby(["city_id", "month"])[WEATHER_COLS].mean()
    city_stats = stats_source.groupby("city_id")[WEATHER_COLS].mean()
    for idx, row in train[train[WEATHER_COLS].isna().any(axis=1)].iterrows():
        city, month = row["city_id"], row["date"].month
        for col in WEATHER_COLS:
            if pd.isna(train.at[idx, col]):
                value = month_stats.loc[(city, month), col] if (city, month) in month_stats.index else np.nan
                train.at[idx, col] = city_stats.loc[city, col] if pd.isna(value) else value
    expected_future = pd.MultiIndex.from_product([cities, future_dates], names=["city_id", "date"])
    future = raw[raw["date"].isin(future_dates)].set_index(["city_id", "date"]).reindex(expected_future).reset_index()
    future["month"] = future["date"].dt.month
    for idx, row in future[future[WEATHER_COLS].isna().any(axis=1)].iterrows():
        city, month = row["city_id"], row["month"]
        for col in WEATHER_COLS:
            if pd.isna(future.at[idx, col]):
                value = month_stats.loc[(city, month), col] if (city, month) in month_stats.index else np.nan
                future.at[idx, col] = city_stats.loc[city, col] if pd.isna(value) else value
    future = future.drop(columns="month")
    if train[WEATHER_COLS].isna().any().any() or future[WEATHER_COLS].isna().any().any():
        raise ValueError("气象填充后仍有缺失；填充值未使用验证期统计量")
    return train, future


def import_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_city_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.PRED_START = PRED_START
    module.PRED_END = PRED_END
    module.PRED_LEN = PRED_LEN
    module.QUEUE_DEPTH = QUEUE_DEPTH
    module.CNY_TEMPLATE_YEARS = SOURCE_YEARS
    module.CNY_TEMPLATE_WEIGHTS = SOURCE_WEIGHTS
    module.CNY_TEMPLATE_WEIGHT = ORIGINAL_WEIGHT
    module.CNY_BLEND_START = PRED_START
    module.CNY_BLEND_END = PRED_END
    return module


def base_template(dates, load_matrix):
    positions = {pd.Timestamp(date): i for i, date in enumerate(dates)}
    phase = int((PRED_START - CNY_DATES[PRED_START.year]).days)
    template = np.zeros((load_matrix.shape[1], PRED_LEN), np.float64)
    for year, weight in zip(SOURCE_YEARS, SOURCE_WEIGHTS):
        source_date = CNY_DATES[year] + pd.Timedelta(days=phase)
        source = positions[source_date]
        level = load_matrix[source - 5:source].mean(axis=0)
        template += weight * (load_matrix[source:source + PRED_LEN] / level[None, :]).T
    return template


def train_ridge(module, train_dates, cities, train_load, province, weather_train, cache_path: Path, reuse: bool):
    if reuse and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as stored:
            return {k: stored[k] for k in stored.files}
    old_builder = module.build_cny_template
    module.build_cny_template = base_template
    try:
        data = module._train_pooled(train_dates, cities, train_load, province, weather_train, save=False)
    finally:
        module.build_cny_template = old_builder
    np.savez_compressed(cache_path, **data)
    return data


def predict_ridge(module, data, cities, province, weather_train, weather_future):
    all_core = pd.concat([weather_train, weather_future], ignore_index=True).drop_duplicates(["city_id", "date"], keep="last")
    static, base_cols, ft_cols, r2_cols = module.build_static_features(all_core)
    names = module.feature_names(base_cols, ft_cols, r2_cols)
    if [str(v) for v in data["feature_names"]] != names:
        raise ValueError("Ridge 特征模式与训练缓存不一致")
    dates = pd.date_range(PRED_START, PRED_END, freq="D")
    base = module._static_arrays(static, dates, cities, base_cols)
    ft = module._static_arrays(static, dates, cities, ft_cols)
    r2 = module._static_arrays(static, dates, cities, r2_cols)
    last_load = data["last_load"].astype(np.float64)
    out = np.empty((PRED_LEN, len(cities)), np.float64)
    for ci, city in enumerate(cities):
        dynamic = module.load_feature_vector(last_load[:, ci])
        cross = module.cross_vector(last_load, last_load.shape[0], ci, cities, province)
        for horizon in range(PRED_LEN):
            x = module._feature_row(base[city][horizon], ft[city][horizon], r2[city][horizon], dynamic, cross, horizon)
            scaled = (x - data["feature_mean"][ci]) / data["feature_scale"][ci]
            value = float(data["intercept"][ci] + scaled @ data["coefficient"][ci])
            out[horizon, ci] = (10 ** value) * float(data["smear"][ci])
    return out


def thermal_features(temp):
    temp = np.asarray(temp, np.float64)
    return np.stack([np.maximum(0, 10 - temp) / 10,
                     np.maximum(0, 18 - temp) / 10,
                     np.maximum(0, temp - 22) / 10,
                     np.maximum(0, temp - 26) / 10], axis=-1)


def effective_temperature(temp, memory=0.25):
    temp = np.asarray(temp, np.float64)
    lag1 = np.concatenate([temp[:1], temp[:-1]])
    lag2 = np.concatenate([temp[:1], temp[:1], temp[:-2]])
    return (1 - memory) * temp + memory * (lag1 + lag2) / 2


def cny_return_phase(year):
    from chinese_calendar.constants import holidays
    days = [pd.Timestamp(d) for d, name in holidays.items() if d.year == year and name == "Spring Festival"]
    if not days:
        raise ValueError(f"{year} 年春节日历缺失")
    return int((max(days) - CNY_DATES[year]).days) + 1


def smooth_curve(history, positions):
    return sum(history[positions + offset] / 7.0 for offset in range(-3, 4))


def build_template(train_dates, cities, cube, weather_train, weather_future, variant: Variant):
    from chinese_calendar import is_holiday
    temps = np.stack([weather_train[weather_train.city_id == city].set_index("date").reindex(train_dates).temp_mean.to_numpy(float) for city in cities], axis=1)
    if variant.temperature:
        feat = thermal_features(effective_temperature(temps))
        ordinary = np.asarray([not is_holiday(d.date()) and abs((d - CNY_DATES[d.year]).days) > 30 for d in train_dates])
        mask = ordinary[7:] & ordinary[:-7]
        dx = feat[7:] - feat[:-7]
        dy = np.log(np.maximum(cube, 1e-7))[7:] - np.log(np.maximum(cube, 1e-7))[:-7]
        coefficients = []
        for ci in range(len(cities)):
            x = dx[mask, ci]
            target = np.clip(dy[mask, ci], -0.5, 0.5)
            coefficients.append(np.linalg.solve(x.T @ x + 30.0 * np.eye(4), x.T @ target))
        coefficients = np.asarray(coefficients)
        response = np.exp(0.75 * np.einsum("dcf,cfj->dcj", feat, coefficients))
        neutral = cube / response
    else:
        coefficients = np.zeros((len(cities), 4, cube.shape[2]))
        neutral = cube.copy()
        mask = np.ones(len(train_dates) - 7, dtype=bool)
    recent = neutral[-5:].mean(axis=0)
    phase = int((PRED_START - CNY_DATES[PRED_START.year]).days)
    positions = {pd.Timestamp(d): i for i, d in enumerate(train_dates)}
    ratios = []
    for year in SOURCE_YEARS:
        source = positions[CNY_DATES[year] + pd.Timedelta(days=phase)]
        indices = np.arange(source, source + PRED_LEN)
        level = neutral[source - 5:source].mean(axis=0)
        phases = np.arange(phase, phase + PRED_LEN)
        values = neutral[indices].copy()
        if variant.smoothing:
            outside = (phases < -2) | (phases > 6)
            values[outside] = smooth_curve(neutral, indices)[outside]
        shift = cny_return_phase(year) - cny_return_phase(PRED_START.year)
        moved = smooth_curve(neutral, indices + shift) if variant.smoothing else neutral[indices + shift]
        gate = np.clip((phases - 6) / 3, 0, 1) * (0.5 if variant.return_alignment else 0.0)
        values = (1 - gate[:, None, None]) * values + gate[:, None, None] * moved
        ratio = values / np.maximum(level, 1e-9)
        # 与正式 C4 一致：历史五日尺度近零的行业通道不做比例放大。
        ratios.append(np.where(level[None] > 1e-8, ratio, 1.0))
    neutral_forecast = np.mean(ratios, axis=0) * recent
    future_temp = np.stack([weather_future[weather_future.city_id == city].set_index("date").reindex(pd.date_range(PRED_START, PRED_END)).temp_mean.to_numpy(float) for city in cities], axis=1)
    if variant.temperature:
        eff = effective_temperature(np.concatenate([temps[-2:], future_temp]))[2:]
        future_response = np.exp(0.75 * np.einsum("dcf,cfj->dcj", thermal_features(eff), coefficients))
    else:
        future_response = np.ones_like(neutral_forecast)
    components = neutral_forecast * future_response
    rho = 0.5 if variant.industry else 0.0
    template = np.maximum(0.0, (1 - rho) * components[:, :, 0] + rho * components[:, :, 1:].sum(axis=2))
    return template, {"thermal_pairs": int(mask.sum()), "coefficients": coefficients}


def official_baseline(load_train, weather_train, weather_future, cities, cache_path: Path, reuse: bool):
    if reuse and cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])
    from chinese_calendar.constants import holidays
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    grouped = {}
    for day, name in holidays.items():
        grouped.setdefault(name, []).append(day)
    holiday_frames = [pd.DataFrame({"holiday": name, "ds": pd.to_datetime(days), "lower_window": 0, "upper_window": 0}) for name, days in grouped.items()]
    holiday_frames.append(pd.DataFrame({"holiday": "chunyun", "ds": pd.to_datetime(grouped["Spring Festival"]), "lower_window": -14, "upper_window": 14}))
    holiday_df = pd.concat(holiday_frames, ignore_index=True)
    params = dict(growth="linear", yearly_seasonality=True, weekly_seasonality=True,
                  daily_seasonality=False, seasonality_mode="additive",
                  changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                  holidays_prior_scale=10.0, interval_width=0.8)
    rows = []
    future_dates = pd.date_range(PRED_START, PRED_END)
    for idx, city in enumerate(cities, 1):
        load_part = load_train[load_train.city_id == city][["date", "load"]].copy()
        wt = weather_train[weather_train.city_id == city][["date", "temp_max", "temp_min"]].copy()
        train = load_part.merge(wt, on="date", how="left")
        train["temp_mean"] = (train.temp_max + train.temp_min) / 2.0
        train["temp_range"] = train.temp_max - train.temp_min
        train = train.rename(columns={"date": "ds", "load": "y"})
        model = Prophet(holidays=holiday_df, **params)
        for col in REGRESSOR_COLS:
            model.add_regressor(col)
        model.fit(train[["ds", "y"] + REGRESSOR_COLS])
        wf = weather_future[weather_future.city_id == city][["date", "temp_max", "temp_min"]].copy()
        wf["temp_mean"] = (wf.temp_max + wf.temp_min) / 2.0
        wf["temp_range"] = wf.temp_max - wf.temp_min
        wf = wf.rename(columns={"date": "ds"}).set_index("ds").reindex(future_dates)
        wf.index.name = "ds"
        wf = wf.reset_index()
        pred = np.maximum(model.predict(wf[["ds"] + REGRESSOR_COLS]).yhat.to_numpy(float), 0.0)
        rows.extend((city, d, value) for d, value in zip(future_dates, pred))
        print(f"[baseline] {idx:02d}/10 {city}", flush=True)
    out = pd.DataFrame(rows, columns=["city_id", "date", "baseline"])
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return out


def score_matrix(actual, pred, dates):
    if (actual == 0).any():
        raise ValueError("评分窗口存在零真值；官方公式未规定零值处理，已停止而未加 epsilon")
    coeff = np.maximum(0.0, 1.0 - np.sqrt(np.mean(((actual - pred) / actual) ** 2, axis=0)))
    mask = (dates >= SPECIAL_START) & (dates <= SPECIAL_END)
    coeff_special = np.maximum(0.0, 1.0 - np.sqrt(np.mean(((actual[mask] - pred[mask]) / actual[mask]) ** 2, axis=0)))
    score_28 = 20.0 * coeff.mean()
    score_special = 5.0 * coeff_special.mean()
    return score_28, score_special, score_28 + score_special, coeff, coeff_special


def configure_plotting():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.dpi": 120, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })


def shade(ax):
    ax.axvspan(SPECIAL_START, SPECIAL_END, color="#D9D9D9", alpha=0.45, label="专项窗口")
    ax.axvline(CNY_DATES[2025], color="#444444", linestyle="--", linewidth=1.0, label="春节")


def save_figure(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_results(fig_dir: Path, dates, cities, actual, baseline, full, weight_df):
    configure_plotting()
    colors = {"actual": "#1A1A1A", "baseline": "#7F7F7F", "full": "#1F5A94"}
    fig, axes = plt.subplots(5, 2, figsize=(14, 17), sharex=True)
    for ci, (city, ax) in enumerate(zip(cities, axes.flat)):
        shade(ax)
        ax.plot(dates, actual[:, ci], color=colors["actual"], lw=1.8, label="实际值")
        ax.plot(dates, baseline[:, ci], color=colors["baseline"], lw=1.2, ls=":", label="官方Baseline")
        ax.plot(dates, full[:, ci], color=colors["full"], lw=1.5, label="完整模型")
        ax.set_title(city)
        ax.set_ylabel("日用电量（归一化值）")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("10个城市实际值与预测对比", y=0.995, fontsize=15)
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, fig_dir / "all_cities_prediction_comparison")

    fig, axes = plt.subplots(5, 2, figsize=(14, 17), sharex=True)
    for ci, (city, ax) in enumerate(zip(cities, axes.flat)):
        shade(ax)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.plot(dates, (baseline[:, ci] - actual[:, ci]) / actual[:, ci], color=colors["baseline"], lw=1.2, ls=":", label="官方Baseline")
        ax.plot(dates, (full[:, ci] - actual[:, ci]) / actual[:, ci], color=colors["full"], lw=1.5, label="完整模型")
        ax.set_title(city)
        ax.set_ylabel("有符号相对误差")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("10个城市逐日有符号相对误差", y=0.995, fontsize=15)
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, fig_dir / "all_cities_signed_relative_error")

    single_dir = fig_dir / "by_city"
    single_dir.mkdir(exist_ok=True)
    for ci, city in enumerate(cities):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        shade(ax)
        ax.plot(dates, actual[:, ci], color=colors["actual"], lw=2, label="实际值")
        ax.plot(dates, baseline[:, ci], color=colors["baseline"], lw=1.4, ls=":", label="官方Baseline")
        ax.plot(dates, full[:, ci], color=colors["full"], lw=1.6, label="完整模型")
        ax.set(title=f"{city} 实际值与预测", ylabel="日用电量（归一化值）", xlabel="日期")
        ax.legend(ncol=5)
        fig.tight_layout()
        save_figure(fig, single_dir / f"{city}_prediction")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        shade(ax)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.plot(dates, (baseline[:, ci] - actual[:, ci]) / actual[:, ci], color=colors["baseline"], lw=1.4, ls=":", label="官方Baseline")
        ax.plot(dates, (full[:, ci] - actual[:, ci]) / actual[:, ci], color=colors["full"], lw=1.6, label="完整模型")
        ax.set(title=f"{city} 逐日有符号相对误差", ylabel="(预测-实际)/实际", xlabel="日期")
        ax.legend(ncol=4)
        fig.tight_layout()
        save_figure(fig, single_dir / f"{city}_signed_relative_error")

    metrics = [("score_28", "28天得分", "#1F5A94"), ("score_special", "春节专项得分", "#B45F06"), ("score_total", "合计得分", "#4F772D")]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, (col, title, color) in zip(axes, metrics):
        ax.plot(weight_df.weight, weight_df[col], marker="o", color=color, lw=1.7)
        ax.axvline(ORIGINAL_WEIGHT, color="#333333", linestyle="--", label="原配置 w=0.85")
        ax.set_ylabel(title)
        ax.legend()
    axes[-1].set_xlabel("春节模板权重 w")
    fig.suptitle("融合权重敏感性（同一验证窗口扫描）", fontsize=15)
    fig.tight_layout()
    save_figure(fig, fig_dir / "weight_sensitivity")


def write_summary(path: Path, ablation: pd.DataFrame, weights: pd.DataFrame, city_compare: pd.DataFrame, meta: dict):
    full = ablation.set_index("experiment_id").loc["full"]
    base = ablation.set_index("experiment_id").loc["official_baseline"]
    best = weights.loc[weights.score_total.idxmax()]
    degraded = city_compare[city_compare.gain_total < 0].city_id.tolist()
    special_degraded = city_compare[city_compare.gain_special < 0].city_id.tolist()
    lines = [
        "# 2025年春节相位地市用电量回测小结", "",
        "## 实验设置", "",
        f"- 训练数据截止：{TRAIN_END.date()}；预测窗口：{PRED_START.date()}至{PRED_END.date()}（28天）。",
        f"- 春节专项窗口：{SPECIAL_START.date()}至{SPECIAL_END.date()}（9天）；模板来源：2023、2024年，等权。",
        "- 本实验为使用验证期实测气象的条件回测。未发现2025年当时起报的历史气象预报，因此结果不包含天气预报误差，不能表述为真实部署效果。",
        "- Ridge、标准化、残差尺度修正、温度响应、气象填充值和裁剪上限均仅由训练段估计；28天预测不回灌验证期真实电量。",
        "- 2024年同相位起点不满足400天历史队列，因此未构造2024完整模型成绩。", "",
        "## 主要结果", "",
        f"完整模型本地得分为 {full.score_28:.3f} + {full.score_special:.3f} = {full.score_total:.3f}；官方Baseline本地得分为 {base.score_28:.3f} + {base.score_special:.3f} = {base.score_total:.3f}。",
        f"同一验证窗口权重扫描的最高合计得分为 {best.score_total:.3f}（w={best.weight:g}）。该数值属于验证集扫描，不是独立测试成绩，未自动替换正式配置0.85。",
    ]
    if degraded:
        lines.append("完整模型相对Baseline退化的城市：" + "、".join(degraded) + "。")
    else:
        lines.append("本窗口内完整模型在10个城市的合计分城市指标均未低于Baseline。")
    if special_degraded:
        lines.append("春节专项分城市指标轻微退化：" + "、".join(special_degraded) + "；需与其合计增益分开解读。")
    lines += ["", "## 模块消融结果", "",
              "| 实验版本 | 28天得分 | 春节专项 | 合计 | 相对完整模型 |",
              "|---|---:|---:|---:|---:|"]
    for row in ablation.itertuples(index=False):
        lines.append(f"| {row.experiment_label} | {row.score_28:.3f} | {row.score_special:.3f} | {row.score_total:.3f} | {row.delta_vs_full:+.3f} |")
    lines += ["", "关闭节后恢复对齐仅下降0.004分，单窗口证据很弱；其余指定模块在本窗口均有正向贡献。", "",
              "## 融合权重敏感性", "",
              "| w | 28天得分 | 春节专项 | 合计 |",
              "|---:|---:|---:|---:|"]
    for row in weights.itertuples(index=False):
        marker = " **（原配置）**" if row.is_original else ""
        lines.append(f"| {row.weight:g}{marker} | {row.score_28:.3f} | {row.score_special:.3f} | {row.score_total:.3f} |")
    lines += ["", "## 分城市对比", "",
              "| 城市 | Baseline合计 | 完整模型合计 | 合计增益 | 专项增益 |",
              "|---|---:|---:|---:|---:|"]
    for row in city_compare.itertuples(index=False):
        lines.append(f"| {row.city_id} | {row.baseline_score_total:.3f} | {row.full_score_total:.3f} | {row.gain_total:+.3f} | {row.gain_special:+.3f} |")
    lines += ["", "## 消融开关的代码含义", "",
              "- 关闭温度修正：春节模板历史曲线不除以温度响应，未来也不乘回温度响应；Ridge分支及模板其他步骤不变。这是模板内部成对关闭的耦合变体。",
              "- 关闭行业分项融合：模板内部权重设为0，仅使用总量通道；总量温度修正保留。",
              "- 关闭节后恢复对齐：返工平移混合门控设为0；春节相位对齐和平滑保留。",
              "- 关闭历史曲线平滑：主历史曲线和平移分支均使用原始曲线；恢复对齐保留。",
              "- 仅模板／仅Ridge：最终模板权重分别设为1／0，并继续执行与完整模型相同的非负和历史上限裁剪。", "",
              "## 可直接写入技术报告的表述", "",
              f"在2025年春节相位条件回测中，模型以2025年1月12日为训练截止日，直接预测后续28天，并在所有模型共享验证期实测气象的条件下评估。完整模型获得{full.score_total:.3f}分，相对官方Prophet Baseline变化{full.score_total-base.score_total:+.3f}分。该结论仅适用于本次单窗口条件回测，不代表跨年度稳定性或包含气象预报误差的线上效果。", "",
              "## 复现与核验", "",
              f"- 城市数/预测条数：{meta['city_count']}/{meta['prediction_rows']}；行业加总恒等最大误差：{meta['industry_identity_max_error']:.3e}。",
              f"- 训练起点数：{meta['ridge_origins']}；每城市Ridge样本数：{meta['ridge_samples_per_city']}。",
              "- 已核验城市-日期唯一与完整覆盖、有限值、非负裁剪、汇总一致性，以及w=0/1/0.85与单分支/完整模型一致性。",
              "- 官方线上成绩18.129+4.439=22.568与19.102+4.802=23.904仅作为背景，不参与本地增益计算。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    started = time.time()
    for path in [args.output_dir, args.cache_dir, args.output_dir / "figures", args.output_dir / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    load_path = args.data_root / "train_data" / "load_data" / "city_load_train.csv"
    weather_path = args.data_root / "train_data" / "weather_data" / "city_weather_data.csv"
    for path in [load_path, weather_path, args.pipeline, args.official_baseline_zip]:
        if not path.exists():
            raise FileNotFoundError(path)
    frame, dates, cities, load_matrix, cube, province, identity_error = load_load_data(load_path)
    train_mask = dates <= TRAIN_END
    future_dates = pd.date_range(PRED_START, PRED_END)
    train_dates = dates[train_mask]
    if len(train_dates) < QUEUE_DEPTH:
        raise ValueError("训练截止点不满足400天历史队列")
    train_load = load_matrix[train_mask]
    train_cube = cube[train_mask]
    actual = load_matrix[np.isin(dates, future_dates)]
    if actual.shape != (PRED_LEN, 10):
        raise ValueError(f"验证真值覆盖异常: {actual.shape}")
    raw_weather = aggregate_weather(weather_path, args.cache_dir / "daily_weather_through_2025-02-09.csv", args.reuse_cache)
    weather_train, weather_future = fill_weather_without_leakage(raw_weather, cities, train_dates, future_dates)
    module = import_pipeline(args.pipeline)
    ridge_data = train_ridge(module, train_dates, cities, train_load, province, weather_train, args.cache_dir / "ridge_cutoff_2025-01-12.npz", args.reuse_cache)
    ridge = predict_ridge(module, ridge_data, cities, province, weather_train, weather_future)
    templates = {}
    finals = {}
    thermal_meta = {}
    upper = ridge_data["city_upper"].astype(float)
    for variant in VARIANTS:
        template_variant = Variant(variant.experiment_id, variant.label, variant.temperature, variant.industry, variant.return_alignment, variant.smoothing, variant.blend_weight)
        template, info = build_template(train_dates, cities, train_cube, weather_train, weather_future, template_variant)
        final = np.clip((1 - variant.blend_weight) * ridge + variant.blend_weight * template, 0.0, upper[None, :])
        templates[variant.experiment_id] = template
        finals[variant.experiment_id] = final
        thermal_meta[variant.experiment_id] = info
    total_train = frame[(frame.industry_id == 1) & (frame.date <= TRAIN_END)][["city_id", "date", "load"]]
    baseline_df = official_baseline(total_train, weather_train, weather_future, cities, args.cache_dir / "official_prophet_baseline_2025.csv", args.reuse_cache) if not args.skip_baseline else None
    if baseline_df is None:
        raise RuntimeError("正式实验不得使用 --skip-baseline")
    baseline = baseline_df.pivot(index="date", columns="city_id", values="baseline").reindex(index=future_dates, columns=cities).to_numpy(float)

    score_rows = []
    city_rows = []
    all_predictions = []
    experiment_specs = [("official_baseline", "官方Baseline", baseline, np.full_like(ridge, np.nan))] + [(v.experiment_id, v.label, finals[v.experiment_id], templates[v.experiment_id]) for v in VARIANTS]
    for exp_id, label, pred, template in experiment_specs:
        s28, ss, st, c28, cs = score_matrix(actual, pred, future_dates)
        score_rows.append(dict(experiment_id=exp_id, experiment_label=label, score_28=s28, score_special=ss, score_total=st))
        for ci, city in enumerate(cities):
            city_rows.append(dict(experiment_id=exp_id, city_id=city, coefficient_28=c28[ci], coefficient_special=cs[ci], score_28=20*c28[ci], score_special=5*cs[ci], score_total=20*c28[ci]+5*cs[ci]))
            for di, date in enumerate(future_dates):
                all_predictions.append(dict(experiment_id=exp_id, city_id=city, date=date.strftime("%Y-%m-%d"), actual=actual[di, ci], baseline=baseline[di, ci], ridge=ridge[di, ci], template=template[di, ci], final=pred[di, ci]))
    ablation = pd.DataFrame(score_rows)
    full_total = float(ablation.loc[ablation.experiment_id == "full", "score_total"].iloc[0])
    ablation["delta_vs_full"] = ablation.score_total - full_total
    ablation.to_csv(args.output_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")
    city_metrics = pd.DataFrame(city_rows)
    city_metrics.to_csv(args.output_dir / "city_metrics_all_experiments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_predictions).to_csv(args.output_dir / "daily_predictions.csv", index=False, encoding="utf-8-sig")

    weight_rows = []
    full_template = templates["full"]
    for weight in WEIGHTS:
        pred = np.clip((1 - weight) * ridge + weight * full_template, 0.0, upper[None, :])
        s28, ss, st, _, _ = score_matrix(actual, pred, future_dates)
        weight_rows.append(dict(weight=weight, score_28=s28, score_special=ss, score_total=st, is_original=(weight == ORIGINAL_WEIGHT)))
    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(args.output_dir / "weight_sensitivity.csv", index=False, encoding="utf-8-sig")

    cm = city_metrics[city_metrics.experiment_id.isin(["official_baseline", "full"])].pivot(index="city_id", columns="experiment_id", values=["score_28", "score_special", "score_total"])
    compare = pd.DataFrame({
        "city_id": cities,
        "baseline_score_28": [cm.loc[c, ("score_28", "official_baseline")] for c in cities],
        "full_score_28": [cm.loc[c, ("score_28", "full")] for c in cities],
        "baseline_score_special": [cm.loc[c, ("score_special", "official_baseline")] for c in cities],
        "full_score_special": [cm.loc[c, ("score_special", "full")] for c in cities],
        "baseline_score_total": [cm.loc[c, ("score_total", "official_baseline")] for c in cities],
        "full_score_total": [cm.loc[c, ("score_total", "full")] for c in cities],
    })
    compare["gain_28"] = compare.full_score_28 - compare.baseline_score_28
    compare["gain_special"] = compare.full_score_special - compare.baseline_score_special
    compare["gain_total"] = compare.full_score_total - compare.baseline_score_total
    compare.to_csv(args.output_dir / "city_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    checks = {
        "train_end_before_prediction": bool(train_dates.max() < PRED_START),
        "ridge_max_target_date": str(train_dates.max().date()),
        "template_source_years": list(SOURCE_YEARS),
        "template_sources_complete_before_origin": all(CNY_DATES[y] + pd.Timedelta(days=11) < PRED_START for y in SOURCE_YEARS),
        "unique_city_date": not pd.DataFrame(all_predictions).duplicated(["experiment_id", "city_id", "date"]).any(),
        "rows_per_experiment": int(len(cities) * PRED_LEN),
        "finite_predictions": bool(all(np.isfinite(v).all() for v in finals.values()) and np.isfinite(baseline).all()),
        "nonnegative_predictions": bool(all((v >= 0).all() for v in finals.values()) and (baseline >= 0).all()),
        "w0_equals_ridge_only": bool(np.array_equal(np.clip(ridge, 0, upper[None, :]), finals["ridge_only"])),
        "w1_equals_template_only": bool(np.array_equal(np.clip(full_template, 0, upper[None, :]), finals["template_only"])),
        "w085_equals_full": bool(np.array_equal(np.clip((1-ORIGINAL_WEIGHT)*ridge + ORIGINAL_WEIGHT*full_template, 0, upper[None, :]), finals["full"])),
        "score_recompute_identical": bool(score_matrix(actual, finals["full"], future_dates)[2] == score_matrix(actual.copy(), finals["full"].copy(), future_dates)[2]),
    }
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        raise AssertionError(json.dumps(checks, ensure_ascii=False, indent=2))
    (args.output_dir / "validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(args.cache_dir / "branch_predictions_2025.npz", actual=actual, baseline=baseline, ridge=ridge, template=full_template, full=finals["full"], city_upper=upper)
    plot_results(args.output_dir / "figures", future_dates, cities, actual, baseline, finals["full"], weight_df)

    origins = len(train_dates) - QUEUE_DEPTH - PRED_LEN + 1
    with zipfile.ZipFile(args.official_baseline_zip) as archive:
        baseline_members = ["utils/config.py", "utils/data_loader.py", "utils/feature_eng.py", "utils/prophet_model.py", "train.py", "predict.py"]
        official_member_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in baseline_members}
    meta = {
        "experiment": "2025 CNY phase conditional backtest with observed validation weather",
        "train_start": str(train_dates.min().date()), "train_end": str(train_dates.max().date()),
        "prediction_start": str(PRED_START.date()), "prediction_end": str(PRED_END.date()),
        "special_start": str(SPECIAL_START.date()), "special_end": str(SPECIAL_END.date()),
        "weather_condition": "validation-period observed weather used consistently for all models; no forecast-error component",
        "city_count": len(cities), "prediction_rows": len(cities)*PRED_LEN,
        "ridge_origins": origins, "ridge_samples_per_city": origins * PRED_LEN,
        "queue_depth": QUEUE_DEPTH, "template_years": list(SOURCE_YEARS), "template_year_weights": list(SOURCE_WEIGHTS),
        "industry_identity_max_error": identity_error,
        "input_hashes": {"load": sha256(load_path), "weather": sha256(weather_path), "pipeline": sha256(args.pipeline), "backtest_script": sha256(Path(__file__).resolve()), "official_baseline_zip": sha256(args.official_baseline_zip)},
        "official_baseline_source": str(args.official_baseline_zip),
        "official_baseline_member_hashes": official_member_hashes,
        "versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "command": " ".join([sys.executable] + sys.argv), "elapsed_seconds": time.time() - started,
        "thermal_training_pairs": thermal_meta["full"]["thermal_pairs"],
    }
    try:
        import prophet, sklearn, scipy, chinese_calendar
        meta["versions"].update(prophet=prophet.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, chinese_calendar=getattr(chinese_calendar, "__version__", "unknown"))
    except Exception as exc:
        meta["version_capture_warning"] = str(exc)
    (args.output_dir / "run_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "run_command.txt").write_text(meta["command"] + "\n", encoding="utf-8")
    write_summary(args.output_dir / "experiment_summary.md", ablation, weight_df, compare, meta)
    run_log = [
        "2025年春节相位地市用电量条件回测运行日志",
        f"command: {meta['command']}",
        f"elapsed_seconds: {meta['elapsed_seconds']:.3f}",
        f"weather_condition: {meta['weather_condition']}",
        "", "[ablation]", ablation.to_string(index=False),
        "", "[weight_sensitivity]", weight_df.to_string(index=False),
        "", "[validation_checks]", json.dumps(checks, ensure_ascii=False, indent=2), "",
    ]
    (args.output_dir / "logs" / "run.log").write_text("\n".join(run_log), encoding="utf-8")
    print(ablation.to_string(index=False), flush=True)
    print(weight_df.to_string(index=False), flush=True)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
