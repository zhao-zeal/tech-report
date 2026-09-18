import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


PRED_START = pd.Timestamp("2026-02-01")
PRED_END = pd.Timestamp("2026-02-28")
PRED_LEN = 28
QUEUE_DEPTH = 400
RIDGE_ALPHA = 30.0
RIDGE_TARGET = "log"
MODEL_NAME = "city_ridge_cny_multiyear_safe.npz"
MANIFEST_NAME = "city_ridge_cny_multiyear_safe.json"
CNY_DATES = {
    2023: pd.Timestamp("2023-01-22"),
    2024: pd.Timestamp("2024-02-10"),
    2025: pd.Timestamp("2025-01-29"),
    2026: pd.Timestamp("2026-02-17"),
}
CNY_TEMPLATE_YEARS = (2023, 2024, 2025)
CNY_TEMPLATE_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
CNY_SCALE_DAYS = 5
CNY_TEMPLATE_WEIGHT = 0.85
CNY_BLEND_START = PRED_START
CNY_BLEND_END = PRED_END

LOAD_FEATURE_COLS = [
    "load_lag1", "load_lag2", "load_lag3", "load_lag4", "load_lag5",
    "load_lag6", "load_lag7", "load_lag8", "load_lag14", "load_lag21",
    "load_lag28", "load_lag35", "load_lag56", "load_lag364",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
    "roll_max_7", "roll_min_7", "sw_mean_4w", "sw_mean_8w", "wow_diff",
    "mom_diff", "yoy_diff", "ratio_1_7", "ratio_7_28", "wow_ratio",
    "mom_ratio", "yoy_ratio", "sw_ratio",
]

WEATHER_BASE_COLS = [
    "temp_max", "temp_min", "temp_mean", "temp_range", "rhum_mean",
    "precip_sum", "hdd10", "hdd18", "hdd22", "cdd22", "cdd26",
    "cdd28", "temp_d1", "temp_d1_abs", "temp_ewma3", "temp_trend3",
    "temp_max_lag1", "is_rain", "heavy_rain", "temp_x_weekend",
    "temp_x_holiday",
]

SAFE_EXTRA_COLS = [
    "thi_mean", "thi_max", "cdd_thi24", "cdd_thi26", "hdd_thi10",
]

INTERACTION_COLS = [
    "temp_mean_sq", "temp_range_sq", "hdd_x_weekend", "cdd_x_weekend",
    "thi_x_weekend", "thi_x_holiday", "precip_x_weekend",
    "precip_x_holiday", "temp_d1_x_cdd",
]

TRAILING_WEATHER_COLS = [
    "temp_ewma7", "cdd26_cum3", "cdd26_cum7", "hdd18_cum3",
    "hdd18_cum7",
]

# Feature layout matches ml_search.feature_row(..., variant="base") exactly:
# [static_base (calendar + weather), dynamic load, static_ft (THI + inter),
#  static_r2 (trailing weather), cross-city, horizon]
CROSS_COLS = ["cross_prov_lag1", "cross_prov_lag7", "cross_all_lag1",
              "cross_all_lag7"]


def natural_city_key(city_id):
    return int(str(city_id).split("_")[-1])


def script_dir():
    current = Path(__file__).resolve().parent
    # The same file is used both beside train.py and as baseline/utils/...
    return current.parent if current.name == "utils" else current


def model_dir():
    value = Path(os.environ.get(
        "MODEL_DIR", str(script_dir() / "model" / "city_models")
    ))
    value.mkdir(parents=True, exist_ok=True)
    return value


def output_dir():
    value = Path(os.environ.get(
        "OUTPUT_DIR", str(script_dir() / "output" / "city_load_forecasting")
    ))
    value.mkdir(parents=True, exist_ok=True)
    return value


def _candidate_city_roots():
    env = os.environ.get("CITY_DATA_ROOT")
    roots = []
    if env:
        roots.append(Path(env))
    base = script_dir()
    roots.extend([
        base / "datasets" / "city_load_forecasting",
        base.parent / "datasets" / "city_load_forecasting",
        Path.cwd() / "datasets" / "city_load_forecasting",
    ])
    unique = []
    for value in roots:
        value = value.resolve()
        if value not in unique:
            unique.append(value)
    return unique


def find_city_root(require_train):
    checked = []
    for root in _candidate_city_roots():
        checked.append(str(root))
        test_ok = (
            (root / "test_data" / "weather_data" /
             "city_1_7_weather_forecast_data.csv").exists()
            or (root / "test_data" /
                "city_1_7_weather_forecast_data.csv").exists()
        )
        train_ok = (
            (root / "train_data" / "load_data" / "city_load_train.csv").exists()
            and (root / "train_data" / "weather_data" /
                 "city_weather_data.csv").exists()
        )
        if test_ok and (train_ok or not require_train):
            return root
    raise FileNotFoundError(
        "Cannot locate city_load_forecasting. Checked: " + "; ".join(checked)
    )


def forecast_path(root, name):
    nested = root / "test_data" / "weather_data" / name
    direct = root / "test_data" / name
    if nested.exists():
        return nested
    if direct.exists():
        return direct
    raise FileNotFoundError(name)


def _calendar_features(dates):
    try:
        from chinese_calendar import is_holiday, is_workday
        from chinese_calendar.constants import holidays
    except Exception as exc:
        raise RuntimeError(
            "chinese_calendar is required by the official baseline image"
        ) from exc

    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=idx)
    out["dow"] = idx.dayofweek.astype(np.float32)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(np.float32)
    out["day"] = idx.day.astype(np.float32)
    out["month"] = idx.month.astype(np.float32)
    out["dayofyear"] = idx.dayofyear.astype(np.float32)
    out["weekofyear"] = idx.isocalendar().week.to_numpy(dtype=np.float32)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)
    out["day_index"] = (idx - pd.Timestamp("2023-01-01")).days.astype(np.float32)
    out["year"] = idx.year.astype(np.float32)

    py_dates = [value.date() for value in idx]
    out["is_holiday"] = np.asarray([is_holiday(d) for d in py_dates], np.float32)
    out["is_workday"] = np.asarray([is_workday(d) for d in py_dates], np.float32)
    out["is_adjusted_workday"] = np.asarray([
        is_workday(d) and not is_holiday(d) and d.weekday() >= 5
        for d in py_dates
    ], np.float32)
    holiday_names = [holidays.get(d, "none") for d in py_dates]

    meta = [
        ("Spring Festival", "sf"), ("National Day", "national"),
        ("Labour Day", "labour"), ("Tomb-sweeping Day", "qingming"),
        ("Dragon Boat Festival", "dragon"),
        ("Mid-autumn Festival", "midautumn"),
        ("New Year's Day", "newyear"),
    ]
    name_dates = {}
    for day, name in holidays.items():
        name_dates.setdefault(name, []).append(pd.Timestamp(day))
    for name, prefix in meta:
        out["hol_" + prefix] = np.asarray(
            [item == name for item in holiday_names], np.float32
        )
        anchors = pd.DatetimeIndex(sorted(set(name_dates.get(name, []))))
        if len(anchors) == 0:
            out["days_to_" + prefix] = 30.0
        else:
            out["days_to_" + prefix] = np.asarray([
                min(((value - anchor).days for anchor in anchors), key=abs)
                for value in idx
            ], np.float32)

    for prefix in ["sf", "national", "labour", "qingming", "dragon", "midautumn"]:
        distance = out["days_to_" + prefix]
        out["pre_" + prefix + "_1"] = (distance == 1).astype(np.float32)
        out["pre_" + prefix + "_2_3"] = (
            (distance >= 2) & (distance <= 3)
        ).astype(np.float32)
        out["pre_" + prefix + "_4_7"] = (
            (distance >= 4) & (distance <= 7)
        ).astype(np.float32)
        out["post_" + prefix + "_1"] = (distance == -1).astype(np.float32)
        out["post_" + prefix + "_2_3"] = (
            (distance <= -2) & (distance >= -3)
        ).astype(np.float32)
        out["post_" + prefix + "_4_7"] = (
            (distance <= -4) & (distance >= -7)
        ).astype(np.float32)

    sf_dist = out["days_to_sf"]
    out["chunyun"] = (np.abs(sf_dist) <= 14).astype(np.float32)
    out["chunyun_pre"] = ((sf_dist > 0) & (sf_dist <= 14)).astype(np.float32)
    out["chunyun_post"] = ((sf_dist < 0) & (sf_dist >= -14)).astype(np.float32)
    result = out.reset_index(names="date")
    return result, [column for column in result.columns if column != "date"]


def load_train_load(root):
    path = root / "train_data" / "load_data" / "city_load_train.csv"
    frame = pd.read_csv(path, usecols=["province_id", "city_id", "industry_id", "date", "load"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["industry_id"] == 1].copy()
    frame = frame.sort_values(["date", "city_id"])
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    cities = sorted(frame["city_id"].unique(), key=natural_city_key)
    pivot = frame.pivot(index="date", columns="city_id", values="load").reindex(
        index=dates, columns=cities
    )
    if pivot.isna().any().any():
        raise ValueError("city load matrix contains missing cells")
    province = (
        frame[["city_id", "province_id"]].drop_duplicates()
        .set_index("city_id")["province_id"].to_dict()
    )
    if dates[0] != pd.Timestamp("2023-01-01") or dates[-1] != pd.Timestamp("2026-01-31"):
        raise ValueError("unexpected training date range")
    if len(cities) != 10:
        raise ValueError("expected 10 cities")
    return dates, cities, pivot.to_numpy(np.float64), province


def load_observed_weather(root):
    path = root / "train_data" / "weather_data" / "city_weather_data.csv"
    columns = ["CITY_ID", "DATETIME", "TEM", "TEM_MAX", "TEM_MIN", "RHUM", "PRECIPITATION"]
    frame = pd.read_csv(path, usecols=columns)
    frame["date"] = pd.to_datetime(frame["DATETIME"].astype(str).str.slice(0, 10))
    for column in ["TEM", "TEM_MAX", "TEM_MIN", "RHUM", "PRECIPITATION"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(9999, np.nan)
    daily = (
        frame.groupby(["CITY_ID", "date"], as_index=False)
        .agg(
            temp_max=("TEM_MAX", "max"), temp_min=("TEM_MIN", "min"),
            temp_mean=("TEM", "mean"), rhum_mean=("RHUM", "mean"),
            precip_sum=("PRECIPITATION", "sum"),
        )
        .rename(columns={"CITY_ID": "city_id"})
        .sort_values(["city_id", "date"])
    )
    value_cols = ["temp_max", "temp_min", "temp_mean", "rhum_mean", "precip_sum"]
    for column in value_cols:
        daily[column] = daily.groupby("city_id")[column].ffill().bfill()
    if daily[value_cols].isna().any().any():
        raise ValueError("observed weather still contains missing values")
    return daily[["city_id", "date"] + value_cols].reset_index(drop=True)


def _melt_wide(frame, start_week, end_week):
    text = str(frame["FILE_BEGIN_TIME"].iloc[0])[:8]
    base = pd.to_datetime(text, format="%Y%m%d")
    parts = []
    for week in range(start_week, end_week + 1):
        part = frame[["CITY_ID", "MAX_TEMP" + str(week), "MIN_TEMP" + str(week)]].copy()
        part.columns = ["city_id", "temp_max", "temp_min"]
        part["date"] = base + pd.Timedelta(days=week - start_week)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def load_future_weather(root, observed):
    short = pd.read_csv(forecast_path(root, "city_1_7_weather_forecast_data.csv"))
    if len(short) != 3024:
        raise ValueError("unexpected 1-7 day row count: " + str(len(short)))
    short["DDATETIME"] = pd.to_datetime(short["DDATETIME"])
    short["actual_time"] = short["DDATETIME"] + pd.to_timedelta(
        pd.to_numeric(short["PREDICTION_TIME"]), unit="h"
    )
    short["date"] = short["actual_time"].dt.normalize()
    for column in ["TEM", "TEM_MAX", "TEM_MIN", "RHUM", "PRECIPITATION"]:
        short[column] = pd.to_numeric(short[column], errors="coerce")
    short[["TEM_MAX", "TEM_MIN"]] = (
        short[["TEM_MAX", "TEM_MIN"]].replace(9999, np.nan).ffill().bfill()
    )
    short_daily = (
        short.groupby(["CITY_ID", "date"], as_index=False)
        .agg(
            temp_max=("TEM_MAX", "max"), temp_min=("TEM_MIN", "min"),
            temp_mean=("TEM", "mean"), rhum_mean=("RHUM", "mean"),
            precip_sum=("PRECIPITATION", "sum"),
        )
        .rename(columns={"CITY_ID": "city_id"})
    )

    medium = pd.read_csv(forecast_path(root, "city_8_15_weather_forecast_data.csv"))
    long = pd.read_csv(forecast_path(root, "city_16_45_weather_forecast_data.csv"))
    if len(medium) != 53 or len(long) != 53:
        raise ValueError("unexpected wide forecast row count")
    medium_daily = _melt_wide(medium, 8, 15)
    long_daily = _melt_wide(long, 16, 28)
    wide = pd.concat([medium_daily, long_daily], ignore_index=True)
    wide["temp_max"] = pd.to_numeric(wide["temp_max"], errors="coerce")
    wide["temp_min"] = pd.to_numeric(wide["temp_min"], errors="coerce")
    wide = wide.groupby(["city_id", "date"], as_index=False).mean(numeric_only=True)
    wide["temp_mean"] = (wide["temp_max"] + wide["temp_min"]) / 2.0
    wide["rhum_mean"] = np.nan
    wide["precip_sum"] = np.nan

    future = pd.concat([short_daily, wide], ignore_index=True)
    future = future[(future["date"] >= PRED_START) & (future["date"] <= PRED_END)]
    future = future.groupby(["city_id", "date"], as_index=False).mean(numeric_only=True)

    climate_source = observed.copy()
    climate_source["month"] = climate_source["date"].dt.month
    climate = climate_source.groupby(["city_id", "month"], as_index=False).agg(
        climate_rhum=("rhum_mean", "mean"),
        climate_precip=("precip_sum", "mean"),
    )
    future["month"] = future["date"].dt.month
    future = future.merge(climate, on=["city_id", "month"], how="left")
    future["rhum_mean"] = future["rhum_mean"].fillna(future["climate_rhum"])
    future["precip_sum"] = future["precip_sum"].fillna(future["climate_precip"])
    future = future.drop(columns=["month", "climate_rhum", "climate_precip"])

    expected_dates = pd.date_range(PRED_START, PRED_END, freq="D")
    cities = sorted(observed["city_id"].unique(), key=natural_city_key)
    expected = pd.MultiIndex.from_product([cities, expected_dates], names=["city_id", "date"])
    actual = pd.MultiIndex.from_frame(future[["city_id", "date"]])
    if len(future) != 280 or not expected.equals(actual.sort_values()):
        future = future.set_index(["city_id", "date"]).reindex(expected).reset_index()
    value_cols = ["temp_max", "temp_min", "temp_mean", "rhum_mean", "precip_sum"]
    future = future.sort_values(["city_id", "date"])
    for column in value_cols:
        future[column] = future.groupby("city_id")[column].ffill().bfill()
    if len(future) != 280 or future[value_cols].isna().any().any():
        raise ValueError("future daily weather coverage check failed")
    return future[["city_id", "date"] + value_cols].reset_index(drop=True)


def build_static_features(core_weather):
    """Build calendar/weather/THI/interaction/trailing features on a daily
    (city_id, date, temp_max, temp_min, temp_mean, rhum_mean, precip_sum)
    frame.  Returns the frame and the three static column groups used by the
    pooled feature layout (static_base, static_ft, static_r2)."""
    frame = core_weather.sort_values(["city_id", "date"]).copy()
    core_cols = ["temp_max", "temp_min", "temp_mean", "rhum_mean", "precip_sum"]
    for column in core_cols:
        frame[column] = frame.groupby("city_id")[column].ffill().bfill()
    calendar, calendar_cols = _calendar_features(sorted(frame["date"].unique()))
    frame = frame.merge(calendar, on="date", how="left")
    group = frame.groupby("city_id", group_keys=False)

    frame["temp_range"] = frame["temp_max"] - frame["temp_min"]
    frame["hdd10"] = np.maximum(0.0, 10.0 - frame["temp_mean"])
    frame["hdd18"] = np.maximum(0.0, 18.0 - frame["temp_mean"])
    frame["hdd22"] = np.maximum(0.0, 22.0 - frame["temp_mean"])
    frame["cdd22"] = np.maximum(0.0, frame["temp_mean"] - 22.0)
    frame["cdd26"] = np.maximum(0.0, frame["temp_mean"] - 26.0)
    frame["cdd28"] = np.maximum(0.0, frame["temp_mean"] - 28.0)
    frame["temp_d1"] = group["temp_mean"].diff()
    frame["temp_d1_abs"] = frame["temp_d1"].abs()
    frame["temp_ewma3"] = group["temp_mean"].transform(lambda s: s.ewm(span=3).mean())
    rolling3 = group["temp_mean"].transform(lambda s: s.rolling(3).mean())
    frame["temp_trend3"] = rolling3 - rolling3.groupby(frame["city_id"]).shift(3)
    frame["temp_max_lag1"] = group["temp_max"].shift(1)
    frame["is_rain"] = (frame["precip_sum"] > 0.5).astype(np.float32)
    frame["heavy_rain"] = (frame["precip_sum"] > 10.0).astype(np.float32)
    frame["temp_x_weekend"] = frame["temp_mean"] * frame["is_weekend"]
    frame["temp_x_holiday"] = frame["temp_mean"] * frame["is_holiday"]

    relative_humidity = frame["rhum_mean"] / 100.0
    frame["thi_mean"] = frame["temp_mean"] - 0.55 * (1 - relative_humidity) * (frame["temp_mean"] - 14.5)
    frame["thi_max"] = frame["temp_max"] - 0.55 * (1 - relative_humidity) * (frame["temp_max"] - 14.5)
    frame["cdd_thi24"] = np.maximum(0.0, frame["thi_mean"] - 24.0)
    frame["cdd_thi26"] = np.maximum(0.0, frame["thi_mean"] - 26.0)
    frame["hdd_thi10"] = np.maximum(0.0, 10.0 - frame["thi_mean"])

    frame["temp_mean_sq"] = frame["temp_mean"] ** 2
    frame["temp_range_sq"] = frame["temp_range"] ** 2
    frame["hdd_x_weekend"] = frame["hdd18"] * frame["is_weekend"]
    frame["cdd_x_weekend"] = frame["cdd26"] * frame["is_weekend"]
    frame["thi_x_weekend"] = frame["thi_mean"] * frame["is_weekend"]
    frame["thi_x_holiday"] = frame["thi_mean"] * frame["is_holiday"]
    frame["precip_x_weekend"] = frame["precip_sum"] * frame["is_weekend"]
    frame["precip_x_holiday"] = frame["precip_sum"] * frame["is_holiday"]
    frame["temp_d1_x_cdd"] = frame["temp_d1_abs"] * frame["cdd26"]

    frame["temp_ewma7"] = group["temp_mean"].transform(lambda s: s.ewm(span=7).mean())
    frame["cdd26_cum3"] = group["cdd26"].transform(lambda s: s.rolling(3).sum())
    frame["cdd26_cum7"] = group["cdd26"].transform(lambda s: s.rolling(7).sum())
    frame["hdd18_cum3"] = group["hdd18"].transform(lambda s: s.rolling(3).sum())
    frame["hdd18_cum7"] = group["hdd18"].transform(lambda s: s.rolling(7).sum())

    static_base_cols = calendar_cols + WEATHER_BASE_COLS
    static_ft_cols = SAFE_EXTRA_COLS + INTERACTION_COLS
    static_r2_cols = TRAILING_WEATHER_COLS
    all_static = static_base_cols + static_ft_cols + static_r2_cols
    for column in all_static:
        frame[column] = frame.groupby("city_id")[column].ffill().bfill().fillna(0.0)
    return frame, static_base_cols, static_ft_cols, static_r2_cols


def load_feature_vector(queue):
    q = list(np.asarray(queue, np.float64))
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 14, 21, 28, 35, 56, 364]
    result = []
    for lag in lags:
        result.append(q[-lag] if len(q) >= lag else q[-1])
    def mean_tail(size):
        return float(np.mean(q[-size:]))
    segment7 = q[-7:]
    roll7 = mean_tail(7)
    roll14 = mean_tail(14)
    roll28 = mean_tail(28)
    result.extend([
        roll7, roll14, roll28, float(np.std(segment7)),
        float(np.max(segment7)), float(np.min(segment7)),
    ])
    sw4 = float(np.mean([q[-7 * i] for i in range(1, 5)]))
    sw8 = float(np.mean([q[-7 * i] for i in range(1, 9)]))
    wow_diff = q[-7] - q[-14]
    mom_diff = q[-28] - q[-56]
    yoy_diff = q[-1] - q[-364]
    eps = 1e-6
    result.extend([
        sw4, sw8, wow_diff, mom_diff, yoy_diff,
        q[-1] / (roll7 + eps), q[-7] / (roll28 + eps),
        q[-7] / (q[-14] + eps), q[-28] / (q[-56] + eps),
        q[-1] / (q[-364] + eps), sw4 / (roll28 + eps),
    ])
    return np.asarray(result, np.float64)


def cross_vector(load_matrix, origin, city_index, cities, province):
    city = cities[city_index]
    partners = [
        index for index, other in enumerate(cities)
        if other != city and province[other] == province[city]
    ]
    others = [index for index in range(len(cities)) if index != city_index]
    source = partners if partners else others
    return np.asarray([
        load_matrix[origin - 1, source].mean(),
        load_matrix[origin - 7, source].mean(),
        load_matrix[origin - 1, others].mean(),
        load_matrix[origin - 7, others].mean(),
    ], np.float64)


def fit_ridge_pooled(x, y, alpha, mode):
    """Single Ridge fit on the pooled sample matrix (already standardized
    internally).  For mode="log" the target is log10(load) and the returned
    bias is the log-space intercept; the smear factor converts the exponent
    back to level."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    scaled = (x - mean) / scale
    if mode == "log":
        target = np.log10(np.maximum(y, 1e-6))
    else:
        target = y.astype(np.float64)
    target_mean = float(target.mean())
    gram = scaled.T @ scaled
    rhs = scaled.T @ (target - target_mean)
    coefficient = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64), rhs
    )
    smear = 1.0
    if mode == "log":
        fitted = target_mean + scaled @ coefficient
        smear = float(np.mean(10 ** (target - fitted)))
    return mean, scale, coefficient, target_mean, smear


def _static_arrays(static, dates, cities, static_cols):
    arrays = {}
    for city in cities:
        part = static[static["city_id"] == city].set_index("date").reindex(dates)
        if part[static_cols].isna().any().any():
            raise ValueError("missing static feature for " + city)
        arrays[city] = part[static_cols].to_numpy(np.float64)
    return arrays


def feature_names(static_base_cols, static_ft_cols, static_r2_cols):
    return (
        static_base_cols + LOAD_FEATURE_COLS + static_ft_cols
        + static_r2_cols + CROSS_COLS + ["horizon"]
    )


def _feature_row(static_base, static_ft, static_r2, dynamic, cross, horizon):
    return np.concatenate([
        static_base, dynamic, static_ft, static_r2, cross,
        np.asarray([horizon], np.float64),
    ])


def build_cny_template(dates, load_matrix):
    """Build a compact city-by-horizon phase template from known history."""
    positions = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    phase_start = int((PRED_START - CNY_DATES[2026]).days)
    template = np.zeros((load_matrix.shape[1], PRED_LEN), np.float64)
    for year, weight in zip(CNY_TEMPLATE_YEARS, CNY_TEMPLATE_WEIGHTS):
        source_origin_date = CNY_DATES[year] + pd.Timedelta(days=phase_start)
        if source_origin_date not in positions:
            raise ValueError("CNY template origin unavailable: " + str(source_origin_date.date()))
        source_origin = positions[source_origin_date]
        if source_origin < CNY_SCALE_DAYS or source_origin + PRED_LEN > len(dates):
            raise ValueError("CNY template window crosses training boundary")
        scale = load_matrix[
            source_origin - CNY_SCALE_DAYS:source_origin
        ].mean(axis=0)
        if (scale <= 0).any():
            raise ValueError("non-positive CNY template scale")
        ratio = (
            load_matrix[source_origin:source_origin + PRED_LEN] / scale[None, :]
        ).T
        template += float(weight) * ratio
    return template


def _train_pooled(dates, cities, load_matrix, province, observed, save=True):
    """Train one pooled Ridge per city on all (origin, horizon) samples and
    return the model arrays.  Saves the npz + manifest when save=True."""
    started = time.time()
    static, static_base_cols, static_ft_cols, static_r2_cols = build_static_features(observed)
    base_arrays = _static_arrays(static, dates, cities, static_base_cols)
    ft_arrays = _static_arrays(static, dates, cities, static_ft_cols)
    r2_arrays = _static_arrays(static, dates, cities, static_r2_cols)
    n_days = len(dates)
    origins = np.arange(QUEUE_DEPTH, n_days - PRED_LEN + 1, dtype=np.int64)
    names = feature_names(static_base_cols, static_ft_cols, static_r2_cols)
    n_features = len(names)
    feature_mean = np.empty((len(cities), n_features), np.float64)
    feature_scale = np.empty((len(cities), n_features), np.float64)
    coefficient = np.empty((len(cities), n_features), np.float64)
    intercept = np.empty(len(cities), np.float64)
    smear = np.empty(len(cities), np.float64)

    for city_index, city in enumerate(cities):
        dynamic = np.stack([
            load_feature_vector(load_matrix[origin - QUEUE_DEPTH:origin, city_index])
            for origin in origins
        ])
        cross = np.stack([
            cross_vector(load_matrix, origin, city_index, cities, province)
            for origin in origins
        ])
        samples = []
        targets = []
        for position, origin in enumerate(origins):
            row_dynamic = dynamic[position]
            row_cross = cross[position]
            for horizon in range(PRED_LEN):
                target_index = origin + horizon
                samples.append(_feature_row(
                    base_arrays[city][target_index],
                    ft_arrays[city][target_index],
                    r2_arrays[city][target_index],
                    row_dynamic, row_cross, horizon,
                ))
                targets.append(load_matrix[target_index, city_index])
        x = np.asarray(samples, np.float64)
        y = np.asarray(targets, np.float64)
        mean, scale, coef, bias, sm = fit_ridge_pooled(x, y, RIDGE_ALPHA, RIDGE_TARGET)
        feature_mean[city_index] = mean
        feature_scale[city_index] = scale
        coefficient[city_index] = coef
        intercept[city_index] = bias
        smear[city_index] = sm
        print("[train] %s complete (n=%d)" % (city, len(samples)), flush=True)

    core_cols = ["temp_max", "temp_min", "temp_mean", "rhum_mean", "precip_sum"]
    observed_sorted = observed.sort_values(["date", "city_id"])
    observed_dates = pd.DatetimeIndex(sorted(observed_sorted["date"].unique()))
    observed_core = np.stack([
        observed[observed["city_id"] == city].set_index("date")
        .reindex(observed_dates)[core_cols].to_numpy(np.float64)
        for city in cities
    ], axis=1)

    city_upper = np.nanquantile(load_matrix, 0.999, axis=0) * 1.5
    cny_template = build_cny_template(dates, load_matrix)
    data = {
        "cities": np.asarray(cities, dtype="U32"),
        "province": np.asarray([province[city] for city in cities], dtype="U32"),
        "feature_names": np.asarray(names, dtype="U64"),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coefficient": coefficient,
        "intercept": intercept,
        "smear": smear,
        "last_load": load_matrix[-QUEUE_DEPTH:].astype(np.float64),
        "cny_template": cny_template,
        "cny_template_years": np.asarray(CNY_TEMPLATE_YEARS, dtype=np.int16),
        "cny_template_weights": np.asarray(CNY_TEMPLATE_WEIGHTS, np.float64),
        "city_upper": city_upper,
        "observed_dates": np.asarray(
            [value.strftime("%Y-%m-%d") for value in observed_dates], dtype="U10"),
        "observed_core": observed_core,
    }
    if save:
        destination = model_dir() / MODEL_NAME
        np.savez_compressed(destination, **{
            key: (value.astype(np.float32) if value.dtype == np.float64
                  else value)
            for key, value in data.items()
        })
        manifest = {
            "model": "per-city pooled direct Ridge (log10 target + smear) "
                     "with full-month multi-year CNY template blend",
            "pooled": True,
            "alpha": RIDGE_ALPHA,
            "target": RIDGE_TARGET,
            "train_start": str(dates[0].date()),
            "train_end": str(dates[-1].date()),
            "prediction_start": str(PRED_START.date()),
            "prediction_end": str(PRED_END.date()),
            "cities": cities,
            "models": len(cities),
            "features": n_features,
            "training_samples_per_city": len(origins) * PRED_LEN,
            "future_weather_used_for_training": False,
            "future_weather_used_for_prediction": True,
            "cny_template_years": list(CNY_TEMPLATE_YEARS),
            "cny_template_weights": list(CNY_TEMPLATE_WEIGHTS),
            "cny_template_scale_days": CNY_SCALE_DAYS,
            "cny_template_blend_weight": CNY_TEMPLATE_WEIGHT,
            "cny_blend_start": str(CNY_BLEND_START.date()),
            "cny_blend_end": str(CNY_BLEND_END.date()),
            "elapsed_seconds": time.time() - started,
        }
        with open(model_dir() / MANIFEST_NAME, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print("[saved]", destination, flush=True)
        print(json.dumps(manifest, indent=2), flush=True)
    return data


def train_city_model():
    root = find_city_root(require_train=True)
    print("[data]", root, flush=True)
    dates, cities, load_matrix, province = load_train_load(root)
    observed = load_observed_weather(root)
    _train_pooled(dates, cities, load_matrix, province, observed, save=True)


def _observed_from_model(data):
    cities = [str(value) for value in data["cities"]]
    dates = pd.to_datetime(data["observed_dates"].astype(str))
    core = data["observed_core"].astype(np.float64)
    columns = ["temp_max", "temp_min", "temp_mean", "rhum_mean", "precip_sum"]
    rows = []
    for city_index, city in enumerate(cities):
        part = pd.DataFrame(core[:, city_index, :], columns=columns)
        part.insert(0, "date", dates)
        part.insert(0, "city_id", city)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def cny_template_predictions(data, future_dates, cities):
    """Scale the stored multi-year phase template by the latest five days."""
    last_load = data["last_load"].astype(np.float64)
    template = data["cny_template"].astype(np.float64)
    if template.shape != (len(cities), len(future_dates)):
        raise ValueError("CNY template shape mismatch")
    recent_scale = last_load[-CNY_SCALE_DAYS:].mean(axis=0)
    if (recent_scale <= 0).any():
        raise ValueError("non-positive recent CNY scale")
    return template * recent_scale[:, None]


def _predict_pooled(data, cities, province, observed, future, save_output=True,
                    pred_start=None, pred_end=None, template_weight=None):
    if pred_start is None:
        pred_start = PRED_START
    if pred_end is None:
        pred_end = PRED_END
    if template_weight is None:
        template_weight = CNY_TEMPLATE_WEIGHT
    all_core = pd.concat([observed, future], ignore_index=True)
    all_core = all_core.drop_duplicates(["city_id", "date"], keep="last")
    static, static_base_cols, static_ft_cols, static_r2_cols = build_static_features(all_core)
    names = feature_names(static_base_cols, static_ft_cols, static_r2_cols)
    stored_names = [str(value) for value in data["feature_names"]]
    if stored_names != names:
        raise ValueError("feature schema mismatch")
    future_dates = pd.date_range(pred_start, pred_end, freq="D")
    base_future = _static_arrays(static, future_dates, cities, static_base_cols)
    ft_future = _static_arrays(static, future_dates, cities, static_ft_cols)
    r2_future = _static_arrays(static, future_dates, cities, static_r2_cols)
    last_load = data["last_load"].astype(np.float64)
    feature_mean = data["feature_mean"].astype(np.float64)
    feature_scale = data["feature_scale"].astype(np.float64)
    coefficient = data["coefficient"].astype(np.float64)
    intercept = data["intercept"].astype(np.float64)
    smear = data["smear"].astype(np.float64)
    city_upper = data["city_upper"].astype(np.float64)
    template = cny_template_predictions(data, future_dates, cities)
    origin = last_load.shape[0]
    rows = []
    for city_index, city in enumerate(cities):
        dynamic = load_feature_vector(last_load[:, city_index])
        cross = cross_vector(last_load, origin, city_index, cities, province)
        for horizon, date in enumerate(future_dates):
            x = _feature_row(
                base_future[city][horizon],
                ft_future[city][horizon],
                r2_future[city][horizon],
                dynamic, cross, horizon,
            )
            scaled = (x - feature_mean[city_index]) / feature_scale[city_index]
            log_value = float(
                intercept[city_index] + scaled @ coefficient[city_index]
            )
            if RIDGE_TARGET == "log":
                ridge_prediction = float((10 ** log_value) * smear[city_index])
            else:
                ridge_prediction = log_value
            if CNY_BLEND_START <= date <= CNY_BLEND_END:
                prediction = float(
                    (1.0 - template_weight) * ridge_prediction
                    + template_weight * template[city_index, horizon]
                )
            else:
                prediction = ridge_prediction
            prediction = float(np.clip(prediction, 0.0, city_upper[city_index]))
            rows.append((city, date.strftime("%Y/%m/%d"), prediction))
    result = pd.DataFrame(rows, columns=["city_id", "date", "pred"])
    if len(result) != 280 or result["pred"].isna().any() or (result["pred"] < 0).any():
        raise ValueError("prediction QA failed")
    if save_output:
        destination = output_dir() / "submit_result.csv"
        result.to_csv(destination, index=False, encoding="utf-8-sig")
        print("[saved]", destination, result.shape, flush=True)
    return result


def predict_city_model(save_output=True):
    model_path = model_dir() / MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError("Run train.py first: " + str(model_path))
    data = np.load(model_path, allow_pickle=False)
    cities = [str(value) for value in data["cities"]]
    province = {
        city: str(value) for city, value in zip(cities, data["province"])
    }
    observed = _observed_from_model(data)
    root = find_city_root(require_train=False)
    future = load_future_weather(root, observed)
    return _predict_pooled(data, cities, province, observed, future, save_output)


def check_data():
    root = find_city_root(require_train=True)
    dates, cities, load_matrix, _ = load_train_load(root)
    observed = load_observed_weather(root)
    future = load_future_weather(root, observed)
    report = {
        "root": str(root),
        "train_dates": [str(dates[0].date()), str(dates[-1].date())],
        "cities": cities,
        "load_shape": list(load_matrix.shape),
        "observed_weather_rows": len(observed),
        "future_weather_rows": len(future),
        "future_nulls": int(future.isna().sum().sum()),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


# This section extends the frozen multi-year city pipeline above. The Ridge
# features, training algorithm, 85/15 blend, upper cap and weather loader stay
# byte-for-byte identical in the copied base section.

MODEL_NAME = "city_cny_industry_thermal_c4.npz"
MANIFEST_NAME = "city_cny_industry_thermal_c4.json"
THERMAL_SCHEMA = 4
THERMAL_ALPHA = 30.0
THERMAL_GAMMA = 0.75
THERMAL_INDUSTRY_WEIGHT = 0.5
CNY_SMOOTH_WINDOW = 7
THERMAL_MEMORY = 0.25
CNY_RETURN_WARP = 0.5
CNY_SMOOTH_EXCLUDE_PHASES = (-2, 6)
THERMAL_INDUSTRIES = (1, 3, 4, 5, 6, 9, 10)


def _thermal_features(temperature):
    temp = np.asarray(temperature, np.float64)
    return np.stack([
        np.maximum(0, 10 - temp) / 10,
        np.maximum(0, 18 - temp) / 10,
        np.maximum(0, temp - 22) / 10,
        np.maximum(0, temp - 26) / 10,
    ], axis=-1)


def _thermal_effective_temperature(temperature, memory):
    temp = np.asarray(temperature, np.float64)
    if temp.ndim != 2 or len(temp) < 2 or not np.isfinite(temp).all():
        raise ValueError("temperature memory requires a finite daily city matrix")
    lag1 = np.concatenate([temp[:1], temp[:-1]])
    lag2 = np.concatenate([temp[:1], temp[:1], temp[:-2]])
    return (1-memory)*temp + memory*(lag1+lag2)/2


def _cny_return_phase(year):
    from chinese_calendar.constants import holidays

    dates = [pd.Timestamp(date) for date, name in holidays.items()
             if date.year == year and name == "Spring Festival"]
    if not dates:
        raise ValueError("Spring Festival calendar missing for " + str(year))
    return int((max(dates) - CNY_DATES[year]).days) + 1


def _smooth_historical_curve(history, positions):
    radius = CNY_SMOOTH_WINDOW // 2
    if positions.min()-radius < 0 or positions.max()+radius >= len(history):
        raise ValueError("seven-day CNY curve crosses historical boundary")
    return sum(history[positions+offset] / CNY_SMOOTH_WINDOW
               for offset in range(-radius, radius+1))


def _load_industry_cube(root, dates, cities):
    frame = pd.read_csv(root / "train_data" / "load_data" / "city_load_train.csv",
                        usecols=["city_id", "date", "industry_id", "load"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["industry_id"].isin(THERMAL_INDUSTRIES)]
    values = []
    for city in cities:
        part = frame[frame["city_id"] == city].pivot(
            index="date", columns="industry_id", values="load"
        ).reindex(index=dates, columns=list(THERMAL_INDUSTRIES))
        values.append(part.to_numpy(np.float64))
    cube = np.stack(values, axis=1)
    if not np.isfinite(cube).all() or (cube < 0).any():
        raise ValueError("industry load matrix must be complete, finite and nonnegative")
    return cube


def _fit_thermal_upgrade(dates, cities, cube, observed, prediction_start=None,
                         source_years=None):
    from chinese_calendar import is_holiday

    start = PRED_START if prediction_start is None else pd.Timestamp(prediction_start)
    source_years = tuple(CNY_TEMPLATE_YEARS) if source_years is None else tuple(source_years)
    if dates[-1] >= start:
        raise ValueError("thermal training includes dates at or after prediction origin")
    temps = np.stack([
        observed[observed["city_id"] == city].set_index("date")
        .reindex(dates)["temp_mean"].to_numpy(np.float64)
        for city in cities
    ], axis=1)
    if not np.isfinite(temps).all():
        raise ValueError("thermal history has missing temperatures")
    ordinary = np.asarray([
        not is_holiday(date.date()) and
        abs((date - CNY_DATES[date.year]).days) > 30
        for date in dates
    ])
    mask = ordinary[7:] & ordinary[:-7]
    if mask.sum() < 30:
        raise ValueError("insufficient ordinary historical weeks for thermal fit")
    feat = _thermal_features(_thermal_effective_temperature(temps, THERMAL_MEMORY))
    log_load = np.log(np.maximum(cube, 1e-7))
    dx = feat[7:] - feat[:-7]
    dy = log_load[7:] - log_load[:-7]
    coefficients = []
    for city_index in range(len(cities)):
        x = dx[mask, city_index]
        target = np.clip(dy[mask, city_index], -0.5, 0.5)
        coefficients.append(np.linalg.solve(
            x.T @ x + THERMAL_ALPHA * np.eye(x.shape[1]), x.T @ target
        ))
    coefficients = np.asarray(coefficients, np.float64)
    response = np.exp(THERMAL_GAMMA * np.einsum(
        "dcf,cfj->dcj", feat, coefficients
    ))
    neutral = cube / response
    recent = neutral[-CNY_SCALE_DAYS:].mean(axis=0)
    phase = int((start - CNY_DATES[start.year]).days)
    ratios = []
    positions = {pd.Timestamp(date): i for i, date in enumerate(dates)}
    for year in source_years:
        source = positions[CNY_DATES[year] + pd.Timedelta(days=phase)]
        if source < CNY_SCALE_DAYS or source + PRED_LEN > len(dates):
            raise ValueError("industry CNY source crosses historical boundary")
        level = neutral[source-CNY_SCALE_DAYS:source].mean(axis=0)
        indices = np.arange(source, source+PRED_LEN)
        values = neutral[indices].copy()
        phases = np.arange(phase, phase+PRED_LEN)
        outside = ((phases < CNY_SMOOTH_EXCLUDE_PHASES[0]) |
                   (phases > CNY_SMOOTH_EXCLUDE_PHASES[1]))
        smoothed = _smooth_historical_curve(neutral, indices)
        values[outside] = smoothed[outside]
        # Blend lunar alignment with alignment to the first day after the
        # statutory CNY holiday. Only phases +7 onward receive this adjustment.
        shift = _cny_return_phase(year) - _cny_return_phase(start.year)
        moved = _smooth_historical_curve(neutral, indices+shift)
        gate = np.clip((phases-6)/3, 0, 1) * CNY_RETURN_WARP
        values = (1-gate[:, None, None])*values + gate[:, None, None]*moved
        ratio = values / np.maximum(level, 1e-9)
        ratios.append(np.where(level[None] > 1e-8, ratio, 1.0))
    neutral_forecast = np.mean(ratios, axis=0) * recent
    return {
        "thermal_schema": np.asarray(THERMAL_SCHEMA, np.int16),
        "thermal_industries": np.asarray(THERMAL_INDUSTRIES, np.int16),
        "thermal_coefficients": coefficients,
        "thermal_neutral_forecast": neutral_forecast,
        "thermal_gamma": np.asarray(THERMAL_GAMMA, np.float64),
        "thermal_smoothing_window": np.asarray(CNY_SMOOTH_WINDOW, np.int16),
        "thermal_memory": np.asarray(THERMAL_MEMORY, np.float64),
        "thermal_temperature_tail": temps[-2:].copy(),
        "thermal_return_warp": np.asarray(CNY_RETURN_WARP, np.float64),
        "thermal_return_phases": np.asarray([
            (year, _cny_return_phase(year)) for year in source_years + (start.year,)
        ], np.int16),
        "thermal_smoothing_excluded_phases": np.asarray(CNY_SMOOTH_EXCLUDE_PHASES, np.int16),
        "thermal_industry_weight": np.asarray(THERMAL_INDUSTRY_WEIGHT, np.float64),
        "thermal_training_pairs": np.asarray(mask.sum(), np.int64),
        "thermal_prediction_start": np.asarray(str(start.date()), dtype="U10"),
    }


def _thermal_template(data, future, cities):
    if "thermal_schema" not in data or int(data["thermal_schema"]) != THERMAL_SCHEMA:
        raise ValueError("C4 requires its own platform-trained model; old models are incompatible")
    start = pd.Timestamp(str(data["thermal_prediction_start"]))
    future_dates = pd.date_range(start, periods=PRED_LEN)
    temp = np.stack([
        future[future["city_id"] == city].set_index("date")
        .reindex(future_dates)["temp_mean"].to_numpy(np.float64)
        for city in cities
    ], axis=1)
    if not np.isfinite(temp).all() or (np.abs(temp) > 100).any():
        raise ValueError("invalid future temperature for thermal response")
    tail = np.asarray(data["thermal_temperature_tail"], np.float64)
    if tail.shape != (2, len(cities)) or not np.isfinite(tail).all():
        raise ValueError("invalid saved temperature history for C4")
    effective = _thermal_effective_temperature(
        np.concatenate([tail, temp]), float(data["thermal_memory"])
    )[2:]
    response = np.exp(float(data["thermal_gamma"]) * np.einsum(
        "dcf,cfj->dcj", _thermal_features(effective),
        np.asarray(data["thermal_coefficients"], np.float64)
    ))
    components = np.asarray(data["thermal_neutral_forecast"], np.float64) * response
    rho = float(data["thermal_industry_weight"])
    return np.maximum(0.0, (1-rho)*components[:, :, 0] + rho*components[:, :, 1:].sum(axis=2))


def _predict_upgraded(data, cities, province, observed, future, save_output=True):
    start = pd.Timestamp(str(data["thermal_prediction_start"]))
    template = _thermal_template(data, future, cities)
    adjusted = dict(data)
    # Reuse the original prediction function, including its final clipping.
    recent_total = np.asarray(data["last_load"], np.float64)[-CNY_SCALE_DAYS:].mean(axis=0)
    adjusted["cny_template"] = (template / recent_total[None]).T
    return _predict_pooled(adjusted, cities, province, observed, future,
                           save_output, start, start+pd.Timedelta(days=PRED_LEN-1),
                           CNY_TEMPLATE_WEIGHT)


def train_city_model():
    import hashlib

    root = find_city_root(require_train=True)
    dates, cities, load_matrix, province = load_train_load(root)
    observed = load_observed_weather(root)
    cube = _load_industry_cube(root, dates, cities)
    if not np.array_equal(cube[:, :, 0], load_matrix):
        raise ValueError("total load differs between loaders")
    data = _train_pooled(dates, cities, load_matrix, province, observed, save=False)
    # Preserve the incumbent's float32 serialization of its own fields.
    data = {key: (value.astype(np.float32) if value.dtype == np.float64 else value)
            for key, value in data.items()}
    data.update(_fit_thermal_upgrade(dates, cities, cube, observed))
    destination = model_dir() / MODEL_NAME
    np.savez_compressed(destination, **data)
    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    manifest = {
        "model": "C4 seven-day CNY template smoothing, temperature memory and holiday-return alignment",
        "schema": THERMAL_SCHEMA,
        "training_end": str(dates[-1].date()),
        "prediction_start": str(PRED_START.date()),
        "source_years": list(CNY_TEMPLATE_YEARS),
        "ridge_weight": 1-CNY_TEMPLATE_WEIGHT,
        "thermal_alpha": THERMAL_ALPHA, "thermal_gamma": THERMAL_GAMMA,
        "historical_smoothing_window": CNY_SMOOTH_WINDOW,
        "thermal_memory": THERMAL_MEMORY,
        "return_alignment_weight": CNY_RETURN_WARP,
        "return_phases": data["thermal_return_phases"].tolist(),
        "smoothing_excluded_lunar_phases": list(CNY_SMOOTH_EXCLUDE_PHASES),
        "industry_weight_within_template": THERMAL_INDUSTRY_WEIGHT,
        "training_pairs_per_city": int(data["thermal_training_pairs"]),
        "industry_identity_max_error": float(np.max(np.abs(cube[:, :, 0]-cube[:, :, 1:].sum(axis=2)))),
        "external_data_or_coefficients": False,
        "code_sha256": sha256(Path(__file__).resolve()),
        "load_sha256": sha256(root/"train_data/load_data/city_load_train.csv"),
        "weather_sha256": sha256(root/"train_data/weather_data/city_weather_data.csv"),
    }
    with open(model_dir()/MANIFEST_NAME, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print("[C4 saved]", destination, json.dumps(manifest, ensure_ascii=False), flush=True)


def predict_city_model(save_output=True):
    with np.load(model_dir()/MODEL_NAME, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    if "thermal_schema" not in data or int(data["thermal_schema"]) != THERMAL_SCHEMA:
        raise ValueError("C4 model schema mismatch: train the C4 model first")
    cities = [str(value) for value in data["cities"]]
    province = {city: str(value) for city, value in zip(cities, data["province"])}
    observed = _observed_from_model(data)
    root = find_city_root(require_train=False)
    future = load_future_weather(root, observed)
    result = _predict_upgraded(data, cities, province, observed, future, save_output)
    print("[C4 predict]", len(result), "rows; gamma=", float(data["thermal_gamma"]),
          "industry_weight=", float(data["thermal_industry_weight"]), flush=True)
    return result

