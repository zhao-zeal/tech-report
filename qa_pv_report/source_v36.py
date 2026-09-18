import json
import os
import re
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
PRED_START = pd.Timestamp('2025-10-01')
PRED_END = pd.Timestamp('2025-10-31')
PRED_LEN = 31
SPECIAL_START = pd.Timestamp('2025-10-04')
SPECIAL_END = pd.Timestamp('2025-10-07')
RECENT_DAYS = 180
ORDINARY_SCALE = 0.965
MODEL_NAME = 'pv_v9_curve_control.npz'
MANIFEST_NAME = 'pv_v9_curve_control_meta.json'
WX_COLS = ['temp_mean', 'temp_peak', 'temp_std', 'temp_range', 'temp_high_sr_mean', 'temp_high_sr_peak', 'sr_mean', 'sr_peak', 'sr_std', 'sr_range', 'sr_total', 'sr_hours_above_mean', 'sr_hours_above_mean_ratio', 'rhu_mean', 'pre_mean', 'pre_sum', 'rain_hours', 'tcc_mean', 'tcc_peak', 'tcc_std', 'tcc_weighted_mean', 'swddir_mean', 'swddir_peak', 'swddir_total', 'swddif_mean', 'swddif_peak', 'swddif_total', 'vis_mean', 'ws_mean']
SR_BIN_EDGES = [-np.inf, 60, 100, 140, 180, 220, 260, 300, np.inf]
FEATURES = ['month', 'day_of_year', 'cos_day', 'sin_day', 'cos_month', 'sin_month', 'dayofweek', 'is_weekend', 'is_holiday', 'days_to_sf', 'solar_declination', 'etr_norm', 'temp_mean', 'temp_peak', 'temp_std', 'temp_range', 'temp_high_sr_mean', 'temp_high_sr_peak', 'temp_high_sr_penalty', 'sr_mean', 'sr_peak', 'sr_std', 'sr_range', 'sr_total', 'sr_hours_above_mean', 'sr_hours_above_mean_ratio', 'sr_mean_bin', 'sr_bin_target_mean', 'tcc_mean', 'tcc_peak', 'tcc_std', 'tcc_weighted_mean', 'swddir_mean', 'swddir_peak', 'swddir_total', 'swddif_mean', 'swddif_peak', 'swddif_total', 'pre_mean', 'pre_sum', 'rain_hours', 'rain_day', 'vis_mean', 'ws_mean', 'sr_volatility', 'sr_temp_interact', 'daylight_cloud_penalty', 'clearness_index', 'diffuse_ratio', 'direct_ratio', 'sr_peak_ratio', 'kt_env', 'kt_env_peak', 'load_lag1', 'load_lag2', 'load_lag3', 'load_lag7', 'load_lag14', 'load_lag28', 'roll_mean3', 'roll_mean5', 'roll_mean7', 'roll_mean14', 'roll_mean28', 'roll_std7', 'roll_std14', 'dow_mean_4w', 'trend7', 'month_profile']
HOURLY_EXTRA = ['tcc_range', 'swddir_std', 'swddir_range', 'swddif_std', 'swddif_range', 'sr_morning', 'sr_afternoon', 'sr_asym', 'sr_peak_hour', 'tcc_at_peak', 'tcc_morning', 'tcc_afternoon', 'pre_day', 'rain_hours_day', 'rhu_midday', 'ws_day', 'cs_daily', 'kt_daily', 'kt_noon', 'kt_morning', 'kt_afternoon', 'cloud_run', 'sunny_hours']
FEATURES = FEATURES + HOURLY_EXTRA
CAL_BASE = 0.94
CAL_WS = -0.021
CAL_LOG_PRE = -0.041
CAL_MIN = 0.65
CAL_MAX = 1.05
ET_PARAMS = dict(n_estimators=500, max_depth=None, min_samples_split=4, min_samples_leaf=2, max_features='sqrt', bootstrap=False, random_state=42, n_jobs=16)
CNY = {2022: pd.Timestamp('2022-02-01'), 2023: pd.Timestamp('2023-01-22'), 2024: pd.Timestamp('2024-02-10'), 2025: pd.Timestamp('2025-01-29'), 2026: pd.Timestamp('2026-02-17'), 2027: pd.Timestamp('2027-02-06')}

def _weather_calibration(row):
    ws_day = float(row.get('ws_day', 4.0))
    pre_sum = max(float(row.get('pre_sum', 0.0)), 0.0)
    value = CAL_BASE + CAL_WS * (ws_day - 4.0) + CAL_LOG_PRE * (np.log1p(pre_sum) - 0.5)
    return float(np.clip(value, CAL_MIN, CAL_MAX))

def script_dir():
    current = Path(__file__).resolve().parent
    return current.parent if current.name == 'utils' else current

def model_dir():
    if os.environ.get('PV_MODEL_DIR'):
        print('[PV control] ignoring legacy PV_MODEL_DIR; using this control package model directory', flush=True)
    value = Path(os.environ.get('PV_V9_CONTROL_MODEL_DIR', str(script_dir() / 'model' / 'pv_models_v9_curve_control')))
    value.mkdir(parents=True, exist_ok=True)
    return value

def output_dir():
    value = Path(os.environ.get('OUTPUT_DIR', str(script_dir() / 'output' / 'pv_load_forecasting')))
    value.mkdir(parents=True, exist_ok=True)
    return value

def _candidate_roots():
    roots = []
    env = os.environ.get('PV_DATA_ROOT')
    if env:
        roots.append(Path(env))
    base = script_dir()
    roots.extend([base / 'datasets' / 'pv_load_forecasting', base.parent / 'datasets' / 'pv_load_forecasting', Path.cwd() / 'datasets' / 'pv_load_forecasting'])
    out = []
    for r in roots:
        r = r.resolve()
        if r not in out:
            out.append(r)
    return out

def fit_cs_curve(hourly_path, q=0.9, halfwin=10):
    hourly = hourly_path.copy() if isinstance(hourly_path, pd.DataFrame) else pd.read_csv(hourly_path)
    dt_col = 'valid_datetime' if 'valid_datetime' in hourly.columns else 'datetime'
    hourly['datetime'] = pd.to_datetime(hourly[dt_col])
    curve = {}
    for ta, g in hourly.groupby('ta_id'):
        g = g.sort_values('datetime')
        doy = g['datetime'].dt.dayofyear.values
        hour = g['datetime'].dt.hour.values
        sr = g['SR'].astype(float).values
        arr = np.full((367, 24), np.nan)
        for i in range(len(g)):
            arr[doy[i], hour[i]] = sr[i]
        out = np.full((367, 24), np.nan)
        s = pd.DataFrame(arr[1:])
        for hh in range(24):
            col = s[hh]
            trip = pd.concat([col, col, col], ignore_index=True)
            sm = trip.rolling(2 * halfwin + 1, center=True, min_periods=1).quantile(q)
            mid = sm.values[366:732]
            out[1:, hh] = pd.Series(mid).ffill().bfill().values
        curve[ta] = out
    return curve

def _hourly_csv_to_daily(path, cs_curve=None):
    hourly = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    dt_col = 'valid_datetime' if 'valid_datetime' in hourly.columns else 'datetime'
    hourly['datetime'] = pd.to_datetime(hourly[dt_col])
    hourly['hour'] = hourly['datetime'].dt.hour
    for c in ['TEM', 'RHU', 'PRE_15m', 'PRE_acc', 'SR', 'TCC', 'SWDDIR', 'SWDDIF', 'VIS', 'WS']:
        if c not in hourly.columns:
            hourly[c] = np.nan
    hourly = hourly.replace(9999, np.nan)

    def stats(v, p):
        v = v.dropna()
        if not len(v):
            return {f'{p}_mean': np.nan, f'{p}_peak': np.nan, f'{p}_std': np.nan, f'{p}_range': np.nan}
        return {f'{p}_mean': float(v.mean()), f'{p}_peak': float(v.max()), f'{p}_std': float(v.std()), f'{p}_range': float(v.max() - v.min())}
    rows = []
    for (ta, d), h in hourly.groupby(['ta_id', hourly['datetime'].dt.date]):
        sr = h['SR'].astype(float)
        tem = h['TEM'].astype(float)
        tcc = h['TCC'].astype(float)
        if 'PRE_15m' in h.columns and h['PRE_15m'].notna().any():
            pre = h['PRE_15m'].astype(float)
        elif 'PRE' in h.columns and h['PRE'].notna().any():
            pre = h['PRE'].astype(float)
        else:
            pre = h['PRE_acc'].astype(float)
        sr_non_na = sr.dropna()
        sr_mean_val = float(sr_non_na.mean()) if len(sr_non_na) else np.nan
        high_sr = sr > sr_mean_val if pd.notna(sr_mean_val) else pd.Series(False, index=h.index)
        active = sr.fillna(0.0) > 1.0
        tcc_w = np.nan
        if active.any() and tcc.notna().any():
            w = sr[active].fillna(0.0).values
            tv = tcc[active].values
            vm = ~np.isnan(tv) & (w > 0)
            if vm.any():
                tcc_w = float(np.average(tv[vm], weights=w[vm]))
        row = {'ta_id': ta, 'date': pd.Timestamp(d), 'temp_mean': float(tem.mean()), 'rhu_mean': float(h['RHU'].mean()), 'pre_mean': float(pre.mean()), 'pre_sum': float(pre.fillna(0.0).sum()), 'rain_hours': int((pre.fillna(0.0) > 0.01).sum()), 'vis_mean': float(h['VIS'].mean()), 'ws_mean': float(h['WS'].mean()), 'sr_total': float(sr.fillna(0.0).sum()), 'sr_hours_above_mean': int(high_sr.fillna(False).sum()), 'sr_hours_above_mean_ratio': float(high_sr.fillna(False).sum() / len(sr_non_na)) if len(sr_non_na) else np.nan, 'tcc_weighted_mean': tcc_w, 'swddir_total': float(h['SWDDIR'].fillna(0.0).sum()), 'swddif_total': float(h['SWDDIF'].fillna(0.0).sum()), 'temp_high_sr_mean': float(tem[high_sr].mean()) if high_sr.any() and tem.notna().any() else np.nan, 'temp_high_sr_peak': float(tem[high_sr].max()) if high_sr.any() and tem.notna().any() else np.nan, **stats(tem, 'temp'), **stats(sr, 'sr'), **stats(tcc, 'tcc'), **stats(h['SWDDIR'], 'swddir'), **stats(h['SWDDIF'], 'swddif')}
        day = h
        hmask_m = (h['hour'] >= 6) & (h['hour'] <= 11)
        hmask_a = (h['hour'] >= 12) & (h['hour'] <= 18)
        sr_m = float(sr[hmask_m].mean()) if sr.notna().any() else np.nan
        sr_a = float(sr[hmask_a].mean()) if sr.notna().any() else np.nan
        row['sr_morning'] = sr_m
        row['sr_afternoon'] = sr_a
        row['sr_asym'] = sr_a - sr_m
        peak_pos = int(np.argmax(sr.values)) if sr.notna().any() else -1
        row['sr_peak_hour'] = int(h['hour'].iloc[peak_pos]) if peak_pos >= 0 else -1
        row['tcc_at_peak'] = float(tcc.iloc[peak_pos]) if peak_pos >= 0 and pd.notna(tcc.iloc[peak_pos]) else np.nan
        row['tcc_morning'] = float(tcc[hmask_m].mean()) if tcc.notna().any() else np.nan
        row['tcc_afternoon'] = float(tcc[hmask_a].mean()) if tcc.notna().any() else np.nan
        row['pre_day'] = float(pre[hmask_m | hmask_a].fillna(0.0).sum())
        row['rain_hours_day'] = int((pre[hmask_m | hmask_a].fillna(0.0) > 0.01).sum())
        midday = (h['hour'] >= 10) & (h['hour'] <= 14)
        row['rhu_midday'] = float(h['RHU'][midday].mean())
        row['ws_day'] = float(h['WS'][hmask_m | hmask_a].mean())
        if cs_curve is not None:
            doy = pd.Timestamp(d).dayofyear
            env = cs_curve[ta][doy]
            cs_sum = float(np.nansum(env))
            row['cs_daily'] = cs_sum
            row['kt_daily'] = float(sr.fillna(0.0).sum() / max(cs_sum, 1e-06))
            kt_h = sr.fillna(0.0).values / np.maximum(env, 1e-06)
            row['kt_noon'] = float(kt_h[12]) if len(kt_h) > 12 else np.nan
            row['kt_morning'] = float(np.mean(kt_h[6:12])) if len(kt_h) > 12 else np.nan
            row['kt_afternoon'] = float(np.mean(kt_h[12:19])) if len(kt_h) > 18 else np.nan
            cloudy = (kt_h < 0.3).astype(int)
            run, max_run = (0, 0)
            for v in cloudy:
                run = run + 1 if v else 0
                max_run = max(max_run, run)
            row['cloud_run'] = float(max_run)
            row['sunny_hours'] = float((kt_h > 0.7).sum())
        rows.append(row)
    return pd.DataFrame(rows)

def fit_clear_env(hist_wx, q=0.9, smooth=7, halfwin=12):
    env = {}
    for ta, g in hist_wx.groupby('ta_id'):
        g = g.sort_values('date')
        sr = g['sr_mean'].values.astype(float)
        doy = g['date'].dt.dayofyear.values
        n = len(g)
        vals = np.full(367, np.nan)
        for i in range(n):
            lo, hi = (max(0, i - halfwin), min(n, i + halfwin + 1))
            vals[doy[i]] = np.quantile(sr[lo:hi], q)
        s = pd.Series(vals[1:])
        sm = pd.Series(np.concatenate([s.values[-smooth:], s.values, s.values[:smooth]]))
        sm = sm.rolling(smooth * 2 + 1, center=True, min_periods=1).mean().values[smooth:-smooth]
        env[ta] = np.concatenate([[np.nan], sm])
    return env

def apply_kt_env(wx_df, clear_env):
    df = wx_df.copy()
    cs = df.apply(lambda r: clear_env.get(r['ta_id'], [np.nan] * 367)[r['date'].dayofyear], axis=1)
    df['cs_env_sr'] = cs
    df['kt_env'] = np.clip(df['sr_mean'] / (df['cs_env_sr'] + 1e-06), 0.0, 1.5)
    df['kt_env_peak'] = np.clip(df['sr_peak'] / (df['cs_env_sr'] + 1e-06), 0.0, 2.0)
    return df

def preprocess_load(load_df):
    df = load_df.copy()
    for pv in df['pv_id'].unique():
        m = df['pv_id'] == pv
        vals = df.loc[m, 'load'].dropna()
        if len(vals):
            lo, hi = (vals.quantile(0.01), vals.quantile(0.99))
            df.loc[m, 'load'] = df.loc[m, 'load'].clip(lower=lo, upper=hi)
    for pv in df['pv_id'].unique():
        m = df['pv_id'] == pv
        ta = df.loc[m, 'ta_id'].iloc[0]
        need = df.loc[m & (df['load'].isna() | (df['load'] < 1e-06))].index
        for i in need:
            d = df.loc[i, 'date']
            others = df[(df['date'] == d) & (df['ta_id'] == ta) & (df['pv_id'] != pv)]['load'].dropna()
            om = others.mean() if len(others) else np.nan
            if np.isfinite(om) and om > 1e-06:
                df.loc[i, 'load'] = om
            else:
                df.loc[i, 'load'] = df.loc[m, 'load'].median()
    return df

def _astro(dates):
    doy = dates.dt.dayofyear.values
    decl = 23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + doy)))
    lat = np.radians(23.0)
    dec = np.radians(decl)
    cos_ws = np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0)
    ws = np.arccos(cos_ws)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    etr = 24.0 * 3600.0 / np.pi * 1367.0 * dr * (cos_ws * np.sin(lat) * np.sin(dec) + ws * np.cos(lat) * np.cos(dec))
    return (decl, etr / 13000.0)

def build_features(load_df, wx_df):
    load_df = preprocess_load(load_df)
    extra = [c for c in wx_df.columns if c not in ('ta_id', 'date') and c not in WX_COLS]
    df = load_df.merge(wx_df[['ta_id', 'date'] + WX_COLS + extra], on=['ta_id', 'date'], how='left')
    for c in WX_COLS:
        df[c] = df[c].ffill().bfill()
    df['month'] = df['date'].dt.month.astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear.astype(int)
    df['dayofweek'] = df['date'].dt.dayofweek.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['days_to_sf'] = df['date'].map(lambda d: int((CNY.get(d.year, CNY.get(d.year - 1, CNY.get(d.year + 1, pd.Timestamp(f'{d.year}-01-29')))) - d).days))
    try:
        from chinese_calendar import is_holiday as _is_holiday
        df['is_holiday'] = df['date'].map(lambda d: int(_is_holiday(d.date()))).astype(int)
    except Exception:
        df['is_holiday'] = 0
    df['solar_declination'], df['etr_norm'] = _astro(df['date'])
    df['sr_volatility'] = df['sr_std'] / (df['sr_mean'] + 1e-08)
    df['sr_temp_interact'] = df['sr_mean'] * df['temp_mean']
    df['temp_high_sr_penalty'] = np.maximum(0.0, df['temp_high_sr_mean'] - 25)
    df['daylight_cloud_penalty'] = df['sr_total'] * (1.0 - df['tcc_weighted_mean'].fillna(df['tcc_mean']).clip(0, 1))
    df['clearness_index'] = df['sr_mean'] / (df['swddir_mean'] + df['swddif_mean'] + 1e-08)
    df['diffuse_ratio'] = df['swddif_mean'] / (df['sr_mean'] + 1e-08)
    df['direct_ratio'] = df['swddir_mean'] / (df['sr_mean'] + 1e-08)
    df['sr_peak_ratio'] = df['sr_peak'] / (df['sr_mean'] + 1e-08)
    df['rain_day'] = (df['rain_hours'] > 0).astype(int)
    df['sr_mean_bin'] = pd.cut(df['sr_mean'].fillna(0), bins=SR_BIN_EDGES, labels=False, include_lowest=True).astype(int)
    for c in ['kt_env', 'kt_env_peak']:
        if c not in df.columns:
            df[c] = 0.0
    df = df.sort_values(['pv_id', 'date']).reset_index(drop=True)
    g = df.groupby('pv_id')['load']
    for k in [1, 2, 3, 7, 14, 28]:
        df[f'load_lag{k}'] = g.shift(k)
    sh = g.shift(1)
    for w in [3, 5, 7, 14, 28]:
        df[f'roll_mean{w}'] = sh.groupby(df['pv_id']).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    df['roll_std7'] = sh.groupby(df['pv_id']).rolling(7, min_periods=1).std().reset_index(level=0, drop=True)
    df['roll_std14'] = sh.groupby(df['pv_id']).rolling(14, min_periods=1).std().reset_index(level=0, drop=True)
    df['dow_mean_4w'] = (g.shift(7) + g.shift(14) + g.shift(21) + g.shift(28)) / 4
    df['trend7'] = g.shift(1) - g.shift(8)
    df['_mkey'] = df['date'].dt.month
    mg = df.groupby(['pv_id', '_mkey'])['load']
    cs, cnt = (mg.cumsum(), mg.cumcount())
    df['month_profile'] = (cs - df['load']) / cnt.replace(0, np.nan)
    df.drop(columns=['_mkey'], inplace=True)
    eg = df.groupby('pv_id')['load']
    ecs, ecnt = (eg.cumsum(), eg.cumcount())
    df['_ever_mean'] = (ecs - df['load']) / ecnt.replace(0, np.nan)
    df['month_profile'] = df['month_profile'].fillna(df['_ever_mean'])
    df.drop(columns=['_ever_mean'], inplace=True)
    return df

def loo_target_encoding(df):
    df = df.copy()
    global_mean = float(df['load'].mean())
    stats = df.groupby(['pv_id', 'sr_mean_bin'])['load'].agg(['sum', 'count', 'mean'])
    stats = stats.rename(columns={'sum': 's', 'count': 'n', 'mean': 'm'}).reset_index()
    df = df.merge(stats, on=['pv_id', 'sr_mean_bin'], how='left')
    df['sr_bin_target_mean'] = np.where(df['n'] > 1, (df['s'] - df['load']) / (df['n'] - 1), global_mean)
    df.drop(columns=['s', 'n', 'm'], inplace=True)
    mapping = {(str(r.pv_key), int(r.sr_mean_bin)): float(r.m) for r in stats.rename(columns={'pv_id': 'pv_key'}).itertuples()}
    return (df, mapping, global_mean)

def _calendar_row(date):
    doy = date.dayofyear
    try:
        from chinese_calendar import is_holiday as _is_holiday
        is_hol = int(_is_holiday(date.date()))
    except Exception:
        is_hol = 0
    return {'month': date.month, 'day_of_year': doy, 'cos_day': float(np.cos(2 * np.pi * doy / 365)), 'sin_day': float(np.sin(2 * np.pi * doy / 365)), 'cos_month': float(np.cos(2 * np.pi * date.month / 12)), 'sin_month': float(np.sin(2 * np.pi * date.month / 12)), 'dayofweek': date.dayofweek, 'is_weekend': int(date.dayofweek >= 5), 'is_holiday': is_hol, 'days_to_sf': int((CNY.get(date.year, CNY.get(date.year - 1, pd.Timestamp(f'{date.year}-01-29'))) - date).days), 'solar_declination': 0.0, 'etr_norm': 0.0}

def _weather_row(wx_dict, clear_env, ta, date):
    row = {}
    wx = wx_dict.get(ta)
    if wx is None or date not in wx.index:
        for c in WX_COLS:
            row[c] = 0.0
        for c in HOURLY_EXTRA:
            row[c] = 0.0
    else:
        for c in WX_COLS:
            row[c] = float(wx.loc[date, c])
        for c in HOURLY_EXTRA:
            row[c] = float(wx.loc[date, c]) if c in wx.columns else 0.0
    row['sr_volatility'] = row['sr_std'] / (row['sr_mean'] + 1e-08)
    row['sr_temp_interact'] = row['sr_mean'] * row['temp_mean']
    row['temp_high_sr_penalty'] = max(0.0, row['temp_high_sr_mean'] - 25)
    row['daylight_cloud_penalty'] = row['sr_total'] * (1.0 - min(max(row.get('tcc_weighted_mean', row['tcc_mean']), 0.0), 1.0))
    row['clearness_index'] = row['sr_mean'] / (row['swddir_mean'] + row['swddif_mean'] + 1e-08)
    row['diffuse_ratio'] = row['swddif_mean'] / (row['sr_mean'] + 1e-08)
    row['direct_ratio'] = row['swddir_mean'] / (row['sr_mean'] + 1e-08)
    row['sr_peak_ratio'] = row['sr_peak'] / (row['sr_mean'] + 1e-08)
    row['rain_day'] = int(row['rain_hours'] > 0)
    row['sr_mean_bin'] = int(pd.cut([row['sr_mean']], bins=SR_BIN_EDGES, labels=False, include_lowest=True)[0])
    env = clear_env.get(ta, np.full(367, np.nan))[date.dayofyear]
    if env is None or (isinstance(env, float) and np.isnan(env)) or env < 1e-06:
        env = 1.0
    row['kt_env'] = min(max(row['sr_mean'] / env, 0.0), 1.5)
    row['kt_env_peak'] = min(max(row['sr_peak'] / env, 0.0), 2.0)
    return row

def _lag_row(queue):
    row = {}
    if len(queue) >= 28:
        row['load_lag1'] = queue[-1]
        row['load_lag2'] = queue[-2]
        row['load_lag3'] = queue[-3]
        row['load_lag7'] = queue[-7]
        row['load_lag14'] = queue[-14]
        row['load_lag28'] = queue[-28]
    else:
        last = queue[-1] if queue else 0.0
        for k in [1, 2, 3, 7, 14, 28]:
            row[f'load_lag{k}'] = queue[-k] if len(queue) >= k else last
    for w in [3, 5, 7, 14, 28]:
        seg = queue[-w:]
        row[f'roll_mean{w}'] = float(np.mean(seg)) if seg else 0.0
    seg7 = queue[-7:]
    seg14 = queue[-14:]
    row['roll_std7'] = float(np.std(seg7)) if seg7 else 0.0
    row['roll_std14'] = float(np.std(seg14)) if seg14 else 0.0
    idx = [-7, -14, -21, -28]
    vals = [queue[i] for i in idx if len(queue) >= -i]
    row['dow_mean_4w'] = float(np.mean(vals)) if vals else 0.0
    row['trend7'] = queue[-1] - queue[-8] if len(queue) >= 8 else 0.0
    return row

def _read_hourly(path, dates, tas):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('V9 curve control requires original V9 hourly forecast CSV: ' + str(path))
    h = pd.read_csv(path)
    if 'datetime' not in h and 'valid_datetime' in h:
        h = h.rename(columns={'valid_datetime': 'datetime'})
    required = ['ta_id', 'datetime', 'TEM', 'RHU', 'PRE_15m', 'SR', 'TCC', 'SWDDIR', 'SWDDIF', 'VIS', 'WS']
    missing = [c for c in required if c not in h]
    if missing:
        raise ValueError(str(path) + ': missing columns ' + str(missing))
    h['datetime'] = pd.to_datetime(h.datetime, errors='raise')
    h = h[h.datetime.between(dates[0], dates[-1] + pd.Timedelta(hours=23))].copy()
    if h.duplicated(['ta_id', 'datetime']).any():
        raise ValueError(str(path) + ': duplicate TA/hour keys')
    expected = pd.MultiIndex.from_product([tas, pd.date_range(dates[0], dates[-1] + pd.Timedelta(hours=23), freq='h')])
    actual = pd.MultiIndex.from_frame(h[['ta_id', 'datetime']])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError(str(path) + ': incomplete or unexpected TA/hour coverage')
    if not np.isfinite(h[required[2:]].to_numpy(float)).all():
        raise ValueError(str(path) + ': non-finite hourly weather')
    return h.sort_values(['ta_id', 'datetime']).reset_index(drop=True)

def _prepare_base(load, hourly):
    curve = fit_cs_curve(hourly)
    daily = _hourly_csv_to_daily(hourly, curve)
    clear = fit_clear_env(daily)
    feat = build_features(load, apply_kt_env(daily, clear)).dropna(subset=['load_lag28', 'month_profile'])
    feat, te, default = loo_target_encoding(feat)
    if not np.isfinite(feat[FEATURES].to_numpy(float)).all():
        raise ValueError('Non-finite original V9 training features')
    return (feat, curve, clear, te, default)

def _future_base(pv, dates, queue, profile, ta, weather, clear, te, default):
    lag = _lag_row(list(queue))
    rows = []
    mult = []
    for d in dates:
        row = _calendar_row(d)
        decl, etr = _astro(pd.Series([d]))
        row.update(solar_declination=float(decl[0]), etr_norm=float(etr[0]))
        row.update(lag)
        row['month_profile'] = float(profile[d.month - 1])
        row.update(_weather_row(weather, clear, ta, d))
        row['sr_bin_target_mean'] = te.get((pv, row['sr_mean_bin']), default)
        rows.append([row[c] for c in FEATURES])
        mult.append(_weather_calibration(row))
    return (np.asarray(rows, dtype=np.float32), np.asarray(mult))

def find_pv_root(require_train=True):
    required = ['train_data/load_data/pv_load_train.csv', 'train_data/weather_data/pv_weather_train_hourly.csv'] if require_train else ['test_data/weather_data/pv_weather_test_hourly.csv']
    for root in _candidate_roots():
        if all(((root / name).is_file() for name in required)):
            return root
    raise FileNotFoundError('V9 curve control cannot locate ' + str(required) + ' under ' + str(_candidate_roots()))

def _load_training(root):
    load = pd.read_csv(root / 'train_data/load_data/pv_load_train.csv', parse_dates=['date'])
    load = load.loc[(load.date >= TRAIN_START) & (load.date < PRED_START)].copy()
    if load.duplicated(['pv_id', 'date']).any():
        raise ValueError('Duplicate original PV labels')
    if load.date.min() != TRAIN_START or load.date.max() != PRED_START - pd.Timedelta(days=1):
        raise ValueError('V9 curve control full training requires PV labels from 2024-10-01 through 2025-09-30')
    pvs = sorted(load.pv_id.unique())
    tas = sorted(load.ta_id.unique())
    if len(pvs) != 100 or tas != [f'ta_{i}' for i in range(1, 10)]:
        raise ValueError('V9 curve control expects the original 100 PV users and 9 TAs')
    if not load.groupby('pv_id').ta_id.nunique().eq(1).all() or not load.groupby('pv_id').installed_capacity.nunique().eq(1).all():
        raise ValueError('PV user changes TA or capacity')
    if not (load.installed_capacity > 0).all():
        raise ValueError('Nonpositive PV capacity')
    wanted = pd.MultiIndex.from_product([pvs, pd.date_range(TRAIN_START, PRED_START - pd.Timedelta(days=1))])
    if len(wanted.difference(pd.MultiIndex.from_frame(load[['pv_id', 'date']]))):
        raise ValueError('Missing original PV user/day rows')
    hourly = _read_hourly(root / 'train_data/weather_data/pv_weather_train_hourly.csv', pd.date_range(TRAIN_START, PRED_START - pd.Timedelta(days=1)), tas)
    return (load, hourly, pvs, tas)
TRAIN_START = pd.Timestamp('2024-10-01')
'Fixed curve_legacy control using the same 200 V9 forests as the prior V22 candidate.\n\nThe original fitted curve-control model identity and training recipe remain.\nAt inference only the hourly curve branch is disabled; daily clear_env remains.\nNo historical profile repair, observed-weather correction or q35 adjustment.\n'
PIPELINE_VERSION = 'v9_no_hourly_curve_control_20260911'
DEFAULT_MODE = 'none_legacy'
MODES = ('none_legacy',)
MODEL_VERSION = 'v9_curve_legacy_control_20260911'
MODEL_DEFAULT_MODE = 'curve_legacy'
MODEL_MODES = ('curve_legacy',)
CURVE_FIELDS = ('cs_daily', 'kt_daily', 'kt_noon', 'kt_morning', 'kt_afternoon', 'cloud_run', 'sunny_hours')
STATE_KEYS = {'pvs', 'tas', 'features', 'last_loads', 'month_profile', 'curve', 'clear', 'te_default'}
BUNDLE_KEYS = {'version', 'pv_id', 'capacity', 'training_dates', 'recent_training_dates', 'full42', 'recent42'}
META_KEYS = {'version', 'allowed_modes', 'default_mode', 'seed', 'feature_count', 'ta_map', 'capacity_map', 'legacy_history_source', 'te_map', 'train_start', 'train_end', 'prediction_start', 'prediction_end', 'policy', 'weather_curve', 'ordinary_scale', 'recent_days', 'forest_count', 'files', 'calendar_environment', 'elapsed_seconds'}

def _resolve_mode(mode=None):
    if mode is not None and mode != DEFAULT_MODE:
        raise ValueError('This control package only supports none_legacy; explicit mode=' + str(mode))
    previous = os.environ.get('PV_PREDICTION_MODE')
    if previous:
        print('[PV control] ignoring legacy PV_PREDICTION_MODE=' + str(previous) + '; fixed mode=none_legacy', flush=True)
    return DEFAULT_MODE

def _calendar_environment():
    import importlib.util
    import importlib.metadata
    available = importlib.util.find_spec('chinese_calendar') is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version('chinese-calendar')
        except importlib.metadata.PackageNotFoundError:
            import chinese_calendar
            version = str(getattr(chinese_calendar, '__version__', 'unreported'))
    return {'available': available, 'version': version}

def _expected_files(pvs):
    return {MODEL_NAME, MANIFEST_NAME, *(f'control_{pv}.joblib' for pv in pvs)}

def _check_model_files(dest, pvs, complete=False):
    expected = _expected_files(pvs)
    invalid = [name for name in expected if (dest / name).exists() and (not (dest / name).is_file()) or (complete and (not (dest / name).is_file()))]
    if invalid:
        raise ValueError('Missing or invalid required V9 curve control model files: ' + str(invalid))

def _check_bundle(bundle, pv, capacity, origin):
    if set(bundle) != BUNDLE_KEYS or bundle['version'] != MODEL_VERSION or bundle['pv_id'] != pv or (bundle['capacity'] != capacity):
        raise ValueError('Wrong V9 curve control PV model bundle: ' + str(pv))
    training = pd.DatetimeIndex(bundle['training_dates'])
    recent = pd.DatetimeIndex(bundle['recent_training_dates'])
    for dates in (training, recent):
        if not len(dates) or dates.hasnans or dates.has_duplicates or (not dates.is_monotonic_increasing) or (dates.min() < TRAIN_START) or (dates.max() >= origin):
            raise ValueError('V9 curve control contains invalid or post-origin training dates: ' + pv)
    expected_recent = training[training >= origin - pd.Timedelta(days=RECENT_DAYS)]
    if not recent.equals(expected_recent):
        raise ValueError('Wrong V9 curve control recent-180 training dates: ' + pv)
    for key in ('full42', 'recent42'):
        model = bundle[key]
        if not isinstance(model, ExtraTreesRegressor) or len(getattr(model, 'estimators_', [])) != 500 or model.n_features_in_ != len(FEATURES):
            raise ValueError('Invalid V9 curve control fitted forest: ' + pv + '/' + key)
        params = model.get_params()
        expected = dict(ET_PARAMS, n_jobs=1, random_state=42)
        if any((params[name] != value for name, value in expected.items())):
            raise ValueError('Wrong V9 curve control forest recipe: ' + pv + '/' + key)

def _predict_user(bundle, x, mult, dates):
    if x.shape != (len(dates), len(FEATURES)) or np.asarray(mult).shape != (len(dates),):
        raise ValueError('Wrong V9 curve control future-feature shape')
    if not np.isfinite(x).all() or not np.isfinite(mult).all():
        raise ValueError('Nonfinite V9 curve control future features or weather calibration')
    predictions = []
    for name in ('full42', 'recent42'):
        model = bundle[name]
        model.set_params(n_jobs=1)
        raw = np.exp(model.predict(x)) - 0.01 * bundle['capacity']
        predictions.append(np.maximum(raw * mult, 0))
    special = dates.day.isin([4, 5, 6, 7])
    result = np.where(special, predictions[1], predictions[0]) * np.where(special, 1.0, ORDINARY_SCALE)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid V9 curve control per-user predictions')
    return result

def _train_control_from_scratch():
    import joblib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    started = time.time()
    root = find_pv_root(require_train=True)
    load, hourly, pvs, tas = _load_training(root)
    if pvs != sorted((f'pv_{i}' for i in range(1, 101))):
        raise ValueError('V9 curve control requires the original pv_1 through pv_100 identifiers')
    if not np.isfinite(load.installed_capacity).all():
        raise ValueError('Nonfinite PV capacity')
    if hourly.datetime.max() >= PRED_START or load.date.max() >= PRED_START:
        raise ValueError('V9 curve control training input extends into prediction period')
    feat, curve, clear, te, default = _prepare_base(load, hourly)
    info = load.groupby('pv_id', sort=False).first()
    groups = {pv: group.sort_values('date') for pv, group in feat.groupby('pv_id')}
    if set(groups) != set(pvs) or feat.date.max() != PRED_START - pd.Timedelta(days=1):
        raise ValueError('Incomplete V9 curve control fitted training features')
    recent_min = feat.date.max() - pd.Timedelta(days=RECENT_DAYS - 1)
    numeric = sorted(pvs, key=lambda pv: int(pv.rsplit('_', 1)[-1]))
    legacy_history_source = dict(zip(numeric, pvs))
    dest = model_dir()
    _check_model_files(dest, pvs)
    manifest = dest / MANIFEST_NAME
    if manifest.exists():
        manifest.unlink()
    workers = max(1, min(16, int(os.environ.get('PV_TRAIN_WORKERS', '4'))))
    print('[PV control train]', 'model_version=' + MODEL_VERSION, 'training_features=original_curve_control', 'seed42 full + recent180', flush=True)
    print('[PV control train] pipeline=', Path(__file__).resolve(), flush=True)
    print('[PV control train] model_dir=', dest.resolve(), flush=True)
    print('[PV control train] load=', (root / 'train_data/load_data/pv_load_train.csv').resolve(), flush=True)
    print('[PV control train] forecast=', (root / 'train_data/weather_data/pv_weather_train_hourly.csv').resolve(), flush=True)
    print('[PV control train] dates=2024-10-01..2025-09-30; hourly_curve=fitted_training_forecast; history_mapping=original_v9', flush=True)

    def train_user(pv):
        group = groups[pv]
        x = group[FEATURES].to_numpy(np.float32)
        capacity = float(info.loc[pv, 'installed_capacity'])
        y = np.log(group.load.to_numpy() + 0.01 * capacity)
        recent = group.date.ge(recent_min).to_numpy()
        if not np.isfinite(y).all() or not recent.any():
            raise ValueError('Invalid V9 curve control historical training targets: ' + pv)
        params = dict(ET_PARAMS, n_jobs=1, random_state=42)
        bundle = dict(version=MODEL_VERSION, pv_id=pv, capacity=capacity, training_dates=group.date.to_numpy(), recent_training_dates=group.loc[recent, 'date'].to_numpy(), full42=ExtraTreesRegressor(**params).fit(x, y), recent42=ExtraTreesRegressor(**params).fit(x[recent], y[recent]))
        _check_bundle(bundle, pv, capacity, PRED_START)
        joblib.dump(bundle, dest / f'control_{pv}.joblib', compress=3)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(train_user, pv) for pv in pvs]):
            future.result()
            done += 1
            if done % 10 == 0:
                print('[V9 curve control train]', done, '/100 users', round(time.time() - started), 'seconds', flush=True)
    state_path = dest / MODEL_NAME
    np.savez_compressed(state_path, pvs=np.asarray(pvs), tas=np.asarray(tas), features=np.asarray(FEATURES), last_loads=np.stack([groups[pv].load.to_numpy()[-60:] for pv in pvs]), month_profile=np.stack([groups[pv].groupby(groups[pv].date.dt.month).load.mean().reindex(range(1, 13), fill_value=0).to_numpy() for pv in pvs]), curve=np.stack([curve[ta] for ta in tas]), clear=np.stack([clear[ta] for ta in tas]), te_default=np.asarray([default], dtype=np.float32))
    meta = dict(version=MODEL_VERSION, allowed_modes=list(MODEL_MODES), default_mode=MODEL_DEFAULT_MODE, seed=42, feature_count=len(FEATURES), ta_map=info.ta_id.to_dict(), capacity_map={pv: float(info.loc[pv, 'installed_capacity']) for pv in pvs}, legacy_history_source=legacy_history_source, te_map={str(pv) + '__' + str(bin_id): value for (pv, bin_id), value in te.items()}, train_start=str(load.date.min().date()), train_end=str(load.date.max().date()), prediction_start=str(PRED_START.date()), prediction_end=str(PRED_END.date()), policy='forecast_only_no_observed_weather', weather_curve='pre_origin_training_hourly_forecast_only', ordinary_scale=ORDINARY_SCALE, recent_days=RECENT_DAYS, forest_count=200, files=102, calendar_environment=_calendar_environment(), elapsed_seconds=time.time() - started)
    if set(meta) != META_KEYS:
        raise ValueError('Unexpected V9 curve control training metadata fields')
    temporary = dest / (MANIFEST_NAME + '.tmp')
    temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(manifest)
    _check_model_files(dest, pvs, complete=True)
    print('[V9 curve control saved]', dest, '102 files; test forecast CSV alone is sufficient for inference', flush=True)
    return state_path

def train_pv_model():
    import joblib
    dest = model_dir()
    if (dest / MANIFEST_NAME).is_file():
        meta, state = _read_fitted_state(dest)
        for pv in state['pvs']:
            bundle = joblib.load(dest / f'control_{pv}.joblib')
            _check_bundle(bundle, pv, float(meta['capacity_map'][pv]), PRED_START)
        print('[PV control reuse] model_version=' + MODEL_VERSION, 'mode=' + DEFAULT_MODE, 'validated 200 existing forests; no fitting or model writes', flush=True)
        return dest / MODEL_NAME
    if any(dest.iterdir()):
        raise RuntimeError('Incomplete curve-control model directory; restore its complete 102 files or select an empty PV_V9_CONTROL_MODEL_DIR: ' + str(dest))
    return _train_control_from_scratch()

def _read_fitted_state(dest):
    manifest = dest / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError('Complete V9 curve control training required: ' + str(manifest))
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    if set(meta) != META_KEYS or meta.get('version') != MODEL_VERSION or meta.get('seed') != 42 or (meta.get('allowed_modes') != list(MODEL_MODES)):
        raise ValueError('Wrong fitted control identity; restore the scored V9 curve-control model directory, or rebuild in an empty PV_V9_CONTROL_MODEL_DIR')
    expected = {'train_start': str(TRAIN_START.date()), 'train_end': str((PRED_START - pd.Timedelta(days=1)).date()), 'prediction_start': str(PRED_START.date()), 'prediction_end': str(PRED_END.date()), 'policy': 'forecast_only_no_observed_weather', 'weather_curve': 'pre_origin_training_hourly_forecast_only', 'ordinary_scale': ORDINARY_SCALE, 'recent_days': RECENT_DAYS, 'forest_count': 200, 'files': 102, 'feature_count': len(FEATURES), 'default_mode': MODEL_DEFAULT_MODE}
    if any((meta.get(key) != value for key, value in expected.items())):
        raise ValueError('Wrong V9 curve control training boundary or fitted recipe')
    if meta['calendar_environment']['available'] != _calendar_environment()['available']:
        raise RuntimeError('V9 curve control training and prediction must use the same chinese_calendar availability')
    with np.load(dest / MODEL_NAME, allow_pickle=False) as source:
        if set(source.files) != STATE_KEYS:
            raise ValueError('Unexpected V9 curve control historical state arrays')
        state = {key: source[key].copy() for key in source.files}
    pvs = list(state['pvs'])
    tas = list(state['tas'])
    if pvs != sorted((f'pv_{i}' for i in range(1, 101))) or tas != [f'ta_{i}' for i in range(1, 10)] or list(state['features']) != FEATURES:
        raise ValueError('V9 curve control entity or feature schema mismatch')
    shapes = {'last_loads': (100, 60), 'month_profile': (100, 12), 'curve': (9, 367, 24), 'clear': (9, 367), 'te_default': (1,)}
    if any((state[key].shape != shape for key, shape in shapes.items())):
        raise ValueError('Wrong V9 curve control historical array shape')
    if any((not np.isfinite(state[key]).all() for key in ('last_loads', 'month_profile', 'te_default'))):
        raise ValueError('Nonfinite V9 curve control fitted history')
    numeric = sorted(pvs, key=lambda pv: int(pv.rsplit('_', 1)[-1]))
    if meta['legacy_history_source'] != dict(zip(numeric, pvs)):
        raise ValueError('Unexpected original V9 historical-state mapping')
    if set(meta['ta_map']) != set(pvs) or set(meta['capacity_map']) != set(pvs):
        raise ValueError('Incomplete V9 curve control PV metadata')
    if any((meta['ta_map'][pv] not in tas or not np.isfinite(meta['capacity_map'][pv]) or meta['capacity_map'][pv] <= 0 for pv in pvs)):
        raise ValueError('Invalid V9 curve control TA or capacity metadata')
    _check_model_files(dest, pvs, complete=True)
    return (meta, state)

def predict_pv_model(save_output=True, mode=None):
    import joblib
    mode = _resolve_mode(mode)
    dest = model_dir()
    meta, state = _read_fitted_state(dest)
    pvs = list(state['pvs'])
    tas = list(state['tas'])
    dates = pd.date_range(PRED_START, PRED_END)
    root = find_pv_root(require_train=False)
    hourly = _read_hourly(root / 'test_data/weather_data/pv_weather_test_hourly.csv', dates, tas)
    use_curve = mode.startswith('curve_')
    correct_history = mode.endswith('_correct')
    curve = {ta: state['curve'][i] for i, ta in enumerate(tas)} if use_curve else None
    clear = {ta: state['clear'][i] for i, ta in enumerate(tas)}
    daily = _hourly_csv_to_daily(hourly, curve)
    weather = {ta: group.set_index('date') for ta, group in daily.groupby('ta_id')}
    te = {(key.rsplit('__', 1)[0], int(key.rsplit('__', 1)[1])): value for key, value in meta['te_map'].items()}
    default = float(state['te_default'][0])
    idx = {pv: i for i, pv in enumerate(pvs)}
    zero_indices = [FEATURES.index(field) for field in CURVE_FIELDS]
    rows = []
    print('[PV control predict]', 'version=' + PIPELINE_VERSION, 'mode=' + mode, 'model_version=' + MODEL_VERSION, flush=True)
    print('[PV control] hourly_curve=absent_seven_defaults daily_clear_env=preserved history_mapping=original_v9', flush=True)
    print('[PV control] pipeline=', Path(__file__).resolve(), flush=True)
    print('[PV control] model_dir=', dest.resolve(), 'manifest=', (dest / MANIFEST_NAME).resolve(), flush=True)
    print('[PV control] test_forecast=', (root / 'test_data/weather_data/pv_weather_test_hourly.csv').resolve(), flush=True)
    for pv in sorted(pvs, key=lambda value: int(value.rsplit('_', 1)[-1])):
        source = pv if correct_history else meta['legacy_history_source'][pv]
        i = idx[source]
        x, mult = _future_base(pv, dates, state['last_loads'][i], state['month_profile'][i], meta['ta_map'][pv], weather, clear, te, default)
        if not use_curve and (not np.array_equal(x[:, zero_indices], np.zeros((len(dates), len(CURVE_FIELDS)), dtype=np.float32))):
            raise ValueError('V9 curve control none mode did not zero exactly the seven curve-dependent feature fields')
        bundle = joblib.load(dest / f'control_{pv}.joblib')
        _check_bundle(bundle, pv, float(meta['capacity_map'][pv]), PRED_START)
        prediction = _predict_user(bundle, x, mult, dates)
        rows.extend(({'pv_id': str(pv), 'date': date.strftime('%Y/%m/%d'), 'pred': float(value)} for date, value in zip(dates, prediction)))
    result = pd.DataFrame(rows, columns=['pv_id', 'date', 'pred'])
    if len(result) != 3100 or result.duplicated(['pv_id', 'date']).any() or (not np.isfinite(result.pred).all()) or result.pred.lt(0).any():
        raise ValueError('Invalid V9 curve control submission output')
    special_rows = pd.to_datetime(result.date).dt.day.isin([4, 5, 6, 7])
    print('[PV control complete] mode=' + mode, 'rows=3100 users=100 dates=2025-10-01..2025-10-31', 'prediction_total=' + format(float(result.pred.sum()), '.17g'), 'special_prediction_total=' + format(float(result.loc[special_rows, 'pred'].sum()), '.17g'), flush=True)
    if save_output:
        path = output_dir() / 'submit_result.csv'
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print('[V9 curve control output]', path, result.shape, flush=True)
    return result
'V25 online trial: fixed half curve-shape mixture at the scored none level.\n\nOriginal V9 training functions, fitted model identity and 102 files are reused.\nNo new models, data files, observations, quantiles or history remapping.\n'
PIPELINE_VERSION = 'v25_none_level_curve_shape_half_20260912'
DEFAULT_MODE = 'v25_none_level_curve_shape_half'
MODES = (DEFAULT_MODE,)
V25_CURVE_WEIGHT = 0.5

def _resolve_mode(mode=None):
    if mode is not None and mode != DEFAULT_MODE:
        raise ValueError('This V25 online trial only supports ' + DEFAULT_MODE + '; explicit mode=' + str(mode))
    previous = os.environ.get('PV_PREDICTION_MODE')
    if previous:
        print('[PV V25 online trial] ignoring old PV_PREDICTION_MODE=' + str(previous), flush=True)
    return DEFAULT_MODE

def _v25_mix_ordinary(none, curve, dates):
    none = np.asarray(none, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    if none.shape != (len(dates),) or curve.shape != none.shape:
        raise ValueError('V25 view predictions have the wrong shape')
    if not np.isfinite(none).all() or not np.isfinite(curve).all() or (none < 0).any() or (curve < 0).any():
        raise ValueError('V25 view predictions must be finite and nonnegative')
    ordinary = ~dates.day.isin([4, 5, 6, 7])
    result = none.copy()
    if ordinary.any():
        mixed = (1.0 - V25_CURVE_WEIGHT) * none[ordinary] + V25_CURVE_WEIGHT * curve[ordinary]
        denominator = mixed.mean()
        if denominator > 1e-12:
            result[ordinary] = mixed / denominator * none[ordinary].mean()
    result[~ordinary] = none[~ordinary]
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid V25 mixture prediction')
    np.testing.assert_array_equal(result[~ordinary], none[~ordinary])
    np.testing.assert_allclose(result[ordinary].sum(), none[ordinary].sum(), atol=1e-12, rtol=0)
    return result

def predict_pv_model(save_output=True, mode=None):
    import joblib
    _resolve_mode(mode)
    dest = model_dir()
    meta, state = _read_fitted_state(dest)
    pvs = list(state['pvs'])
    tas = list(state['tas'])
    dates = pd.date_range(PRED_START, PRED_END)
    root = find_pv_root(require_train=False)
    forecast_path = root / 'test_data/weather_data/pv_weather_test_hourly.csv'
    hourly = _read_hourly(forecast_path, dates, tas)
    curve_reference = {ta: state['curve'][i] for i, ta in enumerate(tas)}
    clear = {ta: state['clear'][i] for i, ta in enumerate(tas)}
    daily_none = _hourly_csv_to_daily(hourly, None)
    daily_curve = _hourly_csv_to_daily(hourly, curve_reference)
    weather_none = {ta: group.set_index('date') for ta, group in daily_none.groupby('ta_id')}
    weather_curve = {ta: group.set_index('date') for ta, group in daily_curve.groupby('ta_id')}
    te = {(key.rsplit('__', 1)[0], int(key.rsplit('__', 1)[1])): value for key, value in meta['te_map'].items()}
    default = float(state['te_default'][0])
    index = {pv: i for i, pv in enumerate(pvs)}
    curve_indices = [FEATURES.index(field) for field in CURVE_FIELDS]
    other_indices = [i for i in range(len(FEATURES)) if i not in curve_indices]
    if len(curve_indices) != 7 or len(other_indices) != 85:
        raise ValueError('V25 requires the original 92-column V9 feature schema')
    ordinary = ~dates.day.isin([4, 5, 6, 7])
    rows = []
    none_predictions, curve_predictions, final_predictions = ([], [], [])
    print('[PV V25 online trial predict] pipeline_version=' + PIPELINE_VERSION, 'mode=' + DEFAULT_MODE, 'base_model_version=' + MODEL_VERSION, 'curve_shape_weight=0.5', 'ONLINE SCORE PENDING; NOT BASELINE PROMOTION', flush=True)
    print('[PV V25 online trial route] same_seed42_forests; ordinary=half_none_plus_half_curve_then_none_total;', 'calendar_4_7=none42_exact; history=original_v9_legacy; forecasts_only', flush=True)
    print('[PV V25 online trial] base=', dest.resolve(), 'test_forecast=', forecast_path.resolve(), flush=True)
    for pv in sorted(pvs, key=lambda value: int(value.rsplit('_', 1)[-1])):
        i = index[meta['legacy_history_source'][pv]]
        history = state['last_loads'][i]
        profile = state['month_profile'][i]
        ta = meta['ta_map'][pv]
        x_none, mult_none = _future_base(pv, dates, history, profile, ta, weather_none, clear, te, default)
        x_curve, mult_curve = _future_base(pv, dates, history, profile, ta, weather_curve, clear, te, default)
        np.testing.assert_array_equal(x_none[:, curve_indices], np.zeros((len(dates), 7), dtype=np.float32))
        np.testing.assert_array_equal(x_none[:, other_indices], x_curve[:, other_indices], err_msg='V25 changed a feature outside the seven hourly curve fields: ' + pv)
        np.testing.assert_array_equal(mult_none, mult_curve, err_msg='V25 weather multipliers differ: ' + pv)
        bundle = joblib.load(dest / f'control_{pv}.joblib')
        _check_bundle(bundle, pv, float(meta['capacity_map'][pv]), PRED_START)
        none = _predict_user(bundle, x_none, mult_none, dates)
        curve = _predict_user(bundle, x_curve, mult_curve, dates)
        final = _v25_mix_ordinary(none, curve, dates)
        none_predictions.append(none)
        curve_predictions.append(curve)
        final_predictions.append(final)
        rows.extend(({'pv_id': str(pv), 'date': date.strftime('%Y/%m/%d'), 'pred': float(value)} for date, value in zip(dates, final)))
    result = pd.DataFrame(rows, columns=['pv_id', 'date', 'pred'])
    if len(result) != 3100 or result.duplicated(['pv_id', 'date']).any() or (not np.isfinite(result.pred).all()) or result.pred.lt(0).any():
        raise ValueError('Invalid V25 output')
    none_all = np.stack(none_predictions)
    curve_all = np.stack(curve_predictions)
    final_all = np.stack(final_predictions)
    np.testing.assert_array_equal(final_all[:, ~ordinary], none_all[:, ~ordinary])
    ordinary_total_error = float(np.max(np.abs(final_all[:, ordinary].sum(axis=1) - none_all[:, ordinary].sum(axis=1))))
    np.testing.assert_allclose(final_all[:, ordinary].sum(axis=1), none_all[:, ordinary].sum(axis=1), atol=1e-12, rtol=0)
    print('[PV V25 online trial totals] none_total=' + format(float(none_all.sum()), '.17g'), 'curve_view_total=' + format(float(curve_all.sum()), '.17g'), 'final_total=' + format(float(final_all.sum()), '.17g'), 'none_special_total=' + format(float(none_all[:, ~ordinary].sum()), '.17g'), 'final_special_total=' + format(float(final_all[:, ~ordinary].sum()), '.17g'), 'max_user_ordinary_total_error=' + format(ordinary_total_error, '.17g'), flush=True)
    print('[PV V25 online trial complete] rows=3100; base_files=102; new_models=0;', 'route=' + DEFAULT_MODE + '; ONLINE SCORE PENDING; NOT BASELINE PROMOTION', flush=True)
    if save_output:
        path = output_dir() / 'submit_result.csv'
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print('[PV V25 online trial output]', path, 'rows=3100', flush=True)
    return result
_V27_RUNTIME_SOURCE = 'import time\n"""Global forecast-conditioned PV network, Paddle native."""\nimport paddle\nfrom paddle import nn\nimport paddle.nn.functional as F\n\nclass BaseNetwork(nn.Layer):\n    def __init__(self, decoder=\'point\', n_users=100):\n        super().__init__();self.decoder=decoder\n        self.history=nn.Sequential(nn.Flatten(),nn.Linear(62*11,128),nn.GELU(),nn.Linear(128,64),nn.GELU())\n        self.user=nn.Embedding(n_users,16)\n        self.input=nn.Linear(64+16+14,64)\n        if decoder==\'point\':\n            self.blocks=nn.LayerList([nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,64)) for _ in range(2)])\n        elif decoder==\'tcn\':\n            self.blocks=nn.LayerList([nn.Sequential(nn.Conv1D(64,64,3,padding=d,dilation=d),nn.GELU(),nn.Conv1D(64,64,1)) for d in [1,3]])\n        else:raise ValueError(decoder)\n        self.norms=nn.LayerList([nn.LayerNorm(64) for _ in range(2)])\n        self.head=nn.Linear(64,1)\n    def forward(self,hist,future,users):\n        n=future.shape[1]\n        h=self.history(hist).unsqueeze(1).expand([-1,n,-1])\n        u=self.user(users).unsqueeze(1).expand([-1,n,-1])\n        x=F.gelu(self.input(paddle.concat([h,u,future],axis=-1)))\n        for block,norm in zip(self.blocks,self.norms):\n            y=block(x) if self.decoder==\'point\' else block(x.transpose([0,2,1])).transpose([0,2,1])\n            x=norm(x+y)\n        valid=hist[:,-14:,1]\n        anchor=(hist[:,-14:,0]*valid).sum(axis=1)/paddle.clip(valid.sum(axis=1),min=1)\n        return self.head(x).squeeze(-1)+anchor.unsqueeze(-1)\n\ndef training_loss(logpred,target,mask,kind):\n    logtarget=paddle.log(target+.05)\n    mse=((logpred-logtarget)**2*mask).sum()/paddle.clip(mask.sum(),min=1)\n    if kind==\'log\':return mse\n    if kind!=\'relative\':raise ValueError(kind)\n    pred=paddle.clip(paddle.exp(paddle.clip(logpred,-8,6))-.05,min=0)\n    relative=(pred-target)/paddle.clip(target,min=.1)\n    absolute=paddle.abs(relative)\n    huber=paddle.where(absolute<=1,.5*relative**2,absolute-.5)\n    return (huber*mask).sum()/paddle.clip(mask.sum(),min=1)+.05*mse\n\nimport paddle\nfrom paddle import nn\nBase = BaseNetwork\nclass Network(Base):\n    def __init__(self,decoder=\'tcn\',n_users=100):\n        super().__init__(decoder,n_users)\n        self.history=nn.Sequential(nn.Flatten(),nn.Linear(62*11,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,64),nn.GELU(),nn.Dropout(.2))\n        self.blocks=nn.LayerList([nn.Sequential(nn.Conv1D(64,64,3,padding=d,dilation=d),nn.GELU(),nn.Dropout(.2),nn.Conv1D(64,64,1)) for d in [1,3]])\n\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nWX=[\'TEM\',\'RHU\',\'PRE_15m\',\'SR\',\'TCC\',\'SWDDIR\',\'SWDDIF\',\'VIS\',\'WS\']\n\ndef read_data(project):\n    root=Path(project)/\'Baseline/datasets/pv_load_forecasting/train_data\'\n    load=pd.read_csv(root/\'load_data/pv_load_train.csv\',parse_dates=[\'date\'])\n    hourly=pd.read_csv(root/\'weather_data/pv_weather_train_hourly.csv\',parse_dates=[\'datetime\'])\n    assert not load.duplicated([\'pv_id\',\'date\']).any() and not hourly.duplicated([\'ta_id\',\'datetime\']).any()\n    dates=pd.date_range(load.date.min(),load.date.max());pvs=sorted(load.pv_id.unique());tas=sorted(load.ta_id.unique())\n    info=load.groupby(\'pv_id\').first().reindex(pvs)\n    y=load.pivot(index=\'pv_id\',columns=\'date\',values=\'load\').reindex(index=pvs,columns=dates).to_numpy(float)\n    hourly[\'date\']=hourly.datetime.dt.normalize();daily=hourly.groupby([\'ta_id\',\'date\'])[WX].mean()\n    w=np.stack([daily.loc[t].reindex(dates).to_numpy(float) for t in tas]);assert np.isfinite(w).all()\n    return dict(y=y,w=w,dates=dates,pvs=np.asarray(pvs),tas=np.asarray(tas),ta=np.array([tas.index(t) for t in info.ta_id]))\n\ndef weather_transform(w):\n    w=w.copy();w[...,2]=np.log1p(np.maximum(w[...,2],0));w[...,7]=np.log1p(np.maximum(w[...,7],0));return w\n\ndef calendar(dates):\n    day=np.asarray(dates.dayofyear);dow=np.asarray(dates.dayofweek)\n    return np.stack([np.sin(day*2*np.pi/365.25),np.cos(day*2*np.pi/365.25),np.sin(dow*2*np.pi/7),np.cos(dow*2*np.pi/7),np.arange(len(dates))/30],axis=-1).astype(\'float32\')\n\ndef future_input(state,raw_weather,dates):\n    # Only the explicitly supplied forecast dates are accepted; no label input.\n    dates=pd.DatetimeIndex(dates);assert raw_weather.shape==(len(state[\'tas\']),len(dates),9) and 1<=len(dates)<=31\n    w=(weather_transform(raw_weather)-state[\'weather_mean\'])/state[\'weather_std\']\n    w=w[state[\'ta\']].astype(\'float32\');cal=np.broadcast_to(calendar(dates),(len(state[\'pvs\']),len(dates),5))\n    future=np.concatenate([w,cal],axis=-1)\n    if len(dates)<31:future=np.concatenate([future,np.repeat(future[:,-1:],31-len(dates),axis=1)],axis=1)\n    assert np.isfinite(future).all()\n    return state[\'history\'].copy(),future.astype(\'float32\'),np.arange(len(state[\'pvs\']),dtype=\'int64\')\n\ndef prepare(d,k,horizon=None):\n    y=d[\'y\'][:,:k].copy();valid=np.isfinite(y)&(y>=0)\n    scale=np.nanmedian(np.where(valid&(y>0),y,np.nan),axis=1);assert np.isfinite(scale).all() and (scale>0).all()\n    target=np.where(valid,np.maximum(y,0)/scale[:,None],0).astype(\'float32\')\n    logs=np.where(valid,np.log(target+.05),0).astype(\'float32\')\n    w=weather_transform(d[\'w\'][:,:k]);mean=w.mean(axis=(0,1));std=w.std(axis=(0,1))+1e-6\n    wn=((w-mean)/std).astype(\'float32\')[d[\'ta\']]\n    hist=np.concatenate([logs[:,:,None],valid[:,:,None].astype(\'float32\'),wn],axis=-1)\n    state=dict(pvs=d[\'pvs\'],tas=d[\'tas\'],ta=d[\'ta\'],scale=scale,weather_mean=mean,weather_std=std,history=hist[:,-62:].copy())\n    groups=[];starts=list(range(62,k-31+1,3))\n    for start in starts:\n        dates=d[\'dates\'][start:start+31]\n        f=np.concatenate([wn[:,start:start+31],np.broadcast_to(calendar(dates),(100,31,5))],axis=-1)\n        groups.append((hist[:,start-62:start],f,np.arange(100,dtype=\'int64\'),target[:,start:start+31],valid[:,start:start+31].astype(\'float32\')))\n    assert groups and starts[-1]+31<=k\n    train=tuple(np.concatenate([g[i] for g in groups]).astype(\'int64\' if i==2 else \'float32\') for i in range(5))\n    ndays=pd.Timestamp(d[\'dates\'][k]).days_in_month if horizon is None else int(horizon)\n    future=future_input(state,d[\'w\'][:,k:k+ndays],d[\'dates\'][k:k+ndays])\n    return train,future,state,str(d[\'dates\'][starts[-1]+30].date())\n\np = paddle\ndef strict(y, pred):\n    with np.errstate(all=\'ignore\'):\n        s = np.maximum(0, 1 - np.sqrt(np.mean(((pred - y) / np.maximum(y, 1e-12)) ** 2, axis=1)))\n    s[~np.isfinite(s)] = 0\n    return float(s.mean())\n\ndef infer(net, future, scale, n):\n    net.eval()\n    with p.no_grad():\n        z = net(*[p.to_tensor(a) for a in future]).numpy()[:, :n]\n    return np.maximum(np.exp(np.clip(z, -8, 6)) - 0.05, 0).astype(\'float64\') * scale[:, None]\n\ndef train_loop(train, seed, epochs, validation=None, batch_size=1024, learning_rate=0.001):\n    """Same mini-batch boundaries; one index transfer and one loss transfer per epoch."""\n    p.seed(seed)\n    rng = np.random.default_rng(seed)\n    net = Network()\n    tensors = [p.to_tensor(a) for a in train]\n    n = len(train[0])\n    opt = p.optimizer.AdamW(learning_rate=learning_rate, parameters=net.parameters(), weight_decay=0.01, grad_clip=p.nn.ClipGradByGlobalNorm(1.0))\n    losses = []\n    vals = []\n    begin = time.perf_counter()\n    if validation is not None:\n        (vf, vs, vy) = validation\n        vt = [p.to_tensor(a) for a in vf]\n    for ep in range(epochs):\n        epoch_begin = time.perf_counter()\n        net.train()\n        lv = []\n        batches = int(np.ceil(n / batch_size))\n        (q, r) = divmod(n, batches)\n        order = p.to_tensor(rng.permutation(n))\n        offset = 0\n        for j in range(batches):\n            size = q + (j < r)\n            ix = order[offset:offset + size]\n            offset += size\n            (h, f, u, y, m) = [p.index_select(t, ix) for t in tensors]\n            loss = training_loss(net(h, f, u), y, m, \'relative\')\n            loss.backward()\n            opt.step()\n            opt.clear_grad()\n            lv.append(loss.detach())\n        values = p.stack(lv).numpy().astype(\'float64\')\n        if not np.isfinite(values).all():\n            raise ValueError(\'Nonfinite neural training loss\')\n        losses.append(float(values.mean()))\n        if validation is not None:\n            net.eval()\n            with p.no_grad():\n                z = net(*vt).numpy()[:, :vy.shape[1]]\n            pred = np.maximum(np.exp(np.clip(z, -8, 6)) - 0.05, 0).astype(\'float64\') * vs[:, None]\n            vals.append(strict(vy, pred) * 20)\n        if True:\n            elapsed = time.perf_counter() - begin\n            print(\'[V27 epoch]\', str(ep + 1) + \'/\' + str(epochs), \'epoch_seconds=\' + str(round(time.perf_counter()-epoch_begin, 2)), \'remaining_minutes=\' + str(round((time.perf_counter()-begin)/(ep+1)*(epochs-ep-1)/60, 2)), \'batch=\' + str(batch_size), \'batches=\' + str(batches), \'seconds=\' + str(round(elapsed, 2)), \'samples_per_second=\' + str(round(n * (ep + 1) / max(elapsed, 1e-09), 1)), flush=True)\n    return (net, losses, vals)\ndef performance_preflight(train,inner_sequences,batch_size=1024,max_seconds=1800):\n    """Estimate neural training only. Isolated model; formal fit resets both RNGs."""\n    import math\n    p.seed(9027);net=Network();net.train();n=len(train[0]);batches=int(math.ceil(n/batch_size))\n    count=int(math.ceil(n/batches));tensors=[p.to_tensor(a[:count]) for a in train]\n    opt=p.optimizer.AdamW(learning_rate=.001,parameters=net.parameters(),weight_decay=.01,grad_clip=p.nn.ClipGradByGlobalNorm(1.))\n    h,f,u,y,m=tensors\n    print(\'[V27 preflight] device=\'+p.get_device(),\'actual_batch=\'+str(count),\'warmup=3 measured=6\',flush=True)\n    def sync():\n        next(iter(net.parameters())).flatten()[0].item()\n    def step(first=False):\n        if first:print(\'[V27 preflight] forward start\',flush=True)\n        loss=training_loss(net(h,f,u),y,m,\'relative\')\n        if first:\n            loss.item();print(\'[V27 preflight] forward complete; backward start\',flush=True)\n        loss.backward()\n        if first:\n            next(iter(net.parameters())).grad.flatten()[0].item()\n            print(\'[V27 preflight] backward complete; AdamW start\',flush=True)\n        opt.step();opt.clear_grad()\n        if first:sync();print(\'[V27 preflight] AdamW complete\',flush=True)\n    warm_start=time.perf_counter()\n    for i in range(3):step(i==0)\n    sync();warm_seconds=time.perf_counter()-warm_start\n    blocks=[]\n    for block in range(2):\n        start=time.perf_counter()\n        for _ in range(3):step()\n        sync();seconds=(time.perf_counter()-start)/3;blocks.append(seconds)\n        print(\'[V27 preflight] timed_block=\'+str(block+1),\'seconds_per_step=\'+str(round(seconds,4)),flush=True)\n    # Small validation forward and output transfer; score calculation overhead is not measured here.\n    net.eval();start=time.perf_counter()\n    with p.no_grad():net(h[:100],f[:100],u[:100]).numpy()\n    validation_seconds=time.perf_counter()-start\n    inner_batches=int(math.ceil(inner_sequences/batch_size));max_steps=2*40*(inner_batches+batches)\n    estimate=1.5*(max(blocks)*max_steps+80*validation_seconds)\n    result=dict(device=p.get_device(),batch_size=batch_size,actual_batch=count,warmup_seconds=warm_seconds,\n        seconds_per_step=blocks,samples_per_second=count/max(blocks),validation_seconds=validation_seconds,\n        maximum_training_steps=max_steps,estimated_neural_seconds=estimate,budget_seconds=max_seconds,\n        estimation_only=True,excludes=\'original forest training, CSV preprocessing, checkpoint I/O\')\n    print(\'[V27 preflight] samples_per_second=\'+str(round(result[\'samples_per_second\'],1)),\n        \'estimated_neural_minutes=\'+str(round(estimate/60,2)),\'budget_minutes=\'+str(round(max_seconds/60,2)),flush=True)\n    if max_seconds>0 and estimate>max_seconds:\n        raise RuntimeError(\'V27 preflight predicts slow neural training: %.1f min exceeds %.1f min budget. No formal neural model was trained. Check device/kernels; PV_V27_MAX_TRAIN_SECONDS changes this budget.\'%(estimate/60,max_seconds/60))\n    return result\n'
_V27_BASE_TRAIN = train_pv_model
_V27_BASE_PREDICT = predict_pv_model
PIPELINE_VERSION = 'v27_tcn_relative_nested25_b1024_20260912'
DEFAULT_MODE = 'v27_tcn_relative_nested25'
MODES = (DEFAULT_MODE,)
V27_ALPHA = 0.25
_V27_MODEL_VERSION = 'v27_tcn_relative_nested_b1024_seed42_2026_20260912'
_V27_FILES = {'neural_seed42.pdparams', 'neural_seed2026.pdparams', 'neural_state.npz', 'neural_meta.json'}
_V27_RUNTIME = None

def _v27_runtime():
    global _V27_RUNTIME
    if _V27_RUNTIME is None:
        namespace = {'__name__': 'v27_embedded_neural'}
        exec(_V27_RUNTIME_SOURCE, namespace)
        _V27_RUNTIME = namespace
    return _V27_RUNTIME

def _v27_model_dir():
    dest = Path(os.environ.get('PV_V27_MODEL_DIR', str(script_dir() / 'model/pv_models_v27_tcn_nested')))
    base = model_dir().resolve()
    actual = dest.resolve()
    if actual == base or actual in base.parents or base in actual.parents:
        raise ValueError('V27 neural directory must be separate from the V9 model directory')
    return dest

def _v27_validate(meta, state):
    if meta.get('version') != _V27_MODEL_VERSION or meta.get('policy') != 'forecast_only_no_observed_weather' or meta.get('seeds') != [42, 2026]:
        raise ValueError('Wrong V27 neural model identity')
    if meta.get('batch_size') != 1024 or meta.get('learning_rate') != 0.001:
        raise ValueError('Wrong V27 training batch or learning rate')
    if meta.get('train_end') != '2025-09-30' or meta.get('origin') != '2025-10-01' or meta.get('alpha') != V27_ALPHA:
        raise ValueError('Wrong V27 time boundary or recipe')
    required = {'pvs', 'tas', 'ta', 'scale', 'weather_mean', 'weather_std', 'history', 'origin', 'horizon'}
    if set(state) != required or str(state['origin']) != '2025-10-01' or int(state['horizon']) != 31:
        raise ValueError('Wrong V27 historical state schema')
    if state['history'].shape != (100, 62, 11) or state['scale'].shape != (100,) or state['ta'].shape != (100,):
        raise ValueError('Wrong V27 historical state shape')
    if len(state['pvs']) != 100 or len(set(state['pvs'])) != 100 or len(state['tas']) != 9:
        raise ValueError('Wrong V27 entity identity')
    if list(state['pvs']) != sorted(['pv_' + str(i) for i in range(1, 101)]):
        raise ValueError('Wrong V27 PV ordering')
    for key in ['history', 'scale', 'weather_mean', 'weather_std', 'ta']:
        if not np.isfinite(state[key]).all():
            raise ValueError('Nonfinite V27 state ' + key)
    if (state['scale'] <= 0).any() or (state['weather_std'] <= 0).any() or (state['ta'] < 0).any() or (state['ta'] >= 9).any():
        raise ValueError('Invalid V27 normalization or mapping')
    for seed in [42, 2026]:
        item = meta.get('training', {}).get(str(seed), {})
        if not 1 <= int(item.get('epochs', 0)) <= 40 or item.get('inner_validation_end') != '2025-09-30':
            raise ValueError('Wrong V27 internal training boundary')

def _v27_load():
    runtime = _v27_runtime()
    paddle = runtime['p']
    dest = _v27_model_dir()
    if not dest.is_dir() or {p.name for p in dest.iterdir()} != _V27_FILES:
        raise FileNotFoundError('V27 requires four platform-trained neural files in ' + str(dest) + '; run train_pv.py with original training CSVs')
    meta = json.loads((dest / 'neural_meta.json').read_text())
    with np.load(dest / 'neural_state.npz', allow_pickle=False) as z:
        state = {k: z[k].copy() for k in z.files}
    _v27_validate(meta, state)
    models = []
    for seed in [42, 2026]:
        net = runtime['Network']()
        weights = paddle.load(str(dest / f'neural_seed{seed}.pdparams'))
        expected = net.state_dict()
        if set(weights) != set(expected) or any((tuple(weights[k].shape) != tuple(expected[k].shape) for k in expected)):
            raise ValueError('Wrong V27 network weights')
        net.set_state_dict(weights)
        net.eval()
        models.append(net)
    return (models, meta, state)

def _v27_training_data():
    runtime = _v27_runtime()
    root = find_pv_root(require_train=True)
    load = pd.read_csv(root / 'train_data/load_data/pv_load_train.csv', parse_dates=['date'])
    hours = pd.read_csv(root / 'train_data/weather_data/pv_weather_train_hourly.csv', parse_dates=['datetime'])
    load = load[load.date.between('2024-10-01', '2025-09-30')].copy()
    hours = hours[hours.datetime.between(pd.Timestamp('2024-10-01'), pd.Timestamp('2025-10-01'), inclusive='left')].copy()
    if load.duplicated(['pv_id', 'date']).any() or hours.duplicated(['ta_id', 'datetime']).any():
        raise ValueError('Duplicate original training keys')
    dates = pd.date_range('2024-10-01', '2025-09-30')
    pvs = sorted(load.pv_id.unique())
    tas = sorted(load.ta_id.unique())
    if len(pvs) != 100 or len(tas) != 9:
        raise ValueError('Incomplete V27 training entities')
    info = load.groupby('pv_id').first().reindex(pvs)
    y = load.pivot(index='pv_id', columns='date', values='load').reindex(index=pvs, columns=dates).to_numpy(float)
    hours['date'] = hours.datetime.dt.normalize()
    daily = hours.groupby(['ta_id', 'date'])[runtime['WX']].mean()
    w = np.stack([daily.loc[ta].reindex(dates).to_numpy(float) for ta in tas])
    if not np.isfinite(w).all():
        raise ValueError('Incomplete original training forecast')
    return dict(y=np.concatenate([y, np.full((100, 31), np.nan)], axis=1), w=np.concatenate([w, np.zeros((9, 31, 9))], axis=1), dates=pd.date_range('2024-10-01', '2025-10-31'), pvs=np.asarray(pvs), tas=np.asarray(tas), ta=np.array([tas.index(t) for t in info.ta_id]))

def _v27_train_new():
    import tempfile, shutil, time
    dest = _v27_model_dir()
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Refuse to overwrite an incomplete V27 model directory: ' + str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    runtime = _v27_runtime()
    paddle = runtime['p']
    d = _v27_training_data()
    k = d['dates'].get_loc(pd.Timestamp('2025-10-01'))
    inner = k - 31
    inner_train, inner_future, inner_state, inner_last = runtime['prepare'](d, inner, 31)
    train, unused, state, last = runtime['prepare'](d, k, 31)
    del unused
    state = dict(state, origin=np.asarray('2025-10-01'), horizon=np.asarray(31))
    actual = d['y'][:, inner:k].copy()
    previous = paddle.get_device()
    device = os.environ.get('PV_V27_TRAIN_DEVICE', 'xpu:0')
    print('[PV V27 training] device=' + device, 'samples=' + str(len(train[0])), 'inner_validation=2025-08-31..2025-09-30', flush=True)
    staging = None
    begin = time.time()
    try:
        paddle.set_device(device)
        performance = runtime['performance_preflight'](train, len(inner_train[0]), max_seconds=float(os.environ.get('PV_V27_MAX_TRAIN_SECONDS', '1800')))
        staging = Path(tempfile.mkdtemp(prefix='v27_build_', dir=str(dest.parent)))
        records = {}
        for seed in [42, 2026]:
            print('[PV V27 inner selection] seed=' + str(seed), flush=True)
            inner_net, inner_losses, values = runtime['train_loop'](inner_train, seed, 40, (inner_future, inner_state['scale'], actual))
            epochs = int(np.argmax(values)) + 1
            del inner_net
            print('[PV V27 full training] seed=' + str(seed), 'selected_epochs=' + str(epochs), flush=True)
            net, losses, _ = runtime['train_loop'](train, seed, epochs)
            paddle.save(net.state_dict(), str(staging / f'neural_seed{seed}.pdparams'))
            del net
            records[str(seed)] = dict(epochs=epochs, inner_scores=values, inner_losses=inner_losses, losses=losses, inner_validation_start='2025-08-31', inner_validation_end='2025-09-30', inner_last_training_target=inner_last, last_training_target=last)
        meta = dict(version=_V27_MODEL_VERSION, policy='forecast_only_no_observed_weather', origin='2025-10-01', train_end='2025-09-30', seeds=[42, 2026], alpha=V27_ALPHA, architecture='global_tcn_relative_dropout02', batch_size=1024, learning_rate=0.001, performance=performance, training=records, training_sequences=len(train[0]), device=device, paddle=paddle.__version__, seconds=time.time() - begin)
        _v27_validate(meta, state)
        np.savez_compressed(staging / 'neural_state.npz', **state)
        (staging / 'neural_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        if dest.exists():
            dest.rmdir()
        staging.rename(dest)
        print('[PV V27 training complete]', dest, flush=True)
        return dest / 'neural_meta.json'
    finally:
        paddle.set_device(previous)
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

def train_pv_model():
    dest = _v27_model_dir()
    if dest.exists() and any(dest.iterdir()) and ({p.name for p in dest.iterdir()} != _V27_FILES):
        raise RuntimeError('Incomplete V27 model directory; existing files preserved: ' + str(dest))
    print('[PV V27 training] validate/reuse original V25 forests', flush=True)
    _V27_BASE_TRAIN()
    if dest.is_dir() and any(dest.iterdir()):
        paddle = _v27_runtime()['p']
        previous = paddle.get_device()
        paddle.set_device('cpu')
        try:
            _v27_load()
        finally:
            paddle.set_device(previous)
        print('[PV V27 reuse] complete neural model; no training CSV read or retraining', flush=True)
        return dest / 'neural_meta.json'
    return _v27_train_new()

def predict_pv_model(save_output=True, mode=None):
    _resolve_mode(mode)
    runtime = _v27_runtime()
    paddle = runtime['p']
    previous = paddle.get_device()
    paddle.set_device('cpu')
    try:
        models, meta, state = _v27_load()
        dates = pd.date_range(PRED_START, PRED_END)
        root = find_pv_root(require_train=False)
        hours = _read_hourly(root / 'test_data/weather_data/pv_weather_test_hourly.csv', dates, list(state['tas']))
        hours['date'] = hours.datetime.dt.normalize()
        daily = hours.groupby(['ta_id', 'date'])[runtime['WX']].mean()
        w = np.stack([daily.loc[ta].reindex(dates).to_numpy(float) for ta in state['tas']])
        arrays = runtime['future_input'](state, w, dates)
        component = (runtime['infer'](models[0], arrays, state['scale'], 31) + runtime['infer'](models[1], arrays, state['scale'], 31)) / 2
        neural = pd.DataFrame(dict(pv_id=np.repeat(state['pvs'], 31), date=np.tile(dates.strftime('%Y/%m/%d'), 100), pred=component.reshape(-1)))
    finally:
        paddle.set_device(previous)
    base = _V27_BASE_PREDICT(save_output=False, mode=None)
    result = base.copy()
    values = neural.set_index(['pv_id', 'date']).pred.reindex(pd.MultiIndex.from_frame(base[['pv_id', 'date']])).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError('V27 prediction keys mismatch')
    ordinary = ~pd.to_datetime(base.date).dt.day.isin([4, 5, 6, 7])
    b = base.pred.to_numpy()
    result.loc[ordinary, 'pred'] = (1 - V27_ALPHA) * b[ordinary.to_numpy()] + V27_ALPHA * values[ordinary.to_numpy()]
    np.testing.assert_array_equal(result.pred.to_numpy()[~ordinary.to_numpy()], b[~ordinary.to_numpy()])
    if len(result) != 3100 or result.duplicated(['pv_id', 'date']).any() or (not np.isfinite(result.pred).all()) or result.pred.lt(0).any():
        raise ValueError('Invalid V27 result')
    print('[PV V27 final] pipeline_version=' + PIPELINE_VERSION, 'mode=' + DEFAULT_MODE, 'ordinary=0.75_V25+0.25_two_seed_TCN', 'calendar_4_7=V25_exact', 'rows=3100', 'total=' + format(float(result.pred.sum()), '.17g'), 'special_total=' + format(float(result.loc[~ordinary, 'pred'].sum()), '.17g'), 'ONLINE SCORE PENDING', flush=True)
    if save_output:
        path = output_dir() / 'submit_result.csv'
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print('[PV V27 output]', path, flush=True)
    return result
_V29_BASE_PREDICT = predict_pv_model
PIPELINE_VERSION = 'v29_v27_gated_low4_070_other102_20260913'
DEFAULT_MODE = 'v29_gated_low4_070_other102'
MODES = (DEFAULT_MODE,)

def _v29_neural_forecast():
    runtime = _v27_runtime()
    paddle = runtime['p']
    previous = paddle.get_device()
    paddle.set_device('cpu')
    try:
        models, meta, state = _v27_load()
        dates = pd.date_range(PRED_START, PRED_END)
        root = find_pv_root(require_train=False)
        hours = _read_hourly(root / 'test_data/weather_data/pv_weather_test_hourly.csv', dates, list(state['tas']))
        hours['date'] = hours.datetime.dt.normalize()
        daily = hours.groupby(['ta_id', 'date'])[runtime['WX']].mean()
        weather = np.stack([daily.loc[ta].reindex(dates).to_numpy(float) for ta in state['tas']])
        arrays = runtime['future_input'](state, weather, dates)
        pred = (runtime['infer'](models[0], arrays, state['scale'], 31) + runtime['infer'](models[1], arrays, state['scale'], 31)) / 2
        return pd.DataFrame(dict(pv_id=np.repeat(state['pvs'], 31), date=np.tile(dates.strftime('%Y/%m/%d'), 100), neural=pred.reshape(-1)))
    finally:
        paddle.set_device(previous)

def predict_pv_model(save_output=True, mode=None):
    _resolve_mode(mode)
    base = _V29_BASE_PREDICT(save_output=False, mode=None)
    component = _v29_neural_forecast()
    ordered = base[['pv_id', 'date', 'pred']].copy()
    ordered['_position'] = np.arange(len(base))
    ordered = ordered.merge(component, on=['pv_id', 'date'], how='left', validate='one_to_one')
    ordered['_day'] = pd.to_datetime(ordered.date)
    ordered = ordered.sort_values(['pv_id', '_day'], kind='stable')
    if len(ordered) != 3100 or ordered.groupby('pv_id').size().ne(31).any() or (not np.isfinite(ordered[['pred', 'neural']].to_numpy()).all()):
        raise ValueError('Invalid V29 input rows')
    neural = ordered.neural.to_numpy().reshape(100, 31)
    values = ordered.pred.to_numpy().reshape(100, 31)
    risk = np.zeros_like(neural, dtype=bool)
    ranks = np.argsort(neural, axis=1, kind='stable')
    np.put_along_axis(risk, ranks[:, :4], True, axis=1)
    active = risk & (neural < values)
    ordered['pred'] = (values * np.where(active, 0.7, 1.02)).reshape(-1)
    ordered['_active'] = active.reshape(-1)
    event = ordered['_day'].dt.day.isin([4, 5, 6, 7])
    date_counts = ordered.groupby('date')['_active'].sum().astype(int).to_dict()
    result = ordered.sort_values('_position')[['pv_id', 'date', 'pred']].reset_index(drop=True)
    if not np.isfinite(result.pred).all() or result.pred.lt(0).any():
        raise ValueError('Invalid V29 output')
    print('[PV V29 final] pipeline_version=' + PIPELINE_VERSION, 'model=original_V27', 'risk_multiplier=0.70 other_multiplier=1.02', 'active_rows=' + str(int(active.sum())), 'active_calendar_4_7_rows=' + str(int(ordered.loc[event, '_active'].sum())), 'calendar_protection=disabled', 'rows=3100', 'total=' + format(float(result.pred.sum()), '.17g'), 'ONLINE SCORE PENDING', flush=True)
    print('[PV V29 risk dates]', json.dumps(date_counts, sort_keys=True), flush=True)
    if save_output:
        path = output_dir() / 'submit_result.csv'
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print('[PV V29 output]', path, flush=True)
    return result
'V36 R1 online probe: repaired hourly-curve view inside the V25 half mixture.\n\nOriginal V29 pipeline (V27 TCN blend, V25 none/curve 0.5 mixture, V9 seed42\nforests, 102+4 scored files) is unchanged. Only the curve-view source of the\nV25 mixture is replaced: the seven hourly-curve fields are recomputed from a\nnight-cleaned copy of the test hourly forecast (SR/SWDDIR/SWDDIF zeroed where\nthe fitted 90th-percentile clear-sky envelope is below 5 W/m2). The none view,\nthe other 85 features, the TCN daily weather, the V29 gate and all multipliers\nare untouched. No retraining, no new data, forecast-only inference.\n'
PIPELINE_VERSION = 'v36_r1_repaired_curve_half_20260913'
DEFAULT_MODE = 'v36_r1_repaired_curve_half'
MODES = (DEFAULT_MODE,)
R1_NIGHT_ENV_EPS = 5.0
_R36_ORIGINAL_HOURLY_TO_DAILY = _hourly_csv_to_daily

def _resolve_mode(mode=None):
    if mode is not None and mode != DEFAULT_MODE:
        raise ValueError('This V36 R1 probe only supports ' + DEFAULT_MODE + '; explicit mode=' + str(mode))
    previous = os.environ.get('PV_PREDICTION_MODE')
    if previous:
        print('[PV V36 R1 probe] ignoring old PV_PREDICTION_MODE=' + str(previous), flush=True)
    return DEFAULT_MODE

def _r1_night_clean(hourly, cs_curve):
    frame = hourly.copy()
    if 'datetime' not in frame.columns:
        frame['datetime'] = pd.to_datetime(frame['valid_datetime'])
    else:
        frame['datetime'] = pd.to_datetime(frame['datetime'])
    doy = frame['datetime'].dt.dayofyear.to_numpy()
    hour = frame['datetime'].dt.hour.to_numpy()
    mask = np.zeros(len(frame), dtype=bool)
    for ta, g in frame.groupby('ta_id', sort=False):
        pos = g.index.to_numpy()
        env = cs_curve[ta]
        vals = env[doy[pos], hour[pos]]
        night = np.where(np.isfinite(vals), vals, np.inf) < R1_NIGHT_ENV_EPS
        mask[pos] = night
    for c in ('SR', 'SWDDIR', 'SWDDIF'):
        frame.loc[mask, c] = 0.0
    return frame

def _hourly_csv_to_daily(path, cs_curve=None):
    if cs_curve is None:
        return _R36_ORIGINAL_HOURLY_TO_DAILY(path, None)
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    dt_col = 'valid_datetime' if 'valid_datetime' in frame.columns else 'datetime'
    if not (pd.to_datetime(frame[dt_col]) >= PRED_START).any():
        return _R36_ORIGINAL_HOURLY_TO_DAILY(path, cs_curve)
    daily = _R36_ORIGINAL_HOURLY_TO_DAILY(path, cs_curve)
    repaired = _R36_ORIGINAL_HOURLY_TO_DAILY(_r1_night_clean(frame, cs_curve), cs_curve)
    if not daily[['ta_id', 'date']].reset_index(drop=True).equals(repaired[['ta_id', 'date']].reset_index(drop=True)):
        raise ValueError('V36 R1 repaired daily aggregation lost rows')
    for c in CURVE_FIELDS:
        daily[c] = repaired[c].to_numpy()
    if not np.isfinite(daily[list(CURVE_FIELDS)].to_numpy(float)).all():
        raise ValueError('V36 R1 repaired curve fields must be finite')
    for c in ('kt_daily', 'kt_noon', 'kt_morning', 'kt_afternoon'):
        if daily[c].to_numpy(float).max() > 5.0:
            raise ValueError('V36 R1 repaired ' + c + ' out of physical range: ' + str(daily[c].to_numpy(float).max()))
    return daily

def _r1_diagnostics():
    dest = model_dir()
    meta, state = _read_fitted_state(dest)
    tas = list(state['tas'])
    dates = pd.date_range(PRED_START, PRED_END)
    root = find_pv_root(require_train=False)
    hours = _read_hourly(root / 'test_data/weather_data/pv_weather_test_hourly.csv', dates, tas)
    curve_ref = {ta: state['curve'][i] for i, ta in enumerate(tas)}
    cleaned = _r1_night_clean(hours, curve_ref)
    masked = np.abs(cleaned['SR'].to_numpy() - hours['SR'].to_numpy()) > 1e-09
    day7 = pd.to_datetime(hours.datetime).dt.day.isin([7, 14, 21, 28]).to_numpy()
    print('[PV V36 R1 night mask] masked_hours=' + str(int(masked.sum())) + '/' + str(len(hours)), 'day7_masked_hours=' + str(int((masked & day7).sum())), flush=True)
    corrupt = _R36_ORIGINAL_HOURLY_TO_DAILY(hours, curve_ref)
    repaired = _R36_ORIGINAL_HOURLY_TO_DAILY(cleaned, curve_ref)
    c7 = corrupt.loc[pd.to_datetime(corrupt.date).dt.day.isin([7, 14, 21, 28])]
    r7 = repaired.loc[pd.to_datetime(repaired.date).dt.day.isin([7, 14, 21, 28])]
    for c in ('kt_morning', 'kt_afternoon', 'kt_daily', 'sunny_hours', 'cloud_run'):
        print('[PV V36 R1 day7 ' + c + '] corrupt_mean=' + format(float(c7[c].mean()), '.6g'), 'repaired_mean=' + format(float(r7[c].mean()), '.6g'), flush=True)

def predict_pv_model(save_output=True, mode=None):
    _resolve_mode(mode)
    print('[PV V36 R1 predict] pipeline_version=' + PIPELINE_VERSION, 'mode=' + DEFAULT_MODE, 'curve_view=night_envelope_repaired; none_view/V27_TCN/V29_gate unchanged; ONLINE SCORE PENDING; NOT BASELINE PROMOTION', flush=True)
    _r1_diagnostics()
    base = _V27_BASE_PREDICT(save_output=False, mode=None)
    rows = base.copy()
    rows['_position'] = np.arange(len(base))
    rows = rows.merge(_v29_neural_forecast(), on=['pv_id', 'date'], validate='one_to_one')
    rows['_day'] = pd.to_datetime(rows.date)
    rows = rows.sort_values(['pv_id', '_day'], kind='stable')
    if len(rows) != 3100 or rows.groupby('pv_id').size().ne(31).any() or (not np.isfinite(rows[['pred', 'neural']].to_numpy()).all()):
        raise ValueError('Invalid V36 input rows')
    b = rows.pred.to_numpy().reshape(100, 31)
    n = rows.neural.to_numpy().reshape(100, 31)
    ordinary = ~rows['_day'].dt.day.isin([4, 5, 6, 7]).to_numpy().reshape(100, 31)
    values = b.copy()
    values[ordinary] = 0.75 * b[ordinary] + 0.25 * n[ordinary]
    risk = np.zeros_like(n, dtype=bool)
    np.put_along_axis(risk, np.argsort(n, axis=1, kind='stable')[:, :4], True, axis=1)
    active = risk & (n < values)
    rows['pred'] = (values * np.where(active, 0.7, 1.02)).reshape(-1)
    rows['_active'] = active.reshape(-1)
    event = rows['_day'].dt.day.isin([4, 5, 6, 7])
    date_counts = rows.groupby('date')['_active'].sum().astype(int).to_dict()
    result = rows.sort_values('_position')[['pv_id', 'date', 'pred']].reset_index(drop=True)
    if not np.isfinite(result.pred).all() or result.pred.lt(0).any():
        raise ValueError('Invalid V36 output')
    special_rows = pd.to_datetime(result.date).dt.day.isin([4, 5, 6, 7])
    print('[PV V36 R1 final] version=' + PIPELINE_VERSION, 'ordinary=0.75_V25(repaired_curve)+0.25_TCN then V29 gate; risk_multiplier=0.70 other_multiplier=1.02', 'active_rows=' + str(int(active.sum())), 'active_calendar_4_7_rows=' + str(int(rows.loc[event, '_active'].sum())), 'rows=3100', 'total=' + format(float(result.pred.sum()), '.17g'), 'special_total=' + format(float(result.loc[special_rows, 'pred'].sum()), '.17g'), 'ONLINE SCORE PENDING', flush=True)
    print('[PV V36 R1 risk dates]', json.dumps(date_counts, sort_keys=True), flush=True)
    if save_output:
        path = output_dir() / 'submit_result.csv'
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print('[PV V36 R1 output]', path, flush=True)
    return result
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V36 R1 repaired-curve forecast-only online probe')
    parser.add_argument('action', choices=['train', 'predict'])
    args = parser.parse_args()
    train_pv_model() if args.action == 'train' else predict_pv_model()