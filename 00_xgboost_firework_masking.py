# -*- coding: utf-8 -*-
"""
敏感性分析：剔除烟花时段后重算 lag_24 / rolling_mean_24 并重训 XGBoost
=====================================================================
审稿人意见：lag/rolling 特征是否包含了烟花开始后的 PM2.5 观测？
          反事实基线不应使用"事后观测"来构造后续事件小时的特征。

本脚本复现 时空极端树模型.ipynb 的核心流水线，唯一区别是：
  变体 A（原始）：lag_24 / rolling_mean_24 直接用原始 PM2.5 计算（现状）
  变体 B（剔除烟花）：先把每年除夕/元宵节当天 00:00–03:00 与 20:00–23:00 的
                   PM2.5 置为 NaN，再计算 shift(24) 与 rolling(24)，缺失处 ffill
                   —— 只遮特征，不遮目标（节日窗口与论文一致）

研究对象为地级行政区（地级市/自治州/地区/盟，共 335 个），已剔除县级市
（五家渠市、石河子市、塔城市、阿勒泰市、和田市、喀什市等）。
数据源：E:/烟花污染/我用的/全部年份_完整地级单位.csv
运行：`python 敏感性分析_剔除烟花重训练.py`
"""
import sys
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split

# 强制 UTF-8 输出，避免 Windows GBK 控制台无法编码 μg/m³ 等字符
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

warnings.filterwarnings('ignore')

# ---------------- 配置 ----------------
# 地级行政区校准后的数据（335 城，已剔除县级市）
DATA_PATH = "E:/烟花污染/我用的/全部年份_完整地级单位.csv"
DATA_COLS = ["NAME", "time_datetime", "PM2.5", "temp_c", "dewpoint_c", "rh",
             "wind_speed", "wind_direction", "surface_pressure", "tp"]

# 节日日期（与原始 notebook 一致）
NEW_YEAR_EVES = ["2020-01-24", "2021-02-11", "2022-01-31",
                 "2023-01-21", "2024-02-09", "2025-01-28"]   # 除夕
LANTERN_FESTIVALS = ["2020-02-08", "2021-02-26", "2022-02-15",
                     "2023-02-05", "2024-02-24", "2025-02-12"]  # 元宵节

PARAMS = {
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae']
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING = 50

BASE_FEATURES = ["wind_direction", "temp_c", "surface_pressure",
                 "wind_speed", "rh", "tp"]
TIME_FEATURES = ["month", "day_of_week", "hour", "is_weekend", "season"]
LAG_FEATURES = ["PM2.5_lag_24", "PM2.5_rolling_mean_24"]


# ---------------- 数据加载 ----------------
def load_data():
    df = pd.read_csv(DATA_PATH, usecols=DATA_COLS)
    print(f"已读取 {DATA_PATH}, 行数: {len(df)}")
    df = df.rename(columns={'time_datetime': 'time'})
    df['time'] = pd.to_datetime(df['time'])
    n_before = len(df)
    df = df.drop_duplicates(subset=['NAME', 'time'], keep='first')
    print(f"去重: {n_before:,} -> {len(df):,} 行 (移除 {n_before - len(df):,} 条重复)")
    df = df.sort_values(['NAME', 'time']).reset_index(drop=True)  # 保证 shift(24)=24小时
    print(f"城市数: {df['NAME'].nunique()}")
    return df


# ---------------- 基础填充（与 notebook 一致） ----------------
def basic_fill(df):
    numeric = ["wind_direction", "temp_c", "surface_pressure",
               "wind_speed", "rh", "tp", "PM2.5"]
    df[numeric] = df.groupby("NAME")[numeric].transform(
        lambda g: g.ffill().bfill().fillna(g.mean()))
    return df


# ---------------- 烟花时段 mask ----------------
def build_festival_events():
    """返回 (start, end, festival_type, festival_year) 列表：
    与论文一致——仅除夕/元宵节当天 00:00–03:00 与 20:00–23:00 两段，不含春节当天。"""
    events = []
    for d in NEW_YEAR_EVES:
        day = pd.Timestamp(d)
        events.append((day + pd.Timedelta(hours=0),
                       day + pd.Timedelta(hours=3),
                       '除夕', day.year))                          # 除夕凌晨
        events.append((day + pd.Timedelta(hours=20),
                       day + pd.Timedelta(hours=23),
                       '除夕', day.year))                          # 除夕夜间
    for d in LANTERN_FESTIVALS:
        day = pd.Timestamp(d)
        events.append((day + pd.Timedelta(hours=0),
                       day + pd.Timedelta(hours=3),
                       '元宵节', day.year))                        # 元宵凌晨
        events.append((day + pd.Timedelta(hours=20),
                       day + pd.Timedelta(hours=23),
                       '元宵节', day.year))                        # 元宵夜间
    return events


def build_firework_windows():
    """返回 (start, end) 列表：节日当天 00:00–03:00 与 20:00–23:00"""
    return [(s, e) for (s, e, _t, _y) in build_festival_events()]


def is_firework_hour(time_series):
    mask = pd.Series(False, index=time_series.index)
    for s, e in build_firework_windows():
        mask |= ((time_series >= s) & (time_series <= e))
    return mask


def festival_labels(time_series):
    """给每个时刻打上 festival_type / festival_year 标签（非节日为空串/0）"""
    ft = pd.Series('', index=time_series.index, dtype=object)
    fy = pd.Series(0, index=time_series.index, dtype='int64')
    for s, e, ftype, fyear in build_festival_events():
        inwin = (time_series >= s) & (time_series <= e)
        ft[inwin] = ftype
        fy[inwin] = fyear
    return ft, fy


# ---------------- 特征工程（与 notebook 的 enhance_features_simple 一致） ----------------
def enhance_features(df, pm25_col='PM2.5'):
    """pm25_col: 用哪一列 PM2.5 来构造 lag/rolling（变体B 传入被 mask 的列）"""
    df = df.copy()
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    df['hour'] = df['time'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['surface_pressure_safe'] = df['surface_pressure'].clip(lower=900, upper=1100)
    df['temp_pressure_ratio'] = df['temp_c'] / df['surface_pressure_safe']
    df['wind_effect'] = df['wind_speed'] * (df['rh'].clip(1, 100) / 100)
    df['day_of_year'] = df['time'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df['season'] = df['month'].map(season_map)

    # 滞后 / 滚动特征（关键：只遮特征，不遮目标 PM2.5）
    df['PM2.5_lag_24'] = df.groupby('NAME')[pm25_col].shift(24)
    df['PM2.5_rolling_mean_24'] = df.groupby('NAME')[pm25_col].transform(
        lambda x: x.rolling(24, min_periods=1).mean())
    df['PM2.5_rolling_std_24'] = df.groupby('NAME')[pm25_col].transform(
        lambda x: x.rolling(24, min_periods=1).std())

    rolling_cols = ['PM2.5_rolling_mean_24', 'PM2.5_rolling_std_24']
    df[rolling_cols] = df.groupby('NAME')[rolling_cols].transform(
        lambda x: x.ffill().bfill().fillna(x.mean()))
    df['PM2.5_lag_24'] = df.groupby('NAME')['PM2.5_lag_24'].transform(
        lambda x: x.ffill().bfill().fillna(x.mean()))
    return df


# ---------------- 异常值处理（与 notebook 一致） ----------------
def detect_and_remove_outliers(df, feature_columns, target_col='PM2.5',
                               target_threshold=3, feature_threshold=3):
    d = df.copy()
    Q1 = d[target_col].quantile(0.25)
    Q3 = d[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - target_threshold * IQR, Q3 + target_threshold * IQR
    d = d[(d[target_col] >= lo) & (d[target_col] <= hi)]

    for f in feature_columns:
        if f in d.columns and d[f].dtype in ['float64', 'int64']:
            z = np.abs((d[f] - d[f].mean()) / d[f].std())
            if (z > feature_threshold).any():
                lp, up = d[f].quantile(0.01), d[f].quantile(0.99)
                d[f] = d[f].clip(lower=lp, upper=up)
    d['PM2.5'] = d['PM2.5'].clip(lower=0)
    d['rh'] = d['rh'].clip(lower=0, upper=100)
    return d


# ---------------- 完整流水线（变体 A / B 共用） ----------------
def run_pipeline(df, pm25_col, label):
    feature_columns = BASE_FEATURES + TIME_FEATURES + LAG_FEATURES

    d = enhance_features(df, pm25_col=pm25_col)
    d = detect_and_remove_outliers(d, feature_columns=feature_columns)
    d = d.sort_values(['NAME', 'time']).reset_index(drop=True)

    # 时间 80/20 划分
    split_date = d['time'].quantile(0.8)
    train = d[d['time'] <= split_date].copy()
    test = d[d['time'] > split_date].copy()

    # 城市编码（对全量城市编码，保证全量可预测）
    le = LabelEncoder()
    all_cities = sorted(d['NAME'].unique())
    le.fit(all_cities)
    train['city_code_enc'] = le.transform(train['NAME'])
    test['city_code_enc'] = le.transform(test['NAME'])
    d['city_code_enc'] = le.transform(d['NAME'])

    final_cols = feature_columns + ['city_code_enc']

    dtrain = xgb.DMatrix(train[final_cols], label=train['PM2.5'],
                         feature_names=final_cols)
    dtest = xgb.DMatrix(test[final_cols], label=test['PM2.5'],
                        feature_names=final_cols)
    model = xgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
                      evals=[(dtrain, 'train'), (dtest, 'test')],
                      early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)

    # 全量预测（含训练期，保证能覆盖早期年份的节日）
    dfull = xgb.DMatrix(d[final_cols], feature_names=final_cols)
    d['PM2.5_pred'] = model.predict(dfull)
    d['PM2.5_diff'] = d['PM2.5'] - d['PM2.5_pred']

    test_rmse = float(np.sqrt(np.mean((test['PM2.5'] -
                                       model.predict(dtest)) ** 2)))
    print(f"[{label}] 测试集 RMSE = {test_rmse:.3f}, "
          f"best_iteration = {model.best_iteration}")
    return d, test_rmse


# ---------------- K 折交叉验证 ----------------
def cross_validate(d, label, n_splits_list=(5, 10)):
    """K 折交叉验证（随机打乱、固定种子）。返回 {k: (rmse, mae, r2)}"""
    feature_columns = BASE_FEATURES + TIME_FEATURES + LAG_FEATURES
    final_cols = feature_columns + ['city_code_enc']
    X = d[final_cols].values
    y = d['PM2.5'].values
    results = {}
    for k in n_splits_list:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        oof = np.empty(len(d))
        for tr_idx, va_idx in kf.split(X):
            # Nested validation: the outer fold remains untouched for final evaluation.
            inner_tr_idx, inner_val_idx = train_test_split(
                tr_idx, test_size=0.2, random_state=42
            )
            dtrain = xgb.DMatrix(X[inner_tr_idx], label=y[inner_tr_idx],
                                  feature_names=final_cols)
            dinner_val = xgb.DMatrix(X[inner_val_idx], label=y[inner_val_idx],
                                     feature_names=final_cols)
            douter_val = xgb.DMatrix(X[va_idx], label=y[va_idx],
                                      feature_names=final_cols)
            model = xgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
                              evals=[(dinner_val, 'inner_val')],
                              early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
            # The outer fold is used only once, after model selection.
            oof[va_idx] = model.predict(douter_val)
        ss_res = np.sum((y - oof) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        rmse = float(np.sqrt(ss_res / len(y)))
        mae = float(np.mean(np.abs(y - oof)))
        r2 = float(1 - ss_res / ss_tot)
        results[k] = (rmse, mae, r2)
        print(f"[{label}] {k}-fold CV: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.4f}")
    return results


# ---------------- 节日时段统计 ----------------
def festival_summary(d):
    mask = is_firework_hour(d['time'])
    fest = d[mask].copy()
    # 烟花贡献非负：把负的 diff（模型高估）截为 0，再参与平均
    fest['PM2.5_diff_pos'] = fest['PM2.5_diff'].clip(lower=0)
    fest['ratio'] = (fest['PM2.5_diff_pos'] /
                     np.clip(fest['PM2.5'], 1, None)) * 100
    city = fest.groupby('NAME').agg(
        diff_mean=('PM2.5_diff_pos', 'mean'),
        ratio_mean=('ratio', 'mean'),
        pm25_mean=('PM2.5', 'mean'),
        pred_mean=('PM2.5_pred', 'mean'),
        n=('PM2.5_diff_pos', 'count')
    ).reset_index()
    return fest, city


def top20(city):
    return city.nlargest(20, 'diff_mean')[['NAME', 'diff_mean', 'ratio_mean']].reset_index(drop=True)


# ---------------- 节日拟合结果导出（仿照原 CSV 表头） ----------------
EXPORT_COLS = ['time_datetime', 'date', 'year', 'month', 'day', 'hour',
               'festival_type', 'festival_year', 'city', 'PM2.5', 'PM2.5_pred',
               'wind_direction', 'temp_c', 'surface_pressure', 'wind_speed',
               'rh', 'tp', 'day_of_week', 'PM2.5_lag_24',
               'PM2.5_rolling_mean_24', 'PM2.5_diff', 'PM2.5_abs_diff',
               'PM2.5_relative_error_pct', 'Records', 'Score', 'Label']


def export_festival_csv(d, label, out_path):
    mask = is_firework_hour(d['time'])
    fest = d[mask].copy().sort_values(['time', 'NAME']).reset_index(drop=True)

    t = fest['time']
    fest['time_datetime'] = [f"{y}/{m}/{dd} {h}:00" for y, m, dd, h in
                             zip(t.dt.year, t.dt.month, t.dt.day, t.dt.hour)]
    fest['date'] = [f"{y}/{m}/{dd}" for y, m, dd in
                    zip(t.dt.year, t.dt.month, t.dt.day)]
    fest['year'] = t.dt.year
    fest['month'] = t.dt.month
    fest['day'] = t.dt.day
    fest['hour'] = t.dt.hour
    fest['city'] = fest['NAME']
    ft, fy = festival_labels(fest['time'])
    fest['festival_type'] = ft
    fest['festival_year'] = fy
    fest['PM2.5_abs_diff'] = fest['PM2.5_diff'].abs()
    fest['PM2.5_relative_error_pct'] = (
        fest['PM2.5_diff'].abs() / np.clip(fest['PM2.5'], 1, None)) * 100
    fest['Records'] = ''
    fest['Score'] = ''
    fest['Label'] = ''

    fest[EXPORT_COLS].to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[{label}] 已保存节日拟合结果: {out_path} ({len(fest)} 行)")
    return fest


# ---------------- 稳健性检验：带符号残差 / 安慰剂 / 城市聚类自助 CI ----------------
def signed_residual_stats(fest):
    """带符号残差统计：负残差（模型高估）的个数与比例，及带符号/截断均值。"""
    d = fest['PM2.5_diff']
    neg = d[d < 0]
    return {
        'n': len(d),
        'mean_signed': float(d.mean()),
        'mean_clipped': float(d.clip(lower=0).mean()),
        'neg_count': int((d < 0).sum()),
        'neg_ratio': float((d < 0).mean()),
        'neg_mean': float(neg.mean()) if len(neg) else 0.0,
        'pos_count': int((d > 0).sum()),
        'pos_ratio': float((d > 0).mean()),
    }


def build_placebo_windows():
    """相邻周安慰剂：每个节日日期 ±7 天，取同样 00:00–03:00 与 20:00–23:00 两段；
    排除与真实节日窗口重叠的时段。"""
    festival_windows = build_firework_windows()
    placebo = []
    for d in NEW_YEAR_EVES + LANTERN_FESTIVALS:
        day = pd.Timestamp(d)
        for offset in (-7, 7):
            base = day + pd.Timedelta(days=offset)
            for sh, eh in ((0, 3), (20, 23)):
                s = base + pd.Timedelta(hours=sh)
                e = base + pd.Timedelta(hours=eh)
                if any((s <= fe) and (e >= fs) for fs, fe in festival_windows):
                    continue
                placebo.append((s, e))
    return placebo


def placebo_summary(d):
    """安慰剂时段（相邻周 ±7 天、同样 8 小时）的残差统计。"""
    windows = build_placebo_windows()
    mask = pd.Series(False, index=d.index)
    for s, e in windows:
        mask |= ((d['time'] >= s) & (d['time'] <= e))
    pl = d[mask].copy()
    pl['PM2.5_diff_pos'] = pl['PM2.5_diff'].clip(lower=0)
    pl['ratio'] = (pl['PM2.5_diff_pos'] / np.clip(pl['PM2.5'], 1, None)) * 100
    return pl


def bootstrap_ci(fest, value_col, n_iter=1000, seed=42, cluster_col='NAME'):
    """城市聚类自助法：按城市整簇有放回重抽样，重复 n_iter 次取均值分布的 2.5%/97.5% 分位。"""
    rng = np.random.default_rng(seed)
    cities = np.array(sorted(fest[cluster_col].unique()))
    n_cities = len(cities)
    ests = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng.choice(cities, size=n_cities, replace=True)
        sub = fest[fest[cluster_col].isin(sample)]
        ests[i] = sub[value_col].mean()
    lo, hi = np.percentile(ests, [2.5, 97.5])
    return float(ests.mean()), float(lo), float(hi)


def diff_bootstrap_ci(fest, placebo, value_col, n_iter=1000, seed=42):
    """节日 vs 安慰剂差值的城市聚类自助 95% CI（核心验证：差值显著>0 才说明真实烟花贡献）"""
    rng = np.random.default_rng(seed)
    cities = np.array(sorted(fest['NAME'].unique()))
    ests = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng.choice(cities, size=len(cities), replace=True)
        f = fest[fest['NAME'].isin(sample)][value_col].mean()
        p = placebo[placebo['NAME'].isin(sample)][value_col].mean()
        ests[i] = f - p
    lo, hi = np.percentile(ests, [2.5, 97.5])
    return float(ests.mean()), float(lo), float(hi)


# ---------------- 主流程 ----------------
if __name__ == '__main__':
    df = load_data()
    df = basic_fill(df)

    firework_mask = is_firework_hour(df['time'])
    n_masked = int(firework_mask.sum())
    print(f"烟花时段行数: {n_masked:,} "
          f"({100 * n_masked / len(df):.2f}% of {len(df):,})")

    # 变体 B 的特征输入列：烟花时段 PM2.5 置 NaN（只遮特征）
    df['PM2.5_masked'] = df['PM2.5'].where(~firework_mask)

    print("\n==== 变体 A：原始特征 ====")
    dA, rmseA = run_pipeline(df, pm25_col='PM2.5', label='A(原始)')
    print("==== 变体 B：剔除烟花时段 ====")
    dB, rmseB = run_pipeline(df, pm25_col='PM2.5_masked', label='B(剔除烟花)')

    festA, cityA = festival_summary(dA)
    festB, cityB = festival_summary(dB)
    tA, tB = top20(cityA), top20(cityB)

    # 导出节日拟合结果（仿照原 CSV 表头）
    export_festival_csv(dA, 'A原始', 'festival_predictions_A_原始.csv')
    export_festival_csv(dB, 'B剔除烟花', 'festival_predictions_B_剔除烟花.csv')

    # 先保存结果（避免打印阶段编码错误导致结果丢失）
    comp = cityA.merge(cityB, on='NAME', suffixes=('_A', '_B'))
    comp.to_csv('敏感性分析_城市对比.csv', index=False, encoding='utf-8-sig')
    tA.to_csv('敏感性分析_top20_A.csv', index=False, encoding='utf-8-sig')
    tB.to_csv('敏感性分析_top20_B.csv', index=False, encoding='utf-8-sig')

    # ---- 稳健性检验：带符号残差 / 安慰剂 / 城市聚类自助 95% CI ----
    print("\n" + "=" * 70)
    print("稳健性检验：带符号残差、安慰剂对照、城市聚类自助 95% CI")
    print("=" * 70)

    sA = signed_residual_stats(festA)
    sB = signed_residual_stats(festB)
    plA = placebo_summary(dA)
    plB = placebo_summary(dB)
    ciA_signed = bootstrap_ci(festA, 'PM2.5_diff')
    ciB_signed = bootstrap_ci(festB, 'PM2.5_diff')
    ciA_clip = bootstrap_ci(festA, 'PM2.5_diff_pos')
    ciB_clip = bootstrap_ci(festB, 'PM2.5_diff_pos')
    diffB = diff_bootstrap_ci(festB, plB, 'PM2.5_diff')

    print("\n---- 带符号残差统计（节日时段） ----")
    print(f"{'指标':<24}{'A 原始':>14}{'B 剔除烟花':>14}")
    print("-" * 52)
    print(f"{'样本数':<24}{sA['n']:>14,}{sB['n']:>14,}")
    print(f"{'带符号均值 (ug/m3)':<24}{sA['mean_signed']:>14.2f}{sB['mean_signed']:>14.2f}")
    print(f"{'截断均值 (ug/m3)':<24}{sA['mean_clipped']:>14.2f}{sB['mean_clipped']:>14.2f}")
    print(f"{'负残差个数':<24}{sA['neg_count']:>14,}{sB['neg_count']:>14,}")
    print(f"{'负残差比例 (%)':<24}{sA['neg_ratio']*100:>12.2f}% {sB['neg_ratio']*100:>12.2f}%")
    print(f"{'正残差个数':<24}{sA['pos_count']:>14,}{sB['pos_count']:>14,}")
    print(f"{'正残差比例 (%)':<24}{sA['pos_ratio']*100:>12.2f}% {sB['pos_ratio']*100:>12.2f}%")

    print("\n---- 安慰剂对照（相邻周 ±7 天，同样 8 小时，排除节日） ----")
    print(f"{'指标':<24}{'A 原始':>14}{'B 剔除烟花':>14}")
    print("-" * 52)
    print(f"{'安慰剂样本数':<24}{len(plA):>14,}{len(plB):>14,}")
    print(f"{'带符号均值 (ug/m3)':<24}{plA['PM2.5_diff'].mean():>14.2f}{plB['PM2.5_diff'].mean():>14.2f}")
    print(f"{'截断均值 (ug/m3)':<24}{plA['PM2.5_diff_pos'].mean():>14.2f}{plB['PM2.5_diff_pos'].mean():>14.2f}")
    print(f"{'节日截断均值 (对照)':<24}{sA['mean_clipped']:>14.2f}{sB['mean_clipped']:>14.2f}")

    print("\n---- 城市聚类自助法 95% CI（节日时段均值, 1000 次, seed=42） ----")
    print(f"{'指标':<20}{'A 原始':>26}{'B 剔除烟花':>26}")
    print("-" * 72)
    print(f"{'带符号均值 CI':<20}"
          f"{f'[{ciA_signed[1]:.2f}, {ciA_signed[2]:.2f}]':>26}"
          f"{f'[{ciB_signed[1]:.2f}, {ciB_signed[2]:.2f}]':>26}")
    print(f"{'截断均值 CI':<20}"
          f"{f'[{ciA_clip[1]:.2f}, {ciA_clip[2]:.2f}]':>26}"
          f"{f'[{ciB_clip[1]:.2f}, {ciB_clip[2]:.2f}]':>26}")
    print(f"\n---- 节日−安慰剂 带符号差值（B 剔除烟花, 城市聚类自助 95% CI） ----")
    print(f"差值均值 = {diffB[0]:.2f} ug/m3, 95% CI = [{diffB[1]:.2f}, {diffB[2]:.2f}]")

    rob_rows = [
        {'指标': '节日样本数', 'A 原始': sA['n'], 'B 剔除烟花': sB['n']},
        {'指标': '带符号均值 (ug/m3)', 'A 原始': round(sA['mean_signed'], 3), 'B 剔除烟花': round(sB['mean_signed'], 3)},
        {'指标': '截断均值 (ug/m3)', 'A 原始': round(sA['mean_clipped'], 3), 'B 剔除烟花': round(sB['mean_clipped'], 3)},
        {'指标': '负残差个数', 'A 原始': sA['neg_count'], 'B 剔除烟花': sB['neg_count']},
        {'指标': '负残差比例 (%)', 'A 原始': round(sA['neg_ratio'] * 100, 2), 'B 剔除烟花': round(sB['neg_ratio'] * 100, 2)},
        {'指标': '正残差个数', 'A 原始': sA['pos_count'], 'B 剔除烟花': sB['pos_count']},
        {'指标': '正残差比例 (%)', 'A 原始': round(sA['pos_ratio'] * 100, 2), 'B 剔除烟花': round(sB['pos_ratio'] * 100, 2)},
        {'指标': '带符号均值 CI 下界', 'A 原始': round(ciA_signed[1], 3), 'B 剔除烟花': round(ciB_signed[1], 3)},
        {'指标': '带符号均值 CI 上界', 'A 原始': round(ciA_signed[2], 3), 'B 剔除烟花': round(ciB_signed[2], 3)},
        {'指标': '截断均值 CI 下界', 'A 原始': round(ciA_clip[1], 3), 'B 剔除烟花': round(ciB_clip[1], 3)},
        {'指标': '截断均值 CI 上界', 'A 原始': round(ciA_clip[2], 3), 'B 剔除烟花': round(ciB_clip[2], 3)},
        {'指标': '安慰剂带符号均值 (ug/m3)', 'A 原始': round(plA['PM2.5_diff'].mean(), 3), 'B 剔除烟花': round(plB['PM2.5_diff'].mean(), 3)},
        {'指标': '安慰剂截断均值 (ug/m3)', 'A 原始': round(plA['PM2.5_diff_pos'].mean(), 3), 'B 剔除烟花': round(plB['PM2.5_diff_pos'].mean(), 3)},
        {'指标': '节日−安慰剂带符号差值均值 (ug/m3)', 'A 原始': None, 'B 剔除烟花': round(diffB[0], 3)},
        {'指标': '节日−安慰剂带符号差值 CI 下界', 'A 原始': None, 'B 剔除烟花': round(diffB[1], 3)},
        {'指标': '节日−安慰剂带符号差值 CI 上界', 'A 原始': None, 'B 剔除烟花': round(diffB[2], 3)},
    ]
    pd.DataFrame(rob_rows).to_csv('敏感性分析_稳健性检验.csv', index=False, encoding='utf-8-sig')

    # ---- K 折交叉验证（审稿人建议：历史重建应使用 CV 而非时间切分） ----
    print("\n==== K 折交叉验证（随机打乱, seed=42） ====")
    cvA = cross_validate(dA, 'A原始')
    cvB = cross_validate(dB, 'B剔除烟花')
    cv_rows = []
    for k in sorted(cvA):
        a, b = cvA[k], cvB[k]
        cv_rows.append({'fold': k,
                        'A_RMSE': round(a[0], 3), 'A_MAE': round(a[1], 3), 'A_R2': round(a[2], 4),
                        'B_RMSE': round(b[0], 3), 'B_MAE': round(b[1], 3), 'B_R2': round(b[2], 4)})
    pd.DataFrame(cv_rows).to_csv('敏感性分析_CV汇总.csv', index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print("K 折交叉验证汇总")
    print("=" * 70)
    print(f"{'fold':<6}{'A RMSE':>10}{'A MAE':>10}{'A R2':>10}"
          f"{'B RMSE':>10}{'B MAE':>10}{'B R2':>10}")
    print("-" * 70)
    for k in sorted(cvA):
        a, b = cvA[k], cvB[k]
        print(f"{k:<6}{a[0]:>10.3f}{a[1]:>10.3f}{a[2]:>10.4f}"
              f"{b[0]:>10.3f}{b[1]:>10.3f}{b[2]:>10.4f}")

    overlap = len(set(tA['NAME']) & set(tB['NAME']))
    rows = [
        ('平均正向diff (ug/m3)', festA['PM2.5_diff_pos'].mean(), festB['PM2.5_diff_pos'].mean()),
        ('最大 diff (ug/m3)', festA['PM2.5_diff'].max(), festB['PM2.5_diff'].max()),
        ('平均相对贡献 (%)', festA['ratio'].mean(), festB['ratio'].mean()),
        ('平均 PM2.5 实际值', festA['PM2.5'].mean(), festB['PM2.5'].mean()),
        ('平均 PM2.5 预测值', festA['PM2.5_pred'].mean(), festB['PM2.5_pred'].mean()),
        ('测试集 RMSE', rmseA, rmseB),
    ]

    # ---- 对比输出 ----
    print("\n" + "=" * 70)
    print("节日时段（除夕/元宵节当天 00:00-03:00 与 20:00-23:00）人为贡献对比")
    print("=" * 70)
    print(f"{'指标':<22}{'A 原始':>12}{'B 剔除烟花':>14}{'变化(B-A)':>14}")
    print("-" * 70)
    for name, a, b in rows:
        print(f"{name:<22}{a:>12.2f}{b:>14.2f}{b - a:>+14.2f}")

    print(f"\n前20城市名单重合数: {overlap} / 20")
    print(f"前20城市平均diff: A={tA['diff_mean'].mean():.2f}, "
          f"B={tB['diff_mean'].mean():.2f}")

    print("\n---- 前20城市（A 原始） ----")
    print(tA.to_string(index=False))
    print("\n---- 前20城市（B 剔除烟花） ----")
    print(tB.to_string(index=False))

    print("\n已保存: 敏感性分析_城市对比.csv / _top20_A.csv / _top20_B.csv / _稳健性检验.csv")
