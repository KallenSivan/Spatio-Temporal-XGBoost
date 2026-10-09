import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from libpysal.weights import Queen, Rook, DistanceBand
from esda.moran import Moran

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"
SHAPEFILES = DATA / "shapefiles"
OUT = str(PROJECT_ROOT / "outputs" / "moran_analysis_updated_B")
CSV = str(DATA / "festival_predictions_B_剔除烟花.csv")
SHP = str(SHAPEFILES / "中国_市.shp")
os.makedirs(OUT, exist_ok=True)
LCC = "+proj=lcc +lat_1=25 +lat_2=47 +lon_0=104 +datum=WGS84 +units=m +no_defs"

def load_csv(path):
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="utf-8", errors="replace")

def norm_name(x):
    s = "" if pd.isna(x) else str(x).strip()
    for suffix in ["省", "市", "区", "县", "自治州", "地区", "特别行政区", "壮族自治区", "回族自治区", "维吾尔自治区", "自治区", "盟", "旗"]:
        s = s.replace(suffix, "")
    return s.replace(" ", "").replace("　", "")

def idw(gdf, value_col="pm25_mean", power=2):
    known = gdf[gdf[value_col].notna()].copy()
    target = gdf[gdf[value_col].isna()].copy()
    if known.empty or target.empty:
        return pd.Series(index=target.index, dtype=float)
    known_xy = np.array([(g.centroid.x, g.centroid.y) for g in known.geometry])
    target_xy = np.array([(g.centroid.x, g.centroid.y) for g in target.geometry])
    dist = cdist(target_xy, known_xy)
    weights = 1.0 / (dist ** power + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    return pd.Series(weights @ known[value_col].to_numpy(), index=target.index)

df = load_csv(CSV)
df["year_int"] = pd.to_numeric(df["festival_year"], errors="coerce").astype("Int64")
df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
df = df[df["year_int"].between(2020, 2025)].copy()
gdf = gpd.read_file(SHP).to_crs(LCC)
name_col = next(c for c in ["NAME_2", "NAME_1", "NAME", "name", "市", "city", "City"] if c in gdf.columns)

shp_names = gdf[name_col].dropna().unique()
mapping = {}
for city in df["city"].dropna().unique():
    nc = norm_name(city)
    scores = []
    for shp_city in shp_names:
        ns = norm_name(shp_city)
        score = 100 if nc == ns else 90 if nc in ns else 85 if ns in nc else 50 * len(set(nc) & set(ns)) / max(len(nc), len(ns), 1)
        scores.append((score, shp_city))
    score, best = max(scores)
    if score >= 60:
        mapping[city] = best

results = []
for year in sorted(df["year_int"].dropna().unique().astype(int)):
    dy = df[df["year_int"] == year]
    city_avg = dy.groupby("city")["PM2.5"].agg(["mean", "count", "std"]).reset_index()
    city_avg.columns = ["csv_city", "pm25_mean", "data_count", "pm25_std"]
    city_avg = city_avg[city_avg.data_count >= 3].copy()

    analysis = gdf[[name_col, "geometry"]].copy().rename(columns={name_col: "city_name"})
    analysis["pm25_mean"] = np.nan
    analysis["data_count"] = 0
    analysis["is_interpolated"] = False
    for _, row in city_avg.iterrows():
        shp_city = mapping.get(row.csv_city)
        if shp_city is not None:
            mask = analysis.city_name == shp_city
            analysis.loc[mask, "pm25_mean"] = row.pm25_mean
            analysis.loc[mask, "data_count"] = row.data_count
    original = analysis.pm25_mean.notna()
    original_count = int(original.sum())
    missing = ~original
    if original_count >= 5 and missing.any():
        analysis.loc[missing, "pm25_mean"] = idw(analysis)[missing]
        analysis.loc[missing, "is_interpolated"] = True
    analysis = analysis.dropna(subset=["pm25_mean"]).copy()
    if len(analysis) < 10:
        continue

    # Match the notebook: try Queen first, then Rook and distance bands.
    weight_specs = [
        ("Queen", lambda: Queen.from_dataframe(analysis, silence_warnings=True)),
        ("Rook", lambda: Rook.from_dataframe(analysis, silence_warnings=True)),
        ("DistanceBand (200km)", lambda: DistanceBand.from_dataframe(analysis, threshold=200000)),
        ("DistanceBand (300km)", lambda: DistanceBand.from_dataframe(analysis, threshold=300000)),
    ]
    w = None
    wname = None
    for candidate, make_w in weight_specs:
        try:
            w = make_w()
            wname = candidate
            break
        except Exception:
            pass
    if w is None:
        continue
    w.transform = "r"
    moran = Moran(analysis.pm25_mean.to_numpy(), w)
    p = moran.p_norm if hasattr(moran, "p_norm") else moran.p_sim
    z = moran.z_norm if hasattr(moran, "z_norm") else moran.z_sim
    results.append({
        "year": year,
        "n_cities": len(analysis),
        "n_original": original_count,
        "n_interpolated": len(analysis) - original_count,
        "weight_matrix": wname,
        "mean_neighbors": w.mean_neighbors,
        "moran_i": moran.I,
        "expected_i": moran.EI,
        "z_score": z,
        "p_value": p,
        "significant": p < 0.05,
        "mean_pm25": analysis.pm25_mean.mean(),
        "min_pm25": analysis.pm25_mean.min(),
        "max_pm25": analysis.pm25_mean.max(),
    })
    analysis[["city_name", "pm25_mean", "data_count", "is_interpolated"]].assign(year=year).to_csv(
        os.path.join(OUT, f"city_pm25_{year}.csv"), index=False, encoding="utf-8-sig"
    )

out = pd.DataFrame(results)
out.to_csv(os.path.join(OUT, "global_moran_2020_2025_updated_B.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "global_moran_report_updated_B.txt"), "w", encoding="utf-8") as f:
    f.write("Annual global Moran's I for updated B data\n")
    f.write(out.to_string(index=False))
print(out.to_string(index=False))
print(f"\nSaved to: {OUT}")
