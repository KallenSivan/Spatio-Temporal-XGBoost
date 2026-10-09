import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

BASE = Path(__file__).resolve().parent / "data"
PANEL = BASE / "城市年面板_2023_2025.csv"
COORD = BASE / "城市坐标.csv"

LCC = "+proj=lcc +lat_1=25 +lat_2=47 +lon_0=104 +datum=WGS84 +units=m +no_defs"

AUTO_PREF = {
    "大理白族自治州": "大理", "德宏傣族景颇族自治州": "德宏", "怒江傈僳族自治州": "怒江",
    "文山壮族苗族自治州": "文山", "楚雄彝族自治州": "楚雄", "红河哈尼族彝族自治州": "红河",
    "西双版纳傣族自治州": "西双版纳", "迪庆藏族自治州": "迪庆", "延边朝鲜族自治州": "延边",
    "凉山彝族自治州": "凉山", "甘孜藏族自治州": "甘孜", "阿坝藏族羌族自治州": "阿坝",
    "恩施土家族苗族自治州": "恩施", "湘西土家族苗族自治州": "湘西",
    "黔东南苗族侗族自治州": "黔东南", "黔南布依族苗族自治州": "黔南", "黔西南布依族苗族自治州": "黔西南",
    "临夏回族自治州": "临夏", "甘南藏族自治州": "甘南", "海北藏族自治州": "海北",
    "海南藏族自治州": "海南", "海西蒙古族藏族自治州": "海西", "玉树藏族自治州": "玉树",
    "黄南藏族自治州": "黄南", "果洛藏族自治州": "果洛", "伊犁哈萨克自治州": "伊犁",
    "克孜勒苏柯尔克孜自治州": "克孜勒苏", "博尔塔拉蒙古自治州": "博尔塔拉",
    "巴音郭楞蒙古自治州": "巴音郭楞", "昌吉回族自治州": "昌吉",
}


def core(name):
    n = str(name).strip()
    if n in AUTO_PREF:
        return AUTO_PREF[n]
    if not n.endswith("地区") and (n.endswith("区") or n.endswith("新区")):
        return None
    for suf in ("自治州", "地区", "盟", "市", "县"):
        if n.endswith(suf):
            return n[:-len(suf)]
    return n


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=0)


def load(path):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise RuntimeError(f"Cannot read {path}")


def main():
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

    df = load(PANEL)
    coord = load(COORD)
    coord["core"] = coord["core"].astype(str)
    df["core"] = df["city"].map(core)
    df = df.merge(coord[["core", "lon", "lat"]], on="core", how="left")
    trans = Transformer.from_crs("EPSG:4326", LCC, always_xy=True)
    df["x"], df["y"] = trans.transform(df["lon"].values, df["lat"].values)

    cols = ["fireworks_pm25_mean", "pd", "Green", "NTL", "Records"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["Records"] == 0, "Records"] = np.nan

    results = []
    for year in (2023, 2024):
        g = df[df["year"] == year].dropna(subset=cols + ["x", "y"]).reset_index(drop=True)
        y = g["fireworks_pm25_mean"].to_numpy(float).reshape(-1, 1)
        rec_log = np.log1p(g["Records"].to_numpy(float))
        X = np.column_stack([
            zscore(g["pd"]), zscore(g["Green"]), zscore(g["NTL"]), zscore(rec_log)
        ])
        coords = g[["x", "y"]].to_numpy(float)
        sel = Sel_BW(coords, y, X, kernel="gaussian", fixed=False, n_jobs=1)
        bw = int(round(sel.search(bw_min=20, bw_max=len(g))))
        res = GWR(coords, y, X, bw=bw, kernel="gaussian", fixed=False, n_jobs=1).fit()
        crit = res.critical_tval()
        sig = np.abs(res.tvalues[:, 4]) > crit
        rec = res.params[:, 4]
        print(f"YEAR={year} N={len(g)} bw={bw} R2={res.R2:.6f} adjR2={res.adj_R2:.6f} AICc={res.aicc:.3f} crit_t={crit:.3f}")
        print(f"  Records log1p-z coefficient: mean={rec.mean():.6f} median={np.median(rec):.6f} P25={np.percentile(rec,25):.6f} P75={np.percentile(rec,75):.6f} min={rec.min():.6f} max={rec.max():.6f}")
        print(f"  negative={int((rec < 0).sum())}/{len(rec)} significant={int(sig.sum())}/{len(rec)} significant_negative={int((sig & (rec < 0)).sum())} significant_positive={int((sig & (rec > 0)).sum())}")
        out = g[["city", "year", "fireworks_pm25_mean", "Records"]].copy()
        out["log1p_Records"] = rec_log
        out["Records_log1p_z_coef"] = rec
        out["Records_log1p_z_t"] = res.tvalues[:, 4]
        out["Records_log1p_z_sig"] = sig
        out.to_csv(Path.cwd() / f"gwr_{year}_log1p_records_firework_rawY.csv", index=False, encoding="utf-8-sig")
        results.append({"year": year, "N": len(g), "bw": bw, "R2": res.R2, "adjR2": res.adj_R2, "AICc": res.aicc, "crit_t": crit, "coef_mean": rec.mean(), "coef_median": np.median(rec), "coef_p25": np.percentile(rec,25), "coef_p75": np.percentile(rec,75), "coef_min": rec.min(), "coef_max": rec.max(), "negative_n": int((rec < 0).sum()), "significant_n": int(sig.sum()), "sig_negative_n": int((sig & (rec < 0)).sum()), "sig_positive_n": int((sig & (rec > 0)).sum())})
    pd.DataFrame(results).to_csv(Path.cwd() / "gwr_log1p_records_firework_rawY_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
