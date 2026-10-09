from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent / "data"
PANEL = BASE / "城市年面板_2023_2025.csv"
OUT = Path.cwd() / "panel_2023_2024_log1p_records_firework.csv"

def load(path):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise RuntimeError(path)

def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)

def fmt(name, model):
    return f"  {name:<30} coef={model.params[name]:>9.4f}  SE={model.bse[name]:>9.4f}  p={model.pvalues[name]:.5f}"

df = load(PANEL)
df = df[df.year.isin([2023, 2024])].copy()
for c in ["fireworks_pm25_mean", "pd", "Green", "NTL", "Records"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.loc[df.Records == 0, "Records"] = np.nan
df["Records_log1p_z"] = zscore(np.log1p(df["Records"]))
for c in ["pd", "Green", "NTL"]:
    df[c + "_z"] = zscore(df[c])
df["year2024"] = (df.year == 2024).astype(int)
df["Records_x_2023"] = df.Records_log1p_z * (df.year == 2023).astype(int)
df["Records_x_2024"] = df.Records_log1p_z * (df.year == 2024).astype(int)
g = df.dropna(subset=["fireworks_pm25_mean", "Records_log1p_z", "pd_z", "Green_z", "NTL_z"]).reset_index(drop=True)

def fit(formula):
    return smf.ols(formula, data=g).fit(cov_type="cluster", cov_kwds={"groups": g["city"]})

mA = fit("fireworks_pm25_mean ~ Records_log1p_z + pd_z + Green_z + NTL_z + year2024 + Records_log1p_z:year2024")
mB = fit("fireworks_pm25_mean ~ Records_x_2023 + Records_x_2024 + pd_z + Green_z + NTL_z + year2024")

print(f"N={len(g)} cities={g.city.nunique()} 2023={int((g.year==2023).sum())} 2024={int((g.year==2024).sum())}")
print(f"Y mean={g.fireworks_pm25_mean.mean():.4f} SD={g.fireworks_pm25_mean.std():.4f}")
print("\nMODEL A interaction (reference year 2023)")
print(f"R2={mA.rsquared:.6f} adjR2={mA.rsquared_adj:.6f}")
for n in ["Records_log1p_z", "pd_z", "Green_z", "NTL_z", "year2024", "Records_log1p_z:year2024"]:
    print(fmt(n, mA))
print("\nMODEL B yearly slopes")
print(f"R2={mB.rsquared:.6f} adjR2={mB.rsquared_adj:.6f}")
for n in ["Records_x_2023", "Records_x_2024", "pd_z", "Green_z", "NTL_z", "year2024"]:
    print(fmt(n, mB))

rows = []
for tag, model, names in [
    ("A_interaction", mA, ["Records_log1p_z", "pd_z", "Green_z", "NTL_z", "year2024", "Records_log1p_z:year2024"]),
    ("B_yearly", mB, ["Records_x_2023", "Records_x_2024", "pd_z", "Green_z", "NTL_z", "year2024"]),
]:
    for n in names:
        rows.append({"model": tag, "term": n, "coef": model.params[n], "se": model.bse[n], "p": model.pvalues[n]})
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUT}")
