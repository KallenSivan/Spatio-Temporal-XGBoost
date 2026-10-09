from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from mgwr.gwr import GWR
from libpysal.weights import KNN
from esda.moran import Moran

BASE = Path(__file__).resolve().parent / "data"
PANEL = BASE / "城市年面板_2023_2025.csv"
COORD = BASE / "城市坐标.csv"
LCC = "+proj=lcc +lat_1=25 +lat_2=47 +lon_0=104 +datum=WGS84 +units=m +no_defs"
AUTO_PREF = {"大理白族自治州":"大理", "德宏傣族景颇族自治州":"德宏", "怒江傈僳族自治州":"怒江", "文山壮族苗族自治州":"文山", "楚雄彝族自治州":"楚雄", "红河哈尼族彝族自治州":"红河", "西双版纳傣族自治州":"西双版纳", "迪庆藏族自治州":"迪庆", "延边朝鲜族自治州":"延边", "凉山彝族自治州":"凉山", "甘孜藏族自治州":"甘孜", "阿坝藏族羌族自治州":"阿坝", "恩施土家族苗族自治州":"恩施", "湘西土家族苗族自治州":"湘西", "黔东南苗族侗族自治州":"黔东南", "黔南布依族苗族自治州":"黔南", "黔西南布依族苗族自治州":"黔西南", "临夏回族自治州":"临夏", "甘南藏族自治州":"甘南", "海北藏族自治州":"海北", "海南藏族自治州":"海南", "海西蒙古族藏族自治州":"海西", "玉树藏族自治州":"玉树", "黄南藏族自治州":"黄南", "果洛藏族自治州":"果洛", "伊犁哈萨克自治州":"伊犁", "克孜勒苏柯尔克孜自治州":"克孜勒苏", "博尔塔拉蒙古自治州":"博尔塔拉", "巴音郭楞蒙古自治州":"巴音郭楞", "昌吉回族自治州":"昌吉"}

def core(name):
    n = str(name).strip()
    if n in AUTO_PREF: return AUTO_PREF[n]
    if not n.endswith("地区") and (n.endswith("区") or n.endswith("新区")): return None
    for suf in ("自治州", "地区", "盟", "市", "县"):
        if n.endswith(suf): return n[:-len(suf)]
    return n

def load(p):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try: return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError: pass

def z(x):
    x = np.asarray(x, float)
    return (x-x.mean())/x.std(ddof=0)

def main():
    df = load(PANEL); co = load(COORD); co["core"] = co["core"].astype(str)
    df["core"] = df["city"].map(core); df = df.merge(co[["core","lon","lat"]], on="core", how="left")
    tr = Transformer.from_crs("EPSG:4326", LCC, always_xy=True)
    df["x"], df["y"] = tr.transform(df["lon"].values, df["lat"].values)
    for c in ["fireworks_pm25_mean","pm25_mean","pd","Green","NTL","Records"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df.Records == 0, "Records"] = np.nan
    selected = {2023:23, 2024:25}
    rows=[]
    for year in (2023, 2024):
        g=df[df.year==year].dropna(subset=["fireworks_pm25_mean","pd","Green","NTL","Records","x","y"]).reset_index(drop=True)
        X=np.column_stack([z(g[c]) for c in ["pd","Green","NTL"]]+[z(np.log1p(g.Records))])
        coords=g[["x","y"]].to_numpy(float)
        for outcome in ["fireworks_pm25_mean","pm25_mean"]:
            y=g[outcome].to_numpy(float).reshape(-1,1)
            base=GWR(coords,y,X,bw=selected[year],kernel="gaussian",fixed=False,n_jobs=1).fit()
            crit=base.critical_tval()
            for j,name in enumerate(["pd","Green","NTL","Records"]):
                c=base.params[:,j+1]; sg=np.abs(base.tvalues[:,j+1])>crit
                rows.append({"year":year,"outcome":outcome,"diagnostic":"local_summary","variable":name,"coef_mean":c.mean(),"coef_median":np.median(c),"coef_min":c.min(),"coef_max":c.max(),"negative_n":int((c<0).sum()),"sig_n":int(sg.sum()),"sig_negative_n":int((sg&(c<0)).sum()),"sig_positive_n":int((sg&(c>0)).sum())})
            for k in (4,8):
                w=KNN.from_array(coords,k=k); w.transform="r"
                # Moran's I for residuals from the selected specification
                mi=Moran(base.resid_response.flatten(),w,permutations=999)
                rows.append({"year":year,"outcome":outcome,"diagnostic":"Moran_KNN","k":k,"I":mi.I,"p_perm":mi.p_sim})
            for bw in sorted(set([max(20,selected[year]-5), selected[year], selected[year]+5, selected[year]+10])):
                for kernel in ["gaussian","bisquare"]:
                    try:
                        res=GWR(coords,y,X,bw=bw,kernel=kernel,fixed=False,n_jobs=1).fit()
                        c=res.params[:,4]; rows.append({"year":year,"outcome":outcome,"diagnostic":"sensitivity","bw":bw,"kernel":kernel,"coef_mean":c.mean(),"coef_median":np.median(c),"negative_n":int((c<0).sum()),"sig_n":int((np.abs(res.tvalues[:,4])>res.critical_tval()).sum()),"sig_negative_n":int(((c<0)&(np.abs(res.tvalues[:,4])>res.critical_tval())).sum()),"sig_positive_n":int(((c>0)&(np.abs(res.tvalues[:,4])>res.critical_tval())).sum())})
                    except Exception as e:
                        rows.append({"year":year,"outcome":outcome,"diagnostic":"sensitivity","bw":bw,"kernel":kernel,"error":str(e)})
    pd.DataFrame(rows).to_csv(Path.cwd()/"gwr_diagnostics_log1p_records.csv",index=False,encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__": main()
