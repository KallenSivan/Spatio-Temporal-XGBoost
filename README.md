[README.md](https://github.com/user-attachments/files/32793776/README.md)
# Firework-related PM2.5 analysis code

This folder contains the analysis scripts used for the revised manuscript on firework-related PM₂.₅ during the Chinese Spring Festival and its association with online attention to drone shows.

## Contents

| File | Purpose |
|---|---|
| `01_moran_updated_B.py` | Annual Global Moran’s I using the updated B-model festival-window data |
| `02_lisa_updated_B.py` | Annual Local Moran’s I/LISA cluster classification and maps |
| `03_gwr_annual_firework.py` | Annual GWR models for 2023 and 2024 with firework-related PM₂.₅ as the outcome |
| `04_gwr_diagnostics.py` | GWR diagnostics, residual Moran’s I, and bandwidth/kernel sensitivity analyses |
| `05_ols_pooled_2023_2024.py` | Pooled city-year OLS with year-specific Records associations |
| `06_ols_window_split.py` | Window-specific pooled OLS for the midnight and evening sub-windows |

## Required input structure

Place the input files in `data/` and the city boundary files in `data/shapefiles/`:

```text
data/
├── festival_predictions_B_剔除烟花.csv
├── 城市年面板_2023_2025.csv
├── 城市坐标.csv
└── shapefiles/
    ├── 中国_市.shp
    ├── 中国_市.shx
    ├── 中国_市.dbf
    ├── 中国_市.prj
    └── 中国_市line.shp and its companion files
```


## Analysis workflow

Run the scripts in the following order:

```bash
python 01_moran_updated_B.py
python 02_lisa_updated_B.py
python 03_gwr_annual_firework.py
python 04_gwr_diagnostics.py
python 05_ols_pooled_2023_2024.py
python 06_ols_window_split.py
```

`02_lisa_updated_B.py` uses the annual files created by `01_moran_updated_B.py`. The other analyses use the city-year panel and prediction files in `data/`.

## Main model specifications

### Spatial autocorrelation

Global and local Moran’s I use row-standardized Queen contiguity weights when available. Cities without valid observations are completed using inverse-distance weighting before the annual spatial analysis. LISA classifications use permutation-based local significance at `P < 0.05`.

### Annual GWR

Separate models are estimated for 2023 and 2024:

```text
Firework-related PM2.5_i
  = β0,i + β1,i z[ln(1 + Records_i)]
    + β2,i z(PD_i) + β3,i z(Green_i) + β4,i z(NTL_i) + ε_i
```

The models use an adaptive Gaussian kernel. The bandwidth is selected by minimizing AICc over 20 to N nearest neighbours. Local significance is assessed with the model-reported multiplicity-adjusted 95% critical t-value. Diagnostic models evaluate residual Moran’s I with eight-nearest-neighbour spatial weights. Sensitivity analyses refit the models under alternative bandwidths and an adaptive bisquare kernel.

### Pooled OLS

The pooled city-year models cover 2023–2024 and use city-clustered standard errors. `Records` is transformed as `ln(1 + Records)` and standardized. Population density, green space and nighttime-light intensity are standardized covariates. The by-year specification estimates separate Records associations for 2023 and 2024.

The window-specific analysis separates the festival window into:

- Midnight: 00:00–03:00
- Evening: 20:00–23:00

The same covariates, year indicator and city-clustered standard errors are used for each window.

## Software environment

The scripts require Python packages including:

```text
numpy
pandas
geopandas
scipy
pyproj
libpysal
esda
splot
mgwr
statsmodels
matplotlib
```

The scripts were developed for the revised analysis and should be run with the same data-cleaning and festival-window definitions used in the manuscript.

