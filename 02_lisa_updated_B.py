# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from libpysal.weights import Queen, Rook, DistanceBand
from esda.moran import Moran_Local
from splot.esda import moran_scatterplot, plot_local_autocorrelation
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SHAPEFILE_DIR = os.path.join(DATA_DIR, "shapefiles")
shp_path = os.path.join(SHAPEFILE_DIR, "中国_市.shp")
nine_line_shp = os.path.join(SHAPEFILE_DIR, "中国_市line.shp")
input_dir = os.path.join(PROJECT_ROOT, "outputs", "moran_analysis_updated_B")
output_dir = os.path.join(input_dir, "local_moran")
os.makedirs(output_dir, exist_ok=True)

# 视觉参数 - 参考你的画图代码
LCC = "+proj=lcc +lat_1=25 +lat_2=47 +lon_0=104 +datum=WGS84 +units=m +no_defs"
EDGE_COLOR, NO_DATA_COLOR = "#444444", "#F4F4F4"

# 布局参数 - 参考你的画图代码
FIG_W, FIG_H, DPI = 16, 10, 100
ML, MB, MW, MH = 0.05, 0.08, 0.82, 0.85 
IN_W, IN_H = 0.11, 0.22                  
IN_X, IN_Y = (ML + MW) - IN_W, MB        
LINEWIDTH = 1.2
X_LIMIT_RIGHT_OFFSET = 55000

# LISA聚类颜色 - 使用英文标签

LISA_COLORS = {
    'High-High': '#FF0000',     # 红色
    'Low-Low': '#0000FF',       # 蓝色
    'High-Low': '#FFC0CB',      # 粉色
    'Low-High': '#87CEEB',      # 浅蓝色
    'Not Significant': '#E0E0E0'  # 浅灰色（比白色深一些）
}

print("="*60)
print("中国城市PM2.5浓度局部莫兰指数分析")
print("="*60)

# ---------------- 1. 加载地理数据 ----------------
print("\n1. 加载地理数据...")
gdf = gpd.read_file(shp_path)

# 确定城市名称列
name_col = None
for col in ["NAME_2", "NAME_1", "NAME", "name", "市", "city", "City"]:
    if col in gdf.columns:
        name_col = col
        break

if name_col is None:
    name_col = gdf.columns[0]

print(f"地理数据城市名称列: '{name_col}'")

# 投影转换
gdf_proj = gdf.to_crs(LCC)

# 加载南海诸岛
nine_line = gpd.read_file(nine_line_shp).to_crs(LCC) if os.path.exists(nine_line_shp) else None

# ---------------- 2. 获取可用年份 ----------------
print("\n2. 查找可用年份数据...")

# 查找所有年份的CSV文件
yearly_files = [f for f in os.listdir(input_dir) if f.startswith('city_pm25_') and f.endswith('.csv')]
years = sorted([int(f.replace('city_pm25_', '').replace('.csv', '')) for f in yearly_files 
                if f.replace('city_pm25_', '').replace('.csv', '').isdigit()])

print(f"找到年份数据: {years}")

if len(years) == 0:
    print("错误: 没有找到年份数据文件")
    exit()

# ---------------- 3. 为每个年份计算局部莫兰指数 ----------------
print("\n3. 计算各年份局部莫兰指数...")

# 用于存储每年结果的字典
yearly_results = {}

for year in years:
    print(f"\n{'='*50}")
    print(f"处理 {year} 年数据...")
    print(f"{'='*50}")
    
    # 读取当年数据
    year_file = os.path.join(input_dir, f"city_pm25_{year}.csv")
    df_year = pd.read_csv(year_file, encoding='utf-8-sig')
    if 'city' not in df_year.columns and 'city_name' in df_year.columns:
        df_year = df_year.rename(columns={'city_name': 'city'})
    
    print(f"  读取到 {len(df_year)} 个城市的数据")
    
    # 合并地理数据
    analysis_gdf = gdf_proj[[name_col, 'geometry']].copy()
    analysis_gdf = analysis_gdf.merge(df_year, left_on=name_col, right_on='city', how='inner')
    
    if len(analysis_gdf) < 10:
        print(f"  {year}年有效城市不足10个 ({len(analysis_gdf)})，跳过")
        continue
    
    print(f"  合并后有效城市: {len(analysis_gdf)}")
    
    # 创建空间权重矩阵
    print(f"  创建空间权重矩阵...")
    
    weight_types = [
        ("Queen", lambda: Queen.from_dataframe(analysis_gdf, silence_warnings=True)),
        ("Rook", lambda: Rook.from_dataframe(analysis_gdf, silence_warnings=True)),
        ("DistanceBand (200km)", lambda: DistanceBand.from_dataframe(analysis_gdf, threshold=200000))
    ]
    
    w = None
    w_name = ""
    
    for name, func in weight_types:
        try:
            w = func()
            w_name = name
            print(f"    ✓ {name}成功")
            break
        except Exception as e:
            continue
    
    if w is None:
        print(f"  {year}年无法创建空间权重矩阵，跳过")
        continue
    
    # 权重矩阵标准化
    w.transform = 'r'
    print(f"  平均邻居数: {w.mean_neighbors:.2f}")
    
    # 计算局部莫兰指数
    y = analysis_gdf['pm25_mean'].values
    moran_local = Moran_Local(y, w)
    
    # 将结果添加到GeoDataFrame
    analysis_gdf['local_I'] = moran_local.Is
    analysis_gdf['p_sim'] = moran_local.p_sim
    analysis_gdf['quadrant'] = moran_local.q  # 象限: 1=HH, 2=LH, 3=LL, 4=HL
    
    # 添加聚类类型（英文标签）
    def get_cluster_type(row):
        if row['p_sim'] < 0.05:
            if row['quadrant'] == 1:
                return 'High-High'
            elif row['quadrant'] == 2:
                return 'Low-High'
            elif row['quadrant'] == 3:
                return 'Low-Low'
            elif row['quadrant'] == 4:
                return 'High-Low'
            else:
                return 'Unknown'
        else:
            return 'Not Significant'
    
    analysis_gdf['cluster_type'] = analysis_gdf.apply(get_cluster_type, axis=1)
    
    # 统计结果
    cluster_counts = analysis_gdf['cluster_type'].value_counts()
    print(f"\n  {year}年聚类结果统计:")
    for cluster_type, count in cluster_counts.items():
        percentage = count / len(analysis_gdf) * 100
        print(f"    {cluster_type}: {count}个 ({percentage:.1f}%)")
    
    # 保存结果
    yearly_results[year] = analysis_gdf
    
    # 保存局部莫兰结果到CSV
    result_df = analysis_gdf[[name_col, 'pm25_mean', 'local_I', 'p_sim', 'quadrant', 'cluster_type', 'is_interpolated']].copy()
    result_df.to_csv(os.path.join(output_dir, f"local_moran_{year}.csv"), 
                     index=False, encoding='utf-8-sig')
    
    # ---------------- 4. 为每个年份绘制LISA聚类图 ----------------
    print(f"\n  绘制{year}年LISA聚类图...")
    
    def render_inset_image(gdf_data, cluster_colors_dict):
        """绘制南海诸岛插图"""
        px_w, px_h = int(IN_W*FIG_W*DPI), int(IN_H*FIG_H*DPI)
        fig_in = plt.figure(figsize=(px_w/DPI, px_h/DPI), dpi=DPI)
        ax_in = fig_in.add_axes([0, 0, 1, 1])
        ax_in.set_axis_off()
        
        # 绘制所有城市（灰色背景）
        gdf_proj.plot(ax=ax_in, color=NO_DATA_COLOR, edgecolor=EDGE_COLOR, linewidth=0.1)
        
        # 绘制聚类
        for cluster_type, color in cluster_colors_dict.items():
            subset = gdf_data[gdf_data['cluster_type'] == cluster_type]
            if not subset.empty:
                subset.plot(ax=ax_in, color=color, edgecolor=EDGE_COLOR, linewidth=0.1)
        
        # 添加南海诸岛
        if nine_line is not None:
            nine_line.plot(ax=ax_in, color="black", linewidth=0.6)
        
        # 设置南海诸岛范围
        ax_in.set_xlim(550000, 2050000)
        ax_in.set_ylim(-200000, 2750000)
        
        # 保存为图像
        import io
        buf = io.BytesIO()
        fig_in.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig_in)
        return img
    
    # 绘制主图
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([ML, MB, MW, MH])
    
    # 绘制所有城市（灰色背景）
    gdf_proj.plot(ax=ax, color=NO_DATA_COLOR, edgecolor=EDGE_COLOR, linewidth=0.2)
    
    # 绘制聚类（按顺序：先不显著，再其他）
    # 先绘制不显著区域
    subset = analysis_gdf[analysis_gdf['cluster_type'] == 'Not Significant']
    if not subset.empty:
        subset.plot(ax=ax, color=LISA_COLORS['Not Significant'], edgecolor=EDGE_COLOR, linewidth=0.25)
    
    # 再绘制其他类型
    for cluster_type in ['High-High', 'Low-Low', 'High-Low', 'Low-High']:
        subset = analysis_gdf[analysis_gdf['cluster_type'] == cluster_type]
        if not subset.empty:
            subset.plot(ax=ax, color=LISA_COLORS[cluster_type], edgecolor=EDGE_COLOR, linewidth=0.25)
    
    # 添加南海诸岛
    if nine_line is not None:
        nine_line.plot(ax=ax, color=EDGE_COLOR, linewidth=0.6)
    
    # 设置视野范围
    b = gdf_proj.total_bounds
    ax.set_xlim(b[0]-300000, b[2] + X_LIMIT_RIGHT_OFFSET) 
    ax.set_ylim(2000000, b[3]+100000)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    
    # 添加南海诸岛插图
    inset_img = render_inset_image(analysis_gdf, LISA_COLORS)
    ax_in = fig.add_axes([IN_X, IN_Y, IN_W, IN_H])
    ax_in.imshow(inset_img, origin='upper')
    ax_in.set_axis_off()
    
    # 添加边框
    main_rect = mpl.patches.Rectangle((ML, MB), MW, MH, transform=fig.transFigure, 
                                      facecolor="none", edgecolor="black", linewidth=LINEWIDTH, zorder=5)
    fig.patches.append(main_rect)
    top_l = mpl.lines.Line2D([IN_X, IN_X+IN_W], [IN_Y+IN_H, IN_Y+IN_H], 
                             transform=fig.transFigure, color="black", linewidth=LINEWIDTH, zorder=10)
    left_l = mpl.lines.Line2D([IN_X, IN_X], [IN_Y, IN_Y+IN_H], 
                              transform=fig.transFigure, color="black", linewidth=LINEWIDTH, zorder=10)
    fig.lines.extend([top_l, left_l])
    
    # 添加图例 - 放在左下方内部
    from matplotlib.patches import Patch
    legend_elements = []
    for cluster_type, color in LISA_COLORS.items():
        if cluster_type != 'Not Significant' and not analysis_gdf[analysis_gdf['cluster_type'] == cluster_type].empty:
            legend_elements.append(Patch(facecolor=color, edgecolor=EDGE_COLOR, label=cluster_type))
    
    # 图例放在左下方内部
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, 
              frameon=True, fancybox=True, framealpha=0.9)
    
    # 标题只保留年份
    ax.set_title(f'{year}', fontsize=14, pad=20)
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, f"lisa_cluster_map_{year}.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # 绘制局部莫兰指数分布图
    print(f"  绘制{year}年局部莫兰指数分布图...")
    
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([ML, MB, MW, MH])
    
    # 绘制局部莫兰指数 - 修复legend_kwds参数
    # 创建ScalarMappable用于colorbar
    norm = mpl.colors.Normalize(vmin=analysis_gdf['local_I'].min(), vmax=analysis_gdf['local_I'].max())
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    
    analysis_gdf.plot(column='local_I', ax=ax, cmap='coolwarm', 
                     edgecolor=EDGE_COLOR, linewidth=0.25,
                     vmin=analysis_gdf['local_I'].min(), vmax=analysis_gdf['local_I'].max())
    
    # 添加colorbar
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, orientation='vertical')
    cbar.set_label('Local Moran I', fontsize=10)
    
    # 添加南海诸岛
    if nine_line is not None:
        nine_line.plot(ax=ax, color=EDGE_COLOR, linewidth=0.6)
    
    # 设置视野范围
    ax.set_xlim(b[0]-300000, b[2] + X_LIMIT_RIGHT_OFFSET) 
    ax.set_ylim(2000000, b[3]+100000)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    
    # 添加南海诸岛插图
    inset_img = render_inset_image(analysis_gdf, LISA_COLORS)
    ax_in = fig.add_axes([IN_X, IN_Y, IN_W, IN_H])
    ax_in.imshow(inset_img, origin='upper')
    ax_in.set_axis_off()
    
    # 添加边框
    fig.patches.append(main_rect)
    fig.lines.extend([top_l, left_l])
    
    # 标题只保留年份
    ax.set_title(f'{year}', fontsize=14, pad=20)
    
    plt.savefig(os.path.join(output_dir, f"local_moran_values_{year}.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # 绘制显著性分布图
    print(f"  绘制{year}年显著性分布图...")
    
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([ML, MB, MW, MH])
    
    # 将p值转换为 -log10(p) 便于可视化
    analysis_gdf['neg_log10_p'] = -np.log10(analysis_gdf['p_sim'].clip(lower=1e-10))
    
    # 创建ScalarMappable用于colorbar
    norm_sig = mpl.colors.Normalize(vmin=analysis_gdf['neg_log10_p'].min(), 
                                    vmax=analysis_gdf['neg_log10_p'].max())
    sm_sig = plt.cm.ScalarMappable(cmap='viridis', norm=norm_sig)
    sm_sig.set_array([])
    
    analysis_gdf.plot(column='neg_log10_p', ax=ax, cmap='viridis', 
                     edgecolor=EDGE_COLOR, linewidth=0.25,
                     vmin=analysis_gdf['neg_log10_p'].min(), vmax=analysis_gdf['neg_log10_p'].max())
    
    # 添加colorbar
    cbar_sig = plt.colorbar(sm_sig, ax=ax, shrink=0.6, orientation='vertical')
    cbar_sig.set_label('-log10(p)', fontsize=10)
    
    # 添加南海诸岛
    if nine_line is not None:
        nine_line.plot(ax=ax, color=EDGE_COLOR, linewidth=0.6)
    
    # 设置视野范围
    ax.set_xlim(b[0]-300000, b[2] + X_LIMIT_RIGHT_OFFSET) 
    ax.set_ylim(2000000, b[3]+100000)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    
    # 添加南海诸岛插图
    ax_in = fig.add_axes([IN_X, IN_Y, IN_W, IN_H])
    ax_in.imshow(inset_img, origin='upper')
    ax_in.set_axis_off()
    
    # 添加边框
    fig.patches.append(main_rect)
    fig.lines.extend([top_l, left_l])
    
    # 标题只保留年份
    ax.set_title(f'{year}', fontsize=14, pad=20)
    
    plt.savefig(os.path.join(output_dir, f"significance_map_{year}.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # 提取热点/冷点城市
    hotspots = analysis_gdf[analysis_gdf['cluster_type'] == 'High-High'].sort_values('pm25_mean', ascending=False)
    coldspots = analysis_gdf[analysis_gdf['cluster_type'] == 'Low-Low'].sort_values('pm25_mean', ascending=True)
    outliers = analysis_gdf[analysis_gdf['cluster_type'].isin(['High-Low', 'Low-High'])]
    
    # 保存列表
    hotspots[[name_col, 'pm25_mean', 'p_sim', 'is_interpolated']].to_csv(
        os.path.join(output_dir, f"hotspots_{year}.csv"), index=False, encoding='utf-8-sig')
    coldspots[[name_col, 'pm25_mean', 'p_sim', 'is_interpolated']].to_csv(
        os.path.join(output_dir, f"coldspots_{year}.csv"), index=False, encoding='utf-8-sig')
    outliers[[name_col, 'pm25_mean', 'cluster_type', 'p_sim', 'is_interpolated']].to_csv(
        os.path.join(output_dir, f"outliers_{year}.csv"), index=False, encoding='utf-8-sig')

# ---------------- 5. 绘制多年对比图 ----------------
if len(yearly_results) > 1:
    print("\n4. 绘制多年对比图...")
    
    # 创建子图网格
    n_years = len(yearly_results)
    n_cols = min(3, n_years)
    n_rows = (n_years + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
    if n_rows == 1:
        axes = axes.flatten() if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for idx, (year, gdf_data) in enumerate(sorted(yearly_results.items())):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # 绘制聚类
        # 先绘制不显著区域
        subset = gdf_data[gdf_data['cluster_type'] == 'Not Significant']
        if not subset.empty:
            subset.plot(ax=ax, color=LISA_COLORS['Not Significant'], edgecolor='black', linewidth=0.2)
        
        # 再绘制其他类型
        for cluster_type in ['High-High', 'Low-Low', 'High-Low', 'Low-High']:
            subset = gdf_data[gdf_data['cluster_type'] == cluster_type]
            if not subset.empty:
                subset.plot(ax=ax, color=LISA_COLORS[cluster_type], edgecolor='black', linewidth=0.2)
        
        # 添加南海诸岛
        if nine_line is not None:
            nine_line.plot(ax=ax, color="black", linewidth=0.5)
        
        # 标题只保留年份
        ax.set_title(f'{year}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(b[0]-300000, b[2] + X_LIMIT_RIGHT_OFFSET)
        ax.set_ylim(2000000, b[3]+100000)
    
    # 隐藏多余的子图
    for idx in range(len(yearly_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lisa_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------- 6. 生成汇总报告（英文） ----------------
print("\n5. 生成局部莫兰分析汇总报告...")

with open(os.path.join(output_dir, "local_moran_summary.txt"), 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("Local Moran's I Analysis Summary (2020-2025)\n")
    f.write("="*60 + "\n\n")
    
    f.write("I. Methodology\n")
    f.write("-"*40 + "\n")
    f.write("1. Local Moran's I (LISA): Identifies spatial clusters and outliers\n")
    f.write("2. Cluster types:\n")
    f.write("   - High-High: High values surrounded by high values\n")
    f.write("   - Low-Low: Low values surrounded by low values\n")
    f.write("   - High-Low: High values surrounded by low values\n")
    f.write("   - Low-High: Low values surrounded by high values\n")
    f.write("3. Significance level: p < 0.05\n\n")
    
    f.write("II. Annual Results\n")
    f.write("-"*40 + "\n")
    
    for year, gdf_data in sorted(yearly_results.items()):
        cluster_counts = gdf_data['cluster_type'].value_counts()
        
        f.write(f"\n{year}:\n")
        f.write(f"  Total cities: {len(gdf_data)}\n")
        f.write(f"  Cluster statistics:\n")
        
        for cluster_type in ['High-High', 'Low-Low', 'High-Low', 'Low-High', 'Not Significant']:
            count = cluster_counts.get(cluster_type, 0)
            percentage = count / len(gdf_data) * 100
            f.write(f"    {cluster_type}: {count} ({percentage:.1f}%)\n")
        
        # Hotspots Top 5
        hotspots = gdf_data[gdf_data['cluster_type'] == 'High-High'].sort_values('pm25_mean', ascending=False)
        if len(hotspots) > 0:
            f.write(f"\n  Top 5 Hotspots (High-High):\n")
            for i, (_, row) in enumerate(hotspots.head(5).iterrows()):
                interp_mark = " (interpolated)" if row['is_interpolated'] else ""
                f.write(f"    {i+1}. {row[name_col]}: {row['pm25_mean']:.1f} μg/m³ (p={row['p_sim']:.4f}){interp_mark}\n")
        
        # Coldspots Top 5
        coldspots = gdf_data[gdf_data['cluster_type'] == 'Low-Low'].sort_values('pm25_mean', ascending=True)
        if len(coldspots) > 0:
            f.write(f"\n  Top 5 Coldspots (Low-Low):\n")
            for i, (_, row) in enumerate(coldspots.head(5).iterrows()):
                interp_mark = " (interpolated)" if row['is_interpolated'] else ""
                f.write(f"    {i+1}. {row[name_col]}: {row['pm25_mean']:.1f} μg/m³ (p={row['p_sim']:.4f}){interp_mark}\n")
        
        f.write("\n" + "-"*30 + "\n")
    
    f.write("\nIII. Policy Recommendations\n")
    f.write("-"*40 + "\n")
    f.write("1. Focus pollution control measures on High-High clusters\n")
    f.write("2. Study and promote successful practices from Low-Low clusters\n")
    f.write("3. Investigate pollution sources in outlier areas (High-Low and Low-High)\n")
    f.write("4. Establish regional joint prevention and control mechanisms\n")
    f.write("5. Optimize air quality monitoring networks, especially in outlier areas\n")

print("\n" + "="*60)
print("Local Moran analysis completed!")
print("="*60)
print(f"\nResults saved to: {output_dir}")
print("\nGenerated files:")
print("  - local_moran_YYYY.csv - Annual local Moran data")
print("  - lisa_cluster_map_YYYY.png - LISA cluster maps")
print("  - local_moran_values_YYYY.png - Local Moran I distribution")
print("  - significance_map_YYYY.png - Significance maps")
print("  - hotspots_YYYY.csv - Hotspot lists")
print("  - coldspots_YYYY.csv - Coldspot lists")
print("  - outliers_YYYY.csv - Outlier lists")
print("  - lisa_comparison.png - Multi-year comparison")
print("  - local_moran_summary.txt - Summary report")
