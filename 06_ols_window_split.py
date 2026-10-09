# -*- coding: utf-8 -*-
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(PROJECT_ROOT, 'data')
CSV = os.path.join(BASE, 'festival_predictions_B_剔除烟花.csv')
PANEL = os.path.join(BASE, '城市年面板_2023_2025.csv')
OUT = os.path.join(os.getcwd(), 'panel_window_split_log1p_results.csv')
TABLE = os.path.join(os.getcwd(), 'panel_window_split_log1p_tableS.csv')
STATS = os.path.join(os.getcwd(), 'panel_window_split_log1p_tableS_stats.csv')

def zscore(a):
    a=np.asarray(a,float); m=np.nanmean(a); s=np.nanstd(a)
    return (a-m)/s if s>0 else (a-m)
def fit(formula,d):
    return smf.ols(formula,data=d).fit(cov_type='cluster',cov_kwds={'groups':d['city']})
def star(p): return '***' if p<.01 else ('**' if p<.05 else ('*' if p<.1 else ''))

def main():
    df=pd.read_csv(CSV,encoding='utf-8-sig')
    df['fw']=(df['PM2.5']-df['PM2.5_pred']).clip(lower=0)
    df['window']=np.where(df['hour']<=3,'midnight','evening')
    w=df[df.year>=2023].groupby(['city','year','window']).agg(fw=('fw','mean'),n=('fw','count')).reset_index()
    wide=w.pivot_table(index=['city','year'],columns='window',values=['fw','n'])
    wide.columns=[f'{a}_{b}' for a,b in wide.columns]; wide=wide.reset_index()
    p=pd.read_csv(PANEL,encoding='gbk'); p=p[p.year.isin([2023,2024])].copy()
    for c in ['pd','Green','NTL','Records']: p[c]=pd.to_numeric(p[c],errors='coerce')
    p.loc[p.Records==0,'Records']=np.nan
    g=p[['city','year','pd','Green','NTL','Records']].merge(wide,on=['city','year'],how='inner')
    for c in ['pd','Green','NTL']: g[c+'_z']=zscore(g[c].values)
    g['log1p_Records']=np.log1p(g['Records']); g['Records_log1p_z']=zscore(g['log1p_Records'].values)
    g['year2024']=(g.year==2024).astype(int)
    g['fw_all']=(g.fw_midnight*g.n_midnight+g.fw_evening*g.n_evening)/(g.n_midnight+g.n_evening)
    outcomes={'overall':'fw_all','midnight':'fw_midnight','evening':'fw_evening'}
    rows=[]; table=[]; stats=[]
    for model in ['main_effect','by_year']:
      for tag,ycol in outcomes.items():
        if model=='main_effect':
          d=g[['city','year',ycol,'Records_log1p_z','pd_z','Green_z','NTL_z','year2024']].dropna()
          formula=f'{ycol} ~ Records_log1p_z + pd_z + Green_z + NTL_z + year2024'
          m=fit(formula,d); terms=['Records_log1p_z']
        else:
          g['d23']=(g.year==2023).astype(int); g['d24']=(g.year==2024).astype(int)
          g['Records_log1p_x_2023']=g.Records_log1p_z*g.d23; g['Records_log1p_x_2024']=g.Records_log1p_z*g.d24
          d=g[['city','year',ycol,'Records_log1p_x_2023','Records_log1p_x_2024','pd_z','Green_z','NTL_z','year2024']].dropna()
          formula=f'{ycol} ~ Records_log1p_x_2023 + Records_log1p_x_2024 + pd_z + Green_z + NTL_z + year2024'
          m=fit(formula,d); terms=['Records_log1p_x_2023','Records_log1p_x_2024']
        stats.append({'Panel': 'Main effect' if model=='main_effect' else 'By-year','Window':tag,'N':int(m.nobs),'R2':m.rsquared,'adjR2':m.rsquared_adj})
        for term in terms:
          b=float(m.params[term]); se=float(m.bse[term]); pv=float(m.pvalues[term])
          rows.append({'model':model,'window':tag,'term':term,'coef':b,'se':se,'p':pv,'n':int(m.nobs),'R2':m.rsquared,'adjR2':m.rsquared_adj})
        if model=='main_effect':
          cells=[('Records_log1p_z', 'Records (log1p)')]
          for term,label in [('Records_log1p_z','Records (log1p)'),('pd_z','pd'),('Green_z','Green'),('NTL_z','NTL'),('year2024','year2024')]:
            b=float(m.params[term]); se=float(m.bse[term]); pv=float(m.pvalues[term]); table.append({'Panel':'Main effect','Variable':label,'window':tag,'cell':f'{b:.3f}{star(pv)} ({se:.3f})','coef':b,'se':se,'p':pv})
        else:
          for term,label in [('Records_log1p_x_2023','Records×2023'),('Records_log1p_x_2024','Records×2024'),('pd_z','pd'),('Green_z','Green'),('NTL_z','NTL'),('year2024','year2024')]:
            b=float(m.params[term]); se=float(m.bse[term]); pv=float(m.pvalues[term]); table.append({'Panel':'By-year','Variable':label,'window':tag,'cell':f'{b:.3f}{star(pv)} ({se:.3f})','coef':b,'se':se,'p':pv})
    pd.DataFrame(rows).to_csv(OUT,index=False,encoding='utf-8-sig')
    t=pd.DataFrame(table); piv=t.pivot(index=['Panel','Variable'],columns='window',values='cell').reset_index(); piv.to_csv(TABLE,index=False,encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(STATS,index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows).to_string(index=False)); print('\nTABLE'); print(piv.to_string(index=False)); print('\nSTATS'); print(pd.DataFrame(stats).to_string(index=False))
if __name__=='__main__': main()
