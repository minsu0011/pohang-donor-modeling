from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def structure_metrics(y,stage1,bias_applied,struct_corr,lam,mask=None):
    base=stage1+bias_applied
    c=np.asarray(struct_corr,float)
    if mask is not None: c=c*np.asarray(mask,float)
    final=base+float(lam)*c
    br=float(r2_score(y,base)); fr=float(r2_score(y,final))
    return {"bias_only_r2":br,"final_r2":fr,"delta_r2":fr-br,"rmse":float(np.sqrt(mean_squared_error(y,final))),"structure_mean":float(np.mean(float(lam)*c))}


def aggregate_structure(raw:pd.DataFrame,cfg:dict)->pd.DataFrame:
    gcfg=cfg["stage2_bias_structure"]["structure_gate"]
    rows=[]
    for keys,g in raw.groupby(["candidate_id","family","profile","params_json","backend","structure_lambda","application_mode"],sort=True):
        cid,fam,prof,pj,backend,lam,mode=keys
        gains=pd.to_numeric(g.delta_r2,errors="coerce").dropna()
        if gains.empty: continue
        med=float(gains.median()); worst=float(gains.min()); std=float(gains.std(ddof=0)); pos=float((gains>0).mean())
        sm=float(pd.to_numeric(g.structure_mean,errors="coerce").abs().median())
        score=med-float(gcfg["std_penalty"])*std-float(gcfg["negative_worst_penalty"])*abs(min(0.0,worst))-float(gcfg["mean_drift_penalty"])*sm
        if fam=="NO_STRUCTURE": eligible=True; score=0.0
        else: eligible=(med>=float(gcfg["median_gain_min"]) and pos>=float(gcfg["positive_share_min"]) and worst>=float(gcfg["worst_gain_min"]) and sm<=float(gcfg["max_abs_structure_mean"]))
        rows.append({"candidate_id":cid,"family":fam,"profile":prof,"params_json":pj,"backend":backend,"structure_lambda":float(lam),"application_mode":mode,"window_count":int(len(gains)),"median_gain":med,"worst_gain":worst,"gain_std":std,"positive_share":pos,"median_abs_structure_mean":sm,"selection_score":score,"eligible":int(eligible)})
    return pd.DataFrame(rows).sort_values(["eligible","selection_score","median_gain"],ascending=[False,False,False]).reset_index(drop=True)


def category_gate_from_windows(rows:pd.DataFrame,cfg:dict)->pd.DataFrame:
    gcfg=cfg["stage2_bias_structure"]["category_abstention_gate"]
    out=[]
    for cat,g in rows.groupby("category",sort=True):
        vals=pd.to_numeric(g.sse_gain_ratio,errors="coerce").dropna()
        if vals.empty: continue
        pos=float((vals>0).mean()); med=float(vals.median()); worst=float(vals.min()); n=int(pd.to_numeric(g.rows,errors="coerce").fillna(0).sum())
        keep=(len(vals)>=int(gcfg["min_windows"]) and n>=int(gcfg["min_rows_total"]) and pos>=float(gcfg["positive_share_min"]) and med>=float(gcfg["median_sse_gain_ratio_min"]) and worst>=float(gcfg["worst_sse_gain_ratio_min"]))
        out.append({"category":cat,"window_count":int(len(vals)),"rows_total":n,"median_sse_gain_ratio":med,"worst_sse_gain_ratio":worst,"positive_share":pos,"structure_enabled":int(keep)})
    return pd.DataFrame(out)
