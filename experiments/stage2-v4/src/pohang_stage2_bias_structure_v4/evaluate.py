from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error

from pohang_stage2_adaptive_v3.evaluate import prepare_outer
from pohang_stage2_adaptive_v3.features import feature_profiles
from .bias import BiasCandidate,estimate_bias
from .models import StructureCandidate,fit_predict_centered


def _metrics(y,s1,bias_pred,final):
    r1=float(r2_score(y,s1)); rb=float(r2_score(y,s1+bias_pred)); rf=float(r2_score(y,final))
    return {"stage1_r2":r1,"bias_only_r2":rb,"final_r2":rf,"bias_delta_r2":rb-r1,"structure_delta_r2":rf-rb,"total_delta_r2":rf-r1,"stage1_rmse":float(np.sqrt(mean_squared_error(y,s1))),"bias_only_rmse":float(np.sqrt(mean_squared_error(y,s1+bias_pred))),"final_rmse":float(np.sqrt(mean_squared_error(y,final))),"final_residual_mean":float(np.mean(y-final))}


def evaluate_selected_fold(spec,frame,frozen,cfg,row:pd.Series,test_end:int,category_gate:pd.DataFrame|None=None,quick=False):
    fold=int(row.fold); ts=int(row.test_start)
    train,test,xtr,xte,y,s1,s1_frame,viol=prepare_outer(spec,frame,frozen,cfg,fold,ts,int(test_end))
    bc=BiasCandidate(**json.loads(str(row.bias_spec_json)))
    braw,bdiag=estimate_bias(bc,xtr); bias_scalar=float(row.bias_lambda)*float(braw); bias_pred=np.full(len(test),bias_scalar,float)
    fam=str(row.structure_family); prof=str(row.structure_profile); slam=float(row.structure_lambda); mode=str(row.structure_application_mode)
    profiles=feature_profiles(spec,cfg); feats,cats=profiles[prof]
    cand=StructureCandidate(fam,prof,str(row.structure_params_json))
    tr=xtr.copy(); tr["__centered_residual"]=pd.to_numeric(tr.__residual,errors="coerce")-bias_scalar
    enabled=None
    if mode=="CATEGORY_GATED":
        cg=category_gate if category_gate is not None else pd.DataFrame()
        enabled=set(cg[cg.structure_enabled.eq(1)].category.astype(str)) if not cg.empty else set()
    def apply_mask(c):
        if enabled is None: return c
        return c*xte.category.astype(str).isin(enabled).astype(float).to_numpy()
    rows=[]; canonical=None
    if fam=="NO_STRUCTURE":
        corr=np.zeros(len(test)); final=s1+bias_pred; m=_metrics(y,s1,bias_pred,final)
        rows.append({"fold":fold,"backend":"NONE","seed":0,"structure_family":fam,"profile":prof,"structure_lambda":slam,"application_mode":mode,"bias_candidate_id":bc.id,"bias_lambda":float(row.bias_lambda),"bias_applied_log":bias_scalar,"structure_train_mean_removed":0.0,"test_start":ts,"test_end":int(test_end),"train_rows":len(xtr),"test_rows":len(test),"leakage_violations":viol,**m})
        canonical=(corr,final,0.0)
    elif fam in {"RIDGE","ELASTICNET","HUBER"}:
        _,c,cent=fit_predict_centered(cand,tr,xte,feats,cats,backend="CPU",seed=0,threads=1)
        c=apply_mask(c); final=s1+bias_pred+slam*c; m=_metrics(y,s1,bias_pred,final)
        rows.append({"fold":fold,"backend":"CPU","seed":0,"structure_family":fam,"profile":prof,"structure_lambda":slam,"application_mode":mode,"bias_candidate_id":bc.id,"bias_lambda":float(row.bias_lambda),"bias_applied_log":bias_scalar,"structure_train_mean_removed":cent,"test_start":ts,"test_end":int(test_end),"train_rows":len(xtr),"test_rows":len(test),"leakage_violations":viol,**m})
        canonical=(c,final,cent)
    else:
        cpu=list(cfg["validation"]["seeds_cpu"]); gpu=list(cfg["validation"]["seeds_gpu"])
        if quick: cpu=cpu[:1]; gpu=[]
        def job(seed,backend):
            _,c,cent=fit_predict_centered(cand,tr,xte,feats,cats,backend=backend,seed=int(seed),threads=int(cfg["hardware_stage2_v4"]["cpu_threads_per_catboost"] if backend=="CPU" else cfg["hardware_stage2_v4"]["gpu_host_threads"]),gpu_ram_part=float(cfg["hardware_stage2_v4"]["gpu_ram_part"]),used_ram_limit=str(cfg["hardware_stage2_v4"]["used_ram_limit"]))
            c=apply_mask(c); final=s1+bias_pred+slam*c
            return int(seed),c,cent,final,_metrics(y,s1,bias_pred,final)
        cres=[]
        with ThreadPoolExecutor(max_workers=min(int(cfg["hardware_stage2_v4"]["cpu_catboost_workers"]),len(cpu))) as ex:
            fs=[ex.submit(job,s,"CPU") for s in cpu]
            for f in as_completed(fs): cres.append(f.result())
        for seed,c,cent,final,m in cres:
            rows.append({"fold":fold,"backend":"CPU","seed":seed,"structure_family":fam,"profile":prof,"structure_lambda":slam,"application_mode":mode,"bias_candidate_id":bc.id,"bias_lambda":float(row.bias_lambda),"bias_applied_log":bias_scalar,"structure_train_mean_removed":cent,"test_start":ts,"test_end":int(test_end),"train_rows":len(xtr),"test_rows":len(test),"leakage_violations":viol,**m})
        cmean=np.mean([x[1] for x in cres],axis=0); centmean=float(np.mean([x[2] for x in cres])); final=s1+bias_pred+slam*cmean; canonical=(cmean,final,centmean)
        for seed in gpu:
            seed,c,cent,final,m=job(seed,"GPU")
            rows.append({"fold":fold,"backend":"GPU","seed":seed,"structure_family":fam,"profile":prof,"structure_lambda":slam,"application_mode":mode,"bias_candidate_id":bc.id,"bias_lambda":float(row.bias_lambda),"bias_applied_log":bias_scalar,"structure_train_mean_removed":cent,"test_start":ts,"test_end":int(test_end),"train_rows":len(xtr),"test_rows":len(test),"leakage_violations":viol,**m})
    corr,final,cent=canonical
    keys=[c for c in ["district","year_month","category","category_major","spend_thousand_krw"] if c in test]
    pred=test[keys].copy(); pred["fold"]=fold; pred["row_index"]=test.index.to_numpy(); pred["actual_log_spend"]=y; pred["stage1_expected_log"]=s1
    pred["stage2a_bias_raw_log"]=float(braw); pred["stage2a_bias_lambda"]=float(row.bias_lambda); pred["stage2a_bias_applied_log"]=bias_scalar
    pred["stage2b_structure_raw_log"]=corr; pred["stage2b_structure_lambda"]=slam; pred["stage2b_structure_applied_log"]=slam*corr
    pred["final_expected_log"]=final; pred["expected_spend_thousand_krw"]=np.expm1(np.clip(final,-30,30)).clip(min=0); pred["actual_minus_expected_log"]=y-final
    pred["selected_bias_family"]=bc.family; pred["selected_structure_family"]=fam; pred["selected_structure_profile"]=prof; pred["structure_application_mode"]=mode
    pred["structure_category_enabled"]=1 if enabled is None else xte.category.astype(str).isin(enabled).astype(int).to_numpy(); pred["policy_use_allowed"]=0
    return pd.DataFrame(rows),pred
