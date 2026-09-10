from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from pohang_stage2_adaptive_v3.windows import build_adaptive_windows, split_window
from pohang_stage2_adaptive_v3.features import prepare_window, feature_profiles
from pohang_stage1_v22.orthogonal_interactions import past_only_orthogonalize
from .bias import BiasCandidate, bias_candidates, estimate_bias, bias_metrics, aggregate_bias
from .models import StructureCandidate, structure_candidates, full_catboost_candidates, fit_predict_centered
from .structure import structure_metrics, aggregate_structure, category_gate_from_windows


def _hash(x:dict)->str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:22]


def _prepare_windows(spec,outer_train,frozen,cfg,quick):
    acfg=cfg["stage2_bias_structure"]
    wins=build_adaptive_windows(outer_train,short_count=int(acfg["quick_short_window_count"] if quick else acfg["short_window_count"]),short_months=int(acfg["short_window_months"]),long_months=int(acfg["long_window_months"]),min_train_months=int(acfg["inner_min_train_months"]))
    s1=frozen.crossfit_training(outer_train)
    orth,aud=past_only_orthogonalize(outer_train,alpha=float(cfg["orthogonal_interactions"]["category_mean_alpha"]))
    out=[]
    for w in wins:
        tr,va=split_window(outer_train,w)
        xtr,xva,y,s1v,viol=prepare_window(spec,cfg,frozen,tr,va,precomputed_stage1=s1,precomputed_orth=orth,precomputed_orth_audit=aud)
        out.append({"window":w,"train":xtr,"valid":xva,"y":y,"stage1":s1v,"leakage_violations":viol})
    return out


def _bias_tournament(prepared,cfg,quick):
    rows=[]; lambdas=[float(x) for x in cfg["stage2_bias_structure"]["bias_lambdas"]]
    for cand in bias_candidates(quick):
        for p in prepared:
            raw,diag=estimate_bias(cand,p["train"])
            for lam in ([0.0] if cand.family=="NO_BIAS" else lambdas):
                m=bias_metrics(p["y"],p["stage1"],raw,lam)
                rows.append({"bias_candidate_id":cand.id,"bias_family":cand.family,"bias_months":cand.months,"bias_half_life":cand.half_life,"bias_lambda":lam,"bias_spec_json":json.dumps(asdict(cand),sort_keys=True),"window":p["window"].name,"leakage_violations":int(p["leakage_violations"]),**diag,**m})
    raw=pd.DataFrame(rows); agg=aggregate_bias(raw,cfg)
    eligible=agg[(agg.eligible.eq(1)) & (pd.to_numeric(agg.selection_score,errors="coerce")>0)]
    if eligible.empty:
        pick=agg[agg.bias_candidate_id.eq("NO_BIAS")].iloc[0]; reason="NO_ELIGIBLE_BIAS"
    else:
        pick=eligible.iloc[0]; reason="ELIGIBLE_BIAS_WINNER"
    specj=str(raw[raw.bias_candidate_id.eq(pick.bias_candidate_id)].bias_spec_json.iloc[0])
    sel={**pick.to_dict(),"bias_spec_json":specj,"selection_reason":reason}
    return raw,agg,pd.Series(sel)


def _cand_task(cand,p,features,cats,bias_cand,bias_lambda,cfg,cache_dir,quick,backend="CPU"):
    key=_hash({"cand":cand.id,"window":asdict(p["window"]),"bias":asdict(bias_cand),"bias_lambda":bias_lambda,"backend":backend,"quick":quick,"ntr":len(p["train"]),"nva":len(p["valid"])})
    cp=cache_dir/f"{key}.json"
    if cp.exists(): return json.loads(cp.read_text(encoding="utf-8"))
    bias_raw,_=estimate_bias(bias_cand,p["train"]); bias_applied=float(bias_lambda)*float(bias_raw)
    tr=p["train"].copy(); tr["__centered_residual"]=pd.to_numeric(tr.__residual,errors="coerce")-bias_applied
    ptr,pv,center=fit_predict_centered(cand,tr,p["valid"],features,cats,backend=backend,seed=int(cfg["stage2_bias_structure"]["screen_seed"]),threads=int(cfg["hardware_stage2_v4"]["cpu_threads_per_catboost"] if backend=="CPU" else cfg["hardware_stage2_v4"]["gpu_host_threads"]),gpu_ram_part=float(cfg["hardware_stage2_v4"]["gpu_ram_part"]),used_ram_limit=str(cfg["hardware_stage2_v4"]["used_ram_limit"]))
    rows=[]
    for lam in ([0.0] if cand.family=="NO_STRUCTURE" else [float(x) for x in cfg["stage2_bias_structure"]["structure_lambdas"]]):
        m=structure_metrics(p["y"],p["stage1"],bias_applied,pv,lam)
        rows.append({"candidate_id":cand.id,"family":cand.family,"profile":cand.profile,"params_json":cand.params_json,"backend":backend,"structure_lambda":lam,"application_mode":"GLOBAL","window":p["window"].name,"bias_applied":bias_applied,"train_prediction_center_removed":center,"leakage_violations":int(p["leakage_violations"]),**m})
    cp.write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)),encoding="utf-8")
    return rows


def _evaluate_structure_candidates(prepared,profiles,bias_cand,bias_lambda,cfg,cache_dir,quick):
    cands=structure_candidates(list(profiles),quick)+full_catboost_candidates(list(profiles),quick)
    # De-duplicate IDs.
    seen=set(); cands=[c for c in cands if not (c.id in seen or seen.add(c.id))]
    linear=[c for c in cands if c.family in {"NO_STRUCTURE","RIDGE","ELASTICNET","HUBER"}]
    cat=[c for c in cands if c.family.startswith("CATBOOST")]
    rows=[]
    def run(c,backend="CPU"):
        f,ca=profiles[c.profile]; rr=[]
        for p in prepared: rr.extend(_cand_task(c,p,f,ca,bias_cand,bias_lambda,cfg,cache_dir,quick,backend))
        return rr
    # GPU shadow is one stream and overlaps CPU screening. CPU remains canonical
    # for selection; GPU rows are retained as a visible robustness/resource audit.
    gpu_enabled=(not quick) and bool(cfg["stage2_bias_structure"].get("gpu_shadow_enabled",False))
    with ThreadPoolExecutor(max_workers=1) as gpu_ex:
        gpu_fs=[gpu_ex.submit(run,c,"GPU") for c in cat] if gpu_enabled else []
        with ThreadPoolExecutor(max_workers=min(int(cfg["hardware_stage2_v4"]["linear_workers"]),max(1,len(linear)))) as ex:
            fs=[ex.submit(run,c,"CPU") for c in linear]
            for f in as_completed(fs): rows.extend(f.result())
        with ThreadPoolExecutor(max_workers=min(int(cfg["hardware_stage2_v4"]["cpu_catboost_workers"]),max(1,len(cat)))) as ex:
            fs=[ex.submit(run,c,"CPU") for c in cat]
            for f in as_completed(fs): rows.extend(f.result())
        for f in as_completed(gpu_fs): rows.extend(f.result())
    raw=pd.DataFrame(rows); agg=aggregate_structure(raw,cfg)
    cpu_agg=agg[agg.backend.eq("CPU")]
    elig=cpu_agg[(cpu_agg.eligible.eq(1)) & (~cpu_agg.family.eq("NO_STRUCTURE")) & (pd.to_numeric(cpu_agg.selection_score,errors="coerce")>0)]
    if elig.empty:
        pick=cpu_agg[cpu_agg.family.eq("NO_STRUCTURE")].iloc[0]; reason="NO_ELIGIBLE_STRUCTURE"
    else:
        pick=elig.iloc[0]; reason="ELIGIBLE_STRUCTURE_WINNER"
    return raw,agg,pick,reason,{c.id:c for c in cands}


def _selected_row_details(prepared,cand,features,cats,bias_cand,bias_lambda,struct_lambda,cfg):
    details=[]; overall=[]
    for p in prepared:
        braw,_=estimate_bias(bias_cand,p["train"]); b=float(bias_lambda)*braw
        tr=p["train"].copy(); tr["__centered_residual"]=pd.to_numeric(tr.__residual,errors="coerce")-b
        _,pv,_=fit_predict_centered(cand,tr,p["valid"],features,cats,backend="CPU",seed=int(cfg["stage2_bias_structure"]["screen_seed"]),threads=int(cfg["hardware_stage2_v4"]["cpu_threads_per_catboost"]),gpu_ram_part=float(cfg["hardware_stage2_v4"]["gpu_ram_part"]),used_ram_limit=str(cfg["hardware_stage2_v4"]["used_ram_limit"]))
        base=p["stage1"]+b; final=base+float(struct_lambda)*pv
        w=p["window"].name
        tmp=pd.DataFrame({"window":w,"category":p["valid"].category.astype(str).to_numpy(),"y":p["y"],"base":base,"corr":float(struct_lambda)*pv,"final":final})
        details.append(tmp)
        for cat,g in tmp.groupby("category"):
            s0=float(np.sum((g.y-g.base)**2)); s1=float(np.sum((g.y-g.final)**2)); ratio=(s0-s1)/max(s0,1e-12)
            overall.append({"window":w,"category":cat,"rows":len(g),"sse_gain_ratio":ratio})
    return pd.concat(details,ignore_index=True),pd.DataFrame(overall)


def _loo_category_gated(detail,cat_summary,cfg):
    rows=[]; windows=sorted(detail.window.unique())
    for w in windows:
        train_cat=cat_summary[~cat_summary.window.eq(w)]
        gate=category_gate_from_windows(train_cat,cfg)
        enabled=set(gate[gate.structure_enabled.eq(1)].category.astype(str)) if not gate.empty else set()
        d=detail[detail.window.eq(w)].copy(); mask=d.category.astype(str).isin(enabled).astype(float).to_numpy()
        base=d["base"].to_numpy(float); y=d["y"].to_numpy(float); corr=d["corr"].to_numpy(float)*mask
        b=float(r2_score(y,base)); f=float(r2_score(y,base+corr))
        rows.append({"window":w,"delta_r2":f-b,"final_r2":f,"bias_only_r2":b,"structure_mean":float(np.mean(corr)),"enabled_categories":len(enabled),"application_mode":"CATEGORY_GATED_LOO"})
    return pd.DataFrame(rows)


def run_fold_tournament(spec,frame,fold,test_start,frozen,cfg,cache_dir:Path,quick=False):
    outer_train=frame[pd.to_numeric(frame.year_month,errors="coerce").lt(int(test_start))].copy()
    prepared=_prepare_windows(spec,outer_train,frozen,cfg,quick)
    if len(prepared)<int(cfg["stage2_bias_structure"]["minimum_inner_windows"]): raise RuntimeError(f"INSUFFICIENT_INNER_WINDOWS:{fold}:{len(prepared)}")
    cache_dir.mkdir(parents=True,exist_ok=True)
    braw,bagg,bsel=_bias_tournament(prepared,cfg,quick)
    bspec=json.loads(str(bsel.bias_spec_json)); bc=BiasCandidate(**bspec); bl=float(bsel.bias_lambda)
    profiles=feature_profiles(spec,cfg)
    sraw,sagg,spick,sreason,lookup=_evaluate_structure_candidates(prepared,profiles,bc,bl,cfg,cache_dir,quick)
    fam=str(spick.family); mode="NO_STRUCTURE" if fam=="NO_STRUCTURE" else "GLOBAL"; cat_gate=pd.DataFrame(); gated_eval=pd.DataFrame()
    if fam!="NO_STRUCTURE":
        cand=lookup[str(spick.candidate_id)]; feats,cats=profiles[cand.profile]
        detail,catsum=_selected_row_details(prepared,cand,feats,cats,bc,bl,float(spick.structure_lambda),cfg)
        gated_eval=_loo_category_gated(detail,catsum,cfg)
        # Compare LOO gated to the already selected GLOBAL candidate on the same windows.
        gg=pd.DataFrame({"candidate_id":[cand.id]*len(gated_eval),"family":[cand.family]*len(gated_eval),"profile":[cand.profile]*len(gated_eval),"params_json":[cand.params_json]*len(gated_eval),"backend":["CPU"]*len(gated_eval),"structure_lambda":[float(spick.structure_lambda)]*len(gated_eval),"application_mode":["CATEGORY_GATED"]*len(gated_eval),"delta_r2":gated_eval.delta_r2,"structure_mean":gated_eval.structure_mean})
        gagg=aggregate_structure(gg,cfg)
        if not gagg.empty and int(gagg.iloc[0].eligible)==1 and float(gagg.iloc[0].selection_score) >= float(spick.selection_score)+float(cfg["stage2_bias_structure"]["category_gate_min_score_gain"]):
            mode="CATEGORY_GATED"; spick=gagg.iloc[0]; sreason="CATEGORY_GATED_LOO_WINNER"
        cat_gate=category_gate_from_windows(catsum,cfg); cat_gate["selected_candidate_id"]=cand.id; cat_gate["selected_structure_lambda"]=float(spick.structure_lambda)
    sel={"fold":int(fold),"test_start":int(test_start),"bias_candidate_id":str(bsel.bias_candidate_id),"bias_spec_json":str(bsel.bias_spec_json),"bias_lambda":float(bsel.bias_lambda),"bias_selection_reason":str(bsel.selection_reason),"bias_median_gain":float(bsel.median_gain),"bias_worst_gain":float(bsel.worst_gain),"bias_positive_share":float(bsel.positive_share),"structure_family":str(spick.family),"structure_profile":str(spick.profile),"structure_params_json":str(spick.params_json),"structure_lambda":float(spick.structure_lambda),"structure_application_mode":mode,"structure_selection_reason":sreason,"structure_median_gain":float(spick.median_gain),"structure_worst_gain":float(spick.worst_gain),"structure_positive_share":float(spick.positive_share),"inner_window_count":len(prepared),"leakage_violations":int(sum(p["leakage_violations"] for p in prepared))}
    return braw,bagg,sraw,sagg,cat_gate,gated_eval,pd.DataFrame([sel])
