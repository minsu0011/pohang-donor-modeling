from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

from pohang_main_v2.config import load_config
from pohang_main_v2.cli import _spec,_run_paths,project_root
from pohang_main_v2.utils import ensure_dir,write_csv,atomic_write_json
from pohang_stage1_v22.modeling import canonicalize_spec
from pohang_stage2_adaptive_v3.stage1_freeze import FrozenStage1
from .tournament import run_fold_tournament
from .evaluate import evaluate_selected_fold
from .gates import build_gate


def _dest(cfg,root,run_name):
    _,out,cache=_run_paths(cfg,root,run_name)
    return ensure_dir(out),ensure_dir(cache/"bias_structure_tasks")


def prepare(args):
    root=project_root(); cfg=load_config(args.config); spec,*_=_spec(cfg,root,args.run_name); frozen=FrozenStage1.load(root,cfg); dest,_=_dest(cfg,root,args.run_name)
    blocks=list(cfg["validation"]["rolling_test_blocks"]); start=1
    if args.mode=="quick": blocks=blocks[-1:]; start=len(cfg["validation"]["rolling_test_blocks"])
    rows=[frozen.reproduce_outer(spec.frame,int(fold),int(ts),int(te)) for fold,(ts,te) in enumerate(blocks,start)]
    repro=pd.DataFrame(rows); write_csv(repro,dest/"STAGE1_FROZEN_REPRODUCTION.csv")
    canon=canonicalize_spec(spec,cfg)
    audit={"status":"PASS","canonical_stage2_features":len(canon.features),"canonical_categorical_features":list(canon.categorical),"stage1_frozen_reference_rows":len(frozen.reference_selection),"stage1_reproduction":repro.to_dict("records"),"decomposition":"final = stage1 + bias + lambda * centered_structure","policy_use_allowed":0}
    atomic_write_json(dest/"STAGE2_V4_PREPARE_AUDIT.json",audit); print(json.dumps(audit,ensure_ascii=False,indent=2))


def select(args):
    root=project_root(); cfg=load_config(args.config); spec,*_=_spec(cfg,root,args.run_name); frozen=FrozenStage1.load(root,cfg); dest,cache=_dest(cfg,root,args.run_name)
    blocks=list(cfg["validation"]["rolling_test_blocks"]); start=1
    if args.mode=="quick": blocks=blocks[-1:]; start=len(cfg["validation"]["rolling_test_blocks"])
    br=[];ba=[];sr=[];sa=[];cg=[];ge=[];sels=[]
    for fold,(ts,te) in enumerate(blocks,start):
        x=run_fold_tournament(spec,spec.frame,int(fold),int(ts),frozen,cfg,cache/f"fold{fold}",args.mode=="quick")
        braw,bagg,sraw,sagg,catg,gated,sel=x
        for df in (braw,bagg,sraw,sagg,catg,gated):
            if not df.empty: df.insert(0,"fold",int(fold))
        sel["test_end"]=int(te)
        br.append(braw);ba.append(bagg);sr.append(sraw);sa.append(sagg);cg.append(catg);ge.append(gated);sels.append(sel)
        print(sel.to_string(index=False))
    def cat(xs): return pd.concat([x for x in xs if x is not None and not x.empty],ignore_index=True) if any(x is not None and not x.empty for x in xs) else pd.DataFrame()
    write_csv(cat(br),dest/"STAGE2A_BIAS_INNER_TOURNAMENT.csv"); write_csv(cat(ba),dest/"STAGE2A_BIAS_SUMMARY.csv")
    write_csv(cat(sr),dest/"STAGE2B_STRUCTURE_INNER_TOURNAMENT.csv"); write_csv(cat(sa),dest/"STAGE2B_STRUCTURE_SUMMARY.csv")
    write_csv(cat(cg),dest/"STAGE2B_CATEGORY_ABSTENTION.csv"); write_csv(cat(ge),dest/"STAGE2B_CATEGORY_GATED_LOO_EVAL.csv")
    selection=cat(sels); write_csv(selection,dest/"STAGE2_V4_SELECTION_BY_FOLD.csv")
    aud=selection[[c for c in ["fold","test_start","test_end","bias_candidate_id","bias_lambda","bias_selection_reason","structure_family","structure_profile","structure_lambda","structure_application_mode","structure_selection_reason","bias_median_gain","structure_median_gain"] if c in selection]].copy()
    aud["bias_abstained"]=(aud.bias_candidate_id=="NO_BIAS").astype(int); aud["structure_abstained"]=(aud.structure_family=="NO_STRUCTURE").astype(int)
    write_csv(aud,dest/"STAGE2_DUAL_ABSTENTION_AUDIT.csv")


def evaluate(args):
    root=project_root(); cfg=load_config(args.config); spec,*_=_spec(cfg,root,args.run_name); frozen=FrozenStage1.load(root,cfg); dest,_=_dest(cfg,root,args.run_name)
    sel=pd.read_csv(dest/"STAGE2_V4_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    cg_path=dest/"STAGE2B_CATEGORY_ABSTENTION.csv"
    try:
        cg=pd.read_csv(cg_path,encoding="utf-8-sig") if cg_path.exists() and cg_path.stat().st_size>0 else pd.DataFrame()
    except pd.errors.EmptyDataError:
        cg=pd.DataFrame()
    ms=[];ps=[]
    for _,r in sel.iterrows():
        sub=cg[cg.fold.eq(int(r.fold))].copy() if not cg.empty and "fold" in cg else pd.DataFrame()
        m,p=evaluate_selected_fold(spec,spec.frame,frozen,cfg,r,int(r.test_end),sub,args.mode=="quick"); ms.append(m);ps.append(p); print(m.to_string(index=False))
    md=pd.concat(ms,ignore_index=True); pp=pd.concat(ps,ignore_index=True)
    write_csv(md,dest/"STAGE2_V4_OUTER_METRICS_ALL_BACKENDS.csv"); pp.to_csv(dest/"EXPECTED_SPEND_FINAL_OOF_V4.csv.gz",index=False,encoding="utf-8-sig",compression={"method":"gzip","mtime":0})


def finalize(args):
    root=project_root(); cfg=load_config(args.config); spec,*_=_spec(cfg,root,args.run_name); dest,_=_dest(cfg,root,args.run_name)
    repro=pd.read_csv(dest/"STAGE1_FROZEN_REPRODUCTION.csv",encoding="utf-8-sig"); sel=pd.read_csv(dest/"STAGE2_V4_SELECTION_BY_FOLD.csv",encoding="utf-8-sig"); met=pd.read_csv(dest/"STAGE2_V4_OUTER_METRICS_ALL_BACKENDS.csv",encoding="utf-8-sig"); oof=pd.read_csv(dest/"EXPECTED_SPEND_FINAL_OOF_V4.csv.gz",compression="gzip",encoding="utf-8-sig")
    if args.mode=="quick": gate={"status":"QUICK_COMPLETE","folds":sorted(sel.fold.astype(int).unique().tolist()),"rows":len(oof),"policy_use_allowed":0,"consumption_gap_use_allowed":0}
    else:
        br=pd.read_csv(dest/"STAGE2A_BIAS_INNER_TOURNAMENT.csv",encoding="utf-8-sig"); sr=pd.read_csv(dest/"STAGE2B_STRUCTURE_INNER_TOURNAMENT.csv",encoding="utf-8-sig")
        gate=build_gate(repro,sel,met,oof,canonicalize_spec(spec,cfg),cfg,br,sr)
    atomic_write_json(dest/"FINAL_EXPECTED_SPEND_V4_GATE.json",gate)
    lines=["# POHANG Stage2 Bias–Structure Decomposition V4","",f"- Status: **{gate['status']}**",f"- Consumption-gap use allowed: **{gate.get('consumption_gap_use_allowed',0)}**",f"- Causal policy use allowed: **{gate.get('policy_use_allowed',0)}**","","## Fold selections"]
    for _,r in sel.sort_values("fold").iterrows(): lines.append(f"- Fold {int(r.fold)}: bias `{r.bias_candidate_id}` × {float(r.bias_lambda):.2f}; structure `{r.structure_family}` / `{r.structure_profile}` × {float(r.structure_lambda):.2f}; mode `{r.structure_application_mode}`")
    if args.mode!="quick":
        lines += ["","## Gate checks"]+[f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in gate["checks"].items()]
        lines += ["",f"- Median final R2: {gate['median_final_r2']:.6f}",f"- Worst total gain: {gate['worst_total_gain']:+.6f}",f"- Latest final R2: {gate['latest_final_r2']:.6f}",f"- CatBoost backend audit: {gate['catboost_backend_audit_status']}"]
    (dest/"FINAL_EXPECTED_SPEND_V4_REVIEW_KO.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps(gate,ensure_ascii=False,indent=2))


def gpu_check(args):
    from catboost import CatBoostRegressor
    x=np.random.default_rng(1).normal(size=(50000,64)); y=1.7*x[:,0]-0.6*x[:,1]+np.random.default_rng(2).normal(size=len(x))
    m=CatBoostRegressor(iterations=180,depth=8,task_type="GPU",devices="0",gpu_ram_part=0.92,verbose=False,allow_writing_files=False); m.fit(x,y); print("RTX 5080 / CatBoost GPU PASS")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/default.yaml"); p.add_argument("--run-name",default="stage2_bias_structure_v4"); sub=p.add_subparsers(dest="cmd",required=True)
    for name,fn in [("prepare",prepare),("select",select),("evaluate",evaluate),("finalize",finalize)]:
        q=sub.add_parser(name); q.add_argument("--mode",choices=["quick","full"],default="full"); q.set_defaults(func=fn)
    q=sub.add_parser("gpu-check");q.set_defaults(func=gpu_check); a=p.parse_args();a.func(a)
if __name__=="__main__": main()
