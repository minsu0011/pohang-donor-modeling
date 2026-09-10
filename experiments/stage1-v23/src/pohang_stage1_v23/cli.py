from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

from pohang_main_v2.config import load_config
from pohang_main_v2.cli import _spec, _prepare, _input_context, _run_paths, project_root
from pohang_main_v2.utils import atomic_write_json, ensure_dir, write_csv
from pohang_stage1_v22.modeling import canonicalize_spec

from .selection import build_feasibility_first_selection
from .modeling import run_backend_baselines_v23
from .ablation import group_units, individual_units, run_stage2_ablation, summarize, fold_diagnostics
from .gates import summarize_experiment_gate


def _paths(cfg,root,run_name):
    _,out,_=_run_paths(cfg,root,run_name)
    return ensure_dir(out/"stage1_v23")


def prepare_cmd(args):
    root=project_root(); cfg=load_config(args.config); _,out,_=_run_paths(cfg,root,args.run_name)
    prepared=_input_context(cfg,root,out); _prepare(cfg,root,args.run_name,prepared)
    spec,*_=_spec(cfg,root,args.run_name); dest=_paths(cfg,root,args.run_name)
    quick=args.mode=="quick"
    sel,s1,screen,agg,windows=build_feasibility_first_selection(spec,cfg,quick=quick)
    write_csv(sel,dest/"STAGE1_SELECTION_BY_FOLD.csv")
    write_csv(s1,dest/"STAGE1_ALL_CANDIDATE_STAGE1_AUDIT.csv")
    write_csv(screen,dest/"STAGE1_ALL_CANDIDATE_STAGE2_SCREEN.csv")
    write_csv(agg,dest/"STAGE1_ALL_CANDIDATE_FEASIBILITY.csv")
    write_csv(windows,dest/"STAGE1_INNER_WINDOW_AUDIT.csv")
    canon=canonicalize_spec(spec,cfg)
    atomic_write_json(dest/"STAGE2_FEATURE_SET.json",{
        "feature_count":len(canon.features),"features":list(canon.features),"categorical":list(canon.categorical),
        "group_count":len(group_units(spec,cfg)),"groups":group_units(spec,cfg),"outer_test_selection_used":0,
    })
    print(sel.to_string(index=False))


def gpu_check_cmd(args):
    from catboost import CatBoostRegressor
    import numpy as np
    X=np.random.default_rng(1).normal(size=(16000,32)); y=X[:,0]*2-X[:,1]+np.random.default_rng(2).normal(size=16000)
    m=CatBoostRegressor(iterations=120,depth=8,task_type="GPU",devices="0",verbose=False,allow_writing_files=False)
    m.fit(X,y); print("RTX GPU PASS")


def baseline_cmd(args):
    root=project_root(); cfg=load_config(args.config); spec,*_=_spec(cfg,root,args.run_name); dest=_paths(cfg,root,args.run_name)
    sel=pd.read_csv(dest/"STAGE1_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    backend=args.backend.upper(); quick=args.mode=="quick"; hw=cfg.get("hardware_v23",{})
    threads=int(hw.get("cpu_threads_per_model",6) if backend=="CPU" else hw.get("gpu_host_threads",8)); workers=int(hw.get("cpu_workers",4) if backend=="CPU" else 1)
    metrics,pred=run_backend_baselines_v23(spec,cfg,sel,backend,quick,threads,workers)
    bd=ensure_dir(dest/backend.lower()); write_csv(metrics,bd/"baseline.csv")
    if not pred.empty: pred.to_csv(bd/"oof_predictions.csv.gz",index=False,encoding="utf-8-sig",compression={"method":"gzip","mtime":0})
    print(metrics[["backend","seed","fold","selection_status","stage1_candidate","stage1_r2","r2","delta_r2_vs_stage1"]].to_string(index=False))


def ablation_cmd(args):
    root=project_root(); cfg=load_config(args.config); spec,*_=_spec(cfg,root,args.run_name); dest=_paths(cfg,root,args.run_name)
    sel=pd.read_csv(dest/"STAGE1_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    backend=args.backend.upper(); quick=args.mode=="quick"; base=pd.read_csv(dest/backend.lower()/"baseline.csv",encoding="utf-8-sig")
    hw=cfg.get("hardware_v23",{}); threads=int(hw.get("cpu_threads_per_model",6) if backend=="CPU" else hw.get("gpu_host_threads",8)); workers=int(hw.get("cpu_workers",4) if backend=="CPU" else 1)
    units=group_units(spec,cfg) if args.kind=="group" else individual_units(spec,cfg)
    if quick and args.kind=="individual":
        units=dict(list(sorted(units.items()))[:int(cfg.get("stage2_v23_ablation",{}).get("quick_individual_count",10))])
    raw=run_stage2_ablation(spec,cfg,sel,base,backend,units,kind=args.kind,quick=quick,workers=workers,threads=threads,cache_dir=Path(cfg["paths"]["cache_dir"])/"stage1_v23"/backend.lower()/args.kind)
    bd=ensure_dir(dest/backend.lower()); write_csv(raw,bd/f"{args.kind}_ablation_raw.csv"); write_csv(summarize(raw),bd/f"{args.kind}_ablation_summary.csv")
    print(summarize(raw).sort_values("median_r2_drop",ascending=False).head(20).to_string(index=False))


def finalize_cmd(args):
    root=project_root(); cfg=load_config(args.config); dest=_paths(cfg,root,args.run_name)
    selections=pd.read_csv(dest/"STAGE1_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    bases=[]; raws=[]
    for b in ["cpu","gpu"]:
        bases.append(pd.read_csv(dest/b/"baseline.csv",encoding="utf-8-sig"))
        for kind in ["group","individual"]:
            raws.append(pd.read_csv(dest/b/f"{kind}_ablation_raw.csv",encoding="utf-8-sig"))
    baseline=pd.concat(bases,ignore_index=True); raw=pd.concat(raws,ignore_index=True)
    write_csv(baseline,dest/"BASELINE_ALL_BACKENDS.csv"); write_csv(raw,dest/"STAGE2_ABLATION_RAW_ALL_BACKENDS.csv")
    summary=summarize(raw); write_csv(summary,dest/"STAGE2_ABLATION_SUMMARY_ALL_BACKENDS.csv")
    diag=fold_diagnostics(raw,float(cfg.get("stage2_v23_ablation",{}).get("harmful_r2_threshold",0.0005))); write_csv(diag,dest/"STAGE2_FOLD_DIAGNOSTICS.csv")
    f2=diag[diag.diagnostic_class.ne("NO_HARM_SIGNAL")].copy() if not diag.empty else pd.DataFrame(); write_csv(f2,dest/"FOLD2_HARMFUL_UNITS.csv")

    # Donor cross-check is descriptive only; donor never overrides Pohang direct ablation.
    registry_path = dest.parent / "POHANG_FEATURE_REGISTRY_FULL_V2.csv"
    donor_rows = []
    if registry_path.exists():
        reg = pd.read_csv(registry_path, encoding="utf-8-sig")
        if "column_name" in reg.columns:
            by_feature = reg.set_index("column_name", drop=False)
            for r in diag.itertuples(index=False):
                if str(r.kind) == "individual" and str(r.unit) in by_feature.index:
                    rr = by_feature.loc[str(r.unit)]
                    if isinstance(rr, pd.DataFrame): rr = rr.iloc[0]
                    donor_rows.append({
                        "kind": r.kind, "unit": r.unit, "concept_group": getattr(rr,"concept_group",""),
                        "donor_concept": getattr(rr,"donor_concept",""), "donor_decision": getattr(rr,"donor_decision",""),
                        "diagnostic_class": r.diagnostic_class, "fold2_cpu_drop": getattr(r,"fold2_cpu_drop",float("nan")),
                        "fold2_gpu_drop": getattr(r,"fold2_gpu_drop",float("nan")), "other_folds_median_drop": getattr(r,"other_folds_median_drop",float("nan")),
                        "note": "Donor is supporting evidence only; Pohang direct ablation has priority.",
                    })
                elif str(r.kind) == "group":
                    part = reg[reg.get("concept_group", pd.Series(dtype=str)).astype(str).eq(str(r.unit))] if "concept_group" in reg.columns else pd.DataFrame()
                    donor_rows.append({
                        "kind": r.kind, "unit": r.unit, "concept_group": r.unit,
                        "donor_concept": "|".join(sorted(set(part.get("donor_concept", pd.Series(dtype=str)).dropna().astype(str)))) if not part.empty else "",
                        "donor_decision": "|".join(sorted(set(part.get("donor_decision", pd.Series(dtype=str)).dropna().astype(str)))) if not part.empty else "",
                        "diagnostic_class": r.diagnostic_class, "fold2_cpu_drop": getattr(r,"fold2_cpu_drop",float("nan")),
                        "fold2_gpu_drop": getattr(r,"fold2_gpu_drop",float("nan")), "other_folds_median_drop": getattr(r,"other_folds_median_drop",float("nan")),
                        "note": "Donor is supporting evidence only; Pohang direct ablation has priority.",
                    })
    write_csv(pd.DataFrame(donor_rows), dest/"DONOR_CROSSCHECK_STAGE2.csv")

    gate=summarize_experiment_gate(selections,baseline,raw,cfg); atomic_write_json(dest/"STAGE1_STAGE2_V23_GATE.json",gate)
    lines=["# Feasibility-first Stage1 + Orthogonal Stage2 Full Ablation V2.3","",f"FINAL STATUS: **{gate['status']}**","","## Stage1 selection"]
    lines.append(f"- Feasible folds: {gate['feasible_folds']}/{gate['total_folds']}")
    lines.append(f"- NO_FEASIBLE_STAGE1 folds: {gate['no_feasible_folds']}")
    lines += ["","## Gate"]+[f"- {'PASS' if v else 'FAIL'} — `{k}`" for k,v in gate["checks"].items()]
    lines += ["","## Fold2 diagnostic","- Fold2 harmful results are post-hoc diagnostics only and are not used to modify Fold2 in this run."]
    if not f2.empty:
        for r in f2.sort_values(["diagnostic_class","fold2_conservative_drop"],na_position="last").head(20).itertuples(index=False):
            lines.append(f"- {r.kind}/{r.unit}: {r.diagnostic_class}, CPU={getattr(r,'fold2_cpu_drop',float('nan')):.6f}, GPU={getattr(r,'fold2_gpu_drop',float('nan')):.6f}")
    lines += ["","## Next-step rule","- Do not prune features in this same run using outer Fold2 results.","- Use FOLD2_HARMFUL_UNITS.csv only to design the next pre-registered Stage2 profile experiment.","- policy_use_allowed remains 0 for this diagnostic experiment."]
    (dest/"V23_REVIEW_KO.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"V2.3 finalize complete: {dest / 'V23_REVIEW_KO.md'}")


def build_parser():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/default.yaml"); p.add_argument("--run-name",default="stage1_feasibility_stage2_ablation_v23")
    s=p.add_subparsers(dest="command",required=True)
    sp=s.add_parser("prepare"); sp.add_argument("--mode",choices=["quick","full"],default="full"); sp.set_defaults(func=prepare_cmd)
    sp=s.add_parser("gpu-check"); sp.set_defaults(func=gpu_check_cmd)
    sp=s.add_parser("baseline"); sp.add_argument("--backend",choices=["CPU","GPU"],required=True); sp.add_argument("--mode",choices=["quick","full"],default="full"); sp.set_defaults(func=baseline_cmd)
    sp=s.add_parser("ablation"); sp.add_argument("--backend",choices=["CPU","GPU"],required=True); sp.add_argument("--kind",choices=["group","individual"],required=True); sp.add_argument("--mode",choices=["quick","full"],default="full"); sp.set_defaults(func=ablation_cmd)
    sp=s.add_parser("finalize"); sp.set_defaults(func=finalize_cmd)
    return p


def main():
    a=build_parser().parse_args(); a.func(a)

if __name__=="__main__": main()
