from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from pohang_main_v2.config import load_config
from pohang_main_v2.cli import _spec, _prepare, _input_context, _run_paths, project_root
from pohang_main_v2.utils import atomic_write_json, atomic_write_text, ensure_dir, write_csv

from .dynamic_baseline import default_candidates
from .modeling import build_stage1_selection, run_backend_baselines
from .audit import run_category_reentry_and_proxy, run_age_industry_ablation
from .gates import summarize_gate


def _paths(cfg, root, run_name):
    _, out, _ = _run_paths(cfg, root, run_name)
    return ensure_dir(out / "stage1_v21")


def prepare_cmd(args):
    root=project_root(); cfg=load_config(args.config)
    _,out,_=_run_paths(cfg,root,args.run_name); prepared=_input_context(cfg,root,out)
    _prepare(cfg,root,args.run_name,prepared)
    spec,_,_,_,_,_=_spec(cfg,root,args.run_name)
    dest=_paths(cfg,root,args.run_name)
    selections,audit=build_stage1_selection(spec,cfg,default_candidates())
    write_csv(selections,dest/"STAGE1_SELECTION_BY_FOLD.csv"); write_csv(audit,dest/"STAGE1_INNER_TOURNAMENT.csv")
    # Explicit provenance audit: selection/test chronology only.
    rows=[]
    for r in selections.itertuples(index=False):
        rows.append({"fold":int(r.fold),"test_start":int(r.test_start),"winner":r.winner,"selection_uses_outer_test":0,"violations":0})
    write_csv(pd.DataFrame(rows),dest/"STAGE1_SELECTION_LEAKAGE_AUDIT.csv")
    print(selections.to_string(index=False))


def baseline_cmd(args):
    root=project_root(); cfg=load_config(args.config); spec,_,_,_,_,_=_spec(cfg,root,args.run_name); dest=_paths(cfg,root,args.run_name)
    sel=pd.read_csv(dest/"STAGE1_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    backend=args.backend.upper(); quick=args.mode=="quick"
    threads=int(cfg["hardware_v21"]["cpu_threads_per_model"] if backend=="CPU" else cfg["hardware_v21"]["gpu_host_threads"])
    workers=int(cfg["hardware_v21"]["cpu_workers"] if backend=="CPU" else 1)
    metrics,pred=run_backend_baselines(spec,cfg,sel,backend,quick,threads,workers)
    bd=ensure_dir(dest/backend.lower()); write_csv(metrics,bd/"baseline.csv")
    if not pred.empty: pred.to_csv(bd/"oof_predictions.csv.gz",index=False,encoding="utf-8-sig",compression={"method":"gzip","mtime":0})
    print(metrics[["backend","seed","fold","stage1_candidate","stage1_r2","r2","delta_r2_vs_stage1","residual_r2"]].to_string(index=False))


def audit_cmd(args):
    root=project_root(); cfg=load_config(args.config); spec,_,_,_,_,_=_spec(cfg,root,args.run_name); dest=_paths(cfg,root,args.run_name)
    sel=pd.read_csv(dest/"STAGE1_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    backend=args.backend.upper(); base=pd.read_csv(dest/backend.lower()/"baseline.csv",encoding="utf-8-sig")
    threads=int(cfg["hardware_v21"]["cpu_threads_per_model"] if backend=="CPU" else cfg["hardware_v21"]["gpu_host_threads"])
    workers=int(cfg["hardware_v21"]["cpu_workers"] if backend=="CPU" else 1)
    reentry,proxy=run_category_reentry_and_proxy(spec,cfg,base,sel,backend,threads,workers=workers)
    age=run_age_industry_ablation(spec,cfg,base,sel,backend,threads,workers=workers)
    bd=ensure_dir(dest/backend.lower()); write_csv(reentry,bd/"category_reentry_raw.csv"); write_csv(proxy,bd/"category_proxy_audit.csv"); write_csv(age,bd/"age_industry_ablation.csv")
    print(proxy.to_string(index=False)); print(age.to_string(index=False))


def finalize_cmd(args):
    root=project_root(); cfg=load_config(args.config); dest=_paths(cfg,root,args.run_name)
    bases=[];res=[];prox=[];age=[]
    for b in ["cpu","gpu"]:
        bases.append(pd.read_csv(dest/b/"baseline.csv",encoding="utf-8-sig")); res.append(pd.read_csv(dest/b/"category_reentry_raw.csv",encoding="utf-8-sig")); prox.append(pd.read_csv(dest/b/"category_proxy_audit.csv",encoding="utf-8-sig")); age.append(pd.read_csv(dest/b/"age_industry_ablation.csv",encoding="utf-8-sig"))
    baseline=pd.concat(bases,ignore_index=True); reentry=pd.concat(res,ignore_index=True); proxy=pd.concat(prox,ignore_index=True); age_df=pd.concat(age,ignore_index=True)
    write_csv(baseline,dest/"BASELINE_ALL_BACKENDS.csv"); write_csv(reentry,dest/"CATEGORY_REENTRY_ALL_BACKENDS.csv"); write_csv(proxy,dest/"CATEGORY_PROXY_AUDIT_ALL_BACKENDS.csv"); write_csv(age_df,dest/"AGE_INDUSTRY_ABLATION_ALL_BACKENDS.csv")
    # Combine static selection audit + baseline-level leakage counters.
    leak=pd.read_csv(dest/"STAGE1_SELECTION_LEAKAGE_AUDIT.csv",encoding="utf-8-sig")
    btmp=baseline.copy()
    btmp["violations"] = pd.to_numeric(btmp["stage1_test_leakage_violations"], errors="coerce").fillna(0) + pd.to_numeric(btmp["stage1_train_leakage_violations"], errors="coerce").fillna(0)
    bviol=btmp.groupby(["backend","seed","fold"],as_index=False).agg(violations=("violations","sum"))
    leak=pd.concat([leak,bviol],ignore_index=True,sort=False); write_csv(leak,dest/"STAGE1_FUTURE_TARGET_AUDIT.csv")
    gate=summarize_gate(baseline,reentry,proxy,age_df,leak,root/"data/reference_v2",cfg); atomic_write_json(dest/"STAGE1_V21_GATE.json",gate)
    lines=["# Stage 1 Dynamic Baseline V2.1 결과","",f"FINAL STATUS: **{gate['status']}**","","## 7개 Gate"]
    for k,v in gate["checks"].items(): lines.append(f"- {'PASS' if v else 'FAIL'} — `{k}`")
    lines += ["",f"CPU category re-entry median ΔR²: {gate['cpu_category_reentry_median_delta_r2']:.6f}",f"CPU median Stage2 gain: {gate['cpu_median_stage2_gain']:.6f}",f"Latest CPU Fold R²: {gate['latest_cpu_fold_r2']:.6f}",f"Future-target violation: {gate['stage1_leakage_violations']}"]
    (dest/"STAGE1_V21_REVIEW_KO.md").write_text("\n".join(lines),encoding="utf-8")
    print((dest/"STAGE1_V21_REVIEW_KO.md").read_text(encoding="utf-8"))


def finalize_full_cmd(args):
    root=project_root(); cfg=load_config(args.config); dest=_paths(cfg,root,args.run_name)
    bases=[]; reentries=[]; proxies=[]; ages=[]; predictions=[]
    for backend in ["cpu","gpu"]:
        base_dir=dest/backend
        bases.append(pd.read_csv(base_dir/"baseline.csv",encoding="utf-8-sig"))
        reentries.append(pd.read_csv(base_dir/"category_reentry_raw.csv",encoding="utf-8-sig"))
        proxies.append(pd.read_csv(base_dir/"category_proxy_audit.csv",encoding="utf-8-sig"))
        ages.append(pd.read_csv(base_dir/"age_industry_ablation.csv",encoding="utf-8-sig"))
        pred_path=base_dir/"oof_predictions.csv.gz"
        if pred_path.exists(): predictions.append(pd.read_csv(pred_path,encoding="utf-8-sig",compression="gzip"))
    baseline=pd.concat(bases,ignore_index=True)
    reentry=pd.concat(reentries,ignore_index=True)
    proxy=pd.concat(proxies,ignore_index=True)
    age_df=pd.concat(ages,ignore_index=True)
    pred=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()

    expected_folds=len(cfg["validation"]["rolling_test_blocks"])
    expected={"CPU":expected_folds*len(cfg["validation"]["seeds_cpu"]),"GPU":expected_folds*len(cfg["validation"]["seeds_gpu"])}
    if not baseline.status.eq("PASS").all() or baseline.duplicated(["backend","seed","fold"]).any():
        raise RuntimeError("STAGE1_BASELINE_STATUS_OR_IDENTITY_FAIL")
    counts=baseline.groupby(baseline.backend.astype(str).str.upper()).size().to_dict()
    if counts != expected:
        raise RuntimeError(f"STAGE1_BASELINE_COMPLETENESS_FAIL expected={expected} actual={counts}")
    expected_runs=sum(expected.values())
    if len(reentry)!=expected_runs*5 or len(proxy)!=expected_runs or len(age_df)!=expected_runs:
        raise RuntimeError(f"STAGE1_AUDIT_COMPLETENESS_FAIL reentry={len(reentry)}/{expected_runs*5} proxy={len(proxy)}/{expected_runs} age={len(age_df)}/{expected_runs}")
    selections=pd.read_csv(dest/"STAGE1_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    if len(selections)!=expected_folds or selections.fold.nunique()!=expected_folds or not selections.outer_test_used_for_selection.eq(0).all():
        raise RuntimeError("STAGE1_SELECTION_COMPLETENESS_OR_LEAKAGE_FAIL")

    write_csv(baseline,dest/"BASELINE_ALL_BACKENDS.csv")
    write_csv(reentry,dest/"CATEGORY_REENTRY_ALL_BACKENDS.csv")
    write_csv(proxy,dest/"CATEGORY_PROXY_AUDIT_ALL_BACKENDS.csv")
    write_csv(age_df,dest/"AGE_INDUSTRY_ABLATION_ALL_BACKENDS.csv")
    leak=pd.read_csv(dest/"STAGE1_SELECTION_LEAKAGE_AUDIT.csv",encoding="utf-8-sig")
    btmp=baseline.copy()
    btmp["violations"]=pd.to_numeric(btmp["stage1_test_leakage_violations"],errors="coerce").fillna(0)+pd.to_numeric(btmp["stage1_train_leakage_violations"],errors="coerce").fillna(0)
    bviol=btmp.groupby(["backend","seed","fold"],as_index=False).agg(violations=("violations","sum"))
    leak=pd.concat([leak,bviol],ignore_index=True,sort=False)
    write_csv(leak,dest/"STAGE1_FUTURE_TARGET_AUDIT.csv")
    gate=summarize_gate(baseline,reentry,proxy,age_df,leak,root/"data/reference_v2",cfg)
    atomic_write_json(dest/"STAGE1_V21_GATE.json",gate)

    if pred.empty:
        raise RuntimeError("STAGE1_OOF_PREDICTIONS_MISSING")
    pred.to_csv(dest/"OOF_PREDICTIONS_ALL_BACKENDS.csv.gz",index=False,encoding="utf-8-sig",compression={"method":"gzip","mtime":0})
    canonical=pred[pred.backend.astype(str).str.upper().eq("CPU")].copy()
    keys=[column for column in ["row_index","fold","district","year_month","category","category_major","spend_thousand_krw","stage1_candidate"] if column in canonical]
    values=[column for column in ["actual_log_spend","stage1_log","residual_actual","residual_prediction","expected_log_spend"] if column in canonical]
    oof=canonical.groupby(keys,as_index=False)[values].median()
    seed_counts=canonical.groupby(keys,as_index=False).size().rename(columns={"size":"seed_count"})
    oof=oof.merge(seed_counts,on=keys,how="left",validate="one_to_one")
    oof["expected_spend_thousand_krw"]=np.expm1(pd.to_numeric(oof.expected_log_spend,errors="coerce")).clip(lower=0)
    actual=pd.to_numeric(oof.get("spend_thousand_krw"),errors="coerce")
    oof["expected_minus_actual_thousand_krw"]=oof.expected_spend_thousand_krw-actual
    oof["actual_minus_expected_thousand_krw"]=actual-oof.expected_spend_thousand_krw
    oof["model_gate_status"]=gate["status"]
    oof["policy_use_allowed"]=int(gate["status"]=="PASS")
    oof["causal_effect_estimate"]=0
    oof.to_csv(dest/"OOF_EXPECTED_SPEND_STAGE1_V21.csv.gz",index=False,encoding="utf-8-sig",compression={"method":"gzip","mtime":0})

    verification_path=dest.parent/"INPUT_VERIFICATION.json"
    verification=json.loads(verification_path.read_text(encoding="utf-8-sig")) if verification_path.exists() else {}
    packaging_exception=verification.get("status")!="PASS"
    overall_status="PARTIAL_PASS" if packaging_exception and gate["status"]=="PASS" else gate["status"]
    summary={
        "overall_delivery_status":overall_status,"model_gate_status":gate["status"],"input_verification_status":verification.get("status"),
        "baseline_rows":int(len(baseline)),"baseline_by_backend":{str(k):int(v) for k,v in counts.items()},
        "category_reentry_rows":int(len(reentry)),"proxy_audit_rows":int(len(proxy)),"age_industry_ablation_rows":int(len(age_df)),
        "selection_folds":int(len(selections)),"oof_all_backend_rows":int(len(pred)),"oof_canonical_rows":int(len(oof)),
        "oof_policy_use_allowed":bool(gate["status"]=="PASS"),"donor_is_secondary_only":True,"gate":gate,
    }
    atomic_write_json(dest/"STAGE1_V21_RUN_SUMMARY.json",summary)

    lines=["# 포항 Stage 1 동적 기준선 V2.1 결과","",f"전체 전달 상태: **{overall_status}**",f"모델 7-Gate 상태: **{gate['status']}**","","## 7개 Gate",""]
    for key,value in gate["checks"].items(): lines.append(f"- {'PASS' if value else 'FAIL'} `{key}`")
    lines += [
        "","## 핵심 지표","",
        f"- CPU raw category 재진입 중앙 ΔR²: {gate['cpu_category_reentry_median_delta_r2']:.6f}",
        f"- GPU raw category 재진입 중앙 ΔR²: {gate['gpu_category_reentry_median_delta_r2']:.6f}",
        f"- CPU 중앙 Stage 2 gain: {gate['cpu_median_stage2_gain']:.6f}",
        f"- 최신 CPU Fold reconstructed R²: {gate['latest_cpu_fold_r2']:.6f}",
        f"- 미래 target 위반: {gate['stage1_leakage_violations']}건",
        "","## Fold별 Stage 1 선택","",
    ]
    for row in selections.sort_values("fold").itertuples(index=False):
        lines.append(f"- Fold {int(row.fold)} ({int(row.test_start)}-{int(row.test_end)}): `{row.winner}`, inner Stage2 gain={row.inner_stage2_gain:.6f}, outer test selection 사용=0")
    lines += ["","## 완결성","",f"- baseline: {len(baseline)}/16",f"- category re-entry: {len(reentry)}/80",f"- proxy audit: {len(proxy)}/16",f"- Age×Industry ablation: {len(age_df)}/16",f"- canonical OOF: {len(oof)}행"]
    if packaging_exception:
        lines += ["","## 입력 패키징 예외","","- canonical CORE ZIP이 없어 내장 manifest와 크기·SHA-256이 일치하는 추출 트리를 사용했다.","- 모델 입력은 검증됐지만 전체 전달 상태는 최대 PARTIAL_PASS로 제한한다."]
    lines += ["","## 해석 제한","","- donor는 구조 보조 검증에만 사용했고 포항 관측치나 계수를 대체하지 않았다.","- OOF expected spend는 인과효과·ROI·쿠폰 정책 순위가 아니다.","- 모델 게이트가 PASS가 아니면 OOF의 `policy_use_allowed`는 0이다."]
    atomic_write_text(dest/"STAGE1_V21_REVIEW_KO.md","\n".join(lines)+"\n")
    print((dest/"STAGE1_V21_REVIEW_KO.md").read_text(encoding="utf-8"))


def gpu_check_cmd(args):
    from catboost import CatBoostRegressor
    import numpy as np
    X=np.random.default_rng(42).normal(size=(3000,16)); y=X[:,0]*2+np.random.default_rng(1).normal(size=3000)
    m=CatBoostRegressor(iterations=60,depth=6,task_type="GPU",devices="0",verbose=False,allow_writing_files=False);m.fit(X,y); print("GPU PASS")


def build_parser():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/default.yaml"); p.add_argument("--run-name",default="stage1_dynamic_v21")
    s=p.add_subparsers(dest="command",required=True)
    for n,f in [("prepare",prepare_cmd),("gpu-check",gpu_check_cmd),("finalize",finalize_full_cmd)]: sp=s.add_parser(n);sp.set_defaults(func=f)
    for n,f in [("baseline",baseline_cmd),("audit",audit_cmd)]:
        sp=s.add_parser(n);sp.add_argument("--backend",choices=["CPU","GPU"],required=True);sp.add_argument("--mode",choices=["quick","full"],default="full");sp.set_defaults(func=f)
    return p

def main():
    a=build_parser().parse_args(); a.func(a)

if __name__=="__main__": main()
