from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pohang_main_v2.config import load_config
from pohang_main_v2.cli import _spec, _prepare, _input_context, _run_paths, project_root
from pohang_main_v2.utils import atomic_write_json, atomic_write_text, ensure_dir, write_csv

from .dynamic_baseline import default_candidates
from .modeling import build_stage1_selection, canonicalize_spec, run_backend_baselines
from .audit import run_category_reentry_and_proxy, run_age_industry_ablation
from .gates import summarize_gate
from .orthogonal_interactions import AGE_BASES, WEATHER_BASES, orthogonalization_identity_audit


def _paths(cfg, root, run_name):
    _, out, _ = _run_paths(cfg, root, run_name)
    return ensure_dir(out / "stage1_v22")


def prepare_cmd(args):
    root = project_root(); cfg = load_config(args.config)
    _, out, _ = _run_paths(cfg, root, args.run_name)
    prepared = _input_context(cfg, root, out)
    _prepare(cfg, root, args.run_name, prepared)
    spec, _, _, _, _, _ = _spec(cfg, root, args.run_name)
    dest = _paths(cfg, root, args.run_name)
    quick = args.mode == "quick"
    selections, stage1_audit, screen_audit, window_audit = build_stage1_selection(spec, cfg, default_candidates(), quick=quick)
    write_csv(selections, dest / "STAGE1_SELECTION_BY_FOLD.csv")
    write_csv(stage1_audit, dest / "STAGE1_MULTIWINDOW_STAGE1_AUDIT.csv")
    write_csv(screen_audit, dest / "STAGE1_MULTIWINDOW_GATE_SCREEN.csv")
    write_csv(window_audit, dest / "STAGE1_INNER_WINDOW_AUDIT.csv")
    canonical = canonicalize_spec(spec, cfg)
    ident = orthogonalization_identity_audit(canonical.frame, canonical.features)
    ident.update({
        "canonical_feature_count": len(canonical.features),
        "age_orthogonal_features": list(canonical.groups.get("age_industry_interaction", [])),
        "weather_orthogonal_features": list(canonical.groups.get("weather_industry_interaction", [])),
        "mode": args.mode,
    })
    atomic_write_json(dest / "ORTHOGONAL_FEATURE_IDENTITY_AUDIT.json", ident)
    defs = []
    for c, b in {**AGE_BASES, **WEATHER_BASES}.items():
        defs.append({"feature": c, "base_feature": b, "definition": "current_value - historical_shrunk_mean_by_category_major", "raw_sparse_onehot": 0})
    write_csv(pd.DataFrame(defs), dest / "ORTHOGONAL_INTERACTION_DEFINITIONS.csv")
    print(selections.to_string(index=False))
    print(json.dumps(ident, ensure_ascii=False, indent=2))


def baseline_cmd(args):
    root = project_root(); cfg = load_config(args.config)
    spec, _, _, _, _, _ = _spec(cfg, root, args.run_name); dest = _paths(cfg, root, args.run_name)
    sel = pd.read_csv(dest / "STAGE1_SELECTION_BY_FOLD.csv", encoding="utf-8-sig")
    backend = args.backend.upper(); quick = args.mode == "quick"
    hw = cfg["hardware_v22"]
    threads = int(hw["cpu_threads_per_model"] if backend == "CPU" else hw["gpu_host_threads"])
    workers = int(hw["cpu_workers"] if backend == "CPU" else 1)
    metrics, pred = run_backend_baselines(spec, cfg, sel, backend, quick, threads, workers)
    bd = ensure_dir(dest / backend.lower()); write_csv(metrics, bd / "baseline.csv")
    if not pred.empty:
        pred.to_csv(bd / "oof_predictions.csv.gz", index=False, encoding="utf-8-sig", compression={"method": "gzip", "mtime": 0})
    print(metrics[["backend", "seed", "fold", "stage1_candidate", "stage1_r2", "r2", "delta_r2_vs_stage1", "residual_r2"]].to_string(index=False))


def audit_cmd(args):
    root = project_root(); cfg = load_config(args.config)
    spec, _, _, _, _, _ = _spec(cfg, root, args.run_name); dest = _paths(cfg, root, args.run_name)
    sel = pd.read_csv(dest / "STAGE1_SELECTION_BY_FOLD.csv", encoding="utf-8-sig")
    backend = args.backend.upper(); base = pd.read_csv(dest / backend.lower() / "baseline.csv", encoding="utf-8-sig")
    hw = cfg["hardware_v22"]
    threads = int(hw["cpu_threads_per_model"] if backend == "CPU" else hw["gpu_host_threads"])
    quick = args.mode == "quick"
    workers = int(hw["cpu_workers"] if backend == "CPU" else 1)
    reentry, proxy, compare = run_category_reentry_and_proxy(spec, cfg, base, sel, backend, threads, quick=quick, workers=workers)
    age = run_age_industry_ablation(spec, cfg, base, sel, backend, threads, quick=quick, workers=workers)
    bd = ensure_dir(dest / backend.lower())
    write_csv(reentry, bd / "category_reentry_raw.csv")
    write_csv(proxy, bd / "category_proxy_audit.csv")
    write_csv(compare, bd / "orth_vs_raw_proxy_comparison.csv")
    write_csv(age, bd / "age_industry_ablation.csv")
    print(proxy.to_string(index=False)); print(age.to_string(index=False))


def finalize_cmd(args):
    root = project_root(); cfg = load_config(args.config); dest = _paths(cfg, root, args.run_name)
    bases=[]; res=[]; prox=[]; comp=[]; age=[]
    for b in ["cpu", "gpu"]:
        bases.append(pd.read_csv(dest / b / "baseline.csv", encoding="utf-8-sig"))
        res.append(pd.read_csv(dest / b / "category_reentry_raw.csv", encoding="utf-8-sig"))
        prox.append(pd.read_csv(dest / b / "category_proxy_audit.csv", encoding="utf-8-sig"))
        comp.append(pd.read_csv(dest / b / "orth_vs_raw_proxy_comparison.csv", encoding="utf-8-sig"))
        age.append(pd.read_csv(dest / b / "age_industry_ablation.csv", encoding="utf-8-sig"))
    baseline=pd.concat(bases,ignore_index=True); reentry=pd.concat(res,ignore_index=True); proxy=pd.concat(prox,ignore_index=True); compare=pd.concat(comp,ignore_index=True); age_df=pd.concat(age,ignore_index=True)
    write_csv(baseline,dest/"BASELINE_ALL_BACKENDS.csv"); write_csv(reentry,dest/"CATEGORY_REENTRY_ALL_BACKENDS.csv")
    write_csv(proxy,dest/"CATEGORY_PROXY_AUDIT_ALL_BACKENDS.csv"); write_csv(compare,dest/"ORTH_VS_RAW_PROXY_ALL_BACKENDS.csv")
    write_csv(age_df,dest/"AGE_INDUSTRY_ABLATION_ALL_BACKENDS.csv")

    selections=pd.read_csv(dest/"STAGE1_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    windows=pd.read_csv(dest/"STAGE1_INNER_WINDOW_AUDIT.csv",encoding="utf-8-sig")
    leak_rows=[]
    for r in selections.itertuples(index=False):
        leak_rows.append({"source":"selection","fold":int(r.fold),"backend":"SELECTION","seed":-1,"violations":int(r.outer_test_used_for_selection),"detail":"outer_test_used_for_selection"})
    for r in windows.itertuples(index=False):
        v = int(not (int(r.valid_before_outer_test)==1 and int(r.train_before_valid)==1))
        leak_rows.append({"source":"inner_window","fold":int(r.fold),"backend":"SELECTION","seed":-1,"violations":v,"detail":f"window_{int(r.window)}"})
    for r in baseline.itertuples(index=False):
        v = int(getattr(r,"stage1_train_leakage_violations",0))+int(getattr(r,"stage1_test_leakage_violations",0))+int(getattr(r,"orth_train_leakage_violations",0))+int(getattr(r,"orth_test_leakage_violations",0))
        leak_rows.append({"source":"baseline","fold":int(r.fold),"backend":str(r.backend),"seed":int(r.seed),"violations":v,"detail":"stage1+orth chronology"})
    leakage=pd.DataFrame(leak_rows); write_csv(leakage,dest/"FUTURE_TARGET_FEATURE_AUDIT.csv")

    gate=summarize_gate(baseline,reentry,proxy,age_df,leakage,selections,windows,root/"data/reference_v21_full",cfg)
    atomic_write_json(dest/"STAGE1_V22_GATE.json",gate)
    lines=["# Stage 1 Multi-Window + Orthogonal Interaction V2.2 결과","",f"FINAL STATUS: **{gate['status']}**","","## Gate"]
    for k,v in gate["checks"].items(): lines.append(f"- {'PASS' if v else 'FAIL'} — `{k}`")
    lines += [
        "", "## 핵심 수치",
        f"- CPU category 재진입 median ΔR²: {gate['cpu_category_reentry_median_delta_r2']:.6f}",
        f"- CPU category 재진입 worst-fold ΔR²: {gate['cpu_category_reentry_worst_fold_delta_r2']:.6f}",
        f"- CPU median Stage2 gain: {gate['cpu_median_stage2_gain']:.6f}",
        f"- Latest CPU Fold R²: {gate['latest_cpu_fold_r2']:.6f}",
        f"- Orth proxy excess median: {gate['orth_proxy_excess_median']}",
        f"- Raw sparse proxy excess median: {gate['raw_proxy_excess_median']}",
        f"- Orth - Raw proxy improvement: {gate['orth_minus_raw_proxy_excess_improvement_median']}",
        f"- Future target/feature violation: {gate['future_target_feature_violations']}",
        "", "## 해석",
        "- Stage 1 후보는 outer-train 내부의 여러 시간창으로만 선택된다.",
        "- canonical Stage 2에는 raw 21개 Age×Industry sparse 열이 들어가지 않는다.",
        "- Age×Industry는 category별 과거 정상 연령구성 대비 deviation 3개로 표현한다.",
        "- raw sparse interaction은 진단 비교용으로만 재학습한다.",
    ]
    (dest/"STAGE1_V22_REVIEW_KO.md").write_text("\n".join(lines),encoding="utf-8")
    print((dest/"STAGE1_V22_REVIEW_KO.md").read_text(encoding="utf-8"))


def finalize_full_cmd(args):
    root=project_root(); cfg=load_config(args.config); dest=_paths(cfg,root,args.run_name)
    bases=[]; reentries=[]; proxies=[]; comparisons=[]; ages=[]; predictions=[]
    for backend in ["cpu","gpu"]:
        base_dir=dest/backend
        bases.append(pd.read_csv(base_dir/"baseline.csv",encoding="utf-8-sig"))
        reentries.append(pd.read_csv(base_dir/"category_reentry_raw.csv",encoding="utf-8-sig"))
        proxies.append(pd.read_csv(base_dir/"category_proxy_audit.csv",encoding="utf-8-sig"))
        comparisons.append(pd.read_csv(base_dir/"orth_vs_raw_proxy_comparison.csv",encoding="utf-8-sig"))
        ages.append(pd.read_csv(base_dir/"age_industry_ablation.csv",encoding="utf-8-sig"))
        predictions.append(pd.read_csv(base_dir/"oof_predictions.csv.gz",encoding="utf-8-sig",compression="gzip"))
    baseline=pd.concat(bases,ignore_index=True); reentry=pd.concat(reentries,ignore_index=True)
    proxy=pd.concat(proxies,ignore_index=True); compare=pd.concat(comparisons,ignore_index=True)
    age_df=pd.concat(ages,ignore_index=True); pred=pd.concat(predictions,ignore_index=True)

    expected_folds=len(cfg["validation"]["rolling_test_blocks"])
    expected={"CPU":expected_folds*len(cfg["validation"]["seeds_cpu"]),"GPU":expected_folds*len(cfg["validation"]["seeds_gpu"])}
    if len(baseline)!=sum(expected.values()) or not baseline.status.eq("PASS").all() or baseline.duplicated(["backend","seed","fold"]).any():
        raise RuntimeError("V22_BASELINE_STATUS_IDENTITY_COMPLETENESS_FAIL")
    counts=baseline.groupby(baseline.backend.astype(str).str.upper()).size().to_dict()
    if counts!=expected or baseline.data_identity.nunique()!=1:
        raise RuntimeError(f"V22_BASELINE_BACKEND_OR_DATA_IDENTITY_FAIL expected={expected} actual={counts}")
    expected_runs=sum(expected.values())
    expected_profiles=7
    if len(reentry)!=expected_runs*expected_profiles or len(proxy)!=expected_runs or len(compare)!=expected_runs or len(age_df)!=expected_runs:
        raise RuntimeError(f"V22_AUDIT_COMPLETENESS_FAIL reentry={len(reentry)}/{expected_runs*expected_profiles} proxy={len(proxy)}/{expected_runs} compare={len(compare)}/{expected_runs} age={len(age_df)}/{expected_runs}")

    selections=pd.read_csv(dest/"STAGE1_SELECTION_BY_FOLD.csv",encoding="utf-8-sig")
    windows=pd.read_csv(dest/"STAGE1_INNER_WINDOW_AUDIT.csv",encoding="utf-8-sig")
    stage1_audit=pd.read_csv(dest/"STAGE1_MULTIWINDOW_STAGE1_AUDIT.csv",encoding="utf-8-sig")
    screen_audit=pd.read_csv(dest/"STAGE1_MULTIWINDOW_GATE_SCREEN.csv",encoding="utf-8-sig")
    if len(selections)!=4 or selections.fold.nunique()!=4 or not selections.outer_test_used_for_selection.eq(0).all():
        raise RuntimeError("V22_SELECTION_COMPLETENESS_OR_OUTER_TEST_FAIL")
    if len(windows)!=12 or not windows.valid_before_outer_test.eq(1).all() or not windows.train_before_valid.eq(1).all():
        raise RuntimeError("V22_MULTIWINDOW_CHRONOLOGY_COMPLETENESS_FAIL")
    if len(stage1_audit)!=180 or len(screen_audit)!=48 or not selections.inner_window_count.eq(3).all():
        raise RuntimeError(f"V22_TOURNAMENT_COMPLETENESS_FAIL stage1={len(stage1_audit)}/180 screen={len(screen_audit)}/48")
    identity=json.loads((dest/"ORTHOGONAL_FEATURE_IDENTITY_AUDIT.json").read_text(encoding="utf-8-sig"))
    if identity.get("raw_sparse_feature_count")!=0 or identity.get("orthogonal_feature_count")!=9 or identity.get("canonical_feature_count")!=78:
        raise RuntimeError(f"V22_CANONICAL_FEATURE_IDENTITY_FAIL: {identity}")

    write_csv(baseline,dest/"BASELINE_ALL_BACKENDS.csv"); write_csv(reentry,dest/"CATEGORY_REENTRY_ALL_BACKENDS.csv")
    write_csv(proxy,dest/"CATEGORY_PROXY_AUDIT_ALL_BACKENDS.csv"); write_csv(compare,dest/"ORTH_VS_RAW_PROXY_ALL_BACKENDS.csv")
    write_csv(age_df,dest/"AGE_INDUSTRY_ABLATION_ALL_BACKENDS.csv")
    leak_rows=[]
    for row in selections.itertuples(index=False):
        leak_rows.append({"source":"selection","fold":int(row.fold),"backend":"SELECTION","seed":-1,"violations":int(row.outer_test_used_for_selection),"detail":"outer_test_used_for_selection"})
    for row in windows.itertuples(index=False):
        violation=int(not(int(row.valid_before_outer_test)==1 and int(row.train_before_valid)==1))
        leak_rows.append({"source":"inner_window","fold":int(row.fold),"backend":"SELECTION","seed":-1,"violations":violation,"detail":f"window_{int(row.window)}"})
    for row in baseline.itertuples(index=False):
        violation=sum(int(getattr(row,column,0)) for column in ["stage1_train_leakage_violations","stage1_test_leakage_violations","orth_train_leakage_violations","orth_test_leakage_violations"])
        leak_rows.append({"source":"baseline","fold":int(row.fold),"backend":str(row.backend),"seed":int(row.seed),"violations":violation,"detail":"stage1+orth chronology"})
    leakage=pd.DataFrame(leak_rows); write_csv(leakage,dest/"FUTURE_TARGET_FEATURE_AUDIT.csv")
    gate=summarize_gate(baseline,reentry,proxy,age_df,leakage,selections,windows,root/"data/reference_v21_full",cfg)
    atomic_write_json(dest/"STAGE1_V22_GATE.json",gate)

    pred.to_csv(dest/"OOF_PREDICTIONS_ALL_BACKENDS.csv.gz",index=False,encoding="utf-8-sig",compression={"method":"gzip","mtime":0})
    canonical=pred[pred.backend.astype(str).str.upper().eq("CPU")].copy()
    selected=selections.set_index("fold").winner.astype(str).to_dict()
    if not canonical.stage1_candidate.astype(str).eq(canonical.fold.map(selected).astype(str)).all():
        raise RuntimeError("V22_OOF_SELECTION_IDENTITY_FAIL")
    keys=[column for column in ["row_index","fold","district","year_month","category","category_major","spend_thousand_krw","stage1_candidate"] if column in canonical]
    values=[column for column in ["actual_log_spend","stage1_log","residual_actual","residual_prediction","expected_log_spend"] if column in canonical]
    oof=canonical.groupby(keys,as_index=False)[values].median()
    oof=oof.merge(canonical.groupby(keys,as_index=False).size().rename(columns={"size":"seed_count"}),on=keys,how="left",validate="one_to_one")
    oof["expected_spend_thousand_krw"]=np.expm1(pd.to_numeric(oof.expected_log_spend,errors="coerce")).clip(lower=0)
    actual=pd.to_numeric(oof.get("spend_thousand_krw"),errors="coerce")
    oof["expected_minus_actual_thousand_krw"]=oof.expected_spend_thousand_krw-actual
    oof["actual_minus_expected_thousand_krw"]=actual-oof.expected_spend_thousand_krw
    oof["model_gate_status"]=gate["status"]; oof["policy_use_allowed"]=int(gate["status"]=="PASS"); oof["causal_effect_estimate"]=0
    oof.to_csv(dest/"OOF_EXPECTED_SPEND_STAGE1_V22.csv.gz",index=False,encoding="utf-8-sig",compression={"method":"gzip","mtime":0})

    verification_path=dest.parent/"INPUT_VERIFICATION.json"
    verification=json.loads(verification_path.read_text(encoding="utf-8-sig")) if verification_path.exists() else {}
    packaging_exception=verification.get("status")!="PASS"
    overall="PARTIAL_PASS" if packaging_exception and gate["status"]=="PASS" else gate["status"]
    summary={"overall_delivery_status":overall,"model_gate_status":gate["status"],"input_verification_status":verification.get("status"),"baseline_rows":len(baseline),"baseline_by_backend":{str(k):int(v) for k,v in counts.items()},"category_reentry_rows":len(reentry),"proxy_rows":len(proxy),"comparison_rows":len(compare),"age_industry_rows":len(age_df),"selection_folds":len(selections),"inner_window_rows":len(windows),"stage1_tournament_rows":len(stage1_audit),"gate_screen_rows":len(screen_audit),"oof_all_backend_rows":len(pred),"oof_canonical_rows":len(oof),"raw_sparse_canonical_features":identity["raw_sparse_feature_count"],"orthogonal_features":identity["orthogonal_feature_count"],"donor_is_secondary_only":True,"gate":gate}
    atomic_write_json(dest/"STAGE1_V22_RUN_SUMMARY.json",summary)

    lines=["# 포항 Stage 1 Multi-Window + Orthogonal Interaction V2.2 결과","",f"전체 전달 상태: **{overall}**",f"모델 Gate 상태: **{gate['status']}**","","## Gate",""]
    for key,value in gate["checks"].items(): lines.append(f"- {'PASS' if value else 'FAIL'} `{key}`")
    lines += ["","## 핵심 지표","",f"- CPU category 재진입 중앙 ΔR²: {gate['cpu_category_reentry_median_delta_r2']:.6f}",f"- CPU category 재진입 worst-fold ΔR²: {gate['cpu_category_reentry_worst_fold_delta_r2']:.6f}",f"- CPU 중앙 Stage 2 gain: {gate['cpu_median_stage2_gain']:.6f}",f"- 최신 CPU Fold R²: {gate['latest_cpu_fold_r2']:.6f}",f"- Orth proxy excess: {gate['orth_proxy_excess_median']}",f"- Raw proxy excess: {gate['raw_proxy_excess_median']}",f"- Orth-Raw proxy improvement: {gate['orth_minus_raw_proxy_excess_improvement_median']}",f"- 미래 target/feature 위반: {gate['future_target_feature_violations']}건","","## Fold별 Stage 1 선택",""]
    for row in selections.sort_values("fold").itertuples(index=False):
        lines.append(f"- Fold {int(row.fold)} ({int(row.test_start)}-{int(row.test_end)}): `{row.winner}`, windows={int(row.inner_window_count)}, median gain={row.inner_median_stage2_gain:.6f}, worst gain={row.inner_worst_stage2_gain:.6f}, outer test selection=0")
    lines += ["","## 완결성","",f"- baseline: {len(baseline)}/16",f"- category re-entry: {len(reentry)}/112",f"- proxy/orth-vs-raw/Age×Industry: {len(proxy)}/16, {len(compare)}/16, {len(age_df)}/16",f"- inner windows: {len(windows)}/12",f"- Stage1 candidate-window 평가: {len(stage1_audit)}/180",f"- Gate screen: {len(screen_audit)}/48",f"- canonical OOF: {len(oof)}행",f"- canonical raw sparse: {identity['raw_sparse_feature_count']}개, orthogonal: {identity['orthogonal_feature_count']}개"]
    if packaging_exception: lines += ["","## 입력 패키징 예외","","- canonical CORE ZIP이 없어 내장 manifest와 크기·SHA-256이 일치하는 추출 트리를 사용했다.","- 모델 입력은 검증됐지만 전체 전달 상태는 최대 PARTIAL_PASS로 제한한다."]
    lines += ["","## 해석 제한","","- donor는 구조 보조 검증에만 사용했고 포항 관측치나 계수를 복사하지 않았다.","- OOF expected spend는 인과효과·ROI·쿠폰 정책 순위가 아니다.","- 모델 게이트가 PASS가 아니면 OOF의 `policy_use_allowed`는 0이다."]
    atomic_write_text(dest/"STAGE1_V22_REVIEW_KO.md","\n".join(lines)+"\n")
    print((dest/"STAGE1_V22_REVIEW_KO.md").read_text(encoding="utf-8"))


def gpu_check_cmd(args):
    from catboost import CatBoostRegressor
    import numpy as np
    X=np.random.default_rng(42).normal(size=(8000,24)); y=X[:,0]*2+np.random.default_rng(1).normal(size=8000)
    m=CatBoostRegressor(iterations=100,depth=7,task_type="GPU",devices="0",verbose=False,allow_writing_files=False);m.fit(X,y); print("GPU PASS")


def build_parser():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/default.yaml"); p.add_argument("--run-name",default="stage1_multiwindow_ortho_v22")
    s=p.add_subparsers(dest="command",required=True)
    sp=s.add_parser("prepare"); sp.add_argument("--mode",choices=["quick","full"],default="full"); sp.set_defaults(func=prepare_cmd)
    for n,f in [("gpu-check",gpu_check_cmd),("finalize",finalize_full_cmd)]: sp=s.add_parser(n); sp.set_defaults(func=f)
    for n,f in [("baseline",baseline_cmd),("audit",audit_cmd)]:
        sp=s.add_parser(n);sp.add_argument("--backend",choices=["CPU","GPU"],required=True);sp.add_argument("--mode",choices=["quick","full"],default="full");sp.set_defaults(func=f)
    return p


def main():
    a=build_parser().parse_args(); a.func(a)

if __name__=="__main__": main()
