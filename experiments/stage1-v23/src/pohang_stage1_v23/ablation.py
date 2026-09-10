from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from pohang_main_v2.residual_modeling import ResidualModelSpec
from pohang_main_v2.splits import explicit_rolling_folds, split_fold
from pohang_main_v2.utils import atomic_write_json, ensure_dir, stable_hash
from pohang_stage1_v22.dynamic_baseline import default_candidates
from pohang_stage1_v22.modeling import canonicalize_spec, fit_dynamic_residual_fixed


def group_units(spec_original: ResidualModelSpec, cfg: dict[str, Any]) -> dict[str, list[str]]:
    spec = canonicalize_spec(spec_original, cfg)
    out = {}
    for name, cols in spec.groups.items():
        active = sorted({c for c in cols if c in spec.features})
        if active:
            out[str(name)] = active
    return out


def individual_units(spec_original: ResidualModelSpec, cfg: dict[str, Any]) -> dict[str, list[str]]:
    spec = canonicalize_spec(spec_original, cfg)
    return {f: [f] for f in sorted(spec.features)}


def _candidate_map(selections: pd.DataFrame):
    cmap = {c.name: c for c in default_candidates()}
    out = {}
    for r in selections.itertuples(index=False):
        n = str(r.effective_candidate)
        if n not in cmap:
            raise RuntimeError(f"UNKNOWN_STAGE1_CANDIDATE:{n}")
        out[int(r.fold)] = cmap[n]
    return out


def _baseline_lookup(baseline: pd.DataFrame, spec: ResidualModelSpec, backend: str) -> dict[tuple, Any]:
    need = {"model","profile","backend","seed","fold","status","data_identity","stage1_candidate","raw_train_rows","train_rows","test_rows","test_start","test_end","best_iterations"}
    missing = need.difference(baseline.columns)
    if missing:
        raise RuntimeError(f"V23_BASELINE_SCHEMA_MISSING:{sorted(missing)}")
    x = baseline[
        baseline.model.eq(spec.name)
        & baseline.profile.eq(spec.profile)
        & baseline.backend.astype(str).str.upper().eq(backend.upper())
        & baseline.status.eq("PASS")
    ].copy()
    keys = ["model","profile","backend","seed","fold"]
    if x.duplicated(keys).any():
        raise RuntimeError("V23_BASELINE_DUPLICATE_IDENTITY")
    return {(str(r.model),str(r.profile),str(r.backend).upper(),int(r.seed),int(r.fold)):r for r in x.itertuples(index=False)}


def run_stage2_ablation(
    spec_original: ResidualModelSpec,
    cfg: dict[str, Any],
    selections: pd.DataFrame,
    baseline: pd.DataFrame,
    backend: str,
    units: dict[str, list[str]],
    *,
    kind: str,
    quick: bool,
    workers: int,
    threads: int,
    cache_dir: Path,
) -> pd.DataFrame:
    spec = canonicalize_spec(spec_original, cfg)
    cmap = _candidate_map(selections)
    lookup = _baseline_lookup(baseline, spec, backend)
    folds = explicit_rolling_folds(spec.frame, spec.period_col, cfg["validation"]["rolling_test_blocks"], int(cfg["validation"]["min_train_months"]))
    seeds = list(cfg["validation"]["seeds_gpu" if backend.upper()=="GPU" else "seeds_cpu"])
    if quick:
        folds = folds[-1:]; seeds = [seeds[0]]
    cache_dir = ensure_dir(cache_dir)
    jobs=[]
    for unit, omitted in units.items():
        remaining=[f for f in spec.features if f not in set(omitted)]
        if not remaining: continue
        cats=[c for c in spec.categorical if c in remaining]
        for fold in folds:
            for seed in seeds:
                key=(spec.name,spec.profile,backend.upper(),int(seed),int(fold.fold))
                if key not in lookup:
                    raise RuntimeError(f"V23_MISSING_BASELINE:{key}")
                jobs.append((unit,omitted,remaining,cats,fold,int(seed),lookup[key],cmap[int(fold.fold)]))

    def one(job):
        unit, omitted, remaining, cats, fold, seed, base, candidate = job
        identity={
            "version":"V23", "model":spec.name,"profile":spec.profile,"backend":backend.upper(),"seed":seed,"fold":int(fold.fold),
            "kind":kind,"unit":unit,"omitted":omitted,"remaining":remaining,"iterations":int(base.best_iterations),
            "stage1_candidate":candidate.name,"data_identity":spec.data_identity,"test_start":int(fold.test_start),"test_end":int(fold.test_end),
        }
        task=stable_hash(identity); path=cache_dir/f"{task}.json"
        if path.exists():
            row=json.loads(path.read_text(encoding="utf-8")); row["cache_hit"]=True; return row
        try:
            _,_,full_train,test,_=split_fold(spec.frame,spec.period_col,fold,int(cfg["validation"]["inner_valid_months"]))
            if len(full_train)!=int(base.raw_train_rows) or len(test)!=int(base.test_rows):
                raise RuntimeError("V23_ABLATION_RAW_ROW_INVARIANT_FAIL")
            if int(base.test_start)!=int(fold.test_start) or int(base.test_end)!=int(fold.test_end):
                raise RuntimeError("V23_ABLATION_PERIOD_INVARIANT_FAIL")
            if str(base.stage1_candidate)!=candidate.name:
                raise RuntimeError("V23_ABLATION_STAGE1_CANDIDATE_INVARIANT_FAIL")
            metric=fit_dynamic_residual_fixed(spec,fold,cfg,candidate,seed,backend,threads,iterations=int(base.best_iterations),features=remaining,categorical=cats)
            if int(metric["raw_train_rows"])!=int(base.raw_train_rows) or int(metric["test_rows"])!=int(base.test_rows):
                raise RuntimeError("V23_ABLATION_REGENERATED_ROW_INVARIANT_FAIL")
            r2_drop=float(base.r2)-float(metric["r2"])
            rmse_inc=float(metric["rmse"])-float(base.rmse)
            if (r2_drop < -1e-10 and rmse_inc > 1e-10) or (r2_drop > 1e-10 and rmse_inc < -1e-10):
                raise RuntimeError("V23_ABLATION_METRIC_COHERENCE_FAIL")
            row={
                "model":spec.name,"profile":spec.profile,"backend":backend.upper(),"seed":seed,"fold":int(fold.fold),"kind":kind,"unit":unit,
                "omitted_columns":"|".join(omitted),"omitted_count":len(omitted),"status":"PASS","stage1_candidate":candidate.name,
                "selection_status":str(base.selection_status) if hasattr(base,"selection_status") else "UNKNOWN",
                "canonical_allowed":int(getattr(base,"canonical_allowed",0)),
                "baseline_r2":float(base.r2),"ablated_r2":float(metric["r2"]),"r2_drop":r2_drop,
                "baseline_rmse":float(base.rmse),"ablated_rmse":float(metric["rmse"]),"rmse_increase":rmse_inc,
                "baseline_spearman":float(base.spearman),"ablated_spearman":float(metric["spearman"]),"spearman_drop":float(base.spearman)-float(metric["spearman"]),
                "baseline_residual_r2":float(base.residual_r2),"ablated_residual_r2":float(metric["residual_r2"]),"residual_r2_drop":float(base.residual_r2)-float(metric["residual_r2"]),
                "baseline_stage2_gain":float(base.delta_r2_vs_stage1),"ablated_stage2_gain":float(metric["delta_r2_vs_stage1"]),"stage2_gain_drop":float(base.delta_r2_vs_stage1)-float(metric["delta_r2_vs_stage1"]),
                "baseline_train_rows":int(base.train_rows),"regenerated_train_rows":int(metric["train_rows"]),"raw_train_rows":int(base.raw_train_rows),"test_rows":int(base.test_rows),
                "best_iterations":int(base.best_iterations),"data_identity":str(base.data_identity),"baseline_invariant_pass":1,"metric_coherence_pass":1,
                "task_hash":task,"cache_hit":False,
            }
        except Exception as exc:
            row={"backend":backend.upper(),"seed":seed,"fold":int(fold.fold),"kind":kind,"unit":unit,"omitted_columns":"|".join(omitted),"status":"FAILED","error":f"{type(exc).__name__}: {exc}","task_hash":task,"cache_hit":False}
        atomic_write_json(path,row); return row

    print(f"[V23 ablation {backend.upper()}] kind={kind} units={len(units)} jobs={len(jobs)} workers={workers}",flush=True)
    rows=[]
    if backend.upper()=="GPU" or workers<=1:
        for i,j in enumerate(jobs,1):
            rows.append(one(j))
            if i%max(1,len(jobs)//20)==0 or i==len(jobs): print(f"[{backend} {kind}] {i}/{len(jobs)}",flush=True)
    else:
        with ThreadPoolExecutor(max_workers=min(workers,len(jobs))) as ex:
            futs=[ex.submit(one,j) for j in jobs]
            for i,f in enumerate(as_completed(futs),1):
                rows.append(f.result())
                if i%max(1,len(jobs)//20)==0 or i==len(jobs): print(f"[{backend} {kind}] {i}/{len(jobs)}",flush=True)
    df=pd.DataFrame(rows)
    if "error" in df.columns:
        fatal=df[df.error.astype(str).str.contains("V23_ABLATION_.*_FAIL",regex=True,na=False)]
        if not fatal.empty:
            raise RuntimeError("V23 ablation invariant failed: "+"; ".join(fatal.error.head(5)))
    return df


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    x=raw[raw.status.eq("PASS")].copy()
    if x.empty: return pd.DataFrame()
    return x.groupby(["backend","kind","unit","omitted_columns"],as_index=False).agg(
        median_r2_drop=("r2_drop","median"),mean_r2_drop=("r2_drop","mean"),min_r2_drop=("r2_drop","min"),max_r2_drop=("r2_drop","max"),
        positive_run_share=("r2_drop",lambda s:float((s>0).mean())),median_rmse_increase=("rmse_increase","median"),median_spearman_drop=("spearman_drop","median"),
        median_residual_r2_drop=("residual_r2_drop","median"),median_stage2_gain_drop=("stage2_gain_drop","median"),folds=("fold","nunique"),seeds=("seed","nunique"),runs=("r2_drop","size"),
        invariant_share=("baseline_invariant_pass","mean"),coherence_share=("metric_coherence_pass","mean"),
    )


def fold_diagnostics(raw: pd.DataFrame, harmful_threshold: float = 0.0005) -> pd.DataFrame:
    x=raw[raw.status.eq("PASS")].copy()
    if x.empty: return pd.DataFrame()
    per=x.groupby(["kind","unit","omitted_columns","backend","fold"],as_index=False).r2_drop.median()
    rows=[]
    for key,g in per.groupby(["kind","unit","omitted_columns"],sort=True):
        rec={"kind":key[0],"unit":key[1],"omitted_columns":key[2]}
        for fold in sorted(per.fold.unique()):
            part=g[g.fold.eq(fold)]
            vals={str(r.backend).upper():float(r.r2_drop) for r in part.itertuples(index=False)}
            rec[f"fold{fold}_cpu_drop"]=vals.get("CPU",np.nan)
            rec[f"fold{fold}_gpu_drop"]=vals.get("GPU",np.nan)
            finite=[v for v in vals.values() if np.isfinite(v)]
            rec[f"fold{fold}_conservative_drop"]=min(finite) if finite else np.nan
        f2=[rec.get("fold2_cpu_drop",np.nan),rec.get("fold2_gpu_drop",np.nan)]
        f2=[v for v in f2 if np.isfinite(v)]
        others=[rec.get(f"fold{k}_conservative_drop",np.nan) for k in [1,3,4]]
        others=[v for v in others if np.isfinite(v)]
        rec["fold2_harmful_consensus"]=int(len(f2)>=2 and max(f2) < -harmful_threshold)
        rec["fold2_harmful_any"]=int(bool(f2) and min(f2) < -harmful_threshold)
        rec["other_folds_median_drop"]=float(np.median(others)) if others else np.nan
        if rec["fold2_harmful_consensus"] and rec["other_folds_median_drop"] > harmful_threshold:
            rec["diagnostic_class"]="REGIME_SPECIFIC_CONFLICT"
        elif rec["fold2_harmful_consensus"]:
            rec["diagnostic_class"]="FOLD2_HARMFUL_CONSENSUS"
        elif rec["fold2_harmful_any"]:
            rec["diagnostic_class"]="FOLD2_HARMFUL_ONE_BACKEND"
        else:
            all_drops=[rec.get(f"fold{k}_conservative_drop",np.nan) for k in [1,2,3,4]]
            all_drops=[v for v in all_drops if np.isfinite(v)]
            rec["diagnostic_class"]="GLOBAL_HARMFUL" if all_drops and np.median(all_drops)<-harmful_threshold else "NO_HARM_SIGNAL"
        rows.append(rec)
    return pd.DataFrame(rows)
