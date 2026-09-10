from __future__ import annotations

from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from pohang_main_v2.splits import explicit_rolling_folds
from pohang_main_v2.residual_modeling import ResidualModelSpec

from .dynamic_baseline import Stage1Candidate, default_candidates
from .modeling import fit_dynamic_residual_fixed


def _fold_map(spec: ResidualModelSpec, cfg: dict[str, Any]):
    return {f.fold: f for f in explicit_rolling_folds(spec.frame, spec.period_col, cfg["validation"]["rolling_test_blocks"], int(cfg["validation"]["min_train_months"]))}


def run_category_reentry_and_proxy(spec: ResidualModelSpec, cfg: dict[str, Any], baseline: pd.DataFrame, selections: pd.DataFrame, backend: str, threads: int, workers: int = 1, candidates: list[Stage1Candidate] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = list(candidates or default_candidates())
    cand = {int(r.fold): next(c for c in candidates if c.name == str(r.winner)) for r in selections.itertuples(index=False)}
    folds = _fold_map(spec, cfg)
    strict = baseline[(baseline.status.eq("PASS")) & baseline.backend.astype(str).str.upper().eq(backend.upper())].copy()
    gated_groups = {"age_industry_interaction", "weather_industry_interaction"}
    gated = sorted({f for g, fs in spec.groups.items() if g in gated_groups for f in fs if f in spec.features})
    no_gated = [f for f in spec.features if f not in set(gated)]
    profiles = {
        "strict_residual": list(spec.features),
        "add_category": list(spec.features) + ["category"],
        "add_category_major": list(spec.features) + ["category_major"],
        "no_category_gated_interactions": no_gated,
        "no_interactions_plus_category_major": no_gated + ["category_major"],
    }
    jobs=[]
    for base in strict.itertuples(index=False):
        fold=folds[int(base.fold)]; candidate=cand[int(base.fold)]
        for name, features in profiles.items():
            jobs.append((base,fold,candidate,name,features))

    def one(job):
        base,fold,candidate,name,features=job
        if name=="strict_residual":
            metric={"r2":float(base.r2),"rmse":float(base.rmse),"residual_r2":float(base.residual_r2)}
        else:
            cats=[c for c in spec.categorical if c in features]
            for key in ["category","category_major"]:
                if key in features and key not in cats: cats.append(key)
            metric=fit_dynamic_residual_fixed(spec, fold, cfg, candidate, int(base.seed), backend, threads, iterations=int(base.best_iterations), features=features, categorical=cats)
        return {"backend":backend.upper(),"seed":int(base.seed),"fold":int(base.fold),"profile":name,"r2":float(metric["r2"]),"rmse":float(metric["rmse"]),"residual_r2":float(metric["residual_r2"]),"stage1_candidate":candidate.name}

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows=[future.result() for future in as_completed(executor.submit(one, job) for job in jobs)]
    else:
        rows=[one(job) for job in jobs]
    raw=pd.DataFrame(rows).sort_values(["backend","seed","fold","profile"]).reset_index(drop=True)
    ref=raw[raw.profile.eq("strict_residual")][["backend","seed","fold","r2"]].rename(columns={"r2":"strict_r2"})
    raw=raw.merge(ref,on=["backend","seed","fold"],how="left",validate="many_to_one")
    raw["r2_gain_vs_strict"]=raw.r2-raw.strict_r2
    pivot=raw.pivot_table(index=["backend","seed","fold"],columns="profile",values="r2",aggfunc="first").reset_index()
    pivot["interaction_value_gain"]=pivot["strict_residual"]-pivot["no_category_gated_interactions"]
    pivot["identity_only_gain"]=pivot["no_interactions_plus_category_major"]-pivot["no_category_gated_interactions"]
    pivot["interaction_excess_over_identity"]=pivot["strict_residual"]-pivot["no_interactions_plus_category_major"]
    denom=pivot["interaction_value_gain"].abs()
    pivot["identity_recovery_share"]=np.where(denom>1e-12,pivot["identity_only_gain"]/denom,np.nan)
    return raw,pivot


def run_age_industry_ablation(spec: ResidualModelSpec, cfg: dict[str, Any], baseline: pd.DataFrame, selections: pd.DataFrame, backend: str, threads: int, workers: int = 1, candidates: list[Stage1Candidate] | None = None) -> pd.DataFrame:
    candidates=list(candidates or default_candidates())
    cand={int(r.fold):next(c for c in candidates if c.name==str(r.winner)) for r in selections.itertuples(index=False)}
    folds=_fold_map(spec,cfg)
    omitted=list(spec.groups.get("age_industry_interaction",[]))
    remaining=[f for f in spec.features if f not in set(omitted)]
    cats=[c for c in spec.categorical if c in remaining]
    strict=baseline[(baseline.status.eq("PASS")) & baseline.backend.astype(str).str.upper().eq(backend.upper())]
    def one(base):
        metric=fit_dynamic_residual_fixed(spec,folds[int(base.fold)],cfg,cand[int(base.fold)],int(base.seed),backend,threads,iterations=int(base.best_iterations),features=remaining,categorical=cats)
        return {"backend":backend.upper(),"seed":int(base.seed),"fold":int(base.fold),"baseline_r2":float(base.r2),"ablated_r2":float(metric["r2"]),"r2_drop":float(base.r2)-float(metric["r2"]),"baseline_residual_r2":float(base.residual_r2),"ablated_residual_r2":float(metric["residual_r2"]),"positive":int(float(base.r2)-float(metric["r2"])>0),"stage1_candidate":cand[int(base.fold)].name}
    bases=list(strict.itertuples(index=False))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows=[future.result() for future in as_completed(executor.submit(one, base) for base in bases)]
    else:
        rows=[one(base) for base in bases]
    return pd.DataFrame(rows).sort_values(["backend","seed","fold"]).reset_index(drop=True)
