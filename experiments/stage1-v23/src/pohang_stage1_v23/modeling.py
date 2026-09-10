from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from pohang_main_v2.residual_modeling import ResidualModelSpec
from pohang_main_v2.splits import explicit_rolling_folds
from pohang_stage1_v22.dynamic_baseline import Stage1Candidate, default_candidates
from pohang_stage1_v22.modeling import fit_dynamic_residual_rolling


def _candidate_map(selections: pd.DataFrame, candidates: list[Stage1Candidate]) -> dict[int, Stage1Candidate]:
    cmap = {c.name: c for c in candidates}
    out = {}
    for row in selections.itertuples(index=False):
        name = str(getattr(row, "effective_candidate"))
        if name not in cmap:
            raise RuntimeError(f"UNKNOWN_STAGE1_CANDIDATE:{name}")
        out[int(row.fold)] = cmap[name]
    return out


def run_backend_baselines_v23(
    spec: ResidualModelSpec,
    cfg: dict[str, Any],
    selections: pd.DataFrame,
    backend: str,
    quick: bool,
    threads: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = default_candidates()
    cmap = _candidate_map(selections, candidates)
    sel_status = {int(r.fold): str(r.selection_status) for r in selections.itertuples(index=False)}
    sel_allowed = {int(r.fold): int(r.canonical_allowed) for r in selections.itertuples(index=False)}
    folds = explicit_rolling_folds(spec.frame, spec.period_col, cfg["validation"]["rolling_test_blocks"], int(cfg["validation"]["min_train_months"]))
    seeds = list(cfg["validation"]["seeds_gpu" if backend.upper() == "GPU" else "seeds_cpu"])
    if quick:
        folds = folds[-1:]
        seeds = [seeds[0]]
    jobs = [(fold, int(seed)) for fold in folds for seed in seeds]
    metrics, preds = [], []

    def one(fold, seed):
        m, p = fit_dynamic_residual_rolling(spec, fold, cfg, cmap[int(fold.fold)], seed, backend, quick, threads, return_predictions=True)
        m["selection_status"] = sel_status[int(fold.fold)]
        m["canonical_allowed"] = sel_allowed[int(fold.fold)]
        m["diagnostic_only"] = int(not bool(sel_allowed[int(fold.fold)]))
        if p is not None:
            p["selection_status"] = sel_status[int(fold.fold)]
            p["canonical_allowed"] = sel_allowed[int(fold.fold)]
        return m, p

    if backend.upper() == "CPU" and workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            futures = [ex.submit(one, f, s) for f, s in jobs]
            for fut in as_completed(futures):
                m, p = fut.result(); metrics.append(m); preds.append(p)
    else:
        for f, s in jobs:
            m, p = one(f, s); metrics.append(m); preds.append(p)
    return pd.DataFrame(metrics).sort_values(["fold", "seed"]).reset_index(drop=True), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
