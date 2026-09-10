from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any
from pathlib import Path
import json

import numpy as np
import pandas as pd

from pohang_main_v2.residual_modeling import ResidualModelSpec
from pohang_main_v2.utils import stable_hash, atomic_write_json, ensure_dir
from pohang_main_v2.splits import explicit_rolling_folds
from pohang_stage1_v22.dynamic_baseline import Stage1Candidate, default_candidates, fit_dynamic_stage1, predict_dynamic_stage1
from pohang_stage1_v22.inner_windows import build_inner_windows, split_inner_window, audit_inner_windows
from pohang_stage1_v22.modeling import canonicalize_spec, _screen_candidate_on_window, _stage1_metrics


def _adaptive_window_months(full_train: pd.DataFrame, period_col: str, cfg: dict[str, Any], quick: bool) -> int:
    scfg = cfg.get("stage1_v23_selection", {})
    if quick:
        return int(scfg.get("quick_inner_valid_months", 3))
    desired = int(scfg.get("inner_valid_months", 6))
    count = int(scfg.get("inner_window_count", 3))
    min_train = int(scfg.get("inner_min_train_months", 12))
    periods = sorted(pd.to_numeric(full_train[period_col], errors="coerce").dropna().astype(int).unique())
    if len(periods) >= min_train + count * desired:
        return desired
    fallback = int(scfg.get("inner_valid_months_fallback", 4))
    if len(periods) >= min_train + count * fallback:
        return fallback
    return max(2, int((len(periods) - min_train) // max(count, 1)))


def _aggregate(rows: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    scfg = cfg.get("stage1_v23_selection", {})
    min_gain = float(scfg.get("feasible_median_stage2_gain_min", 0.003))
    min_worst_gain = float(scfg.get("feasible_worst_stage2_gain_min", 0.0))
    min_pos = float(scfg.get("feasible_stage2_positive_share_min", 1.0))
    max_cat_med = float(scfg.get("feasible_category_reentry_median_max", 0.005))
    max_cat_worst = float(scfg.get("feasible_category_reentry_worst_max", 0.010))

    out = []
    for name, p in rows.groupby("name", sort=True):
        stage2 = pd.to_numeric(p.stage2_gain, errors="coerce").to_numpy(float)
        cat = pd.to_numeric(p.category_reentry_gain, errors="coerce").to_numpy(float)
        strict = pd.to_numeric(p.strict_r2, errors="coerce").to_numpy(float)
        age = pd.to_numeric(p.age_industry_drop, errors="coerce").to_numpy(float)
        proxy = pd.to_numeric(p.interaction_excess_over_identity, errors="coerce").to_numpy(float)
        rec = {
            "name": str(name),
            "window_count": int(len(p)),
            "median_strict_r2": float(np.nanmedian(strict)),
            "worst_strict_r2": float(np.nanmin(strict)),
            "median_stage2_gain": float(np.nanmedian(stage2)),
            "worst_stage2_gain": float(np.nanmin(stage2)),
            "stage2_positive_share": float(np.mean(stage2 > 0)),
            "stage2_std": float(np.nanstd(stage2)),
            "median_category_reentry_gain": float(np.nanmedian(cat)),
            "worst_category_reentry_gain": float(np.nanmax(cat)),
            "median_age_industry_drop": float(np.nanmedian(age)),
            "median_proxy_excess": float(np.nanmedian(proxy)),
            "worst_proxy_excess": float(np.nanmin(proxy)),
        }
        constraints = {
            "C1_all_windows_stage2_positive": rec["stage2_positive_share"] >= min_pos,
            "C2_median_stage2_gain": rec["median_stage2_gain"] >= min_gain,
            "C3_worst_stage2_gain": rec["worst_stage2_gain"] > min_worst_gain,
            "C4_category_reentry_median": rec["median_category_reentry_gain"] <= max_cat_med,
            "C5_category_reentry_worst": rec["worst_category_reentry_gain"] <= max_cat_worst,
        }
        rec.update(constraints)
        rec["feasible"] = int(all(constraints.values()))
        rec["constraint_fail_count"] = int(sum(not v for v in constraints.values()))
        # Diagnostic ranking only. Never converts an infeasible row into a canonical winner.
        rec["diagnostic_score"] = float(
            -rec["median_strict_r2"]
            + 12.0 * max(0.0, min_gain - rec["median_stage2_gain"])
            + 16.0 * max(0.0, min_worst_gain - rec["worst_stage2_gain"])
            + 8.0 * max(0.0, rec["median_category_reentry_gain"] - max_cat_med)
            + 6.0 * max(0.0, rec["worst_category_reentry_gain"] - max_cat_worst)
            + 2.0 * rec["stage2_std"]
        )
        out.append(rec)
    return pd.DataFrame(out).sort_values(
        ["feasible", "constraint_fail_count", "diagnostic_score", "median_strict_r2"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def build_feasibility_first_selection(
    spec_original: ResidualModelSpec,
    cfg: dict[str, Any],
    candidates: list[Stage1Candidate] | None = None,
    *,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = list(candidates or default_candidates())
    spec = canonicalize_spec(spec_original, cfg)
    folds = explicit_rolling_folds(
        spec.frame, spec.period_col, cfg["validation"]["rolling_test_blocks"], int(cfg["validation"]["min_train_months"])
    )
    if quick:
        folds = folds[-1:]
    scfg = cfg.get("stage1_v23_selection", {})
    window_count = int(scfg.get("quick_inner_window_count" if quick else "inner_window_count", 2 if quick else 3))
    min_inner_train = int(scfg.get("inner_min_train_months", 12))
    workers = int(scfg.get("quick_screen_workers" if quick else "screen_workers", 2 if quick else 6))

    selections: list[dict[str, Any]] = []
    stage1_rows: list[dict[str, Any]] = []
    screen_rows_all: list[pd.DataFrame] = []
    agg_rows_all: list[pd.DataFrame] = []
    window_audits: list[pd.DataFrame] = []

    for fold in folds:
        periods = pd.to_numeric(spec.frame[spec.period_col], errors="coerce")
        full_train = spec.frame[periods.lt(int(fold.test_start))].copy()
        outer_test = spec.frame[periods.ge(int(fold.test_start)) & periods.le(int(fold.test_end))].copy()
        valid_months = _adaptive_window_months(full_train, spec.period_col, cfg, quick)
        windows = build_inner_windows(
            full_train,
            spec.period_col,
            count=window_count,
            valid_months=valid_months,
            step_months=valid_months,
            min_train_months=min_inner_train,
        )
        if len(windows) < min(window_count, 2):
            raise RuntimeError(f"INSUFFICIENT_FEASIBILITY_WINDOWS:fold={fold.fold}:got={len(windows)}")
        wa = audit_inner_windows(windows, int(fold.test_start))
        wa.insert(0, "fold", int(fold.fold))
        wa["requested_window_count"] = window_count
        wa["effective_valid_months"] = valid_months
        window_audits.append(wa)
        if not bool(wa.valid_before_outer_test.eq(1).all() and wa.train_before_valid.eq(1).all()):
            raise RuntimeError("FEASIBILITY_WINDOW_CHRONOLOGY_VIOLATION")

        # Stage-1-only audit for every candidate/window.
        for w in windows:
            train, valid = split_inner_window(full_train, spec.period_col, w)
            for candidate in candidates:
                state = fit_dynamic_stage1(train, candidate, target_col=spec.target_col)
                pred_frame = predict_dynamic_stage1(state, valid)
                pred = pd.to_numeric(pred_frame["stage1_baseline_log"], errors="coerce").to_numpy(float)
                m = _stage1_metrics(valid, pred, spec.target_col)
                stage1_rows.append({
                    "fold": int(fold.fold), "name": candidate.name, **w.to_dict(), **m,
                    "future_target_violations": int(pred_frame["stage1_future_target_violation"].sum()),
                })

        if stage1_rows and sum(int(r.get("future_target_violations", 0)) for r in stage1_rows if int(r.get("fold", -1)) == int(fold.fold)) != 0:
            raise RuntimeError("V23_STAGE1_FUTURE_TARGET_LEAKAGE")

        # Feasibility screen: every one of the 15 candidates gets Stage-2/category/age/proxy evaluation.
        jobs = []
        for candidate in candidates:
            for w in windows:
                train, valid = split_inner_window(full_train, spec.period_col, w)
                jobs.append((candidate, w, train, valid))

        screen_cache = ensure_dir(Path(cfg.get("paths", {}).get("cache_dir", ".cache/tasks")) / "stage1_v23" / "selection")

        def run_job(job):
            candidate, w, tr, va = job
            identity = {
                "version": "V23", "kind": "stage1_candidate_window_screen", "fold": int(fold.fold),
                "candidate": candidate.to_dict(), "window": w.to_dict(), "quick": bool(quick),
                "data_identity": spec.data_identity, "catboost": cfg.get("catboost", {}),
                "selection_cfg": cfg.get("stage1_v23_selection", {}),
                "screen_cfg": cfg.get("stage1_v22_selection", {}),
            }
            task = stable_hash(identity)
            path = screen_cache / f"{task}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8")); row["cache_hit"] = True; return row
            row = {"fold": int(fold.fold), "name": candidate.name, **_screen_candidate_on_window(spec, cfg, tr, va, candidate, window=w, quick=quick), "cache_hit": False, "task_hash": task}
            atomic_write_json(path, row)
            return row

        local: list[dict[str, Any]] = []
        if workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                futures = [ex.submit(run_job, j) for j in jobs]
                for fut in as_completed(futures):
                    local.append(fut.result())
        else:
            local = [run_job(j) for j in jobs]
        screen_df = pd.DataFrame(local)
        screen_rows_all.append(screen_df)
        agg = _aggregate(screen_df, cfg)
        agg.insert(0, "fold", int(fold.fold))
        agg_rows_all.append(agg)

        feasible = agg[agg.feasible.eq(1)].copy()
        if not feasible.empty:
            # Feasibility first. Among feasible candidates maximize worst/median stage2, then strict R2.
            feasible = feasible.sort_values(
                ["worst_stage2_gain", "median_stage2_gain", "median_strict_r2", "worst_category_reentry_gain"],
                ascending=[False, False, False, True],
            )
            chosen = feasible.iloc[0]
            winner = str(chosen["name"])
            status = "FEASIBLE_SELECTED"
            diagnostic = winner
        else:
            chosen = agg.iloc[0]
            winner = ""
            status = "NO_FEASIBLE_STAGE1"
            diagnostic = str(chosen["name"])

        cand = next(c for c in candidates if c.name == diagnostic)
        selections.append({
            "fold": int(fold.fold), "test_start": int(fold.test_start), "test_end": int(fold.test_end),
            "selection_status": status,
            "winner": winner,
            "diagnostic_candidate": diagnostic,
            "effective_candidate": diagnostic,
            "effective_candidate_json": str(asdict(cand)),
            "feasible_candidate_count": int(len(feasible)),
            "candidate_count_evaluated": int(len(candidates)),
            "inner_window_count": int(chosen.window_count),
            "inner_valid_months": int(valid_months),
            "inner_median_stage2_gain": float(chosen.median_stage2_gain),
            "inner_worst_stage2_gain": float(chosen.worst_stage2_gain),
            "inner_stage2_positive_share": float(chosen.stage2_positive_share),
            "inner_median_category_reentry_gain": float(chosen.median_category_reentry_gain),
            "inner_worst_category_reentry_gain": float(chosen.worst_category_reentry_gain),
            "constraint_fail_count": int(chosen.constraint_fail_count),
            "diagnostic_score": float(chosen.diagnostic_score),
            "full_train_rows": int(len(full_train)), "outer_test_rows": int(len(outer_test)),
            "outer_test_used_for_selection": 0,
            "canonical_allowed": int(status == "FEASIBLE_SELECTED"),
        })

    return (
        pd.DataFrame(selections),
        pd.DataFrame(stage1_rows),
        pd.concat(screen_rows_all, ignore_index=True),
        pd.concat(agg_rows_all, ignore_index=True),
        pd.concat(window_audits, ignore_index=True),
    )
