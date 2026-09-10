from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score

from pohang_main_v2.modeling import _prepare_catboost, catboost_params, regression_metrics
from pohang_main_v2.residual_modeling import ResidualModelSpec, _category_bias_mae, _calibration_stats
from pohang_main_v2.splits import TimeFold, explicit_rolling_folds, split_fold

from .dynamic_baseline import (
    Stage1Candidate,
    default_candidates,
    fit_dynamic_stage1,
    past_only_dynamic_stage1,
    predict_dynamic_stage1,
)


def _stage1_metrics(frame: pd.DataFrame, pred: np.ndarray, target_col: str) -> dict[str, float]:
    y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return {"r2": np.nan, "rmse": np.inf, "category_bias_mae": np.inf, "mean_error": np.inf, "score": np.inf}
    yy, pp = y[mask], pred[mask]
    rmse = float(np.sqrt(mean_squared_error(yy, pp)))
    r2 = float(r2_score(yy, pp))
    cb = float(_category_bias_mae(frame.iloc[np.flatnonzero(mask)], yy, pp))
    me = float(abs(np.mean(yy - pp)))
    score = rmse + 0.35 * cb + 0.10 * me
    return {"r2": r2, "rmse": rmse, "category_bias_mae": cb, "mean_error": me, "score": score}


def select_stage1_candidate(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    candidates: list[Stage1Candidate],
    *,
    target_col: str,
) -> tuple[Stage1Candidate, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        state = fit_dynamic_stage1(fit, candidate, target_col=target_col)
        p = predict_dynamic_stage1(state, valid)
        pred = pd.to_numeric(p["stage1_baseline_log"], errors="coerce").to_numpy(float)
        m = _stage1_metrics(valid, pred, target_col)
        rows.append({**candidate.to_dict(), **m, "future_target_violations": int(p["stage1_future_target_violation"].sum())})
    audit = pd.DataFrame(rows).sort_values(["score", "rmse", "name"]).reset_index(drop=True)
    if audit.empty or not np.isfinite(audit.iloc[0]["score"]):
        raise RuntimeError("STAGE1_CANDIDATE_SELECTION_FAILED")
    if int(audit["future_target_violations"].sum()) != 0:
        raise RuntimeError("STAGE1_INNER_VALID_FUTURE_TARGET_LEAKAGE")
    name = str(audit.iloc[0]["name"])
    return next(c for c in candidates if c.name == name), audit



def _inner_candidate_residual_screen(
    spec: ResidualModelSpec,
    cfg: dict[str, Any],
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    candidate: Stage1Candidate,
) -> dict[str, float]:
    """Gate-aware candidate screen using inner validation only.

    It checks three desired properties before the untouched outer test exists:
    Stage-2 incremental gain, raw-category re-entry, and Age×Industry group value.
    """
    min_history = int(cfg["category_baseline"].get("minimum_category_history", 12))
    fit_resid, _ = _prepare_residual_training(fit, candidate, spec.target_col, min_history)
    state = fit_dynamic_stage1(fit, candidate, target_col=spec.target_col)
    s1f = predict_dynamic_stage1(state, valid)
    if int(s1f["stage1_future_target_violation"].sum()) != 0:
        raise RuntimeError("STAGE1_SELECTION_VALID_LEAKAGE")
    s1 = s1f["stage1_baseline_log"].to_numpy(float)
    y = pd.to_numeric(valid[spec.target_col], errors="coerce").to_numpy(float)
    yres = pd.to_numeric(fit_resid["__residual"], errors="coerce").to_numpy(float)

    selection_cfg = cfg.get("stage1_v21_selection", {})
    iterations = int(selection_cfg.get("screen_iterations", 140))
    threads = int(selection_cfg.get("screen_threads_per_model", 4))

    def fit_profile(features: list[str], categorical: list[str]) -> float:
        cats = [c for c in categorical if c in features]
        idx = [features.index(c) for c in cats]
        params = catboost_params(cfg, 4242, "CPU", True, threads)
        params["iterations"] = iterations
        model = CatBoostRegressor(**params)
        Xtr = _prepare_catboost(fit_resid, features, cats)
        Xv = _prepare_catboost(valid, features, cats)
        model.fit(Pool(Xtr, yres, cat_features=idx), verbose=False)
        rp = model.predict(Pool(Xv, cat_features=idx))
        return float(r2_score(y, s1 + rp))

    strict_features = list(spec.features)
    strict_cat = list(spec.categorical)
    strict_r2 = fit_profile(strict_features, strict_cat)
    stage1_r2 = float(r2_score(y, s1))
    addcat_features = strict_features + (["category"] if "category" not in strict_features else [])
    addcat_cat = strict_cat + (["category"] if "category" not in strict_cat else [])
    addcat_r2 = fit_profile(addcat_features, addcat_cat)
    age_cols = list(spec.groups.get("age_industry_interaction", []))
    noage_features = [f for f in strict_features if f not in set(age_cols)]
    noage_cat = [c for c in strict_cat if c in noage_features]
    noage_r2 = fit_profile(noage_features, noage_cat)
    stage2_gain = strict_r2 - stage1_r2
    category_gain = addcat_r2 - strict_r2
    age_drop = strict_r2 - noage_r2
    desired_gain = float(selection_cfg.get("desired_inner_stage2_gain", 0.010))
    desired_cat = float(selection_cfg.get("desired_inner_category_reentry_max", 0.005))
    desired_age = float(selection_cfg.get("desired_inner_age_industry_drop", 0.003))
    # Large penalties apply only when a requested structural condition is missed.
    penalty = (
        6.0 * max(0.0, desired_gain - stage2_gain)
        + 8.0 * max(0.0, category_gain - desired_cat)
        + 4.0 * max(0.0, desired_age - age_drop)
        - strict_r2
    )
    return {
        "inner_stage1_r2": stage1_r2,
        "inner_strict_r2": strict_r2,
        "inner_stage2_gain": stage2_gain,
        "inner_category_reentry_gain": category_gain,
        "inner_age_industry_drop": age_drop,
        "gate_aware_score": float(penalty),
    }

def build_stage1_selection(spec: ResidualModelSpec, cfg: dict[str, Any], candidates: list[Stage1Candidate] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = list(candidates or default_candidates())
    folds = explicit_rolling_folds(spec.frame, spec.period_col, cfg["validation"]["rolling_test_blocks"], int(cfg["validation"]["min_train_months"]))
    selections: list[dict] = []
    audits: list[pd.DataFrame] = []
    top_k = int(cfg.get("stage1_v21_selection", {}).get("top_k_stage1_candidates", 5))
    workers = int(cfg.get("stage1_v21_selection", {}).get("screen_workers", 4))
    for fold in folds:
        fit, valid, full_train, test, valid_periods = split_fold(spec.frame, spec.period_col, fold, int(cfg["validation"]["inner_valid_months"]))
        _, stage1_audit = select_stage1_candidate(fit, valid, candidates, target_col=spec.target_col)
        top_names = stage1_audit.head(max(1, top_k))["name"].tolist()
        shortlist = [c for c in candidates if c.name in top_names]
        screen_rows: list[dict[str, Any]] = []
        def screen(c: Stage1Candidate) -> dict[str, Any]:
            return {"name": c.name, **_inner_candidate_residual_screen(spec, cfg, fit, valid, c)}
        if workers > 1 and len(shortlist) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(shortlist))) as ex:
                futures = {ex.submit(screen, c): c for c in shortlist}
                for fut in as_completed(futures): screen_rows.append(fut.result())
        else:
            screen_rows = [screen(c) for c in shortlist]
        screen_df = pd.DataFrame(screen_rows).sort_values(["gate_aware_score", "inner_strict_r2"], ascending=[True, False])
        winner_name = str(screen_df.iloc[0]["name"])
        winner = next(c for c in candidates if c.name == winner_name)
        audit = stage1_audit.merge(screen_df, on="name", how="left")
        audit.insert(0, "fold", int(fold.fold))
        audit.insert(1, "valid_start", int(min(valid_periods)))
        audit.insert(2, "valid_end", int(max(valid_periods)))
        audit["shortlisted_for_gate_screen"] = audit["name"].isin(top_names).astype(int)
        audit["selected"] = audit["name"].eq(winner.name).astype(int)
        audits.append(audit)
        wr = screen_df[screen_df.name.eq(winner.name)].iloc[0]
        selections.append({
            "fold": int(fold.fold), "test_start": int(fold.test_start), "test_end": int(fold.test_end),
            "winner": winner.name, "winner_json": str(winner.to_dict()),
            "inner_stage2_gain": float(wr.inner_stage2_gain),
            "inner_category_reentry_gain": float(wr.inner_category_reentry_gain),
            "inner_age_industry_drop": float(wr.inner_age_industry_drop),
            "fit_rows": int(len(fit)), "valid_rows": int(len(valid)), "full_train_rows": int(len(full_train)), "test_rows": int(len(test)),
            "outer_test_used_for_selection": 0,
        })
    return pd.DataFrame(selections), pd.concat(audits, ignore_index=True)


def candidate_from_selection(selection: pd.Series | Any, candidates: list[Stage1Candidate]) -> Stage1Candidate:
    name = str(getattr(selection, "winner", selection["winner"] if isinstance(selection, pd.Series) else ""))
    return next(c for c in candidates if c.name == name)


def _prepare_residual_training(frame: pd.DataFrame, candidate: Stage1Candidate, target_col: str, min_history: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = past_only_dynamic_stage1(frame, candidate, target_col=target_col)
    if int(p["stage1_future_target_violation"].sum()) != 0:
        raise RuntimeError("STAGE1_TRAIN_FUTURE_TARGET_LEAKAGE")
    out = frame.copy()
    out["__stage1"] = p["stage1_baseline_log"]
    out["__history_count"] = p["stage1_category_history_count"]
    out["__residual"] = pd.to_numeric(out[target_col], errors="coerce") - pd.to_numeric(out["__stage1"], errors="coerce")
    mask = pd.to_numeric(out["__history_count"], errors="coerce").fillna(0).ge(int(min_history)) & np.isfinite(pd.to_numeric(out["__residual"], errors="coerce"))
    return out.loc[mask].copy(), p


def _metric_bundle(test: pd.DataFrame, y: np.ndarray, s1: np.ndarray, residual_pred: np.ndarray) -> dict[str, float]:
    final = s1 + residual_pred
    total = regression_metrics(y, final)
    base = regression_metrics(y, s1)
    residual = regression_metrics(y - s1, residual_pred)
    out = dict(total)
    out.update({
        "stage1_r2": float(base["r2"]), "stage1_rmse": float(base["rmse"]),
        "residual_r2": float(residual["r2"]), "residual_rmse": float(residual["rmse"]),
        "delta_r2_vs_stage1": float(total["r2"] - base["r2"]),
        "category_bias_mae": float(_category_bias_mae(test, y, final)),
    })
    out.update(_calibration_stats(y, final))
    return out


def fit_dynamic_residual_rolling(
    spec: ResidualModelSpec,
    fold: TimeFold,
    cfg: dict[str, Any],
    candidate: Stage1Candidate,
    seed: int,
    backend: str,
    quick: bool,
    threads: int,
    *,
    features: list[str] | None = None,
    categorical: list[str] | None = None,
    return_predictions: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    features = list(features or spec.features)
    categorical = list(categorical if categorical is not None else [c for c in spec.categorical if c in features])
    fit, valid, full_train, test, valid_periods = split_fold(spec.frame, spec.period_col, fold, int(cfg["validation"]["inner_valid_months"]))
    min_history = int(cfg["category_baseline"].get("minimum_category_history", 12))

    fit_resid, _ = _prepare_residual_training(fit, candidate, spec.target_col, min_history)
    valid_state = fit_dynamic_stage1(fit, candidate, target_col=spec.target_col)
    valid_s1_frame = predict_dynamic_stage1(valid_state, valid)
    if int(valid_s1_frame["stage1_future_target_violation"].sum()) != 0:
        raise RuntimeError("STAGE1_VALID_LEAKAGE")
    valid_s1 = valid_s1_frame["stage1_baseline_log"].to_numpy(float)
    yv = pd.to_numeric(valid[spec.target_col], errors="coerce").to_numpy(float)
    yr_fit = pd.to_numeric(fit_resid["__residual"], errors="coerce").to_numpy(float)

    Xfit = _prepare_catboost(fit_resid, features, categorical)
    Xvalid = _prepare_catboost(valid, features, categorical)
    cat_idx = [features.index(c) for c in categorical]
    params = catboost_params(cfg, int(seed), backend, quick, threads)
    model = CatBoostRegressor(**params)
    started = time.time()
    model.fit(Pool(Xfit, yr_fit, cat_features=cat_idx), eval_set=Pool(Xvalid, yv - valid_s1, cat_features=cat_idx), early_stopping_rounds=int(cfg["catboost"]["early_stopping_rounds"]), verbose=False)
    best = int(model.get_best_iteration()) + 1
    if best <= 0:
        best = int(params["iterations"])

    full_resid, train_stage1 = _prepare_residual_training(full_train, candidate, spec.target_col, min_history)
    test_state = fit_dynamic_stage1(full_train, candidate, target_col=spec.target_col)
    test_s1_frame = predict_dynamic_stage1(test_state, test)
    if int(test_s1_frame["stage1_future_target_violation"].sum()) != 0:
        raise RuntimeError("STAGE1_TEST_FUTURE_TARGET_LEAKAGE")
    test_s1 = test_s1_frame["stage1_baseline_log"].to_numpy(float)
    yt = pd.to_numeric(test[spec.target_col], errors="coerce").to_numpy(float)
    Xfull = _prepare_catboost(full_resid, features, categorical)
    Xtest = _prepare_catboost(test, features, categorical)
    p2 = dict(params); p2["iterations"] = int(best)
    final = CatBoostRegressor(**p2)
    final.fit(Pool(Xfull, pd.to_numeric(full_resid["__residual"], errors="coerce").to_numpy(float), cat_features=cat_idx), verbose=False)
    rp = final.predict(Pool(Xtest, cat_features=cat_idx))
    metrics = _metric_bundle(test, yt, test_s1, rp)
    metrics.update({
        "model": spec.name, "profile": spec.profile, "backend": backend.upper(), "seed": int(seed), "fold": int(fold.fold),
        "test_start": int(fold.test_start), "test_end": int(fold.test_end), "train_rows": int(len(full_resid)), "raw_train_rows": int(len(full_train)), "test_rows": int(len(test)),
        "best_iterations": int(best), "features": len(features), "categorical_features": len(categorical), "status": "PASS", "data_identity": spec.data_identity,
        "stage1_candidate": candidate.name, "stage1_candidate_json": str(candidate.to_dict()),
        "stage1_train_leakage_violations": int(train_stage1["stage1_future_target_violation"].sum()),
        "stage1_test_leakage_violations": int(test_s1_frame["stage1_future_target_violation"].sum()),
        "stage1_max_history_period": int(test_s1_frame["stage1_max_history_period"].max()),
        "elapsed_seconds": float(time.time() - started),
        "inner_valid_start": int(min(valid_periods)), "inner_valid_end": int(max(valid_periods)),
    })
    pred = None
    if return_predictions:
        keys = [c for c in ["district", "year_month", "category", "category_major", "spend_thousand_krw"] if c in test]
        pred = test[keys].copy()
        pred["row_index"] = test.index.to_numpy()
        pred["actual_log_spend"] = yt
        pred["stage1_log"] = test_s1
        pred["residual_actual"] = yt - test_s1
        pred["residual_prediction"] = rp
        pred["expected_log_spend"] = test_s1 + rp
        pred["fold"] = int(fold.fold); pred["backend"] = backend.upper(); pred["seed"] = int(seed); pred["stage1_candidate"] = candidate.name
    return metrics, pred


def fit_dynamic_residual_fixed(spec: ResidualModelSpec, fold: TimeFold, cfg: dict[str, Any], candidate: Stage1Candidate, seed: int, backend: str, threads: int, *, iterations: int, features: list[str], categorical: list[str]) -> dict[str, Any]:
    _, _, full_train, test, _ = split_fold(spec.frame, spec.period_col, fold, int(cfg["validation"]["inner_valid_months"]))
    min_history = int(cfg["category_baseline"].get("minimum_category_history", 12))
    full_resid, train_s1 = _prepare_residual_training(full_train, candidate, spec.target_col, min_history)
    state = fit_dynamic_stage1(full_train, candidate, target_col=spec.target_col)
    s1f = predict_dynamic_stage1(state, test)
    if int(s1f["stage1_future_target_violation"].sum()) != 0 or int(train_s1["stage1_future_target_violation"].sum()) != 0:
        raise RuntimeError("STAGE1_FIXED_LEAKAGE")
    s1 = s1f["stage1_baseline_log"].to_numpy(float)
    y = pd.to_numeric(test[spec.target_col], errors="coerce").to_numpy(float)
    cats = [c for c in categorical if c in features]
    cat_idx = [features.index(c) for c in cats]
    Xtr = _prepare_catboost(full_resid, features, cats)
    Xte = _prepare_catboost(test, features, cats)
    params = catboost_params(cfg, int(seed), backend, False, threads); params["iterations"] = int(iterations)
    model = CatBoostRegressor(**params)
    model.fit(Pool(Xtr, pd.to_numeric(full_resid["__residual"], errors="coerce").to_numpy(float), cat_features=cat_idx), verbose=False)
    rp = model.predict(Pool(Xte, cat_features=cat_idx))
    m = _metric_bundle(test, y, s1, rp)
    m.update({"train_rows": len(full_resid), "raw_train_rows": len(full_train), "test_rows": len(test), "stage1_candidate": candidate.name, "stage1_leakage_violations": 0})
    return m


def run_backend_baselines(spec: ResidualModelSpec, cfg: dict[str, Any], selections: pd.DataFrame, backend: str, quick: bool, threads: int, workers: int, candidates: list[Stage1Candidate] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = list(candidates or default_candidates())
    folds = explicit_rolling_folds(spec.frame, spec.period_col, cfg["validation"]["rolling_test_blocks"], int(cfg["validation"]["min_train_months"]))
    seeds = list(cfg["validation"]["seeds_gpu" if backend.upper() == "GPU" else "seeds_cpu"])
    if quick:
        folds = folds[-1:]; seeds = [seeds[0]]
    sel = {int(r.fold): next(c for c in candidates if c.name == str(r.winner)) for r in selections.itertuples(index=False)}
    jobs = [(fold, int(seed)) for fold in folds for seed in seeds]
    rows, preds = [], []
    def one(job):
        fold, seed = job
        return fit_dynamic_residual_rolling(spec, fold, cfg, sel[int(fold.fold)], seed, backend, quick, threads, return_predictions=True)
    if backend.upper() == "CPU" and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, j) for j in jobs]
            for fut in as_completed(futs):
                m,p = fut.result(); rows.append(m); preds.append(p)
    else:
        for j in jobs:
            m,p = one(j); rows.append(m); preds.append(p)
    return pd.DataFrame(rows), pd.concat([p for p in preds if p is not None], ignore_index=True) if preds else pd.DataFrame()
