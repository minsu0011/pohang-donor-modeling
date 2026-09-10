from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..diagnostics import build_correlation_map, finalize_model_diagnostics
from ..io import DataPaths
from ..modeling import fit_regressor, importance_sum, run_feature_ablation
from ..plots import save_actual_vs_predicted, save_feature_importance
from ..utils import chronological_split, ensure_dir, minmax_score, parse_ym, write_json


def _rank_stability(profile: pd.DataFrame, group_col: str, item_col: str, value_col: str) -> float:
    pivot = profile.pivot_table(index=item_col, columns=group_col, values=value_col, aggfunc="mean")
    if pivot.shape[1] < 2:
        return 0.5
    correlations: list[float] = []
    columns = list(pivot.columns)
    for left_index in range(len(columns)):
        for right_index in range(left_index + 1, len(columns)):
            pair = pivot[[columns[left_index], columns[right_index]]].dropna()
            if len(pair) < 3:
                continue
            corr = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).correlation
            if np.isfinite(corr):
                correlations.append(float(corr))
    if not correlations:
        return 0.5
    return float(np.clip((np.mean(correlations) + 1.0) / 2.0, 0.0, 1.0))


def _peak_stability(frame: pd.DataFrame) -> float:
    yearly = (
        frame.groupby(["industry_name", "year", "tmzon_cd"], as_index=False)["relative_band_strength"]
        .mean()
    )
    if yearly.empty:
        return 0.5
    peaks = yearly.loc[yearly.groupby(["industry_name", "year"])["relative_band_strength"].idxmax()]
    scores: list[float] = []
    for _, group in peaks.groupby("industry_name"):
        counts = group["tmzon_cd"].value_counts(normalize=True)
        if not counts.empty:
            scores.append(float(counts.iloc[0]))
    return float(np.mean(scores)) if scores else 0.5


def _candidate(
    name: str,
    family: str,
    pohang_feature: str,
    importance: float,
    stability: float,
    coverage: float,
    evidence: str,
    transfer_rule: str,
    caution: str,
) -> dict[str, Any]:
    return {
        "source_model": "gyeonggi_time",
        "candidate_feature": name,
        "feature_family": family,
        "pohang_feature_name": pohang_feature,
        "importance_score": float(np.clip(importance, 0.0, 1.0)),
        "stability_score": float(np.clip(stability, 0.0, 1.0)),
        "coverage_score": float(np.clip(coverage, 0.0, 1.0)),
        "evidence": evidence,
        "transfer_rule": transfer_rule,
        "caution": caution,
        "donor_only": False,
    }


def run(paths: DataPaths, config: dict[str, Any], output_dir: str | Path, quick: bool = False) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    state = pd.read_csv(paths["gyeonggi_time_industry"], low_memory=False)
    state = state[state["tmzon_cd"].astype(str) != "TOT"].copy()
    period_parts = parse_ym(state["std_ym"])
    state["year"] = period_parts["year"]
    state["month"] = period_parts["month"]
    state["sales_amt_mean"] = pd.to_numeric(state["sales_amt_mean"], errors="coerce")
    state = state[state["sales_amt_mean"].notna() & (state["sales_amt_mean"] >= 0)].copy()
    state["target_log_sales_mean"] = np.log1p(state["sales_amt_mean"])
    median_by_cell = state.groupby(["std_ym", "mdclass_indutype_cd"])["sales_amt_mean"].transform("median")
    state["relative_band_strength"] = np.where(
        median_by_cell > 0,
        100.0 * state["sales_amt_mean"] / median_by_cell,
        np.nan,
    )
    state["sample_weight"] = np.sqrt(
        pd.to_numeric(state["sales_nonmissing_count"], errors="coerce").fillna(1).clip(lower=1)
    )

    train, test, test_periods = chronological_split(
        state,
        "std_ym",
        int(config["validation"].get("gyeonggi_test_months", 12)),
    )
    feature_columns = [
        "year",
        "month",
        "tmzon_cd",
        "time_zone_label",
        "mdclass_indutype_cd",
        "industry_name",
        "major_category",
    ]
    categorical = [
        "tmzon_cd",
        "time_zone_label",
        "mdclass_indutype_cd",
        "industry_name",
        "major_category",
    ]
    industry_model = fit_regressor(
        train,
        test,
        feature_columns,
        categorical,
        "target_log_sales_mean",
        output / "industry_time_model",
        config,
        sample_weight_column="sample_weight",
        key_columns=["std_ym", "tmzon_cd", "industry_name", "major_category"],
        quick=quick,
    )
    save_feature_importance(
        industry_model.importance,
        output / "industry_time_model" / "feature_importance.png",
    )
    save_actual_vs_predicted(
        industry_model.predictions,
        output / "industry_time_model" / "actual_vs_predicted.png",
    )
    industry_correlation = build_correlation_map(
        train,
        feature_columns,
        categorical,
        "target_log_sales_mean",
        output / "industry_time_model" / "correlation",
        config,
        quick=quick,
    )
    industry_ablation = run_feature_ablation(
        train,
        test,
        feature_columns,
        categorical,
        "target_log_sales_mean",
        output / "industry_time_model" / "feature_ablation",
        config,
        industry_model.metrics,
        sample_weight_column="sample_weight",
        quick=quick,
    )
    finalize_model_diagnostics(output / "industry_time_model")

    profile = (
        state.groupby(
            ["tmzon_cd", "time_zone_label", "mdclass_indutype_cd", "industry_name", "major_category"],
            as_index=False,
        )
        .agg(
            mean_relative_strength=("relative_band_strength", "mean"),
            median_relative_strength=("relative_band_strength", "median"),
            std_relative_strength=("relative_band_strength", "std"),
            month_count=("std_ym", "nunique"),
            raw_sales_mean=("sales_amt_mean", "mean"),
        )
    )
    profile["rank_within_industry"] = profile.groupby("industry_name")["mean_relative_strength"].rank(
        method="dense", ascending=False
    )
    profile.to_csv(output / "industry_time_profile.csv", index=False, encoding="utf-8-sig")

    city = pd.read_csv(paths["gyeonggi_time_city"], low_memory=False)
    city = city[city["tmzon_cd"].astype(str) != "TOT"].copy()
    city_parts = parse_ym(city["std_ym"])
    city["year"] = city_parts["year"]
    city["month"] = city_parts["month"]
    city["sales_amt_mean"] = pd.to_numeric(city["sales_amt_mean"], errors="coerce")
    city = city[city["sales_amt_mean"].notna() & (city["sales_amt_mean"] >= 0)].copy()
    city["target_log_sales_mean"] = np.log1p(city["sales_amt_mean"])
    city["sample_weight"] = np.sqrt(
        pd.to_numeric(city["sales_nonmissing_count"], errors="coerce").fillna(1).clip(lower=1)
    )
    city_train, city_test, city_test_periods = chronological_split(
        city,
        "std_ym",
        int(config["validation"].get("gyeonggi_test_months", 12)),
    )
    city_features = ["year", "month", "primary_city_name", "tmzon_cd", "time_zone_label"]
    city_categorical = ["primary_city_name", "tmzon_cd", "time_zone_label"]
    city_model = fit_regressor(
        city_train,
        city_test,
        city_features,
        city_categorical,
        "target_log_sales_mean",
        output / "city_time_model",
        config,
        sample_weight_column="sample_weight",
        key_columns=["primary_city_name", "std_ym", "tmzon_cd"],
        quick=quick,
    )
    save_feature_importance(city_model.importance, output / "city_time_model" / "feature_importance.png")
    city_correlation = build_correlation_map(
        city_train,
        city_features,
        city_categorical,
        "target_log_sales_mean",
        output / "city_time_model" / "correlation",
        config,
        quick=quick,
    )
    city_ablation = run_feature_ablation(
        city_train,
        city_test,
        city_features,
        city_categorical,
        "target_log_sales_mean",
        output / "city_time_model" / "feature_ablation",
        config,
        city_model.metrics,
        sample_weight_column="sample_weight",
        quick=quick,
    )
    finalize_model_diagnostics(output / "city_time_model")

    city_median = city.groupby(["primary_city_name", "std_ym"])["sales_amt_mean"].transform("median")
    city["relative_band_strength"] = np.where(
        city_median > 0, 100.0 * city["sales_amt_mean"] / city_median, np.nan
    )
    city_profile = (
        city.groupby(["primary_city_name", "tmzon_cd", "time_zone_label"], as_index=False)
        .agg(
            mean_relative_strength=("relative_band_strength", "mean"),
            median_relative_strength=("relative_band_strength", "median"),
            month_count=("std_ym", "nunique"),
        )
    )
    city_profile.to_csv(output / "city_time_profile.csv", index=False, encoding="utf-8-sig")

    time_importance = importance_sum(industry_model.importance, ["tmzon_cd", "time_zone_label"])
    industry_importance = importance_sum(
        industry_model.importance,
        ["mdclass_indutype_cd", "industry_name", "major_category"],
    )
    season_importance = importance_sum(industry_model.importance, ["month", "year"])
    city_importance = importance_sum(city_model.importance, ["primary_city_name"])
    time_stability = _rank_stability(
        state.groupby(["year", "tmzon_cd"], as_index=False)["relative_band_strength"].mean(),
        "year",
        "tmzon_cd",
        "relative_band_strength",
    )
    interaction_stability = _peak_stability(state)
    city_stability = _rank_stability(
        city.groupby(["year", "primary_city_name"], as_index=False)["sales_amt_mean"].mean(),
        "year",
        "primary_city_name",
        "sales_amt_mean",
    )
    coverage = min(1.0, state["std_ym"].nunique() / 72.0)

    candidates = pd.DataFrame(
        [
            _candidate(
                "time_band",
                "time",
                "time_band",
                min(1.0, time_importance * 2.5),
                time_stability,
                coverage,
                "경기 월×업종×시간대 카드매출 예측 및 연도별 시간대 순위 안정성",
                "포항 자료에서 오전/점심/오후/저녁/야간 band를 다시 계산",
                "경기의 시간대 계수와 peak 시각을 포항에 직접 이식하지 않음",
            ),
            _candidate(
                "industry_time_interaction",
                "interaction",
                "industry_x_time_band",
                min(1.0, (time_importance + industry_importance) * 1.6),
                interaction_stability,
                coverage,
                "업종별 peak 시간대의 연도 반복성과 CatBoost 중요도",
                "포항의 업종대분류×시간대 교차피처로 재산출",
                "시간구간 폭이 다르므로 원자료 단순합 금지",
            ),
            _candidate(
                "monthly_seasonality",
                "calendar",
                "month_and_season",
                min(1.0, season_importance * 3.0),
                time_stability,
                coverage,
                "72개월 시간대 패널의 월·연도 설명력",
                "포항 월/계절/비수기 지표로 계산",
                "월 효과는 포항 날씨·축제와 별도 통제",
            ),
            _candidate(
                "regional_time_heterogeneity",
                "interaction",
                "district_type_x_time_band",
                min(1.0, (city_importance + time_importance) * 1.8),
                city_stability,
                min(1.0, city["std_ym"].nunique() / 72.0),
                "경기 주요시×시간대 보조모델",
                "포항 상권유형×시간대 상호작용으로 변환",
                "도시명이 아니라 상권유형 구조만 이전",
            ),
            _candidate(
                "major_industry_category",
                "industry",
                "industry_major_category",
                min(1.0, industry_importance * 2.2),
                interaction_stability,
                coverage,
                "42개 업종·대분류의 반복적 설명력",
                "포항 쿠폰후보 점포를 공통 업종대분류로 매핑",
                "경기 업종 매출규모 자체는 포항 결과가 아님",
            ),
        ]
    )
    mean_r2 = float(np.mean([industry_model.metrics.get("r2", 0.0), city_model.metrics.get("r2", 0.0)]))
    candidates["evidence_quality_score"] = float(np.clip((mean_r2 + 0.1) / 1.1, 0.0, 1.0))
    candidates.to_csv(output / "feature_candidates.csv", index=False, encoding="utf-8-sig")
    summary = {
        "module": "gyeonggi_time",
        "rows_used_industry": int(len(state)),
        "rows_used_city": int(len(city)),
        "industry_test_periods": [int(value) for value in test_periods],
        "city_test_periods": [int(value) for value in city_test_periods],
        "industry_model_metrics": industry_model.metrics,
        "city_model_metrics": city_model.metrics,
        "diagnostics": {
            "industry_time": {
                "correlation": industry_correlation,
                "ablation": industry_ablation,
            },
            "city_time": {
                "correlation": city_correlation,
                "ablation": city_ablation,
            },
        },
        "candidate_count": int(len(candidates)),
        "interpretation_guardrail": "시간대/업종 구조만 donor로 사용하며 경기의 계수·peak를 포항에 복사하지 않는다.",
    }
    write_json(summary, output / "summary.json")
    return summary
