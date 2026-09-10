from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..diagnostics import build_correlation_map, finalize_model_diagnostics
from ..io import DataPaths
from ..modeling import fit_regressor, importance_sum, run_feature_ablation
from ..plots import save_actual_vs_predicted, save_feature_importance
from ..utils import chronological_split, ensure_dir, parse_ym, write_json


DETAILED_AGE_CODES = {
    "F10", "F20", "F30", "F40", "F50", "F60O",
    "M10", "M20", "M30", "M40", "M50", "M60O",
}


def _prepare_source(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["sex_age_cd"] = frame["sex_age_cd"].astype(str)
    frame = frame[frame["sex_age_cd"].isin(DETAILED_AGE_CODES)].copy()
    frame["sales_amt_sum"] = pd.to_numeric(frame["sales_amt_sum"], errors="coerce").fillna(0.0)
    totals = frame.groupby(["std_ym", "mdclass_indutype_cd"])["sales_amt_sum"].transform("sum")
    frame[f"{prefix}_share_pct"] = np.where(totals > 0, 100.0 * frame["sales_amt_sum"] / totals, np.nan)
    return frame[
        [
            "std_ym",
            "sex_age_cd",
            "mdclass_indutype_cd",
            "sex_age_label",
            "industry_name",
            "major_category",
            "sales_amt_sum",
            f"{prefix}_share_pct",
        ]
    ].rename(columns={"sales_amt_sum": f"{prefix}_sales_amt_sum"})


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
        "source_model": "gyeonggi_age",
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
    general = _prepare_source(paths["gyeonggi_card_age_industry"], "general_card")
    local = _prepare_source(paths["gyeonggi_local_age_industry"], "local_currency")
    keys = ["std_ym", "sex_age_cd", "mdclass_indutype_cd"]
    merged = local.merge(
        general,
        on=keys,
        how="inner",
        suffixes=("_local", "_general"),
        validate="one_to_one",
    )
    merged["sex_age_label"] = merged["sex_age_label_local"].fillna(merged["sex_age_label_general"])
    merged["industry_name"] = merged["industry_name_local"].fillna(merged["industry_name_general"])
    merged["major_category"] = merged["major_category_local"].fillna(merged["major_category_general"])
    merged["share_difference_pp"] = (
        merged["local_currency_share_pct"] - merged["general_card_share_pct"]
    )
    parts = parse_ym(merged["std_ym"])
    merged["year"] = parts["year"]
    merged["month"] = parts["month"]
    merged["sample_weight"] = np.sqrt(
        merged["local_currency_sales_amt_sum"].clip(lower=0)
        + merged["general_card_sales_amt_sum"].clip(lower=0)
        + 1.0
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["share_difference_pp"])
    merged.to_csv(output / "age_industry_comparison_model_table.csv", index=False, encoding="utf-8-sig")

    train, test, test_periods = chronological_split(
        merged,
        "std_ym",
        int(config["validation"].get("gyeonggi_test_months", 12)),
    )
    feature_columns = [
        "year",
        "month",
        "sex_age_cd",
        "sex_age_label",
        "mdclass_indutype_cd",
        "industry_name",
        "major_category",
    ]
    categorical = [
        "sex_age_cd",
        "sex_age_label",
        "mdclass_indutype_cd",
        "industry_name",
        "major_category",
    ]
    model = fit_regressor(
        train,
        test,
        feature_columns,
        categorical,
        "share_difference_pp",
        output / "age_industry_model",
        config,
        sample_weight_column="sample_weight",
        key_columns=["std_ym", "sex_age_cd", "industry_name"],
        quick=quick,
    )
    save_feature_importance(model.importance, output / "age_industry_model" / "feature_importance.png")
    save_actual_vs_predicted(model.predictions, output / "age_industry_model" / "actual_vs_predicted.png")
    correlation = build_correlation_map(
        train,
        feature_columns,
        categorical,
        "share_difference_pp",
        output / "age_industry_model" / "correlation",
        config,
        quick=quick,
    )
    ablation = run_feature_ablation(
        train,
        test,
        feature_columns,
        categorical,
        "share_difference_pp",
        output / "age_industry_model" / "feature_ablation",
        config,
        model.metrics,
        sample_weight_column="sample_weight",
        quick=quick,
    )
    finalize_model_diagnostics(output / "age_industry_model")

    merged["positive_difference"] = (merged["share_difference_pp"] > 0).astype(int)
    profile = (
        merged.groupby(["sex_age_cd", "sex_age_label", "industry_name", "major_category"], as_index=False)
        .agg(
            mean_difference_pp=("share_difference_pp", "mean"),
            median_difference_pp=("share_difference_pp", "median"),
            std_difference_pp=("share_difference_pp", "std"),
            positive_month_share=("positive_difference", "mean"),
            month_count=("std_ym", "nunique"),
            local_sales_sum=("local_currency_sales_amt_sum", "sum"),
            general_sales_sum=("general_card_sales_amt_sum", "sum"),
        )
    )
    profile["sign_consistency"] = np.maximum(
        profile["positive_month_share"], 1.0 - profile["positive_month_share"]
    )
    profile.to_csv(output / "age_industry_profile.csv", index=False, encoding="utf-8-sig")

    yearly = (
        merged.groupby(["year", "sex_age_cd"], as_index=False)["share_difference_pp"].mean()
    )
    age_stability_rows: list[dict[str, Any]] = []
    for age_code, group in yearly.groupby("sex_age_cd"):
        positive_share = float((group["share_difference_pp"] >= 0).mean())
        age_stability_rows.append(
            {
                "sex_age_cd": age_code,
                "mean_difference_pp": float(group["share_difference_pp"].mean()),
                "std_across_years": float(group["share_difference_pp"].std(ddof=0)),
                "sign_consistency": max(positive_share, 1.0 - positive_share),
                "year_count": int(group["year"].nunique()),
            }
        )
    age_stability = pd.DataFrame(age_stability_rows).sort_values(
        "sign_consistency", ascending=False
    )
    age_stability.to_csv(output / "age_band_year_stability.csv", index=False, encoding="utf-8-sig")

    age_importance = importance_sum(model.importance, ["sex_age_cd", "sex_age_label"])
    industry_importance = importance_sum(
        model.importance, ["mdclass_indutype_cd", "industry_name", "major_category"]
    )
    season_importance = importance_sum(model.importance, ["month", "year"])
    age_stability_score = float(age_stability["sign_consistency"].mean()) if not age_stability.empty else 0.5
    interaction_stability = float(profile["sign_consistency"].mean()) if not profile.empty else 0.5
    coverage = min(1.0, merged["std_ym"].nunique() / 69.0)

    candidates = pd.DataFrame(
        [
            _candidate(
                "visitor_age_mix",
                "demographic",
                "visitor_age_share_by_band",
                min(1.0, age_importance * 2.8),
                age_stability_score,
                coverage,
                "일반카드와 지역화폐의 12개 상세 성연령 구성 차이",
                "포항 관광지별 방문 연령구성을 별도 피처로 유지",
                "방문 연령은 소비 연령이 아니며 효과계수는 이전 금지",
            ),
            _candidate(
                "age_industry_interaction",
                "interaction",
                "visitor_age_x_industry_type",
                min(1.0, (age_importance + industry_importance) * 1.7),
                interaction_stability,
                coverage,
                "성연령×42개 업종 share 차이의 반복성",
                "포항 상권유형/관광지별 연령구성×업종구성 교차피처",
                "지역화폐-일반카드 차이를 쿠폰 인과효과로 해석하지 않음",
            ),
            _candidate(
                "sex_mix",
                "demographic",
                "visitor_sex_share",
                min(1.0, age_importance * 1.8),
                age_stability_score,
                coverage,
                "성별 연령밴드 구성의 월별 안정성",
                "포항 관광지/축제의 남녀 방문비중으로 계산",
                "성별 소비액이 아닌 방문구성임을 명시",
            ),
            _candidate(
                "demographic_seasonality",
                "interaction",
                "visitor_age_x_season",
                min(1.0, (age_importance + season_importance) * 1.5),
                age_stability_score,
                coverage,
                "월·연도와 성연령 차이의 보조모델 중요도",
                "포항 월/계절×관광지 연령구성 교차피처",
                "표본이 작은 관광지는 축약/partial pooling 필요",
            ),
            _candidate(
                "payment_instrument_heterogeneity",
                "policy_design",
                "coupon_response_heterogeneity_hypothesis",
                min(1.0, (age_importance + industry_importance) * 1.4),
                interaction_stability,
                coverage,
                "결제수단별 연령·업종 구성이 동일하지 않음",
                "파일럿에서 연령×업종 subgroup을 사전지정",
                "파일럿 전에는 실제 쿠폰 CATE로 사용 금지",
            ),
        ]
    )
    candidates["evidence_quality_score"] = float(
        np.clip((float(model.metrics.get("r2", 0.0)) + 0.1) / 1.1, 0.0, 1.0)
    )
    candidates.to_csv(output / "feature_candidates.csv", index=False, encoding="utf-8-sig")
    summary = {
        "module": "gyeonggi_age",
        "rows_used": int(len(merged)),
        "months_used": int(merged["std_ym"].nunique()),
        "test_periods": [int(value) for value in test_periods],
        "model_metrics": model.metrics,
        "diagnostics": {"correlation": correlation, "ablation": ablation},
        "candidate_count": int(len(candidates)),
        "interpretation_guardrail": "지역화폐와 일반카드 구성 차이는 기술적 이질성 증거이며 쿠폰 인과효과가 아니다.",
    }
    write_json(summary, output / "summary.json")
    return summary
