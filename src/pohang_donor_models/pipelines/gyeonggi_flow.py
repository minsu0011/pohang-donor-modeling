from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..diagnostics import build_correlation_map, finalize_model_diagnostics
from ..io import DataPaths
from ..modeling import fit_regressor, importance_sum, run_feature_ablation
from ..plots import save_actual_vs_predicted, save_feature_importance
from ..utils import chronological_split, ensure_dir, hhi, parse_ym, weighted_entropy, write_json


def _network_features(group: pd.DataFrame) -> pd.Series:
    sales = pd.to_numeric(group["sales_amt_sum"], errors="coerce").fillna(0.0).clip(lower=0)
    total = float(sales.sum())
    group_name = getattr(group, "name", None)
    base_city = str(group_name[0] if isinstance(group_name, tuple) else group_name)
    self_mask = group["inflow_primary_city_name"].astype(str) == base_city
    self_sales = float(sales[self_mask].sum())
    external = group.loc[~self_mask].copy()
    external_sales = pd.to_numeric(external["sales_amt_sum"], errors="coerce").fillna(0.0).clip(lower=0)
    external_total = float(external_sales.sum())
    top1 = float(external_sales.max()) if len(external_sales) else 0.0
    entropy = weighted_entropy(external_sales)
    return pd.Series(
        {
            "total_relationship_sales": total,
            "self_sales": self_sales,
            "external_sales": external_total,
            "self_share": self_sales / total if total > 0 else np.nan,
            "external_share": external_total / total if total > 0 else np.nan,
            "external_partner_count": int((external_sales > 0).sum()),
            "external_top1_share": top1 / external_total if external_total > 0 else 0.0,
            "external_hhi": hhi(external_sales),
            "external_entropy": entropy,
            "effective_external_partners": float(np.exp(entropy)) if external_total > 0 else 0.0,
        }
    )


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
        "source_model": "gyeonggi_flow",
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
    pair = pd.read_csv(paths["gyeonggi_flow_city_pair"], low_memory=False)
    parts = parse_ym(pair["std_ym"])
    pair["year"] = parts["year"]
    pair["month"] = parts["month"]
    pair["same_primary_city"] = (
        pair["base_primary_city_name"].astype(str) == pair["inflow_primary_city_name"].astype(str)
    ).astype(int)
    pair["sales_amt_sum"] = pd.to_numeric(pair["sales_amt_sum"], errors="coerce")
    pair = pair[pair["sales_amt_sum"].notna() & (pair["sales_amt_sum"] >= 0)].copy()
    pair["target_log_link_sales"] = np.log1p(pair["sales_amt_sum"])
    pair["sample_weight"] = np.sqrt(
        pd.to_numeric(pair["sales_nonmissing_count"], errors="coerce").fillna(1).clip(lower=1)
    )

    train, test, test_periods = chronological_split(
        pair,
        "std_ym",
        int(config["validation"].get("gyeonggi_test_months", 12)),
    )
    feature_columns = [
        "year",
        "month",
        "base_primary_city_name",
        "inflow_primary_city_name",
        "same_primary_city",
    ]
    categorical = ["base_primary_city_name", "inflow_primary_city_name"]
    model = fit_regressor(
        train,
        test,
        feature_columns,
        categorical,
        "target_log_link_sales",
        output / "link_strength_model",
        config,
        sample_weight_column="sample_weight",
        key_columns=["base_primary_city_name", "inflow_primary_city_name", "std_ym"],
        quick=quick,
    )
    save_feature_importance(model.importance, output / "link_strength_model" / "feature_importance.png")
    save_actual_vs_predicted(model.predictions, output / "link_strength_model" / "actual_vs_predicted.png")
    correlation = build_correlation_map(
        train,
        feature_columns,
        categorical,
        "target_log_link_sales",
        output / "link_strength_model" / "correlation",
        config,
        quick=quick,
    )
    ablation = run_feature_ablation(
        train,
        test,
        feature_columns,
        categorical,
        "target_log_link_sales",
        output / "link_strength_model" / "feature_ablation",
        config,
        model.metrics,
        sample_weight_column="sample_weight",
        quick=quick,
    )
    finalize_model_diagnostics(output / "link_strength_model")

    network = (
        pair.groupby(["base_primary_city_name", "std_ym"], group_keys=False)
        .apply(_network_features, include_groups=False)
        .reset_index()
    )
    network_parts = parse_ym(network["std_ym"])
    network["year"] = network_parts["year"]
    network["month"] = network_parts["month"]
    network.to_csv(output / "city_month_network_features.csv", index=False, encoding="utf-8-sig")

    stability_rows: list[dict[str, Any]] = []
    for city, group in network.groupby("base_primary_city_name"):
        stability_rows.append(
            {
                "base_primary_city_name": city,
                "months": int(group["std_ym"].nunique()),
                "external_share_mean": float(group["external_share"].mean()),
                "external_share_std": float(group["external_share"].std(ddof=0)),
                "external_share_stability": float(
                    1.0
                    - min(
                        1.0,
                        group["external_share"].std(ddof=0)
                        / (abs(group["external_share"].mean()) + 1e-6),
                    )
                ),
                "partner_count_mean": float(group["external_partner_count"].mean()),
                "hhi_mean": float(group["external_hhi"].mean()),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output / "city_network_stability.csv", index=False, encoding="utf-8-sig")

    relationship = pd.read_csv(paths["gyeonggi_flow_relationship_industry"], low_memory=False)
    relationship["sales_amt_sum"] = pd.to_numeric(relationship["sales_amt_sum"], errors="coerce").fillna(0.0)
    totals = relationship.groupby(["std_ym", "major_category"])["sales_amt_sum"].transform("sum")
    relationship["relationship_share_pct"] = np.where(
        totals > 0, 100.0 * relationship["sales_amt_sum"] / totals, np.nan
    )
    relation_profile = (
        relationship.groupby(["major_category", "relationship_class"], as_index=False)
        .agg(
            mean_relationship_share_pct=("relationship_share_pct", "mean"),
            median_relationship_share_pct=("relationship_share_pct", "median"),
            std_relationship_share_pct=("relationship_share_pct", "std"),
            month_count=("std_ym", "nunique"),
        )
    )
    relation_profile.to_csv(
        output / "industry_relationship_profile.csv", index=False, encoding="utf-8-sig"
    )

    pair_importance = importance_sum(
        model.importance,
        ["base_primary_city_name", "inflow_primary_city_name", "same_primary_city"],
    )
    time_importance = importance_sum(model.importance, ["month", "year"])
    network_stability = float(stability["external_share_stability"].mean()) if not stability.empty else 0.5
    relationship_stability = float(
        1.0
        - min(
            1.0,
            relation_profile["std_relationship_share_pct"].mean()
            / (abs(relation_profile["mean_relationship_share_pct"].mean()) + 1e-6),
        )
    )
    coverage = min(1.0, pair["std_ym"].nunique() / 68.0)
    candidates = pd.DataFrame(
        [
            _candidate(
                "self_containment_share",
                "spatial_network",
                "local_capture_or_anchor_strength",
                min(1.0, pair_importance * 2.0),
                network_stability,
                coverage,
                "base 도시 내 동일도시 관계 비중과 월별 안정성",
                "포항에서는 앵커상권 내부 포획력/자체소비 비중으로 재정의",
                "유입관계는 개인 이동 OD가 아님",
            ),
            _candidate(
                "external_connection_share",
                "spatial_network",
                "external_connection_potential",
                min(1.0, pair_importance * 2.2),
                network_stability,
                coverage,
                "도시쌍 링크강도 모델과 외부관계 비중",
                "포항 연관관광지·후보축·거리그래프에서 별도 계산",
                "경기 외부유입 비율을 포항 수치로 복사하지 않음",
            ),
            _candidate(
                "partner_diversity",
                "spatial_network",
                "connected_destination_diversity",
                min(1.0, pair_importance * 1.8),
                network_stability,
                coverage,
                "외부 파트너 수·entropy·effective partner count",
                "포항 후보축의 연결 관광지/상권 다양성으로 계산",
                "관계 수가 많다고 실제 파급효과가 보장되지 않음",
            ),
            _candidate(
                "top_partner_concentration",
                "spatial_network",
                "top_receiver_concentration",
                min(1.0, pair_importance * 1.7),
                network_stability,
                coverage,
                "상위 연결지 비중과 HHI",
                "포항 앵커→리시버 후보축 집중도로 계산",
                "집중도는 구조피처이며 인과효과가 아님",
            ),
            _candidate(
                "industry_spatial_scope",
                "interaction",
                "industry_x_spatial_band",
                min(1.0, (pair_importance + time_importance) * 1.4),
                relationship_stability,
                min(1.0, relationship["std_ym"].nunique() / 68.0),
                "업종대분류별 same/cross 관계 구성 차이",
                "포항에서 업종별 250m/500m/1km 및 연결축 exposure로 구현",
                "거리반경은 파일럿 이후 포항 자료에서 재추정",
            ),
        ]
    )
    predictive_quality = float(np.clip((float(model.metrics.get("r2", 0.0)) + 0.2) / 1.2, 0.0, 1.0))
    descriptive_quality = float(np.clip((network_stability + relationship_stability) / 2.0, 0.0, 1.0))
    candidates["evidence_quality_score"] = 0.35 * predictive_quality + 0.65 * descriptive_quality
    candidates.to_csv(output / "feature_candidates.csv", index=False, encoding="utf-8-sig")
    summary = {
        "module": "gyeonggi_flow",
        "pair_rows_used": int(len(pair)),
        "network_rows": int(len(network)),
        "test_periods": [int(value) for value in test_periods],
        "model_metrics": model.metrics,
        "diagnostics": {"correlation": correlation, "ablation": ablation},
        "candidate_count": int(len(candidates)),
        "interpretation_guardrail": "유입관계는 공간피처 설계 donor이며 개인 OD·포항 spillover 추정치가 아니다.",
    }
    write_json(summary, output / "summary.json")
    return summary
