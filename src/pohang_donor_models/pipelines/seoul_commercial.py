from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..diagnostics import build_correlation_map, finalize_model_diagnostics
from ..io import DataPaths
from ..modeling import fit_regressor, importance_sum, run_feature_ablation
from ..plots import save_actual_vs_predicted, save_cluster_profile_heatmap, save_feature_importance
from ..utils import (
    chronological_split,
    ensure_dir,
    hhi,
    quarter_to_index,
    weighted_entropy,
    write_json,
    yearly_sign_stability,
)


SALES_AMOUNT_COLUMNS = [
    "reported_sales_amount",
    "weekday_sales_amount",
    "weekend_sales_amount",
    "time_00_06_sales_amount",
    "time_06_11_sales_amount",
    "time_11_14_sales_amount",
    "time_14_17_sales_amount",
    "time_17_21_sales_amount",
    "time_21_24_sales_amount",
    "male_sales_amount",
    "female_sales_amount",
    "age_10_sales_amount",
    "age_20_sales_amount",
    "age_30_sales_amount",
    "age_40_sales_amount",
    "age_50_sales_amount",
    "age_60_plus_sales_amount",
]

SALES_USE_COLUMNS = [
    "quarter_code",
    "district_type_code",
    "district_type_name",
    "district_code",
    "district_name",
    "service_industry_code",
    "service_industry_name",
    "reported_sales_count",
] + SALES_AMOUNT_COLUMNS

STORE_USE_COLUMNS = [
    "quarter_code",
    "district_type_code",
    "district_type_name",
    "district_code",
    "district_name",
    "service_industry_code",
    "service_industry_name",
    "store_count",
    "similar_industry_store_count",
    "opening_store_count",
    "closing_store_count",
    "franchise_store_count",
]


def _industry_group(code: pd.Series) -> pd.Series:
    text = code.astype("string").fillna("")
    return np.select(
        [text.str.startswith("CS100"), text.str.startswith("CS200"), text.str.startswith("CS300")],
        ["food", "service", "retail"],
        default="other",
    )


def _read_and_filter_districts(
    frame: pd.DataFrame,
    quick: bool,
    max_districts: int,
    seed: int,
) -> pd.DataFrame:
    if not quick or frame["district_code"].nunique() <= max_districts:
        return frame
    unique_codes = np.array(sorted(frame["district_code"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(unique_codes, size=max_districts, replace=False).tolist())
    return frame[frame["district_code"].astype(str).isin(selected)].copy()


def _aggregate_sales(path: Path, quick: bool, max_districts: int, seed: int) -> pd.DataFrame:
    sales = pd.read_csv(path, usecols=SALES_USE_COLUMNS, low_memory=False)
    sales["district_code"] = sales["district_code"].astype(str)
    sales["quarter_code"] = pd.to_numeric(sales["quarter_code"], errors="coerce").astype("Int64")
    sales = sales.dropna(subset=["quarter_code", "district_code"])
    sales = _read_and_filter_districts(sales, quick, max_districts, seed)
    for column in SALES_AMOUNT_COLUMNS + ["reported_sales_count"]:
        sales[column] = pd.to_numeric(sales[column], errors="coerce").fillna(0.0).clip(lower=0)
    sales["industry_group"] = _industry_group(sales["service_industry_code"])
    keys = ["quarter_code", "district_code"]
    metadata = (
        sales.groupby(keys, as_index=False)
        .agg(
            district_type_code=("district_type_code", "first"),
            district_type_name=("district_type_name", "first"),
            district_name=("district_name", "first"),
            service_industry_count=("service_industry_code", "nunique"),
        )
    )
    sums = sales.groupby(keys, as_index=False)[SALES_AMOUNT_COLUMNS + ["reported_sales_count"]].sum()
    industry_stats = (
        sales.groupby(keys, group_keys=False)["reported_sales_amount"]
        .agg(industry_hhi=hhi, industry_entropy=weighted_entropy, top_industry_sales="max")
        .reset_index()
    )
    grouped_sales = (
        sales.groupby(keys + ["industry_group"], as_index=False)["reported_sales_amount"].sum()
        .pivot_table(index=keys, columns="industry_group", values="reported_sales_amount", fill_value=0.0)
        .reset_index()
    )
    grouped_sales.columns.name = None
    for group_name in ["food", "service", "retail", "other"]:
        if group_name not in grouped_sales.columns:
            grouped_sales[group_name] = 0.0
        grouped_sales = grouped_sales.rename(columns={group_name: f"{group_name}_sales_amount"})

    result = metadata.merge(sums, on=keys, validate="one_to_one")
    result = result.merge(industry_stats, on=keys, validate="one_to_one")
    result = result.merge(grouped_sales, on=keys, validate="one_to_one")
    total = result["reported_sales_amount"].replace(0, np.nan)
    result["top_industry_share"] = result["top_industry_sales"] / total
    result["weekday_share"] = result["weekday_sales_amount"] / total
    result["weekend_share"] = result["weekend_sales_amount"] / total
    for prefix in [
        "time_00_06",
        "time_06_11",
        "time_11_14",
        "time_14_17",
        "time_17_21",
        "time_21_24",
        "male",
        "female",
        "age_10",
        "age_20",
        "age_30",
        "age_40",
        "age_50",
        "age_60_plus",
        "food",
        "service",
        "retail",
        "other",
    ]:
        amount_col = f"{prefix}_sales_amount"
        result[f"{prefix}_share"] = result[amount_col] / total
    result["evening_share"] = (
        result["time_17_21_sales_amount"] + result["time_21_24_sales_amount"]
    ) / total
    result["avg_ticket"] = result["reported_sales_amount"] / result["reported_sales_count"].replace(0, np.nan)
    result["log_sales"] = np.log1p(result["reported_sales_amount"])
    return result


def _aggregate_stores(path: Path, selected_districts: set[str] | None = None) -> pd.DataFrame:
    stores = pd.read_csv(path, usecols=STORE_USE_COLUMNS, low_memory=False)
    stores["district_code"] = stores["district_code"].astype(str)
    stores["quarter_code"] = pd.to_numeric(stores["quarter_code"], errors="coerce").astype("Int64")
    stores = stores.dropna(subset=["quarter_code", "district_code"])
    if selected_districts is not None:
        stores = stores[stores["district_code"].isin(selected_districts)].copy()
    numeric = [
        "store_count",
        "similar_industry_store_count",
        "opening_store_count",
        "closing_store_count",
        "franchise_store_count",
    ]
    for column in numeric:
        stores[column] = pd.to_numeric(stores[column], errors="coerce").fillna(0.0).clip(lower=0)
    result = (
        stores.groupby(["quarter_code", "district_code"], as_index=False)
        .agg(
            store_count=("store_count", "sum"),
            similar_industry_store_count=("similar_industry_store_count", "sum"),
            opening_store_count=("opening_store_count", "sum"),
            closing_store_count=("closing_store_count", "sum"),
            franchise_store_count=("franchise_store_count", "sum"),
            store_industry_count=("service_industry_code", "nunique"),
        )
    )
    base = result["store_count"].replace(0, np.nan)
    result["opening_intensity"] = result["opening_store_count"] / base
    result["closing_intensity"] = result["closing_store_count"] / base
    result["store_turnover_intensity"] = (
        result["opening_store_count"] + result["closing_store_count"]
    ) / base
    result["franchise_share"] = result["franchise_store_count"] / base
    return result


def _auto_cluster_label(row: pd.Series, quantiles: dict[str, float]) -> str:
    if row.get("age_60_plus_share", 0) >= quantiles.get("age_60_plus_share", 1):
        label = "고령·전통형"
    elif (
        row.get("age_20_share", 0) >= quantiles.get("age_20_share", 1)
        and row.get("evening_share", 0) >= quantiles.get("evening_share", 1)
    ):
        label = "청년·야간형"
    elif (
        row.get("weekend_share", 0) >= quantiles.get("weekend_share", 1)
        and row.get("food_share", 0) >= quantiles.get("food_share", 1)
    ):
        label = "관광·주말외식형"
    elif row.get("time_11_14_share", 0) >= quantiles.get("time_11_14_share", 1):
        label = "점심·업무형"
    elif row.get("retail_share", 0) >= quantiles.get("retail_share", 1):
        label = "소매중심형"
    else:
        label = "생활·혼합형"
    return f"{label}-C{int(row.name)}"


def _cluster_districts(
    features: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    output = ensure_dir(output)
    cluster_columns = [
        "age_20_share",
        "age_30_share",
        "age_60_plus_share",
        "evening_share",
        "weekend_share",
        "food_share",
        "service_share",
        "retail_share",
        "store_density_per_10k_m2",
        "industry_hhi",
        "top_industry_share",
        "avg_ticket",
        "opening_intensity",
        "closing_intensity",
    ]
    available = [column for column in cluster_columns if column in features.columns]
    model_frame = features[available].replace([np.inf, -np.inf], np.nan)
    preprocessing = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    matrix = preprocessing.fit_transform(model_frame)
    k_min = int(config["seoul"].get("cluster_k_min", 4))
    k_max = int(config["seoul"].get("cluster_k_max", 8))
    seed = int(config["project"].get("random_seed", 42))
    score_rows: list[dict[str, float]] = []
    best_model: KMeans | None = None
    best_score = -1.0
    best_k = k_min
    for k in range(k_min, min(k_max, len(features) - 1) + 1):
        candidate = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = candidate.fit_predict(matrix)
        score = float(silhouette_score(matrix, labels)) if len(set(labels)) > 1 else -1.0
        score_rows.append({"k": k, "silhouette": score, "inertia": float(candidate.inertia_)})
        if score > best_score:
            best_score = score
            best_model = candidate
            best_k = k
    if best_model is None:
        raise RuntimeError("서울 상권 군집모델을 학습하지 못했습니다.")
    labels = best_model.predict(matrix)
    assignments = features[
        [
            "district_code",
            "district_name",
            "district_type_name",
            "sigungu_name",
            "latest_change_indicator",
        ]
    ].copy()
    assignments["cluster"] = labels

    standardized = pd.DataFrame(matrix, columns=available, index=features.index)
    standardized["cluster"] = labels
    standardized_profile = standardized.groupby("cluster", as_index=False)[available].mean()
    raw_profile = features.assign(cluster=labels).groupby("cluster", as_index=False)[available].mean()
    quantiles = {
        column: float(raw_profile[column].quantile(0.70))
        for column in [
            "age_60_plus_share",
            "age_20_share",
            "evening_share",
            "weekend_share",
            "food_share",
            "time_11_14_share",
            "retail_share",
        ]
        if column in raw_profile.columns
    }
    # _auto_cluster_label uses row.name as cluster id.
    label_lookup: dict[int, str] = {}
    for cluster_id, row in raw_profile.set_index("cluster").iterrows():
        row.name = cluster_id
        label_lookup[int(cluster_id)] = _auto_cluster_label(row, quantiles)
    assignments["cluster_label"] = assignments["cluster"].map(label_lookup)
    raw_profile["cluster_label"] = raw_profile["cluster"].map(label_lookup)
    standardized_profile["cluster_label"] = standardized_profile["cluster"].map(label_lookup)

    assignments.to_csv(output / "seoul_district_cluster_assignments.csv", index=False, encoding="utf-8-sig")
    raw_profile.to_csv(output / "seoul_cluster_profiles_raw.csv", index=False, encoding="utf-8-sig")
    standardized_profile.to_csv(
        output / "seoul_cluster_profiles_standardized.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(score_rows).to_csv(output / "cluster_k_scores.csv", index=False, encoding="utf-8-sig")
    save_cluster_profile_heatmap(
        standardized_profile.drop(columns=["cluster_label"]),
        output / "cluster_profile_heatmap.png",
    )
    return {
        "best_k": best_k,
        "silhouette": best_score,
        "cluster_rows": int(len(assignments)),
        "profile": raw_profile,
    }


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
        "source_model": "seoul_commercial",
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
    seed = int(config["project"].get("random_seed", 42))
    max_districts = int(config["seoul"].get("quick_max_districts", 500))
    sales = _aggregate_sales(paths["seoul_sales"], quick, max_districts, seed)
    selected = set(sales["district_code"].astype(str).unique()) if quick else None
    stores = _aggregate_stores(paths["seoul_stores"], selected)
    frame = sales.merge(
        stores,
        on=["quarter_code", "district_code"],
        how="left",
        validate="one_to_one",
    )

    area = pd.read_csv(paths["seoul_area"], low_memory=False)
    area = area.rename(
        columns={
            "TRDAR_CD": "district_code",
            "RELM_AR": "area_m2",
            "SIGNGU_CD_": "sigungu_name",
            "ADSTRD_CD_": "admdong_name",
        }
    )
    area["district_code"] = area["district_code"].astype(str)
    area = area[["district_code", "area_m2", "sigungu_name", "admdong_name"]].drop_duplicates(
        "district_code"
    )
    frame = frame.merge(area, on="district_code", how="left", validate="many_to_one")
    frame["area_m2"] = pd.to_numeric(frame["area_m2"], errors="coerce")
    frame["store_density_per_10k_m2"] = (
        10000.0 * frame["store_count"] / frame["area_m2"].replace(0, np.nan)
    )
    frame["area_log"] = np.log1p(frame["area_m2"])

    change = pd.read_csv(paths["seoul_change"], low_memory=False)
    change["district_code"] = change["district_code"].astype(str)
    latest_change = (
        change.sort_values("quarter_code")
        .groupby("district_code", as_index=False)
        .tail(1)[["district_code", "change_indicator_name"]]
        .rename(columns={"change_indicator_name": "latest_change_indicator"})
    )
    frame = frame.merge(latest_change, on="district_code", how="left", validate="many_to_one")
    frame["quarter_code"] = frame["quarter_code"].astype(int)
    frame["quarter_index"] = frame["quarter_code"].map(quarter_to_index)
    frame["year"] = frame["quarter_code"].astype(str).str[:4].astype(int)
    frame["quarter"] = frame["quarter_code"].astype(str).str[4].astype(int)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.to_csv(
        output / "seoul_district_quarter_features.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )

    frame = frame.sort_values(["district_code", "quarter_index"]).reset_index(drop=True)
    frame["next_quarter_index"] = frame.groupby("district_code")["quarter_index"].shift(-1)
    frame["next_quarter_code"] = frame.groupby("district_code")["quarter_code"].shift(-1)
    frame["next_log_sales"] = frame.groupby("district_code")["log_sales"].shift(-1)
    frame["target_next_quarter_log_growth"] = frame["next_log_sales"] - frame["log_sales"]
    frame["transition"] = (
        frame["quarter_code"].astype("Int64").astype(str)
        + "->"
        + frame["next_quarter_code"].astype("Int64").astype(str)
    )
    modeling = frame[
        (frame["next_quarter_index"] - frame["quarter_index"] == 1)
        & frame["target_next_quarter_log_growth"].notna()
    ].copy()
    excluded = set(config["seoul"].get("excluded_target_transitions", []))
    if excluded:
        modeling = modeling[~modeling["transition"].isin(excluded)].copy()

    feature_columns = [
        "year",
        "quarter",
        "district_type_name",
        "sigungu_name",
        "log_sales",
        "reported_sales_count",
        "avg_ticket",
        "weekday_share",
        "weekend_share",
        "time_00_06_share",
        "time_06_11_share",
        "time_11_14_share",
        "time_14_17_share",
        "time_17_21_share",
        "time_21_24_share",
        "evening_share",
        "male_share",
        "female_share",
        "age_10_share",
        "age_20_share",
        "age_30_share",
        "age_40_share",
        "age_50_share",
        "age_60_plus_share",
        "food_share",
        "service_share",
        "retail_share",
        "service_industry_count",
        "industry_hhi",
        "industry_entropy",
        "top_industry_share",
        "store_count",
        "similar_industry_store_count",
        "store_industry_count",
        "opening_intensity",
        "closing_intensity",
        "store_turnover_intensity",
        "franchise_share",
        "store_density_per_10k_m2",
        "area_log",
    ]
    feature_columns = [column for column in feature_columns if column in modeling.columns]
    categorical = [column for column in ["district_type_name", "sigungu_name"] if column in feature_columns]
    train, test, test_periods = chronological_split(
        modeling,
        "quarter_index",
        int(config["validation"].get("seoul_test_quarters", 4)),
    )
    model = fit_regressor(
        train,
        test,
        feature_columns,
        categorical,
        "target_next_quarter_log_growth",
        output / "next_quarter_growth_model",
        config,
        key_columns=["district_code", "district_name", "quarter_code", "next_quarter_code"],
        quick=quick,
    )
    save_feature_importance(model.importance, output / "next_quarter_growth_model" / "feature_importance.png")
    save_actual_vs_predicted(model.predictions, output / "next_quarter_growth_model" / "actual_vs_predicted.png")
    correlation = build_correlation_map(
        train,
        feature_columns,
        categorical,
        "target_next_quarter_log_growth",
        output / "next_quarter_growth_model" / "correlation",
        config,
        quick=quick,
    )
    ablation = run_feature_ablation(
        train,
        test,
        feature_columns,
        categorical,
        "target_next_quarter_log_growth",
        output / "next_quarter_growth_model" / "feature_ablation",
        config,
        model.metrics,
        quick=quick,
    )
    finalize_model_diagnostics(output / "next_quarter_growth_model")

    latest_count = int(config["seoul"].get("latest_quarters_for_clustering", 4))
    latest_periods = sorted(frame["quarter_index"].unique())[-latest_count:]
    latest = frame[frame["quarter_index"].isin(latest_periods)].copy()
    cluster_numeric = [
        column
        for column in feature_columns
        if column not in categorical + ["year", "quarter"]
    ]
    cluster_frame = (
        latest.groupby(
            [
                "district_code",
                "district_name",
                "district_type_name",
                "sigungu_name",
                "latest_change_indicator",
            ],
            dropna=False,
            as_index=False,
        )[cluster_numeric]
        .mean()
    )
    cluster_result = _cluster_districts(cluster_frame, config, output / "clustering")

    family_features = {
        "age_mix": [
            "age_10_share",
            "age_20_share",
            "age_30_share",
            "age_40_share",
            "age_50_share",
            "age_60_plus_share",
        ],
        "time_mix": [
            "time_00_06_share",
            "time_06_11_share",
            "time_11_14_share",
            "time_14_17_share",
            "time_17_21_share",
            "time_21_24_share",
            "evening_share",
        ],
        "industry_diversity": [
            "service_industry_count",
            "industry_hhi",
            "industry_entropy",
            "top_industry_share",
            "food_share",
            "service_share",
            "retail_share",
        ],
        "store_capacity": [
            "store_count",
            "similar_industry_store_count",
            "store_density_per_10k_m2",
            "area_log",
        ],
        "business_dynamics": [
            "opening_intensity",
            "closing_intensity",
            "store_turnover_intensity",
            "franchise_share",
        ],
        "weekend_mix": ["weekday_share", "weekend_share"],
        "district_type": ["district_type_name", "sigungu_name"],
    }
    representative = {
        "age_mix": "age_20_share",
        "time_mix": "evening_share",
        "industry_diversity": "industry_hhi",
        "store_capacity": "store_density_per_10k_m2",
        "business_dynamics": "store_turnover_intensity",
        "weekend_mix": "weekend_share",
    }
    stability_map: dict[str, float] = {}
    for family, feature in representative.items():
        if feature in modeling.columns:
            stability_map[family] = yearly_sign_stability(
                modeling,
                feature,
                "target_next_quarter_log_growth",
                "year",
                minimum_rows=50,
            )["stability"]
        else:
            stability_map[family] = 0.5
    stability_map["district_type"] = float(np.clip((cluster_result["silhouette"] + 1.0) / 2.0, 0, 1))
    coverage = min(1.0, frame["quarter_code"].nunique() / 20.0)
    mapping = {
        "age_mix": ("demographic", "visitor_age_share_by_band", "상권별 연령구성 구조"),
        "time_mix": ("time", "time_activity_mix", "상권별 시간대 매출구성"),
        "industry_diversity": ("industry", "industry_diversity_and_mix", "업종 다양성·집중도"),
        "store_capacity": ("capacity", "coupon_eligible_store_capacity", "점포수·밀도·수용력"),
        "business_dynamics": ("business_dynamics", "store_open_close_dynamics", "개폐업·프랜차이즈 구조"),
        "weekend_mix": ("calendar", "weekend_activity_share", "주중/주말 구성"),
        "district_type": ("segmentation", "pohang_district_type", "상권유형·지역 이질성"),
    }
    candidates: list[dict[str, Any]] = []
    for family, features in family_features.items():
        importance = importance_sum(model.importance, [f for f in features if f in feature_columns])
        feature_family, pohang_name, evidence = mapping[family]
        cluster_quality = float(np.clip((cluster_result["silhouette"] + 1.0) / 2.0, 0.0, 1.0))
        scaled_importance = min(1.0, importance * 2.5)
        if family == "district_type":
            scaled_importance = max(scaled_importance, cluster_quality)
        candidates.append(
            _candidate(
                family,
                feature_family,
                pohang_name,
                scaled_importance,
                stability_map.get(family, 0.5),
                coverage,
                f"서울 2021~2025 상권 next-quarter 성장 보조모델 및 {evidence}",
                "서울 값을 복사하지 않고 포항에서 동일 개념을 재계산",
                "서울 상권경계·시장구조·계수를 포항에 직접 이식하지 않음",
            )
        )
    candidate_frame = pd.DataFrame(candidates)
    growth_quality = float(np.clip((float(model.metrics.get("r2", 0.0)) + 0.1) / 1.1, 0.0, 1.0))
    cluster_quality = float(np.clip((cluster_result["silhouette"] + 1.0) / 2.0, 0.0, 1.0))
    candidate_frame["evidence_quality_score"] = 0.45 * growth_quality + 0.55 * cluster_quality
    candidate_frame.loc[candidate_frame["candidate_feature"] == "district_type", "evidence_quality_score"] = cluster_quality
    candidate_frame.to_csv(output / "feature_candidates.csv", index=False, encoding="utf-8-sig")

    summary = {
        "module": "seoul_commercial",
        "district_quarter_rows": int(len(frame)),
        "modeling_rows": int(len(modeling)),
        "district_count": int(frame["district_code"].nunique()),
        "test_quarter_indices": [int(value) for value in test_periods],
        "excluded_transitions": sorted(excluded),
        "model_metrics": model.metrics,
        "diagnostics": {"correlation": correlation, "ablation": ablation},
        "clustering": {
            "best_k": int(cluster_result["best_k"]),
            "silhouette": float(cluster_result["silhouette"]),
            "districts": int(cluster_result["cluster_rows"]),
        },
        "candidate_count": int(len(candidate_frame)),
        "interpretation_guardrail": "서울 모델은 상권 피처와 유형화 구조를 검증하는 donor이며 포항 매출·ROI 예측모델이 아니다.",
    }
    write_json(summary, output / "summary.json")
    return summary
