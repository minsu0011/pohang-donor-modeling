from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .plots import save_ablation_plot, save_correlation_heatmap
from .utils import ensure_dir, write_json


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    table = pd.crosstab(left[valid].astype(str), right[valid].astype(str))
    n = int(table.to_numpy().sum())
    if n == 0 or min(table.shape) < 2:
        return 0.0
    chi2 = float(chi2_contingency(table, correction=False)[0])
    phi2 = chi2 / n
    rows, columns = table.shape
    corrected_phi2 = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / max(n - 1, 1))
    corrected_rows = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    corrected_columns = columns - ((columns - 1) ** 2) / max(n - 1, 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return float(np.sqrt(corrected_phi2 / denominator)) if denominator > 0 else 0.0


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    valid = categories.notna() & values.notna()
    category = categories[valid].astype(str)
    numeric = pd.to_numeric(values[valid], errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    category = category.loc[finite]
    numeric = numeric.loc[finite]
    if numeric.empty or numeric.nunique() < 2 or category.nunique() < 2:
        return 0.0
    grand_mean = float(numeric.mean())
    denominator = float(((numeric - grand_mean) ** 2).sum())
    if denominator <= 0:
        return 0.0
    numerator = 0.0
    for _, group in numeric.groupby(category):
        numerator += len(group) * (float(group.mean()) - grand_mean) ** 2
    return float(np.sqrt(max(0.0, numerator / denominator)))


def _pair_association(
    left: pd.Series,
    right: pd.Series,
    left_categorical: bool,
    right_categorical: bool,
) -> tuple[float, str]:
    if left_categorical and right_categorical:
        return _cramers_v(left, right), "bias_corrected_cramers_v"
    if left_categorical:
        return _correlation_ratio(left, right), "correlation_ratio_eta"
    if right_categorical:
        return _correlation_ratio(right, left), "correlation_ratio_eta"
    numeric = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(numeric) < 3 or numeric.iloc[:, 0].nunique() < 2 or numeric.iloc[:, 1].nunique() < 2:
        return 0.0, "spearman"
    value = float(numeric.iloc[:, 0].corr(numeric.iloc[:, 1], method="spearman"))
    return (value if np.isfinite(value) else 0.0), "spearman"


def build_correlation_map(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    output_dir: str | Path,
    config: dict[str, Any],
    quick: bool = False,
) -> dict[str, Any]:
    """Create a mixed-type association map on training rows only."""
    output = ensure_dir(output_dir)
    columns = list(dict.fromkeys([*feature_columns, target_column]))
    selected = frame[columns].copy()
    configured_max = int(config["validation"].get("correlation_max_rows", 20000))
    quick_max = int(config["validation"].get("quick_correlation_max_rows", 5000))
    max_rows = min(configured_max, quick_max) if quick else configured_max
    sampled = len(selected) > max_rows
    if sampled:
        selected = selected.sample(
            n=max_rows,
            random_state=int(config["project"].get("random_seed", 42)),
        )
    categorical = set(categorical_columns)
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    pair_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(columns):
        for right_index in range(left_index + 1, len(columns)):
            right = columns[right_index]
            value, method = _pair_association(
                selected[left],
                selected[right],
                left in categorical,
                right in categorical,
            )
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
            pair_rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "association": value,
                    "absolute_association": abs(value),
                    "method": method,
                    "includes_target": target_column in {left, right},
                    "high_association": abs(value) >= 0.80,
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values(
        "absolute_association", ascending=False, ignore_index=True
    )
    matrix.to_csv(output / "correlation_map.csv", encoding="utf-8-sig")
    pairs.to_csv(output / "correlation_pairs.csv", index=False, encoding="utf-8-sig")
    save_correlation_heatmap(
        matrix,
        output / "correlation_heatmap.png",
        "Training-only feature association map",
    )
    metadata = {
        "scope": "training_rows_only",
        "source_rows": int(len(frame)),
        "rows_used": int(len(selected)),
        "sampled": sampled,
        "feature_count": int(len(feature_columns)),
        "target_column": target_column,
        "methods": {
            "numeric_numeric": "Spearman rank correlation (signed)",
            "categorical_categorical": "bias-corrected Cramer's V (unsigned)",
            "categorical_numeric": "correlation ratio eta (unsigned)",
        },
        "warning": "연관성은 인과효과가 아니며 donor 값이나 임계값을 포항으로 직접 이전하지 않는다.",
    }
    write_json(metadata, output / "metadata.json")
    return metadata


def finalize_model_diagnostics(model_dir: str | Path) -> None:
    model_path = Path(model_dir)
    ablation_path = model_path / "feature_ablation" / "feature_ablation.csv"
    if ablation_path.exists():
        frame = pd.read_csv(ablation_path, low_memory=False)
        save_ablation_plot(frame, model_path / "feature_ablation" / "feature_ablation_r2_drop.png")


def aggregate_run_diagnostics(
    module_dirs: list[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    ablation_frames: list[pd.DataFrame] = []
    correlation_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    for module_dir in map(Path, module_dirs):
        for path in sorted(module_dir.rglob("feature_ablation.csv")):
            frame = pd.read_csv(path, low_memory=False)
            model_name = path.parent.parent.name
            frame.insert(0, "source_module", module_dir.name)
            frame.insert(1, "model_name", model_name)
            ablation_frames.append(frame)
        for path in sorted(module_dir.rglob("correlation_pairs.csv")):
            frame = pd.read_csv(path, low_memory=False)
            model_name = path.parent.parent.name
            frame.insert(0, "source_module", module_dir.name)
            frame.insert(1, "model_name", model_name)
            correlation_frames.append(frame)
            inventory_rows.append(
                {
                    "source_module": module_dir.name,
                    "model_name": model_name,
                    "map_csv": str(path.parent / "correlation_map.csv"),
                    "heatmap_png": str(path.parent / "correlation_heatmap.png"),
                    "pair_count": int(len(frame)),
                    "high_association_pair_count": int(frame["high_association"].astype(str).str.lower().isin(["true", "1"]).sum()),
                }
            )
    all_ablation = pd.concat(ablation_frames, ignore_index=True) if ablation_frames else pd.DataFrame()
    all_pairs = pd.concat(correlation_frames, ignore_index=True) if correlation_frames else pd.DataFrame()
    inventory = pd.DataFrame(inventory_rows)
    all_ablation.to_csv(output / "all_feature_ablation.csv", index=False, encoding="utf-8-sig")
    all_pairs.to_csv(output / "all_correlation_pairs.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(output / "correlation_map_inventory.csv", index=False, encoding="utf-8-sig")
    failed = int((all_ablation.get("status", pd.Series(dtype=str)) == "FAILED").sum())
    summary = {
        "model_count": int(len(inventory)),
        "ablation_test_count": int(len(all_ablation)),
        "ablation_failed_count": failed,
        "correlation_pair_count": int(len(all_pairs)),
        "interpretation_guardrail": "상관·연관 및 이탈 성능차이는 인과효과가 아니며 포항에 donor 값을 직접 이전하지 않는다.",
    }
    write_json(summary, output / "summary.json")
    with (output / "MODEL_DIAGNOSTICS_REPORT_KO.md").open("w", encoding="utf-8") as handle:
        handle.write("# DONOR 모델 상관관계·전체 피처 이탈 진단\n\n")
        handle.write("상관관계 지도는 학습기간 행만 사용했다. 수치형 쌍은 Spearman, 범주형 쌍은 "
                     "Cramér's V, 혼합 쌍은 correlation ratio를 사용했다.\n\n")
        handle.write(f"- 진단 모델: {summary['model_count']}개\n")
        handle.write(f"- leave-one-feature-out 재학습: {summary['ablation_test_count']}건\n")
        handle.write(f"- 이탈 테스트 실패: {summary['ablation_failed_count']}건\n")
        handle.write(f"- 상관·연관 피처 쌍: {summary['correlation_pair_count']}건\n\n")
        handle.write("`r2_drop > 0` 또는 `rmse_increase > 0`은 해당 피처를 제거했을 때 동일한 "
                     "시간 홀드아웃 성능이 나빠졌다는 뜻이다. 포항 효과크기나 쿠폰 인과효과를 뜻하지 않는다.\n")
    return summary
