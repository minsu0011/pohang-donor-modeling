from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from pohang_main_v2.category_baseline import (
    fit_category_baseline,
    past_only_category_baseline,
    predict_category_baseline,
)
from pohang_main_v2.io import verify_core_manifest_tree
from pohang_main_v2.utils import atomic_write_json, sha256_file
from pohang_main_v2.config import load_config
from pohang_main_v2.feature_registry import active_features, build_feature_registry
from pohang_main_v2.residual_ablation import redundancy_units
from pohang_main_v2.residual_diagnostics import aggregate_oof_predictions


def _toy_panel() -> pd.DataFrame:
    rows = []
    for ym, a, b in [(202401, 1.0, 3.0), (202402, 2.0, 4.0), (202403, 2.5, 4.5)]:
        rows.extend(
            [
                {"district": "남구", "year_month": ym, "month": ym % 100, "category": "A", "category_major": "M1", "target_log_spend": a},
                {"district": "북구", "year_month": ym, "month": ym % 100, "category": "B", "category_major": "M2", "target_log_spend": b},
            ]
        )
    return pd.DataFrame(rows)


def test_past_only_baseline_does_not_read_same_month_target() -> None:
    original = _toy_panel()
    changed = original.copy()
    changed.loc[changed.year_month.eq(202402), "target_log_spend"] += 1000.0
    base_a = past_only_category_baseline(original, "category", period_col="year_month")
    base_b = past_only_category_baseline(changed, "category", period_col="year_month")
    feb = original.year_month.eq(202402)
    assert np.allclose(
        base_a.loc[feb, "category_baseline_log"],
        base_b.loc[feb, "category_baseline_log"],
        equal_nan=True,
    )
    # March is allowed to change because February is then part of history.
    march = original.year_month.eq(202403)
    assert not np.allclose(
        base_a.loc[march, "category_baseline_log"],
        base_b.loc[march, "category_baseline_log"],
        equal_nan=True,
    )


def test_frozen_category_baseline_ignores_target_frame_target() -> None:
    frame = _toy_panel()
    train = frame[frame.year_month.le(202402)]
    test_a = frame[frame.year_month.eq(202403)].copy()
    test_b = test_a.copy()
    test_b["target_log_spend"] += 9999.0
    state = fit_category_baseline(train, "category_month")
    pred_a = predict_category_baseline(state, test_a)
    pred_b = predict_category_baseline(state, test_b)
    assert np.allclose(pred_a.category_baseline_log, pred_b.category_baseline_log)


def test_category_keys_are_stage1_only_not_residual_predictors(tmp_path: Path) -> None:
    panel = _toy_panel()
    panel["district_signal"] = [1, 2, 1, 2, 1, 2]
    registry = build_feature_registry(
        {"main": panel, "screening": pd.DataFrame(), "site": pd.DataFrame(), "corridor": pd.DataFrame()},
        tmp_path,
    )
    residual, categorical, _ = active_features(registry, "main", "residual")
    structural, _, _ = active_features(registry, "main", "structural")
    assert "category" in structural and "category_major" in structural
    assert "category" not in residual and "category_major" not in residual
    assert "district" in residual and "district" in categorical
    roles = registry.set_index("column_name")["model_role"].to_dict()
    assert roles["category"] == "STAGE1_BASELINE_KEY"
    assert roles["category_major"] == "STAGE1_BASELINE_KEY"


def test_canonical_stage1_modes_do_not_absorb_district_identity() -> None:
    cfg = load_config("config/default.yaml")
    assert cfg["category_baseline"]["candidate_modes"] == ["category", "category_month"]
    assert "category_district" not in cfg["category_baseline"]["candidate_modes"]
    assert "category_district_month" not in cfg["category_baseline"]["candidate_modes"]


def test_redundancy_ablation_only_returns_true_multi_feature_clusters() -> None:
    features = ["a", "b", "c", "d"]
    pairs = pd.DataFrame(
        {
            "feature_a": ["a", "a"],
            "feature_b": ["b", "c"],
            "abs_association": [0.98, 0.10],
            "same_concept_group": [True, True],
            "status": ["WITHIN_CONCEPT_REDUNDANCY", "CROSS_CONCEPT_SIGNAL"],
        }
    )
    units = redundancy_units(features, pairs, 0.94)
    assert list(units.values()) == [["a", "b"]]
    assert all(len(columns) > 1 for columns in units.values())


def test_oof_aggregation_uses_canonical_cpu_and_preserves_all_backends(tmp_path: Path) -> None:
    base = {
        "district": "남구",
        "year_month": 202501,
        "category": "A",
        "category_major": "M1",
        "spend_thousand_krw": 100.0,
        "row_index": 1,
        "fold": 4,
        "actual_log_spend": np.log1p(100.0),
        "category_baseline_log": np.log1p(80.0),
        "residual_actual": 0.1,
        "residual_prediction": 0.05,
        "actual_minus_expected_log": -0.1,
        "expected_minus_actual_log": 0.1,
        "actual_minus_expected_thousand_krw": -10.0,
    }
    rows = []
    for backend, seed, expected in [("CPU", 42, 110.0), ("CPU", 137, 120.0), ("GPU", 42, 999.0)]:
        row = dict(base)
        row.update(
            {
                "backend": backend,
                "seed": seed,
                "expected_spend_thousand_krw": expected,
                "expected_log_spend": np.log1p(expected),
            }
        )
        rows.append(row)
    out = aggregate_oof_predictions(pd.DataFrame(rows), tmp_path, "PASS", canonical_backend="CPU")
    assert len(out) == 1
    assert np.isclose(out.loc[0, "expected_spend_thousand_krw"], 115.0)
    assert out.loc[0, "seed_count"] == 2
    assert out.loc[0, "canonical_backend"] == "CPU"
    assert np.isclose(out.loc[0, "expected_minus_actual_thousand_krw"], 15.0)
    assert (tmp_path / "OOF_EXPECTED_SPEND_V2_ALL_BACKENDS.csv.gz").exists()
    assert (tmp_path / "OOF_EXPECTED_SPEND_V2.csv.gz").exists()


def test_manifest_verified_core_fallback_is_strict(tmp_path: Path) -> None:
    data = tmp_path / "02_NEW_INTEGRATED" / "analysis.csv"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"model-input\n")
    manifest = tmp_path / "PACKAGE_CONTENT_MANIFEST.csv"
    manifest.write_text(
        "relative_path,size_bytes,sha256\n"
        f"02_NEW_INTEGRATED/analysis.csv,{data.stat().st_size},{sha256_file(data)}\n"
        "docs/guide.md,12,0000000000000000000000000000000000000000000000000000000000000000\n",
        encoding="utf-8-sig",
    )
    passed = verify_core_manifest_tree(tmp_path)
    assert passed["status"] == "PASS_MODEL_INPUTS_WITH_PACKAGING_EXCEPTION"
    assert passed["verified_files"] == 1
    assert passed["allowed_missing_non_model_files"] == ["docs/guide.md"]
    data.write_bytes(b"tampered\n")
    failed = verify_core_manifest_tree(tmp_path)
    assert failed["status"] == "FAIL"
    assert failed["mismatched_files"]


def test_atomic_json_writes_are_concurrency_safe(tmp_path: Path) -> None:
    path = tmp_path / "shared.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: atomic_write_json(path, {"value": value}), range(64)))
    payload = pd.read_json(path, typ="series")
    assert int(payload["value"]) in range(64)
    assert not list(tmp_path.glob("*.tmp"))
