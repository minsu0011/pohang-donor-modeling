from __future__ import annotations

from pathlib import Path

import pandas as pd

from pohang_donor_models.diagnostics import build_correlation_map
from pohang_donor_models.modeling import run_feature_ablation


def _config() -> dict:
    return {
        "project": {"random_seed": 42},
        "model": {
            "backend": "sklearn",
            "learning_rate": 0.05,
            "l2_leaf_reg": 1.0,
        },
        "validation": {
            "max_permutation_rows": 100,
            "correlation_max_rows": 100,
            "quick_correlation_max_rows": 50,
        },
    }


def test_mixed_correlation_map_writes_all_outputs(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "numeric": range(20),
            "category": ["a"] * 10 + ["b"] * 10,
            "target": range(20),
        }
    )
    result = build_correlation_map(
        frame,
        ["numeric", "category"],
        ["category"],
        "target",
        tmp_path,
        _config(),
    )
    assert result["scope"] == "training_rows_only"
    assert (tmp_path / "correlation_map.csv").exists()
    assert (tmp_path / "correlation_pairs.csv").exists()
    matrix = pd.read_csv(tmp_path / "correlation_map.csv", index_col=0)
    assert matrix.loc["numeric", "target"] == 1.0


def test_feature_ablation_covers_every_feature(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": list(range(80)),
            "x2": [value % 4 for value in range(80)],
            "category": ["a", "b"] * 40,
        }
    )
    frame["target"] = frame["x1"] * 0.5 + frame["x2"]
    train = frame.iloc[:60].copy()
    test = frame.iloc[60:].copy()
    baseline = {"mae": 1.0, "rmse": 1.0, "r2": 0.0}
    result = run_feature_ablation(
        train,
        test,
        ["x1", "x2", "category"],
        ["category"],
        "target",
        tmp_path,
        _config(),
        baseline,
        quick=True,
    )
    assert result["tested_feature_count"] == 3
    output = pd.read_csv(tmp_path / "feature_ablation.csv")
    assert set(output["omitted_feature"]) == {"x1", "x2", "category"}
    assert set(output["status"]) == {"PASS"}
