from __future__ import annotations

from pathlib import Path

import pandas as pd

from pohang_donor_models.pipelines.feature_gate import run


def test_feature_gate_aggregates_cross_source(tmp_path: Path) -> None:
    columns = {
        "candidate_feature": ["a"],
        "feature_family": ["x"],
        "pohang_feature_name": ["shared"],
        "importance_score": [0.8],
        "stability_score": [0.8],
        "coverage_score": [1.0],
        "evidence": ["e"],
        "transfer_rule": ["t"],
        "caution": ["c"],
        "donor_only": [False],
    }
    module_dirs = []
    for index, source in enumerate(["s1", "s2"]):
        directory = tmp_path / source
        directory.mkdir()
        frame = pd.DataFrame(columns)
        frame["source_model"] = source
        frame.to_csv(directory / "feature_candidates.csv", index=False)
        module_dirs.append(directory)
    config = {
        "feature_gate": {
            "accept_threshold": 0.6,
            "conditional_threshold": 0.42,
            "weights": {"importance": 0.35, "stability": 0.30, "coverage": 0.2, "evidence_quality": 0.15},
        }
    }
    result = run(module_dirs, config, tmp_path / "output")
    assert result["accepted_count"] == 1
    whitelist = pd.read_csv(tmp_path / "output" / "donor_feature_whitelist.csv")
    assert whitelist.loc[0, "source_count"] == 2
