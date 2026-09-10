from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import ensure_dir, write_json


REQUIRED_COLUMNS = {
    "source_model",
    "candidate_feature",
    "feature_family",
    "pohang_feature_name",
    "importance_score",
    "stability_score",
    "coverage_score",
    "evidence",
    "transfer_rule",
    "caution",
    "donor_only",
}


def run(module_dirs: list[str | Path], config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    frames: list[pd.DataFrame] = []
    for module_dir in module_dirs:
        path = Path(module_dir) / "feature_candidates.csv"
        if not path.exists():
            raise FileNotFoundError(f"피처 후보 파일이 없습니다: {path}")
        frame = pd.read_csv(path, low_memory=False)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} 필수 컬럼 누락: {sorted(missing)}")
        frames.append(frame)
    detailed = pd.concat(frames, ignore_index=True)
    if "evidence_quality_score" not in detailed.columns:
        detailed["evidence_quality_score"] = 0.5
    for column in ["importance_score", "stability_score", "coverage_score", "evidence_quality_score"]:
        detailed[column] = pd.to_numeric(detailed[column], errors="coerce").fillna(0.0).clip(0, 1)
    weights = config["feature_gate"].get("weights", {})
    detailed["gate_score"] = (
        float(weights.get("importance", 0.45)) * detailed["importance_score"]
        + float(weights.get("stability", 0.35)) * detailed["stability_score"]
        + float(weights.get("coverage", 0.20)) * detailed["coverage_score"]
        + float(weights.get("evidence_quality", 0.15)) * detailed["evidence_quality_score"]
    )
    accept_threshold = float(config["feature_gate"].get("accept_threshold", 0.60))
    conditional_threshold = float(config["feature_gate"].get("conditional_threshold", 0.42))
    donor_only = detailed["donor_only"].astype(str).str.lower().isin({"1", "true", "yes"})
    detailed["decision"] = np.select(
        [
            donor_only,
            detailed["gate_score"] >= accept_threshold,
            detailed["gate_score"] >= conditional_threshold,
        ],
        ["DONOR_ONLY", "ACCEPT", "CONDITIONAL"],
        default="REJECT",
    )
    detailed = detailed.sort_values(
        ["decision", "gate_score"],
        ascending=[True, False],
        ignore_index=True,
    )
    detailed.to_csv(output / "donor_feature_gate_detailed.csv", index=False, encoding="utf-8-sig")

    aggregate_rows: list[dict[str, Any]] = []
    for pohang_feature, group in detailed.groupby("pohang_feature_name", dropna=False):
        decisions = set(group["decision"])
        if "ACCEPT" in decisions:
            final_decision = "ACCEPT"
        elif "CONDITIONAL" in decisions:
            final_decision = "CONDITIONAL"
        elif "DONOR_ONLY" in decisions:
            final_decision = "DONOR_ONLY"
        else:
            final_decision = "REJECT"
        source_count = int(group["source_model"].nunique())
        cross_source_bonus = min(0.10, 0.05 * max(0, source_count - 1))
        aggregate_rows.append(
            {
                "pohang_feature_name": pohang_feature,
                "final_decision": final_decision,
                "aggregate_score": float(min(1.0, group["gate_score"].mean() + cross_source_bonus)),
                "source_count": source_count,
                "source_models": " | ".join(sorted(group["source_model"].astype(str).unique())),
                "candidate_features": " | ".join(sorted(group["candidate_feature"].astype(str).unique())),
                "feature_families": " | ".join(sorted(group["feature_family"].astype(str).unique())),
                "transfer_rule": " / ".join(dict.fromkeys(group["transfer_rule"].astype(str))),
                "caution": " / ".join(dict.fromkeys(group["caution"].astype(str))),
                "evidence": " / ".join(dict.fromkeys(group["evidence"].astype(str))),
            }
        )
    whitelist = pd.DataFrame(aggregate_rows).sort_values(
        ["final_decision", "aggregate_score"], ascending=[True, False], ignore_index=True
    )
    whitelist.to_csv(output / "donor_feature_whitelist.csv", index=False, encoding="utf-8-sig")

    accepted = whitelist[whitelist["final_decision"].isin(["ACCEPT", "CONDITIONAL"])].copy()
    with (output / "DONOR_FEATURE_GATE_REPORT_KO.md").open("w", encoding="utf-8") as handle:
        handle.write("# 서울·경기 Donor Feature Gate 결과\n\n")
        handle.write(
            "이 결과는 포항의 최종 계수나 ROI가 아니다. 서울·경기에서 반복적으로 확인된 구조 중 "
            "포항에서 같은 개념으로 다시 계산할 피처 후보만 정리한다.\n\n"
        )
        handle.write(f"- 상세 후보: {len(detailed):,}개\n")
        handle.write(f"- 포항 대응 피처: {len(whitelist):,}개\n")
        handle.write(f"- ACCEPT/CONDITIONAL: {len(accepted):,}개\n\n")
        if not accepted.empty:
            columns = [
                "pohang_feature_name",
                "final_decision",
                "aggregate_score",
                "source_count",
                "source_models",
                "transfer_rule",
                "caution",
            ]
            handle.write(accepted[columns].to_markdown(index=False))
            handle.write("\n")
    summary = {
        "module": "feature_gate",
        "detailed_candidate_count": int(len(detailed)),
        "pohang_feature_count": int(len(whitelist)),
        "accepted_count": int((whitelist["final_decision"] == "ACCEPT").sum()),
        "conditional_count": int((whitelist["final_decision"] == "CONDITIONAL").sum()),
        "rejected_count": int((whitelist["final_decision"] == "REJECT").sum()),
        "guardrail": "피처 정의만 포항으로 이전하며 donor 계수·효과크기·군집경계는 이전하지 않는다.",
    }
    write_json(summary, output / "summary.json")
    return summary
