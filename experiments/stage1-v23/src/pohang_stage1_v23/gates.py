from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def summarize_experiment_gate(selections: pd.DataFrame, baseline: pd.DataFrame, ablation_raw: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    scfg=cfg.get("stage1_v23_selection",{})
    feasible_folds=int(selections.selection_status.eq("FEASIBLE_SELECTED").sum()) if not selections.empty else 0
    total_folds=int(len(selections))
    all_candidates=int(pd.to_numeric(selections.candidate_count_evaluated,errors="coerce").eq(int(scfg.get("candidate_count_expected",15))).all()) if not selections.empty else 0
    outer_unused=int(pd.to_numeric(selections.outer_test_used_for_selection,errors="coerce").fillna(1).eq(0).all()) if not selections.empty else 0
    b=baseline[baseline.status.eq("PASS")].copy()
    if not b.empty:
        b["backend_norm"] = b.backend.astype(str).str.upper()
        gain=b.groupby(["backend_norm","fold"],as_index=False).delta_r2_vs_stage1.median().rename(columns={"delta_r2_vs_stage1":"gain"})
    else:
        gain=pd.DataFrame()
    fold2=gain[gain.fold.eq(2)] if not gain.empty else pd.DataFrame()
    inv=ablation_raw[ablation_raw.status.eq("PASS")] if not ablation_raw.empty else pd.DataFrame()
    invariant_ok=int((pd.to_numeric(inv.get("baseline_invariant_pass",pd.Series(dtype=float)),errors="coerce").fillna(0).eq(1).all() and pd.to_numeric(inv.get("metric_coherence_pass",pd.Series(dtype=float)),errors="coerce").fillna(0).eq(1).all())) if not inv.empty else 0
    checks={
        "E1_all_15_candidates_stage2_screened":bool(all_candidates),
        "E2_outer_test_never_used_for_stage1_selection":bool(outer_unused),
        "E3_at_least_one_feasible_stage1_fold":bool(feasible_folds>0),
        "E4_ablation_invariants_all_pass":bool(invariant_ok),
        "E5_stage2_group_and_individual_completed":bool(set(inv.kind.astype(str).unique()) >= {"group","individual"}) if not inv.empty else False,
    }
    status="PASS" if all(checks.values()) and feasible_folds==total_folds else "PARTIAL_PASS"
    return {
        "status":status,"checks":checks,"feasible_folds":feasible_folds,"total_folds":total_folds,
        "no_feasible_folds":selections.loc[selections.selection_status.eq("NO_FEASIBLE_STAGE1"),"fold"].astype(int).tolist() if not selections.empty else [],
        "fold_stage2_gain":gain.to_dict("records") if not gain.empty else [],
        "fold2_stage2_gain":fold2.to_dict("records") if not fold2.empty else [],
        "policy_use_allowed":0,
        "note":"This run is diagnostic. Fold-2 ablation is never used to alter the same outer-fold model.",
    }
