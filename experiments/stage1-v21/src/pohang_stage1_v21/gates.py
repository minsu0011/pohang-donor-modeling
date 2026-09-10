from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def summarize_gate(
    baselines: pd.DataFrame,
    reentry: pd.DataFrame,
    proxy: pd.DataFrame,
    age: pd.DataFrame,
    leakage: pd.DataFrame,
    reference_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    gcfg = cfg.get("stage1_v21_gate", {})
    ok = baselines[baselines.status.eq("PASS")].copy()
    ok["backend_norm"] = ok.backend.astype(str).str.upper()
    cpu = ok[ok.backend_norm.eq("CPU")]
    if cpu.empty:
        raise RuntimeError("NO_CPU_BASELINE")

    stage2_fold = (
        ok.groupby(["backend_norm", "fold"], as_index=False)
        .delta_r2_vs_stage1.median()
    )
    cpu_med = float(cpu.delta_r2_vs_stage1.median())
    latest_fold = int(cpu.fold.max())
    latest = float(cpu[cpu.fold.eq(latest_fold)].r2.median())

    reentry = reentry.copy()
    reentry["backend_norm"] = reentry.backend.astype(str).str.upper()
    re_cat = (
        reentry[reentry.profile.eq("add_category")]
        .groupby("backend_norm").r2_gain_vs_strict.median().to_dict()
    )
    cpu_cat = float(re_cat.get("CPU", np.nan))
    gpu_cat = float(re_cat.get("GPU", np.nan))

    proxy = proxy.copy()
    proxy["backend_norm"] = proxy.backend.astype(str).str.upper()
    proxy_med = proxy.groupby("backend_norm").interaction_excess_over_identity.median().to_dict()

    age = age.copy()
    age["backend_norm"] = age.backend.astype(str).str.upper()
    age_stats = age.groupby("backend_norm").agg(
        median_drop=("r2_drop", "median"),
        positive_share=("positive", "mean"),
    ).to_dict("index")
    refs: dict[str, float] = {}
    for backend in ["CPU", "GPU"]:
        p = reference_dir / f"V2_GROUP_ABLATION_{backend}.csv"
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8-sig")
            x = df[df.unit.eq("age_industry_interaction")]
            if not x.empty:
                refs[backend] = float(x.median_r2_drop.iloc[0])

    age_checks = {}
    base_floor = float(gcfg.get("age_industry_min_delta_r2", 0.003))
    retention = float(gcfg.get("age_industry_reference_retention_fraction", 0.50))
    pos_min = float(gcfg.get("age_industry_positive_run_share_min", 0.75))
    for backend in ["CPU", "GPU"]:
        cur = age_stats.get(backend, {"median_drop": np.nan, "positive_share": 0.0})
        ref = refs.get(backend, base_floor)
        floor = max(base_floor, retention * ref)
        passed = bool(
            np.isfinite(cur["median_drop"])
            and float(cur["median_drop"]) >= floor
            and float(cur["positive_share"]) >= pos_min
        )
        age_checks[backend] = {
            "median_drop": float(cur["median_drop"]),
            "positive_share": float(cur["positive_share"]),
            "reference": float(ref),
            "required_floor": float(floor),
            "pass": passed,
        }

    leak_viol = int(
        pd.to_numeric(leakage.get("violations", pd.Series(dtype=float)), errors="coerce")
        .fillna(0).sum()
    ) if not leakage.empty else 10**9

    required_backends = {"CPU", "GPU"}
    fold_backends = set(stage2_fold.backend_norm.astype(str))
    all_fold_positive = (
        required_backends.issubset(fold_backends)
        and bool((stage2_fold.delta_r2_vs_stage1 > 0).all())
    )

    checks = {
        "G1_category_reentry_cpu_median_le_0_005": bool(
            np.isfinite(cpu_cat)
            and cpu_cat <= float(gcfg.get("category_reentry_cpu_max_median_delta_r2", 0.005))
        ),
        "G2_proxy_excess_cpu_gpu_nonnegative": bool(
            all(
                np.isfinite(proxy_med.get(b, np.nan))
                and float(proxy_med.get(b)) >= float(gcfg.get("proxy_interaction_excess_min", 0.0))
                for b in ["CPU", "GPU"]
            )
        ),
        "G3_stage2_gain_all_folds_all_backends_positive": all_fold_positive,
        "G4_cpu_median_stage2_gain_ge_0_01": bool(
            cpu_med >= float(gcfg.get("cpu_median_stage2_gain_min", 0.01))
        ),
        "G5_latest_cpu_fold_r2_ge_0_826": bool(
            latest >= float(gcfg.get("latest_cpu_fold_r2_min", 0.826))
        ),
        "G6_age_industry_maintained_cpu_gpu": bool(
            all(age_checks.get(b, {}).get("pass", False) for b in ["CPU", "GPU"])
        ),
        "G7_stage1_future_target_violations_zero": bool(
            leak_viol <= int(gcfg.get("future_target_violation_max", 0))
        ),
    }
    status = "PASS" if all(checks.values()) else "PARTIAL_PASS"
    return {
        "status": status,
        "checks": checks,
        "cpu_category_reentry_median_delta_r2": cpu_cat,
        "gpu_category_reentry_median_delta_r2": gpu_cat,
        "proxy_interaction_excess_median": {k: float(v) for k, v in proxy_med.items()},
        "cpu_median_stage2_gain": cpu_med,
        "latest_cpu_fold": latest_fold,
        "latest_cpu_fold_r2": latest,
        "age_industry": age_checks,
        "stage1_leakage_violations": leak_viol,
        "fold_stage2_gain": stage2_fold.to_dict("records"),
    }
