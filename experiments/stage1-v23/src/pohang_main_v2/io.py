from __future__ import annotations

import fnmatch
import csv
import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import ensure_dir, sha256_file, write_json


@dataclass(frozen=True)
class PreparedInputs:
    core_zip: Path
    donor_zip: Path
    integration_root: Path
    core_package_root: Path
    donor_reference_dir: Path
    core_sha256: str
    donor_sha256: str
    core_source_mode: str = "ZIP"
    core_manifest_verification: dict | None = None


def resolve_input(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    if any(ch in str(p) for ch in "*?["):
        parent = p.parent if p.parent.exists() else root
        matches = sorted(x for x in parent.iterdir() if fnmatch.fnmatch(x.name, p.name))
        if not matches:
            raise FileNotFoundError(f"Input glob did not match: {p}")
        return matches[0].resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p.resolve()


def extract_zip_cached(path: Path, cache_root: Path, force: bool = False) -> Path:
    digest = sha256_file(path)
    target = cache_root / f"{path.stem}_{digest[:12]}"
    marker = target / ".complete.json"
    if force and target.exists():
        shutil.rmtree(target)
    if marker.exists():
        return target
    if target.exists():
        shutil.rmtree(target)
    ensure_dir(target)
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {path.name} member={bad}")
        zf.extractall(target)
    write_json({"source": str(path), "sha256": digest}, marker)
    return target


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {root}")
    return matches[0]


def find_integration_root(extracted: Path) -> Path:
    matches = [p for p in extracted.rglob("pohang_coupon_integration_20260811") if p.is_dir()]
    if not matches:
        raise FileNotFoundError("pohang_coupon_integration_20260811 not found inside core ZIP")
    return sorted(matches, key=lambda p: len(str(p)))[0]


def find_core_package_root(integration_root: Path) -> Path:
    # integration_root/.../01_BASELINE_20260811/pohang_coupon_integration_20260811
    candidate = integration_root.parent.parent
    if (candidate / "02_NEW_INTEGRATED").exists():
        return candidate
    matches = list(integration_root.parents)
    for p in matches:
        if (p / "02_NEW_INTEGRATED").exists():
            return p
    raise FileNotFoundError("02_NEW_INTEGRATED not found relative to integration root")


def find_donor_reference_dir(extracted: Path) -> Path:
    matches = list(extracted.rglob("donor_feature_gate_v21.csv"))
    if not matches:
        raise FileNotFoundError("donor_feature_gate_v21.csv not found inside donor V2.1 handoff ZIP")
    return matches[0].parent


def verify_core_manifest_tree(core_root: Path) -> dict:
    manifest = core_root / "PACKAGE_CONTENT_MANIFEST.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"CORE content manifest missing: {manifest}")
    checked = 0
    verified = 0
    missing: list[str] = []
    mismatched: list[dict[str, str | int]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rel = str(row["relative_path"]).replace("\\", "/")
            path = core_root / Path(rel)
            checked += 1
            if not path.exists():
                missing.append(rel)
                continue
            actual_size = path.stat().st_size
            actual_sha = sha256_file(path)
            if actual_size != int(row["size_bytes"]) or actual_sha.lower() != str(row["sha256"]).lower():
                mismatched.append({
                    "relative_path": rel,
                    "expected_sha256": str(row["sha256"]),
                    "actual_sha256": actual_sha,
                    "expected_size": int(row["size_bytes"]),
                    "actual_size": actual_size,
                })
            else:
                verified += 1
    allowed_missing = [rel for rel in missing if "/docs/" in f"/{rel}" or "/links/" in f"/{rel}"]
    disallowed_missing = sorted(set(missing) - set(allowed_missing))
    mismatch_payload = "\n".join(
        f"{row['relative_path']}|{row['expected_size']}|{row['actual_size']}|{row['expected_sha256']}|{row['actual_sha256']}"
        for row in sorted(mismatched, key=lambda value: str(value["relative_path"]))
    )
    return {
        "status": "PASS_MODEL_INPUTS_WITH_PACKAGING_EXCEPTION" if not mismatched and not disallowed_missing else "FAIL",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_rows": checked,
        "verified_files": verified,
        "missing_files": missing,
        "allowed_missing_non_model_files": allowed_missing,
        "disallowed_missing_files": disallowed_missing,
        "mismatched_files": mismatched,
        "mismatch_identity": hashlib.sha256(mismatch_payload.encode("utf-8")).hexdigest(),
    }


def prepare_inputs(root: Path, cfg: dict, force: bool = False) -> PreparedInputs:
    donor_zip = resolve_input(root, cfg["inputs"]["donor_v21_zip"])
    cache = Path(cfg["inputs"]["extracted_dir"])
    if not cache.is_absolute():
        cache = root / cache
    cache = ensure_dir(cache)
    donor_extract = extract_zip_cached(donor_zip, cache, force=force)
    core_candidate = Path(cfg["inputs"]["core_zip"])
    if not core_candidate.is_absolute():
        core_candidate = root / core_candidate
    manifest_verification: dict | None = None
    if core_candidate.exists():
        core_zip = core_candidate.resolve()
        core_extract = extract_zip_cached(core_zip, cache, force=force)
        integration_root = find_integration_root(core_extract)
        core_sha256 = sha256_file(core_zip)
        core_source_mode = "ZIP"
    else:
        fallback_value = cfg["inputs"].get("core_extracted_dir")
        if not fallback_value or not bool(cfg["inputs"].get("allow_manifest_verified_core_fallback", False)):
            raise FileNotFoundError(core_candidate)
        fallback = Path(fallback_value)
        if not fallback.is_absolute():
            fallback = root / fallback
        fallback = fallback.resolve()
        manifest_verification = verify_core_manifest_tree(fallback)
        if manifest_verification["status"] != "PASS_MODEL_INPUTS_WITH_PACKAGING_EXCEPTION":
            raise RuntimeError(f"CORE_MANIFEST_TREE_INVALID: {manifest_verification}")
        core_zip = core_candidate.resolve()
        integration_root = find_integration_root(fallback)
        core_sha256 = "CANONICAL_ZIP_UNAVAILABLE"
        core_source_mode = "MANIFEST_VERIFIED_EXTRACTED_TREE"
    return PreparedInputs(
        core_zip=core_zip,
        donor_zip=donor_zip,
        integration_root=integration_root,
        core_package_root=find_core_package_root(integration_root),
        donor_reference_dir=find_donor_reference_dir(donor_extract),
        core_sha256=core_sha256,
        donor_sha256=sha256_file(donor_zip),
        core_source_mode=core_source_mode,
        core_manifest_verification=manifest_verification,
    )


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    errors: list[Exception] = []
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            errors.append(exc)
    if errors:
        raise errors[-1]
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_required(root: Path, relative_path: str, **kwargs) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return read_csv_auto(path, **kwargs)


def find_required(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def list_existing(root: Path, paths: Iterable[str]) -> list[Path]:
    return [root / p for p in paths if (root / p).exists()]
