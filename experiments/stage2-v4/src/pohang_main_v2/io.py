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
    core_source_mode: str
    core_verification: dict


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_extracted_core_tree(root: Path) -> dict:
    """Strictly verify a pre-extracted CORE tree when the canonical ZIP is unavailable.

    Only absent documentation and URL shortcut members are tolerated; every available
    model/data member must match both the manifest size and SHA256.
    """
    manifest = root / "PACKAGE_CONTENT_MANIFEST.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"CORE manifest missing: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"relative_path", "size_bytes", "sha256"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("CORE manifest schema is invalid")
    missing, allowed_missing, disallowed_missing, mismatched = [], [], [], []
    verified = 0
    for row in rows:
        relative = str(row["relative_path"]).replace("\\", "/")
        target = root.joinpath(*relative.split("/"))
        if not target.is_file():
            missing.append(relative)
            non_model = target.suffix.lower() in {".md", ".url"} and (
                "/docs/" in f"/{relative}" or "/links/" in f"/{relative}"
            )
            (allowed_missing if non_model else disallowed_missing).append(relative)
            continue
        actual_size, actual_sha = target.stat().st_size, _sha256_path(target)
        expected_size, expected_sha = int(row["size_bytes"]), str(row["sha256"]).lower()
        if actual_size != expected_size or actual_sha.lower() != expected_sha:
            mismatched.append({
                "relative_path": relative,
                "expected_size": expected_size,
                "actual_size": actual_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            })
        else:
            verified += 1
    status = "PASS_MODEL_INPUTS_WITH_PACKAGING_EXCEPTION" if not disallowed_missing and not mismatched else "FAIL"
    result = {
        "status": status,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256_path(manifest),
        "manifest_rows": len(rows),
        "verified_files": verified,
        "missing_files": missing,
        "allowed_missing_non_model_files": allowed_missing,
        "disallowed_missing_files": disallowed_missing,
        "mismatched_files": mismatched,
    }
    if status == "FAIL":
        raise RuntimeError(f"CORE_EXTRACTED_TREE_VERIFICATION_FAILED: {result}")
    return result


def prepare_inputs(root: Path, cfg: dict, force: bool = False) -> PreparedInputs:
    configured_core = Path(cfg["inputs"]["core_zip"])
    core_zip = configured_core if configured_core.is_absolute() else root / configured_core
    donor_zip = resolve_input(root, cfg["inputs"]["donor_v21_zip"])
    cache = Path(cfg["inputs"]["extracted_dir"])
    if not cache.is_absolute():
        cache = root / cache
    cache = ensure_dir(cache)
    donor_extract = extract_zip_cached(donor_zip, cache, force=force)
    if core_zip.is_file():
        core_zip = core_zip.resolve()
        core_extract = extract_zip_cached(core_zip, cache, force=force)
        integration_root = find_integration_root(core_extract)
        core_package_root = find_core_package_root(integration_root)
        core_sha256 = sha256_file(core_zip)
        core_source_mode = "CANONICAL_ZIP"
        core_verification = {"status": "PASS", "zip_crc": "PASS"}
    else:
        if not bool(cfg["inputs"].get("allow_manifest_verified_core_fallback", False)):
            raise FileNotFoundError(core_zip)
        configured_tree = Path(cfg["inputs"]["core_extracted_dir"])
        tree = configured_tree if configured_tree.is_absolute() else root / configured_tree
        tree = tree.resolve()
        core_verification = verify_extracted_core_tree(tree)
        integration_root = find_integration_root(tree)
        core_package_root = find_core_package_root(integration_root)
        core_sha256 = "CANONICAL_ZIP_UNAVAILABLE"
        core_source_mode = "MANIFEST_VERIFIED_EXTRACTED_TREE"
    return PreparedInputs(
        core_zip=core_zip,
        donor_zip=donor_zip,
        integration_root=integration_root,
        core_package_root=core_package_root,
        donor_reference_dir=find_donor_reference_dir(donor_extract),
        core_sha256=core_sha256,
        donor_sha256=sha256_file(donor_zip),
        core_source_mode=core_source_mode,
        core_verification=core_verification,
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
