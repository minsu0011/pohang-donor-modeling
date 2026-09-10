from __future__ import annotations
import csv, hashlib, zipfile
from pathlib import Path
import pandas as pd


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def find_member(zf: zipfile.ZipFile, suffix: str) -> str:
    matches=[n for n in zf.namelist() if n.replace('\\','/').endswith(suffix)]
    if len(matches)!=1:
        raise RuntimeError(f"SPEND_MEMBER_MATCH_COUNT={len(matches)} suffix={suffix}")
    return matches[0]


def verify_manifest_tree(root: Path) -> dict:
    manifest = root / "PACKAGE_CONTENT_MANIFEST.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"CORE content manifest missing: {manifest}")
    checked = verified = 0
    missing: list[str] = []
    mismatched: list[dict] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rel = str(row["relative_path"]).replace("\\", "/")
            path = root / Path(rel)
            checked += 1
            if not path.is_file():
                missing.append(rel)
                continue
            actual_size = path.stat().st_size
            actual_sha = sha256_file(path)
            if actual_size != int(row["size_bytes"]) or actual_sha.lower() != str(row["sha256"]).lower():
                mismatched.append({
                    "relative_path": rel,
                    "expected_size": int(row["size_bytes"]),
                    "actual_size": actual_size,
                    "expected_sha256": str(row["sha256"]),
                    "actual_sha256": actual_sha,
                })
            else:
                verified += 1
    allowed_missing = [rel for rel in missing if "/docs/" in f"/{rel}" or "/links/" in f"/{rel}"]
    disallowed_missing = sorted(set(missing) - set(allowed_missing))
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
    }


def find_tree_member(root: Path, suffix: str) -> Path:
    normalized = suffix.replace("\\", "/")
    matches = [path for path in root.rglob(Path(normalized).name) if path.as_posix().endswith(normalized)]
    if len(matches) != 1:
        raise RuntimeError(f"SPEND_TREE_MATCH_COUNT={len(matches)} suffix={suffix}")
    return matches[0]


def load_spend_from_core(
    core_zip: Path,
    expected_sha: str,
    suffix: str,
    *,
    fallback_dir: Path | None = None,
    allow_manifest_fallback: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if core_zip.exists():
        actual=sha256_file(core_zip)
        if expected_sha and actual.lower()!=expected_sha.lower():
            raise RuntimeError(f"CORE_SHA256_MISMATCH expected={expected_sha} actual={actual}")
        with zipfile.ZipFile(core_zip) as zf:
            bad=zf.testzip()
            if bad:
                raise RuntimeError(f"CORE_ZIP_CRC_FAIL member={bad}")
            member=find_member(zf, suffix)
            with zf.open(member) as f:
                df=pd.read_csv(f)
        return df, {"input_verification_status":"PASS","core_source_mode":"ZIP","core_sha256":actual,"member":member,"rows":int(len(df))}
    if not allow_manifest_fallback or fallback_dir is None:
        raise FileNotFoundError(core_zip)
    fallback_dir = fallback_dir.resolve()
    verification = verify_manifest_tree(fallback_dir)
    if verification["status"] != "PASS_MODEL_INPUTS_WITH_PACKAGING_EXCEPTION":
        raise RuntimeError(f"CORE_MANIFEST_TREE_INVALID: {verification}")
    member_path = find_tree_member(fallback_dir, suffix)
    df = pd.read_csv(member_path)
    return df, {
        "input_verification_status": verification["status"],
        "core_source_mode": "MANIFEST_VERIFIED_EXTRACTED_TREE",
        "core_sha256": "CANONICAL_ZIP_UNAVAILABLE",
        "expected_core_sha256": expected_sha,
        "member": str(member_path),
        "member_sha256": sha256_file(member_path),
        "rows": int(len(df)),
        "core_manifest_verification": verification,
    }
