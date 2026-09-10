from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import ensure_dir, sha256_file, write_json


REQUIRED_CORE_SUFFIXES = {
    "gyeonggi_time_industry": "02_NEW_INTEGRATED/gyeonggi_card_time/gyeonggi_card_time_state_month_time_industry.csv",
    "gyeonggi_time_city": "02_NEW_INTEGRATED/gyeonggi_card_time/gyeonggi_card_time_primary_city_month_time_total.csv",
    "gyeonggi_card_age_industry": "02_NEW_INTEGRATED/gyeonggi_cardage/gyeonggi_cardage_state_month_sexage_industry.csv",
    "gyeonggi_local_age_industry": "02_NEW_INTEGRATED/gyeonggi_localage/gyeonggi_localage_state_month_sexage_industry.csv",
    "gyeonggi_age_city_compare": "02_NEW_INTEGRATED/donor_harmonization/gyeonggi_age_share_primary_city_month_comparison_wide.csv",
    "gyeonggi_flow_city_pair": "02_NEW_INTEGRATED/gyeonggi_localflow/gyeonggi_localflow_primary_city_pair_month_total.csv",
    "gyeonggi_flow_relationship_industry": "02_NEW_INTEGRATED/gyeonggi_localflow/gyeonggi_localflow_state_month_relationship_major_category.csv",
    "quality_register": "03_METADATA/POHANG_QUALITY_ISSUE_REGISTER_V2.csv",
}

REQUIRED_SEOUL_SUFFIXES = {
    "seoul_sales": "integrated/seoul_sales_2021_2025_canonical.csv.gz",
    "seoul_stores": "integrated/seoul_stores_2021_2025_canonical.csv.gz",
    "seoul_area": "integrated/seoul_commercial_district_area_attributes.csv",
    "seoul_change": "integrated/seoul_commercial_district_change_indicator_canonical.csv",
}

EXPECTED_COLUMNS = {
    "gyeonggi_time_industry": {"std_ym", "tmzon_cd", "industry_name", "sales_amt_mean"},
    "gyeonggi_time_city": {"primary_city_name", "std_ym", "tmzon_cd", "sales_amt_mean"},
    "gyeonggi_card_age_industry": {"std_ym", "sex_age_cd", "mdclass_indutype_cd", "sales_amt_sum"},
    "gyeonggi_local_age_industry": {"std_ym", "sex_age_cd", "mdclass_indutype_cd", "sales_amt_sum"},
    "gyeonggi_flow_city_pair": {"base_primary_city_name", "inflow_primary_city_name", "std_ym", "sales_amt_sum"},
    "gyeonggi_flow_relationship_industry": {"std_ym", "relationship_class", "major_category", "sales_amt_sum"},
    "seoul_sales": {"quarter_code", "district_code", "reported_sales_amount", "age_20_sales_amount"},
    "seoul_stores": {"quarter_code", "district_code", "store_count"},
    "seoul_area": {"TRDAR_CD", "RELM_AR"},
    "seoul_change": {"quarter_code", "district_code", "change_indicator_name"},
}


@dataclass(frozen=True)
class DataPaths:
    files: dict[str, Path]
    manifest_path: Path

    def __getitem__(self, item: str) -> Path:
        return self.files[item]


def _find_member(names: Iterable[str], suffix: str) -> str:
    normalized_suffix = suffix.replace("\\", "/")
    matches = [name for name in names if name.replace("\\", "/").endswith(normalized_suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"ZIP 내부에서 필요한 파일을 정확히 하나 찾지 못했습니다: {suffix} / matches={matches}"
        )
    return matches[0]


def _extract_selected(zip_path: Path, mapping: dict[str, str], destination: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for logical_name, suffix in mapping.items():
            member = _find_member(names, suffix)
            extension = "".join(Path(member).suffixes)
            target = destination / f"{logical_name}{extension}"
            ensure_dir(target.parent)
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            outputs[logical_name] = target
    return outputs


def _windows_extended_path(path: Path) -> Path:
    absolute = path.absolute()
    if str(absolute).startswith("\\\\?\\"):
        return absolute
    return Path("\\\\?\\" + str(absolute))


def _select_from_directory(
    root: Path, mapping: dict[str, str], destination: Path
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    wrapper_dirs = [path for path in root.iterdir() if path.is_dir()]
    for logical_name, suffix in mapping.items():
        candidates = [root / suffix, *[wrapper / suffix for wrapper in wrapper_dirs]]
        matches = [
            _windows_extended_path(path) for path in candidates if _windows_extended_path(path).is_file()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"디렉터리에서 필요한 파일을 정확히 하나 찾지 못했습니다: "
                f"{suffix} / matches={matches}"
            )
        extension = "".join(Path(suffix).suffixes)
        target = destination / f"{logical_name}{extension}"
        ensure_dir(target.parent)
        with matches[0].open("rb") as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        outputs[logical_name] = target
    return outputs


def _source_metadata(path: Path) -> dict[str, str | None]:
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "zip",
        "sha256": None if path.is_dir() else sha256_file(path),
    }


def prepare_inputs(
    core_zip: str | Path,
    seoul_zip: str | Path,
    extracted_dir: str | Path,
    force: bool = False,
) -> DataPaths:
    core_path = Path(core_zip).expanduser().resolve()
    seoul_path = Path(seoul_zip).expanduser().resolve()
    if not core_path.exists():
        raise FileNotFoundError(f"CORE 입력(ZIP 또는 디렉터리)이 없습니다: {core_path}")
    if not seoul_path.exists():
        raise FileNotFoundError(f"서울 donor 입력(ZIP 또는 디렉터리)이 없습니다: {seoul_path}")
    if not (core_path.is_dir() or zipfile.is_zipfile(core_path)):
        raise ValueError(f"CORE 입력이 ZIP 또는 디렉터리가 아닙니다: {core_path}")
    if not (seoul_path.is_dir() or zipfile.is_zipfile(seoul_path)):
        raise ValueError(f"서울 donor 입력이 ZIP 또는 디렉터리가 아닙니다: {seoul_path}")

    destination = ensure_dir(extracted_dir)
    marker = destination / "input_manifest.json"
    if force and destination.exists():
        for child in destination.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    core_dir = ensure_dir(destination / "core")
    seoul_dir = ensure_dir(destination / "seoul")
    files = {}
    files.update(
        _select_from_directory(core_path, REQUIRED_CORE_SUFFIXES, core_dir)
        if core_path.is_dir()
        else _extract_selected(core_path, REQUIRED_CORE_SUFFIXES, core_dir)
    )
    files.update(
        _select_from_directory(seoul_path, REQUIRED_SEOUL_SUFFIXES, seoul_dir)
        if seoul_path.is_dir()
        else _extract_selected(seoul_path, REQUIRED_SEOUL_SUFFIXES, seoul_dir)
    )

    manifest = {
        "core_input": _source_metadata(core_path),
        "seoul_input": _source_metadata(seoul_path),
        "core_zip": str(core_path),
        "core_zip_sha256": None if core_path.is_dir() else sha256_file(core_path),
        "seoul_zip": str(seoul_path),
        "seoul_zip_sha256": None if seoul_path.is_dir() else sha256_file(seoul_path),
        "extracted_files": {
            key: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in sorted(files.items())
        },
    }
    write_json(manifest, marker)
    validate_inputs(files)
    return DataPaths(files=files, manifest_path=marker)


def validate_inputs(files: dict[str, Path]) -> None:
    failures: list[str] = []
    for logical_name, expected in EXPECTED_COLUMNS.items():
        if logical_name not in files:
            failures.append(f"누락: {logical_name}")
            continue
        try:
            columns = set(pd.read_csv(files[logical_name], nrows=0).columns)
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(f"읽기 실패: {logical_name}: {exc}")
            continue
        missing = expected - columns
        if missing:
            failures.append(f"컬럼 누락: {logical_name}: {sorted(missing)}")
    if failures:
        raise ValueError("입력 검증 실패:\n- " + "\n- ".join(failures))
