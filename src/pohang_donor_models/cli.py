from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_config
from .diagnostics import aggregate_run_diagnostics
from .io import prepare_inputs
from .pipelines import feature_gate, gyeonggi_age, gyeonggi_flow, gyeonggi_time, seoul_commercial
from .utils import ensure_dir, set_seed, setup_logging, write_json


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()




def _resolve_archive(
    project_root: Path,
    explicit_value: str | None,
    configured_value: str,
    glob_pattern: str,
) -> Path:
    if explicit_value:
        return _resolve_path(project_root, explicit_value)
    configured = _resolve_path(project_root, configured_value)
    if configured.exists():
        return configured
    unpacked = configured.with_suffix("")
    if unpacked.is_dir():
        return unpacked.resolve()
    raw_dir = (project_root / "data" / "raw").resolve()
    matches = sorted(raw_dir.glob(glob_pattern)) if raw_dir.exists() else []
    if len(matches) == 1:
        return matches[0].resolve()
    return configured


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="포항 관광쿠폰 프로젝트용 서울·경기 donor 보조모델 파이프라인"
    )
    parser.add_argument(
        "command",
        choices=["verify", "gyeonggi-time", "gyeonggi-age", "gyeonggi-flow", "seoul", "gate", "all"],
        help="실행 단계",
    )
    parser.add_argument("--config", default="config/default.yaml", help="YAML 설정 파일")
    parser.add_argument("--core-zip", default=None, help="POHANG_HANDOFF_V2_CORE ZIP 또는 풀린 디렉터리 경로")
    parser.add_argument("--seoul-zip", default=None, help="POHANG_HANDOFF_V2_SEOUL_DONOR ZIP 또는 풀린 디렉터리 경로")
    parser.add_argument("--output-dir", default=None, help="결과 루트 폴더")
    parser.add_argument("--run-name", default=None, help="실행 폴더명")
    parser.add_argument("--quick", action="store_true", help="빠른 검증 모드")
    parser.add_argument("--force-extract", action="store_true", help="입력 추출본 덮어쓰기")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default=None, help="CatBoost 실행 장치")
    return parser


def _write_run_summary(run_dir: Path, summaries: dict[str, Any], quick: bool) -> None:
    with (run_dir / "RUN_SUMMARY_KO.md").open("w", encoding="utf-8") as handle:
        handle.write("# 서울·경기 Donor 보조모델 실행 요약\n\n")
        handle.write(f"- 실행 모드: {'QUICK' if quick else 'FULL'}\n")
        handle.write(f"- 실행 폴더: `{run_dir}`\n\n")
        handle.write("## 모듈 상태\n\n")
        for name, summary in summaries.items():
            handle.write(f"### {name}\n\n")
            handle.write("```json\n")
            handle.write(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
            handle.write("\n```\n\n")
        handle.write("## 해석 제한\n\n")
        handle.write(
            "이 실행은 서울·경기에서 피처 정의와 상호작용 구조를 검증한다. "
            "모델의 계수, peak 시간, 군집경계, 예측값을 포항 결과로 직접 사용하지 않는다. "
            "포항 최종 모델에는 `feature_gate/donor_feature_whitelist.csv`에서 통과한 개념 중 "
            "포항 직접자료로 재계산 가능한 피처만 입력한다.\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = Path.cwd().resolve()
    config_path = _resolve_path(project_root, args.config)
    config = load_config(config_path)
    if args.task_type:
        config["model"]["task_type"] = args.task_type
    config["project"]["quick"] = bool(args.quick)
    seed = int(config["project"].get("random_seed", 42))
    set_seed(seed)

    core_zip = _resolve_archive(
        project_root,
        args.core_zip,
        config["inputs"]["core_zip"],
        "POHANG_HANDOFF_V2_CORE_20260812*.zip",
    )
    seoul_zip = _resolve_archive(
        project_root,
        args.seoul_zip,
        config["inputs"]["seoul_zip"],
        "POHANG_HANDOFF_V2_SEOUL_DONOR_20260812*.zip",
    )
    extracted_dir = _resolve_path(project_root, config["inputs"]["extracted_dir"])
    output_root = _resolve_path(project_root, args.output_dir or "outputs")
    run_name = args.run_name
    if not run_name:
        base_name = str(config["project"].get("output_run_name", "donor_aux_run"))
        run_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(output_root / run_name)
    logger = setup_logging(run_dir / "run.log")
    logger.info("프로젝트 루트: %s", project_root)
    logger.info("CORE ZIP: %s", core_zip)
    logger.info("SEOUL ZIP: %s", seoul_zip)
    logger.info("출력: %s", run_dir)

    try:
        data_paths = prepare_inputs(
            core_zip,
            seoul_zip,
            extracted_dir,
            force=bool(args.force_extract),
        )
        logger.info("입력 추출 및 스키마 검증 완료")
        if args.command == "verify":
            write_json(
                {
                    "status": "PASS",
                    "input_manifest": str(data_paths.manifest_path),
                    "files": {key: str(value) for key, value in data_paths.files.items()},
                },
                run_dir / "verify_result.json",
            )
            print(f"PASS: {run_dir / 'verify_result.json'}")
            return 0

        summaries: dict[str, Any] = {}
        module_dirs = {
            "gyeonggi_time": run_dir / "01_gyeonggi_time",
            "gyeonggi_age": run_dir / "02_gyeonggi_age",
            "gyeonggi_flow": run_dir / "03_gyeonggi_flow",
            "seoul_commercial": run_dir / "04_seoul_commercial",
        }

        if args.command in {"gyeonggi-time", "all"}:
            logger.info("경기 시간대 보조모델 시작")
            summaries["gyeonggi_time"] = gyeonggi_time.run(
                data_paths, config, module_dirs["gyeonggi_time"], args.quick
            )
        if args.command in {"gyeonggi-age", "all"}:
            logger.info("경기 성연령×업종 보조모델 시작")
            summaries["gyeonggi_age"] = gyeonggi_age.run(
                data_paths, config, module_dirs["gyeonggi_age"], args.quick
            )
        if args.command in {"gyeonggi-flow", "all"}:
            logger.info("경기 공간관계 보조모델 시작")
            summaries["gyeonggi_flow"] = gyeonggi_flow.run(
                data_paths, config, module_dirs["gyeonggi_flow"], args.quick
            )
        if args.command in {"seoul", "all"}:
            logger.info("서울 상권 보조모델 시작")
            summaries["seoul_commercial"] = seoul_commercial.run(
                data_paths, config, module_dirs["seoul_commercial"], args.quick
            )
        if args.command in {"gate", "all"}:
            missing = [str(path) for path in module_dirs.values() if not (path / "feature_candidates.csv").exists()]
            if missing:
                raise FileNotFoundError(
                    "Feature Gate 실행 전 네 donor 모듈 결과가 모두 필요합니다. 누락: " + ", ".join(missing)
                )
            logger.info("Donor Feature Gate 시작")
            summaries["feature_gate"] = feature_gate.run(
                list(module_dirs.values()), config, run_dir / "05_feature_gate"
            )
        if args.command == "all":
            logger.info("상관관계 지도 및 전체 피처 이탈테스트 통합")
            summaries["model_diagnostics"] = aggregate_run_diagnostics(
                list(module_dirs.values()), run_dir / "06_model_diagnostics"
            )

        write_json(
            {
                "status": "PASS",
                "command": args.command,
                "quick": args.quick,
                "run_dir": str(run_dir),
                "input_manifest": str(data_paths.manifest_path),
                "summaries": summaries,
            },
            run_dir / "run_manifest.json",
        )
        _write_run_summary(run_dir, summaries, args.quick)
        (output_root / "LATEST_RUN.txt").write_text(str(run_dir), encoding="utf-8")
        logger.info("모든 요청 단계 완료")
        print(f"완료: {run_dir}")
        return 0
    except Exception as exc:
        logger.exception("실행 실패: %s", exc)
        write_json(
            {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
            run_dir / "FAILED.json",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
