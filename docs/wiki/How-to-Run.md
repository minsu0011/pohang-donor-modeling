# 코드 읽기와 실행

Donor core는 Python 3.10 이상 3.14 미만을 대상으로 합니다.

```bash
pip install -e .
pip install -r requirements-dev.txt
python -m pohang_donor_models.cli --help
python -m pytest tests -q
```

명령은 저장소 최상위에서 실행합니다. 입력은 [default.yaml](../../config/default.yaml)의 core/seoul ZIP과 맞춰 준비합니다. 데이터 준비와 feature gate는 [src/pohang_donor_models](../../src/pohang_donor_models)에 있습니다.

후속 실험을 읽을 때는 V2.1의 기준선 선택, V2.2의 multi-window와 orthogonal 특징, V2.3 gate, V3/V4의 보정 선택 순서가 좋습니다. [experiments](../../experiments)에 각 패키지가 분리돼 있습니다.

실험들에는 동일 이름의 모듈이 있지만 서로 같은 구현이라고 가정하면 안 됩니다. 특히 일부 공통 모듈과 individuality의 profiles/atlas가 없어 후속 실험 전체 실행은 제공하지 않습니다. 다른 버전의 파일로 대체하면 과거 결과와 다른 모델이 될 수 있습니다.

[전체 구조](../architecture.md)와 [검증 경계](../validation.md)를 함께 읽으면 donor core와 후속 실험의 역할을 구분할 수 있습니다.
