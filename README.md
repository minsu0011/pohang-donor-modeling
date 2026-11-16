# Pohang Donor Modeling

포항 지역·업종·월별 관광소비의 기준선을 만들고, 공통 패턴으로 설명되지 않는 잔차를 분석하는 연구입니다. 서울·경기 자료는 특징의 의미와 안정성을 검토하는 donor로 사용합니다.

## 모델

- **Stage1**: 업종·지역·계절과 과거 소비 수준으로 기대소비 기준선 계산
- **Stage2**: 잔차의 일정한 수준 차이인 Bias와 특징별 Structure 보정 분리
- **선택 기준**: 시간창별 검증에서 보정이 불안정하면 적용하지 않음

Python, pandas, NumPy, scikit-learn, CatBoost, SciPy와 PyYAML을 사용합니다. Ridge·ElasticNet·Huber는 잔차 보정의 비교 후보입니다.

## 실행

```bash
pip install -e .
pip install -r requirements-dev.txt
python -m pohang_donor_models.cli --help
python -m pytest tests -q
```

실제 계산에는 `config/default.yaml`에 맞는 입력 ZIP이 필요합니다. [Donor 패키지](src/pohang_donor_models), [설정](config), [후속 실험](experiments)을 구분해 읽을 수 있습니다. 일부 후속 실험은 별도 공통 모듈이 필요합니다.

## 결과

보존된 V4 기록의 최종 R² 중앙값은 약 0.8773이며, 최악 fold의 기준선 대비 변화는 약 −0.00285입니다. 모든 fold에서 개선된 것은 아닙니다. 소비 잔차는 지역 분석의 단서이며 정책의 인과 효과로 해석하지 않습니다.

[연구 과정](docs/wiki/Development-Journey.md) · [모델별 역할](docs/wiki/Model-Evolution.md) · [결과 해석](docs/wiki/Validation-and-Results.md) · [입력 데이터](data/README.md)

[출처 및 이용 조건](ATTRIBUTION.md)
