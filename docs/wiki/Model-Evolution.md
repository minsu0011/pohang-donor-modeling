# 모델별 역할과 데이터 경계

## 입력과 target

포항의 지역·업종·월 패널에서 로그 관광소비를 설명합니다. 과거 수준, 계절성, 연령구성, 기상 등 성격이 다른 특징을 구분합니다. Donor 자료는 서울·경기에서 특징 개념을 연구하는 용도이며 포항 관측행으로 합치지 않습니다.

설정은 끝난 집계창만 사용하도록 제한합니다. 업종별 과거 평균과 residual target을 만들 때도 그 fold에서 이용할 수 있는 history만 사용해야 합니다.

## Stage1과 Stage2를 나눈 이유

Stage1은 정상적인 업종·지역·계절 수준입니다. Stage2는 그 수준으로 설명되지 않는 잔차입니다. Raw category가 잔차 모델에서 다시 강하게 작동한다면 기준선의 역할이 충분했는지 먼저 점검해야 합니다.

| 구성 | 역할 | 과도하게 해석하면 안 되는 것 |
|---|---|---|
| Donor feature gate | 특징 개념의 우선순위 | 포항 성능 보증 |
| Dynamic / multi-window Stage1 | 시기별 기준선과 후보 선택 | 정책 인과 효과 |
| Individuality | 지역 간 차이·peer 비교 | 지역별 독립 정답 |
| V3 selector | 잔차 모델과 강도 선택 | 항상 보정해야 한다는 결론 |
| V4 Bias | 일정한 수준 차이 보정 | 특징 기반 구조 |
| V4 Structure | 중심화한 조건부 잔차 보정 | 새로운 절편 |
| Abstention | 불안정한 보정 생략 | 데이터가 정상이라는 판정 |

## V4의 계산

```text
최종 기대소비(log)
= 고정 Stage1 + 선택한 Bias + λ × 중심화한 Structure
```

Bias는 최근 잔차 평균 등의 간단한 후보입니다. Structure는 Ridge·ElasticNet·Huber·CatBoost를 비교합니다. 단순한 후보와 보정 없음이 복잡한 모델과 같은 경쟁에 들어갑니다.

기본 설정의 Bias gate는 inner gain의 중앙값, 양의 창 비율, 최악 gain과 drift를 함께 봅니다. Structure에는 correction 평균의 이동 제한도 있습니다. 단일 평균 성능만으로 모든 업종에 같은 보정을 적용하지 않습니다.

## 시간 경계

이미 선택된 Stage1 scope라도 선택 시점보다 과거의 residual을 만드는 데 소급 사용하면 미래 정보가 됩니다. V3/V4는 scope가 유효한 시점 이후에만 재생하고 더 이전 구간에는 사전 정의한 fallback을 사용하도록 설계했습니다.

코드는 [후속 실험](../../experiments), [donor 파이프라인](../../src/pohang_donor_models), [V4 설정](../../experiments/stage2-v4/config/default.yaml)에 나뉘어 있습니다. 각 실험의 같은 이름 모듈을 서로 바꿔 끼우지 않습니다.
