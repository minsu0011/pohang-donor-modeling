# 기준선과 잔차 보정

Donor core는 ZIP 입력을 준비하고 특징·모델 후보, ablation, 안정성 진단과 특징 선택을 연결합니다. 포항 확장은 지역·업종·월 패널에서 과거 특징을 만들고 여러 시간창의 기준선과 잔차 보정을 비교합니다.

`src/pohang_donor_models`는 독립 실행 패키지입니다. `experiments`의 동명 모듈은 실험별 구현이 다르므로 경로를 합쳐 사용하지 않습니다. 후속 실험 일부에 공통 모듈이 없어 전체 실행은 제공하지 않습니다.

[모델별 계산](wiki/Model-Evolution.md) · [실행](wiki/How-to-Run.md)
