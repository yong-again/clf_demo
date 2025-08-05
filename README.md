# Image Classification Demo

이 프로젝트는 CIFAR-10 / Food-101 등의 이미지 데이터셋을 기반으로  
ResNet, EfficientNet 등 사전 학습(Pretrained) 모델을 활용하여  
5개 클래스 분류 모델을 학습하고 평가하는 구조로 되어 있습니다.

## 1. 프로젝트 구조

```angular2html
classification/
├── data/                   # 데이터셋 다운로드 경로
├── dataloader/              # 데이터 로더 모듈
│   ├── init.py
│   └── data_loader.py
├── models/                  # 모델 정의
│   ├── init.py
│   └── model.py
├── results/                 # 학습 결과 저장
│   └── YYYYMMDD/            # 날짜별
│       └── exp*/            # 실험별 (exp, exp2…)
│           ├── config.json
│           ├── best_model.pth
│           ├── logs/        # TensorBoard 로그
│           └── test_results.json
├── utils/                   # 유틸리티 함수
│   └── util.py
├── train.py                 # 학습 루프
├── test.py                  # 테스트 및 결과 저장
└── run.py                   # 메인 실행 스크립트

```

---

## 2.설치

- ### Requirements 설치

```bash
pip install -r requirements.txt
```
---
## 3. 학습 실행
```bash
python run.py
```

- results/YYYYMMDD/exp/ 폴더가 생성되며:
  - config.json: 실험 설정 저장
  - best_model.pth: 최고 정확도 모델
  - logs/: TensorBoard 로그
  - test_results.json: 테스트 결과 (loss, accuracy, predictions)

---
## 4. 테스트 실행
```bash
python test.py
```
---
## 5. Config 관리
- config.py에 Config 클래스 포함
- Optimizer, Scheduler, Loss 등 선택가능

예시:
```angular2html
cfg.MODEL_NAME = "resnet18"
cfg.OPTIMIZER = "AdamW"
cfg.SCHEDULER_NAME = "CosineAnnealingLR"
cfg.NUM_CLASSES = 5
```
---
## 6. 결과 저장 구조
```
results/
└── 20250805/
    └── exp/
        ├── config.json
        ├── best_model.pth
        ├── logs/
        └── test_results.json
```
---
## ⚡ 주요 기능
- ✅ Pretrained Model 기반 Fine-Tuning
- ✅ CIFAR-10 지원 (라벨 0~4로 자동 리매핑)
- ✅ Albumentations 기반 빠른 데이터 로딩
- ✅ Optimizer, Scheduler, Loss를 Config에서 선택 가능
- ✅ TensorBoard로 학습 곡선 및 입력 이미지 시각화
- ✅ 테스트 결과 JSON 저장 (실제 vs 예측 라벨)


