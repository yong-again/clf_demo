# Image Classification Demo

이 프로젝트는 CIFAR-10 / Food-101 등의 이미지 데이터셋을 기반으로  
ResNet, EfficientNet 등 사전 학습(Pretrained) 모델을 활용하여  
5개 클래스 분류 모델을 학습하고 평가하는 구조로 되어 있습니다.

## 1. 프로젝트 구조

```angular2html
classification/
├── data/                   # 데이터셋 다운로드 경로
├── dataloader/             # 데이터 로더 모듈
│   ├── init.py
│   └── data_loader.py
│
├── models/                  # 모델 정의
│   ├── init.py
│   └── model.py
│
├── results/                 # 학습 결과 저장
│   └── YYYYMMDD/            # 날짜별
│       └── exp*/            # 실험별 (exp, exp2…)
│           ├── config.json
│           ├── best_model.pth
│           ├── logs/        # TensorBoard 로그
│           └── test_results.json
│
├── utils/                   # 유틸리티 함수
│    ├── device_check.py     # GPU/MPS/CPU 감지
│    └── util.py             # loss builder, optim builder, ... 등 포함
│
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
**참고** : pytorch version은 cuda버전 확인 후 설치 필요
```angular2html
nvcc --version # cuda version 확인
or
nvidia-smi
```
- 링크 : [이전버전](https://pytorch.org/get-started/locally/?__hstc=76629258.724dacd2270c1ae797f3a62ecd655d50.1746547368336.1746547368336.1746547368336.1&__hssc=76629258.9.1746547368336&__hsfp=2230748894)
- 링크 : [최신버전](https://pytorch.org/get-started/locally/?__hstc=76629258.724dacd2270c1ae797f3a62ecd655d50.1746547368336.1746547368336.1746547368336.1&__hssc=76629258.9.1746547368336&__hsfp=2230748894)
---
## 3. 학습 실행 및 테스트 실행
```bash
python run.py
```

- results/YYYYMMDD/exp/ 폴더가 생성되며:
  - config.json: 실험 설정 저장
  - best_model.pth: 최고 정확도 모델
  - logs/: TensorBoard 로그
  - test_results.json: 테스트 결과 (loss, accuracy, architecture, predictions)

---
## 4. Config 관리
- config.py에 Config 클래스 포함
- Optimizer, Scheduler, Loss 등 선택가능

예시:
```angular2html
cpu_count = multiprocessing.cpu_count() - 1
# ----------------------------
# Path Settings
# ----------------------------
self.ROOT_PATH = Path(__file__).resolve().parent
self.DATA_DIR = self.ROOT_PATH / "data"
self.MODEL_DIR = self.ROOT_PATH / "weights"

# ----------------------------
# Model Settings
# ----------------------------
self.MODEL_NAME = "resnet18" # 참조: https://docs.pytorch.org/vision/stable/models.html
self.PRETRAINED = True
self.NUM_CLASSES = 5
self.FREEZE_BACKBONE = False

# ----------------------------
# Optimizer Settings
# ----------------------------
self.OPTIMIZER = "Adam"  # "SGD", "AdamW" 등 선택 가능
self.BASE_LR = 1e-4
self.WEIGHT_DECAY = 1e-5
self.MOMENTUM = 0.9  # SGD 전용

# ----------------------------
# Scheduler
# ----------------------------
# Scheduler Settings
self.SCHEDULER_NAME = "StepLR"  # "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"
self.SCHEDULER_STEP_SIZE = 7
self.SCHEDULER_GAMMA = 0.1
self.SCHEDULER_TMAX = 10            # CosineAnnealingLR
self.SCHEDULER_ETA_MIN = 1e-6       # CosineAnnealingLR
self.SCHEDULER_FACTOR = 0.5         # ReduceLROnPlateau
self.SCHEDULER_PATIENCE = 3         # ReduceLROnPlateau

# ----------------------------
# Training Options
# ----------------------------
self.BATCH_SIZE = 64
self.NUM_WORKERS = cpu_count
self.NUM_EPOCHS = 30
self.DEVICE = get_device()
self.AMP_DTYPE = get_amp_dtye(self.DEVICE)
self.USE_AMP = True
self.GRADIENT_ACCUMULATION_STEPS = 1
self.SAVE_BEST_MODEL = True
self.EARLY_STOPPING_PATIENCE = 5

# ----------------------------
# Loss Function
# ----------------------------
self.LOSS_NAME = "CrossEntropyLoss"
self.LOSS_PARAMS = {"label_smoothing": 0.05}

# ----------------------------
# Checkpoint Path
# ----------------------------
self.CHECKPOINT_PATH = "best_model.pth"
```
### config 설명

1. Path Settings
- ROOT_PATH: 프로젝트 루트 경로
- DATA_DIR: 학습/테스트 데이터 저장 경로
- MODEL_DIR: 모델 가중치 파일(.pth) 저장 경로

2. Model Settings
- MODEL_NAME: 사용할 모델 이름 (PyTorch Vision에서 지원하는 ResNet, EfficientNet 등)
- PRETRAINED: 사전 학습된(pretrained) 가중치 사용 여부
- NUM_CLASSES: 분류할 클래스 개수
- FREEZE_BACKBONE: 백본(특징 추출기) 가중치 동결 여부

3.Optimizer Settings
- OPTIMIZER: 옵티마이저 종류 (SGD, Adam, AdamW 등)
- BASE_LR: 학습률
- WEIGHT_DECAY: 가중치 감쇠(L2 정규화)
- MOMENTUM: SGD 전용 모멘텀 값

4. Scheduler Settings
- SCHEDULER_NAME: 스케줄러 종류 (StepLR, CosineAnnealingLR, ReduceLROnPlateau)
- SCHEDULER_STEP_SIZE: StepLR에서 학습률 감소 주기
- SCHEDULER_GAMMA: 학습률 감소 비율
- SCHEDULER_TMAX: CosineAnnealingLR에서 한 주기의 최대 epoch 수
- SCHEDULER_ETA_MIN: CosineAnnealingLR에서 최소 학습률
- SCHEDULER_FACTOR: ReduceLROnPlateau에서 학습률 감소 비율
- SCHEDULER_PATIENCE: ReduceLROnPlateau에서 성능 개선이 없을 때 대기 epoch 수

5.Training Options
- BATCH_SIZE: 학습 시 한 배치에 사용할 이미지 개수
- NUM_WORKERS: DataLoader 병렬 로딩을 위한 CPU 프로세스 수
- NUM_EPOCHS: 전체 학습 반복 횟수
- DEVICE: 학습에 사용할 장치(GPU/CPU 자동 감지)
- AMP_DTYPE: AMP(Mixed Precision Training)에서 사용할 dtype
- USE_AMP: AMP 사용 여부
- GRADIENT_ACCUMULATION_STEPS: 그래디언트 누적 스텝 수
- SAVE_BEST_MODEL: 검증 성능이 최고일 때 모델을 저장할지 여부
- EARLY_STOPPING_PATIENCE: 성능 향상이 없을 때 학습을 조기 종료하는 patience

6. Loss Function
- LOSS_NAME: 손실 함수 이름 (CrossEntropyLoss, FocalLoss 등)
- LOSS_PARAMS: 손실 함수에 전달할 추가 파라미터 (예: 라벨 스무딩)

7.Checkpoint Path
- CHECKPOINT_PATH: 모델 가중치를 저장할 파일명

---
## 5. 결과 저장 구조
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

