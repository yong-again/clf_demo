import multiprocessing
from pathlib import Path
import torch
from utils.device_check import get_device, get_amp_dtye

class Config:
    def __init__(self):
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

    def make_dirs(self):
        """필요한 폴더 생성"""
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        #self.LOG_DIR.mkdir(parents=True, exist_ok=True)