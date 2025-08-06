import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, Flowers102, Food101

def build_loss(cfg):
    """
    Config에 따라 Loss 객체 생성
    """
    if not hasattr(nn, cfg.LOSS_NAME):
        raise ValueError(f"Unsupported loss function: {cfg.LOSS_NAME}")

    loss_class = getattr(nn, cfg.LOSS_NAME)
    return loss_class(**cfg.LOSS_PARAMS)

def build_optimizer(cfg, model):
    """
    Config에 따라 Optimizer 선택
    """
    params = filter(lambda p: p.requires_grad, model.parameters())

    if cfg.OPTIMIZER.lower() == "sgd":
        return optim.SGD(params, lr=cfg.BASE_LR, momentum=cfg.MOMENTUM,
                         weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.lower() == "adam":
        return optim.Adam(params, lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.lower() == "adamw":
        return optim.AdamW(params, lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER}")

def build_scheduler(cfg, optimizer):
    """
    Config에 따라 scheduler 선택
    """
    scheduler_name = cfg.SCHEDULER_NAME.lower()
    print(f"Building scheduler: {cfg.SCHEDULER_NAME}")  # 디버깅 로그

    if scheduler_name == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.SCHEDULER_STEP_SIZE,
            gamma=cfg.SCHEDULER_GAMMA
        )
    elif scheduler_name == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SCHEDULER_TMAX,
            eta_min=cfg.SCHEDULER_ETA_MIN
        )
    elif scheduler_name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.SCHEDULER_FACTOR,
            patience=cfg.SCHEDULER_PATIENCE
        )
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.SCHEDULER_NAME}")

def evaluate_model(cfg, model, dataloader, return_loss=False):
    """
    validation or test data로 모델 평가 코드
    """
    model.eval()
    criterion = build_loss(cfg)
    correct, total = 0, 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            logits = model(images)
            if criterion:
                loss = criterion(logits, labels)
                running_loss += loss.item() * images.size(0)

            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    print("Validation Loss: {:.4f} Acc: {:.4f}".format(avg_loss, acc))

    if return_loss:
        return acc, avg_loss
    return acc

def create_experiment_path(cfg, root="results"):
    """
    모델 훈련 결과 및 테스트 결과 저장 경로 생성
    """
    date_dir = datetime.now().strftime("%Y%m%d")
    base_dir = cfg.ROOT_PATH / Path(root) / date_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    exp_num = 1
    while True:
        exp_dir = base_dir / (f"exp" if exp_num == 1 else f"exp{exp_num}")
        if not exp_dir.exists():
            exp_dir.mkdir()
            return exp_dir
        exp_num += 1

