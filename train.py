import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from models import ModelBuilder
from utils import build_loss, build_optimizer, build_scheduler, evaluate_model
from torch.utils.tensorboard import SummaryWriter

def train_model(cfg, train_loader, val_loader, checkpoint_path):
    """
    Fine-tuning ResNet
    """
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    #------------
    # Build Model
    # ------------
    model_builder = ModelBuilder(
        model_name=cfg.MODEL_NAME,
        num_classes=cfg.NUM_CLASSES,
        pretrained=cfg.PRETRAINED,
        freeze_backbone=cfg.FREEZE_BACKBONE,
    )
    model = model_builder.build().to(cfg.DEVICE)

    # --------------------------
    # Loss, Optimizer, Scheduler
    # --------------------------
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = GradScaler()

    # ----------------
    # Training Loop
    # ----------------

    best_acc = 0
    patience_counter = 0

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{cfg.NUM_EPOCHS}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # tensorboard
        if epoch == 0:
            images, _ = next(iter(train_loader))
            img_grid = images[:16]
            writer.add_images("Train/Input Images", img_grid, global_step=0)

        for step, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            optimizer.zero_grad()

            with autocast(dtype=cfg.AMP_DTYPE, enabled=cfg.USE_AMP):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # -------------------
        # tensorboard Logging
        # -------------------
        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/Accuracy", epoch_acc, epoch)

        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Validation
        val_acc, val_loss = evaluate_model(cfg, model, val_loader, return_loss=True)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        if cfg.SCHEDULER_NAME.lower() == "reducelronplateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # save best model
        if cfg.SAVE_BEST_MODEL and val_acc > best_acc:
            torch.save(model.state_dict(), checkpoint_path)
            best_acc = val_acc
            patience_counter = 0
            print("✅ Best model saved.")

        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print("⏹️ Early stopping triggered.")

    writer.close()

    return model





