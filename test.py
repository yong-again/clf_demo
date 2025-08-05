import torch
import json
from pathlib import Path
from tqdm import tqdm
from models import ModelBuilder
from utils import build_loss

def test_model(cfg, test_loader, exp_path):
    """
    Load Best model and run inference on test dataset
    Save predictions as JSON
    """
    # ------------------
    # model init and load weights
    # ------------------
    model_builder = ModelBuilder(
                                    model_name=cfg.MODEL_NAME,
                                    num_classes=cfg.NUM_CLASSES,
                                    pretrained=False,
                                    freeze_backbone=False
                                )
    model = model_builder.build().to(cfg.DEVICE)
    print(exp_path)

    model_path = exp_path / cfg.CHECKPOINT_PATH

    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()

    criterion = build_loss(cfg)

    predictions = {}
    running_loss, correct, total = 0.0, 0, 0

    # ---------------------
    # Test Loop
    # ---------------------
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
            images, labels =  images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            _, preds = torch.max(logits, 1)

            for i in range(len(preds)):
                predictions[f"sample_{batch_idx}_{i}"] = {
                    "true_label": labels[i].item(),
                    "pred_label": int(preds[i].item())
                }

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total

    results = {
        "loss" : avg_loss,
        "acc" : acc,
        "predictions" : predictions
    }

    json_path = Path(exp_path) / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f)

    print(f"✅ Test results saved to {json_path}")
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

