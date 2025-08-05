import json
from config import Config
from train import train_model
from test import test_model
from utils import create_experiment_path
from dataloader import get_dataloaders

def main():
    # ---------------
    # config setting
    # ----------------
    cfg = Config()

    exp_path = create_experiment_path(cfg, root="results")
    cfg.MODEL_DIR = exp_path
    cfg.LOG_DIR = exp_path / "logs"
    cfg.CHECKPOINT_DIR = exp_path / "best_model.pth"

    cfg.make_dirs()

    # ---------------
    # DataLoader
    # ---------------
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # --------------
    # Save Config
    # --------------
    config_file = exp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(cfg.__dict__, f, indent=4, default=str)

    # ------------------
    # Train
    # ------------------
    train_model(cfg, train_loader, val_loader, cfg.CHECKPOINT_DIR)

    # ------------------
    # Test
    # ------------------
    test_model(cfg, test_loader, exp_path)

    print(f"✅ Experiment completed. Results saved in {exp_path}")

if __name__ == "__main__":
    main()

