import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class RemappedDataset(Dataset):
    """
     CIFAR-10 Subset에서 라벨을 0~N-1로 매핑
     """

    def __init__(self, dataset, indices, class_to_idx, transform=None):
        self.samples = [(np.array(dataset[i][0]), class_to_idx[dataset[i][1]]) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

def get_dataloaders(cfg):
    # Albumentations transform
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ])

    # Dataset 로드
    train_dataset = datasets.CIFAR10(root=cfg.DATA_DIR, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=cfg.DATA_DIR, train=False, download=True)

    # 클래스별 샘플 개수 계산
    train_counts = Counter([label for _, label in train_dataset])
    selected_classes = [cls for cls, _ in train_counts.most_common(cfg.NUM_CLASSES)]
    print("Selected top classes:", selected_classes)

    class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}

    def filter_indices(dataset):
        return [i for i, (_, label) in enumerate(dataset) if label in selected_classes]

    train_indices = filter_indices(train_dataset)
    test_indices = filter_indices(test_dataset)

    # RemappedDataset 사용
    train_subset = RemappedDataset(train_dataset, train_indices, class_to_idx, transform=train_transform)
    test_subset = RemappedDataset(test_dataset, test_indices, class_to_idx, transform=test_transform)

    # 검증 세트 분리
    val_size = int(0.2 * len(train_subset))
    train_size = len(train_subset) - val_size
    train_subset, val_subset = random_split(train_subset, [train_size, val_size])

    # DataLoader 최적화
    loader_args = {
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": min(8, cfg.NUM_WORKERS),
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True
    }

    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_args)

    print(f"Train size: {len(train_loader.dataset)}, "
          f"Val size: {len(val_loader.dataset)}, "
          f"Test size: {len(test_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")

    return train_loader, val_loader, test_loader


class CustomDataset(Dataset):
    pass