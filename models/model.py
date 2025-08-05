import torch.nn as nn
from torchvision import models

class ModelBuilder:
    def __init__(self, model_name, num_classes, pretrained=True, freeze_backbone=False):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model = None

    def _load_model(self):
        model_class = getattr(models, self.model_name)
        self.model = model_class(pretrained=self.pretrained)  #freeze_backbone 제거

    def _replace_classifier(self):
        if hasattr(self.model, 'fc'):  # ResNet 계열
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.model, 'classifier'):  # EfficientNet, MobileNet
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, self.num_classes)

    def _freeze_backbone(self):
        if self.freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False

    def build(self):
        self._load_model()
        self._replace_classifier()
        self._freeze_backbone()
        return self.model