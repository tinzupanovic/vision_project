from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_head(model):
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_layer4_and_head(model):
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
