from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def collect_predictions(model, loader, device, return_images=False):
    model.eval()
    all_true = []
    all_pred = []
    all_images = [] if return_images else None

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

        if return_images:
            all_images.append(images.cpu())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    if return_images:
        imgs = torch.cat(all_images, dim=0)
        return y_true, y_pred, imgs
    return y_true, y_pred


def confusion_matrix_np(y_true, y_pred, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm):
    correct = np.diag(cm)
    total = cm.sum(axis=1)
    acc = np.zeros_like(correct, dtype=np.float64)
    np.divide(correct, total, out=acc, where=total != 0)
    return acc, total
