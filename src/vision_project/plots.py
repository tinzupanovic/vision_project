from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD_T = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def plot_training_curves(history, outdir: Path, prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_acc.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_names, outpath: Path, normalize=True):
    cm_plot = cm.astype(np.float64)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm_plot)
        np.divide(cm_plot, row_sums, out=cm_norm, where=row_sums != 0)
        cm_plot = cm_norm

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90, fontsize=7)
    plt.yticks(ticks, class_names, fontsize=7)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def unnormalize(img_batch: torch.Tensor) -> torch.Tensor:
    return (img_batch * IMAGENET_STD_T) + IMAGENET_MEAN_T


def save_error_gallery(images_norm, y_true, y_pred, class_names, outpath: Path, n=16):
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        return

    chosen = wrong_idx[:n]
    imgs = images_norm[chosen]
    imgs = unnormalize(imgs).clamp(0, 1)

    cols = 4
    rows = int(np.ceil(n / cols))
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 3 * rows))
    for i, idx in enumerate(chosen):
        plt.subplot(rows, cols, i + 1)
        img = imgs[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        t = class_names[y_true[idx]]
        p = class_names[y_pred[idx]]
        plt.title(f"T: {t}\nP: {p}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
