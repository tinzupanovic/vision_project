import torch
import torch.nn as nn
import torch.optim as optim

from vision_project.io_utils import save_class_names

from vision_project.config import TrainConfig, MODELS_DIR, FIGURES_DIR, TABLES_DIR
from vision_project.data import (
    build_transforms,
    load_datasets,
    make_splits,
    make_loaders,
)
from vision_project.models import build_resnet18, freeze_all, unfreeze_head
from vision_project.engine import fit
from vision_project.metrics import (
    collect_predictions,
    confusion_matrix_np,
    per_class_accuracy,
)
from vision_project.plots import (
    plot_training_curves,
    plot_confusion_matrix,
    save_error_gallery,
)
from vision_project.artifacts import save_per_class_accuracy


def main():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tfms, eval_tfms = build_transforms()
    trainval_ds, trainval_ds_eval, test_ds = load_datasets(train_tfms, eval_tfms)
    train_ds, val_ds = make_splits(trainval_ds, trainval_ds_eval, cfg)
    train_loader, val_loader, _ = make_loaders(train_ds, val_ds, test_ds, cfg)

    class_names = trainval_ds.classes

    save_class_names(class_names, TABLES_DIR / "class_names.json")

    num_classes = len(class_names)

    model = build_resnet18(num_classes).to(device)

    freeze_all(model)
    unfreeze_head(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    ckpt_path = MODELS_DIR / "best_resnet18_head_only.pth"
    history, best_val_acc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=5,
        ckpt_path=ckpt_path,
        scheduler=None,
    )

    plot_training_curves(history, FIGURES_DIR, prefix="baseline_head_only")

    y_true_val, y_pred_val, val_imgs = collect_predictions(
        model, val_loader, device, return_images=True
    )
    cm_val = confusion_matrix_np(y_true_val, y_pred_val, num_classes)

    plot_confusion_matrix(
        cm_val,
        class_names,
        FIGURES_DIR / "confusion_matrix_val_baseline.png",
        normalize=True,
    )
    acc_val, sup_val = per_class_accuracy(cm_val)
    save_per_class_accuracy(
        class_names,
        acc_val,
        sup_val,
        TABLES_DIR / "per_class_accuracy_val_baseline.csv",
    )
    save_error_gallery(
        val_imgs,
        y_true_val,
        y_pred_val,
        class_names,
        FIGURES_DIR / "error_gallery_val_baseline.png",
        n=16,
    )

    print(f"Best val acc (baseline): {best_val_acc:.4f}")
    print(f"Saved baseline checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
