import torch
import torch.nn as nn
import torch.optim as optim

from vision_project.io_utils import save_class_names

from vision_project.config import (
    TrainConfig,
    FineTuneConfig,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)
from vision_project.data import (
    build_transforms,
    load_datasets,
    make_splits,
    make_loaders,
)
from vision_project.models import build_resnet18, freeze_all, unfreeze_layer4_and_head
from vision_project.engine import fit, evaluate
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
    ft = FineTuneConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tfms, eval_tfms = build_transforms()
    trainval_ds, trainval_ds_eval, test_ds = load_datasets(train_tfms, eval_tfms)
    train_ds, val_ds = make_splits(trainval_ds, trainval_ds_eval, cfg)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg)

    class_names = trainval_ds.classes

    save_class_names(class_names, TABLES_DIR / "class_names.json")

    num_classes = len(class_names)

    model = build_resnet18(num_classes).to(device)

    freeze_all(model)
    unfreeze_layer4_and_head(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": ft.lr_layer4},
            {"params": model.fc.parameters(), "lr": ft.lr_head},
        ],
        weight_decay=ft.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=ft.step_size, gamma=ft.gamma
    )

    ckpt_path = MODELS_DIR / "best_resnet18_layer4.pth"
    history, best_val_acc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=ft.epochs,
        ckpt_path=ckpt_path,
        scheduler=scheduler,
    )

    plot_training_curves(history, FIGURES_DIR, prefix="layer4_finetune")

    y_true_val, y_pred_val, val_imgs = collect_predictions(
        model, val_loader, device, return_images=True
    )
    cm_val = confusion_matrix_np(y_true_val, y_pred_val, num_classes)
    plot_confusion_matrix(
        cm_val, class_names, FIGURES_DIR / "confusion_matrix_val.png", normalize=True
    )
    acc_val, sup_val = per_class_accuracy(cm_val)
    save_per_class_accuracy(
        class_names, acc_val, sup_val, TABLES_DIR / "per_class_accuracy_val.csv"
    )
    save_error_gallery(
        val_imgs,
        y_true_val,
        y_pred_val,
        class_names,
        FIGURES_DIR / "error_gallery_val.png",
        n=16,
    )

    y_true_test, y_pred_test = collect_predictions(
        model, test_loader, device, return_images=False
    )
    test_acc = (y_true_test == y_pred_test).mean()
    test_loss, _ = (0.0, 0.0)
    test_loss, _ = evaluate(model, test_loader, criterion, device)

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Final TEST acc: {test_acc:.4f}")
    print(f"Final TEST loss: {test_loss:.4f}")

    cm_test = confusion_matrix_np(y_true_test, y_pred_test, num_classes)
    plot_confusion_matrix(
        cm_test, class_names, FIGURES_DIR / "confusion_matrix_test.png", normalize=True
    )
    acc_test, sup_test = per_class_accuracy(cm_test)
    save_per_class_accuracy(
        class_names, acc_test, sup_test, TABLES_DIR / "per_class_accuracy_test.csv"
    )


if __name__ == "__main__":
    main()
