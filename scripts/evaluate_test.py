import argparse
import torch
import torch.nn as nn

from vision_project.config import TrainConfig, MODELS_DIR, FIGURES_DIR, TABLES_DIR
from vision_project.data import (
    build_transforms,
    load_datasets,
    make_splits,
    make_loaders,
)
from vision_project.models import build_resnet18
from vision_project.engine import evaluate
from vision_project.metrics import (
    collect_predictions,
    confusion_matrix_np,
    per_class_accuracy,
)
from vision_project.plots import plot_confusion_matrix
from vision_project.artifacts import save_per_class_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(MODELS_DIR / "best_resnet18_layer4.pth"),
        help="Path to checkpoint .pth",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tfms, eval_tfms = build_transforms()
    trainval_ds, trainval_ds_eval, test_ds = load_datasets(train_tfms, eval_tfms)
    train_ds, val_ds = make_splits(trainval_ds, trainval_ds_eval, cfg)
    _, _, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg)

    class_names = trainval_ds.classes
    num_classes = len(class_names)

    model = build_resnet18(num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"TEST loss: {test_loss:.4f}")
    print(f"TEST acc:  {test_acc:.4f}")

    y_true_test, y_pred_test = collect_predictions(
        model, test_loader, device, return_images=False
    )
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
