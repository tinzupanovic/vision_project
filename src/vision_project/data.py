from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import DATA_DIR, TrainConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def load_datasets(train_tfms, eval_tfms):
    trainval_ds = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        download=True,
        transform=train_tfms,
    )

    trainval_ds_eval = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        download=False,
        transform=eval_tfms,
    )

    test_ds = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        target_types="category",
        download=True,
        transform=eval_tfms,
    )
    return trainval_ds, trainval_ds_eval, test_ds


def make_splits(trainval_ds, trainval_ds_eval, cfg: TrainConfig):
    g = torch.Generator().manual_seed(cfg.seed)
    n_total = len(trainval_ds)
    n_val = int(n_total * cfg.val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(trainval_ds, [n_train, n_val], generator=g)

    val_ds.dataset = trainval_ds_eval
    return train_ds, val_ds


def make_loaders(train_ds, val_ds, test_ds, cfg: TrainConfig):
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
