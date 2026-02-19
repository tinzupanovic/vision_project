from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42
    val_ratio: float = 0.2


@dataclass(frozen=True)
class FineTuneConfig:
    lr_head: float = 1e-3
    lr_layer4: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    step_size: int = 3
    gamma: float = 0.1
