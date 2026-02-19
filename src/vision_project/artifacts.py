from __future__ import annotations
from pathlib import Path
import pandas as pd


def save_per_class_accuracy(class_names, acc, support, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"class": class_names, "accuracy": acc, "support": support}
    ).sort_values("accuracy")
    df.to_csv(outpath, index=False)
    return df
