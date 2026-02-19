from __future__ import annotations

import json
from pathlib import Path
from typing import List


def save_class_names(class_names: List[str], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)


def load_class_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
