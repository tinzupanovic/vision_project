from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from vision_project.config import MODELS_DIR, TABLES_DIR
from vision_project.data import build_transforms
from vision_project.io_utils import load_class_names
from vision_project.models import build_resnet18


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict pet breed (Oxford-IIIT Pets) from an image."
    )
    p.add_argument(
        "--image", type=str, required=True, help="Path to input image (jpg/png)."
    )
    p.add_argument(
        "--topk", type=int, default=3, help="Number of top predictions to show."
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(MODELS_DIR / "best_resnet18_layer4.pth"),
        help="Path to checkpoint .pth (default: models/best_resnet18_layer4.pth)",
    )
    p.add_argument(
        "--classes",
        type=str,
        default=str(TABLES_DIR / "class_names.json"),
        help="Path to class names JSON (default: reports/tables/class_names.json)",
    )
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    image_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    classes_path = Path(args.classes)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Class names JSON not found: {classes_path}\n"
            "Run the training script once to generate it (reports/tables/class_names.json)."
        )

    class_names = load_class_names(classes_path)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet18(num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, eval_tfms = build_transforms(img_size=224)

    img = Image.open(image_path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1)

    topk = max(1, min(args.topk, num_classes))
    top_probs, top_idx = torch.topk(probs, k=topk, dim=1)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_idx = top_idx.squeeze(0).cpu().tolist()

    print(f"Image: {image_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Top-{topk} predictions:")
    for rank, (p, idx) in enumerate(zip(top_probs, top_idx), start=1):
        print(f"{rank:>2}. {class_names[idx]:<25}  prob={p:.4f}")


if __name__ == "__main__":
    main()
