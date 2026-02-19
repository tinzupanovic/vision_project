import torch

from vision_project.data import build_transforms
from vision_project.models import build_resnet18


def test_eval_transform_output_shape():
    _, eval_tfms = build_transforms(img_size=224)
    assert eval_tfms is not None


def test_model_forward_shape_cpu():
    num_classes = 37
    model = build_resnet18(num_classes)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, num_classes)
