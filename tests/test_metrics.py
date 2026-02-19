import numpy as np

from vision_project.metrics import confusion_matrix_np, per_class_accuracy


def test_confusion_matrix_shape_and_counts():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 0])
    cm = confusion_matrix_np(y_true, y_pred, num_classes=3)

    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)

    assert cm[0, 0] == 2
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1


def test_per_class_accuracy_values():
    cm = np.array(
        [
            [5, 0, 0],
            [1, 3, 0],
            [0, 2, 0],
        ]
    )

    acc, support = per_class_accuracy(cm)

    assert support.tolist() == [5, 4, 2]
    assert np.isclose(acc[0], 1.0)
    assert np.isclose(acc[1], 3 / 4)
    assert np.isclose(acc[2], 0.0)
