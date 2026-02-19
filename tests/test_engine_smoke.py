import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from vision_project.engine import train_one_epoch, evaluate


def test_train_one_epoch_smoke():
    device = torch.device("cpu")

    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 5, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 5)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    train_loss, train_acc = train_one_epoch(model, loader, criterion, optimizer, device)
    _, val_acc = evaluate(model, loader, criterion, device)

    assert isinstance(train_loss, float) or hasattr(train_loss, "__float__")
    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= val_acc <= 1.0
