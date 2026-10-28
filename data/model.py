import torch.nn as nn


def build_model(num_classes):
    return nn.Sequential(
        nn.Linear(63, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )