import torch.nn as nn


def build_model(num_classes):
    return nn.Sequential(
        nn.Linear(63, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )


class SeqModel(nn.Module): #GRU is being used to get continious frame data
    def __init__(self, num_classes, in_dim=65, hidden=64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, num_classes)

    def forward (self, x):
        d, h = self.gru(x)
        return self.out(h[-1])