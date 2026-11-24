import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from model import SeqModel

T = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "seq_data.csv"), header=None)
print(df[0].value_counts())

labels = sorted(df[0].unique())
label_to_id = {l: i for i, l in enumerate(labels)}
x = df.iloc[:, 1:].values.astype(np.float32).reshape(-1, T, 65)
y = df[0].map(label_to_id).values
#Usual training split 80/20


x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
x_train, x_val = torch.tensor(x_train), torch.tensor(x_val)
y_train, y_val = torch.tensor(y_train), torch.tensor(y_val)

model = SeqModel(len(labels))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

for epoch in range(150):
    model.train()
    perm = torch.randperm(len(x_train))
    for i in range(0, len(perm), 32):
        idx = perm[i:i + 32]
        optimizer.zero_grad()
        loss = loss_function(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0 or epoch == 149:
        model.eval()
        with torch.no_grad():
            acc = (model(x_val).argmax(1) == y_val).float().mean()
        print(f"epoch {epoch}  val acc: {acc:.3f}")

model.eval()
with torch.no_grad():
    preds = model(x_val).argmax(1)
for i, l in enumerate(labels):
    mask = y_val == i
    print(l, f"{(preds[mask] == i).float().mean():.2f}")

torch.save({"state": model.state_dict(), "labels": labels},
           os.path.join(BASE_DIR, "models", "seq.pt"))
print("saved")