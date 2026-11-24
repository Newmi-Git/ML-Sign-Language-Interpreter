import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from model import WordModel

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "words.csv"), header=None)
print(df[0].value_counts().sort_index())

labels = [str(l) for l in sorted(df[0].unique())]
to_id = {l: i for i, l in enumerate(labels)}
y = df[0].astype(str).map(to_id).values
x = df.iloc[:, 2:].values.astype(np.float32).reshape(-1, 30, 138)
sessions = df[1].astype(str).values

# Split by session so the test clips come from sessions the model never trained on.
if len(set(sessions)) >= 3:
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(x, y, groups=sessions))
else:
    print("WARNING: fewer than 3 sessions, using a random split. Accuracy will look better than it is.")
    tr, va = train_test_split(np.arange(len(x)), test_size=0.2, stratify=y, random_state=42)

x_train, x_val = torch.tensor(x[tr]), torch.tensor(x[va])
y_train, y_val = torch.tensor(y[tr]), torch.tensor(y[va])

model = WordModel(len(labels))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

for epoch in range(150):
    model.train()
    perm = torch.randperm(len(x_train))
    for i in range(0, len(perm), 32):
        idx = perm[i:i + 32]
        xb = x_train[idx] + 0.01 * torch.randn_like(x_train[idx])   # small noise = cheap augmentation
        optimizer.zero_grad()
        loss = loss_function(model(xb), y_train[idx])
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
    if mask.sum() > 0:
        print(l, f"{(preds[mask] == i).float().mean():.2f}")

os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
torch.save({"state": model.state_dict(), "labels": labels}, os.path.join(HERE, "models", "words.pt"))
print("saved models/words.pt")