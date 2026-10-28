import pandas as pd 
import torch 
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "data.csv"), header=None)
print(df[0].value_counts().sort_index())


labels = sorted(df[0].unique())
label_to_id = {l: i for i,l in enumerate(labels)}

x = df.iloc[:, 1:].values.astype(np.float32)
y = df[0].map(label_to_id).values 

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

x_train, x_val = torch.tensor(x_train), torch.tensor(x_val)
y_train, y_val = torch.tensor(y_train), torch.tensor(y_val)


model = nn.Sequential (
    nn.Linear(63, 128),
    nn.ReLU(),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, len(labels))
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()


for epoch in range(100):
    model.train()
    perm = torch.randperm(len(x_train))
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]
        optimizer.zero_grad()
        loss = loss_function(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0 or epoch == 99:
        model.eval()
        with torch.no_grad():
            acc = (model(x_val).argmax(1) == y_val).float().mean()
        print(f"epoch {epoch}  val acc: {acc:.3f}")

model.eval()
with torch.no_grad():
    predictions = model(x_val).argmax(1)
for i, l in enumerate(labels):
    mask = y_val == i
    print(l, f"{(predictions[mask] == i).float().mean():.2f}")

os.makedirs("models", exist_ok=True)
torch.save({"state": model.state_dict(), "labels": labels}, "models/letters.pt")
print("saved")