import pandas as pd 
import torch 
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv", header=None)
print(df[0].value_counts().sort_index())


labels = sorted(df[0].unique())
label_to_id = {l: i for i,l in enumerate(labels)}

x = df.iloc[:, :1].values.astype(np.float32)
y = df[0].map(label_to_id).values 

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

x_train, x_val = torch.tensor(x_train), torch.tensor(x_val)
y_train, y_val = torch.tensor(y_train), torch.tensor(y_val)


model = nn.Sequential (
    nn.Linear(63, 128),
    nn.RelU()

    nn.Linear(128, 63),
    nn.RelU()

    nn.Linear(128, len(labels))
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()


for epoch in range(1000):
    model.train()
    