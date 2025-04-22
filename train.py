# train.py — обучение нейросети для агента

import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# === Настройки ===
BUFFER_PATH = "demo_buffer.json"
MODEL_PATH = "agent_model.pt"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# === Загрузка данных ===
with open(BUFFER_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

action_to_idx = {"left": 0, "right": 1, "up": 2, "down": 3}
idx_to_action = ["left", "right", "up", "down"]

X = []
y = []
for record in data:
    flat = sum(record["state"], [])  # flatten 5x5 -> 25
    X.append(flat)
    y.append(action_to_idx[record["action"]])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Нормализация входов (GID могут быть большими)
X /= X.max()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Модель ===
class AgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 действия
        )

    def forward(self, x):
        return self.net(x)

model = AgentNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Обучение ===
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val)

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train_t.size()[0])
    for i in range(0, X_train_t.size()[0], BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        val_preds = val_outputs.argmax(dim=1)
        accuracy = (val_preds == y_val_t).float().mean().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {val_loss:.4f} | Acc: {accuracy:.2%}")

# === Сохранение модели ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"[OK] Модель сохранена в {MODEL_PATH}")