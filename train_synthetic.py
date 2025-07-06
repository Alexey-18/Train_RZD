import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
import pickle
import json

from model import PriceNet

print("🔁 Starting training pipeline…")

# 1) Load & one-hot encode
df = pd.read_csv("data/synthetic_tickets.csv")
df = pd.get_dummies(
    df,
    columns=["weekday", "train_type", "season"],
    prefix=["wd", "tt", "se"]
)

# 2) Prepare X and y
y = df["price"].values.reshape(-1, 1)
X = df.drop(columns=["price"]).values

# 3) Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Save scaler for predict.py
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved")

# 5) Train/val/test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1, random_state=42
)

# 6) To tensors
dtype = torch.float32
torch.manual_seed(42)
X_train_t = torch.tensor(X_train, dtype=dtype)
y_train_t = torch.tensor(y_train, dtype=dtype)
X_val_t   = torch.tensor(X_val,   dtype=dtype)
y_val_t   = torch.tensor(y_val,   dtype=dtype)
X_test_t  = torch.tensor(X_test,  dtype=dtype)
y_test_t  = torch.tensor(y_test,  dtype=dtype)

# Function for quick train & val (Grid Search)
def train_and_evaluate(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train_t.shape[1]
    # передаём hidden1 из params
    hidden1 = params["hidden1"]
    model = PriceNet(n_features, hidden1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # DataLoaders
    bs = params["batch_size"]
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_ds   = TensorDataset(X_val_t,   y_val_t)
    val_dl   = DataLoader(val_ds,   batch_size=bs)

    # Quick training
    epochs = params.get("epochs", 50)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Validation loss
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            total += criterion(model(xb), yb).item() * xb.size(0)
            count += xb.size(0)
    return total / count

# 7) Grid Search
param_grid = {
    "lr": [1e-1, 1e-2, 1e-3, 1e-4],
    "batch_size": [16, 32, 64], # после добавления оперативной памяти добавить 128 и 256
    # количество эпох для быстрой проверки
    "epochs": [30, 50, 100],
    # размер первого скрытого слоя fc1(все остальное будет стоиться у меня относительно него)
    "hidden1": [64, 128, 256],
}
best_loss = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    loss = train_and_evaluate(params)
    print(f"Params {params} -> Val MSE: {loss:.4f}")
    if loss < best_loss:
        best_loss = loss
        best_params = params

print("▶ Best params:", best_params)
with open("best.params.json", "w") as f:
  json.dump(best_params, f)
print("✅ Saved best_params.json")

#8) Full training with best hyperparameters, scheduler & early stopping

#Extracting the best parameters
bs = best_params["batch_size"]
lr = best_params["lr"]
hidden1 = best_params["hidden1"]
n_epochs = 200         # можно увеличить число эпох для финального обучения
patience = 20          # патience для early stopping

#Translate everything to the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = X_train_t.shape[1]
hidden1 = best_params["hidden1"]
model = PriceNet(n_features, hidden1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=50, eta_min=1e-5)

#Cooking DataLoader for a full train+val
train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_ds = TensorDataset(X_val_t, y_val_t)
val_dl = DataLoader(val_ds, batch_size=bs)

best_val = float("inf")
counter = 0

print("\n🚀 Starting full training…")
for epoch in range(1, n_epochs+1):
  #Train
  model.train()
  for xb, yb in train_dl:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    optimizer.step()
    
  #Validation
  model.eval()
  val_losses = []
  with torch.no_grad():
    for xb, yb in val_dl:
      xb, yb = xb.to(device), yb.to(device)
      val_losses.append(criterion(model(xb), yb).item())
  avg_val = sum(val_losses) / len(val_losses)
  scheduler.step()
  
  print(f"Epoch {epoch:03d} | Val Loss: {avg_val:.2f}")
  
  #Early stopping 
  if avg_val < best_val:
    best_val = avg_val
    counter = 0
    torch.save(model.state_dict(), "best_model.pth")
  else:
    counter += 1
    if counter >= patience:
      print(f"🔒 Early stopping at epoch {epoch}")
      break
    
# 9 Test evaluation 
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
with torch.no_grad():
  X_test_t = X_test_t.to(device)
  y_test_t = y_test_t.to(device)
  preds = model(X_test_t)
  test_mse = criterion(preds, y_test_t).item()
print(f"\n📉 Final Test MSE: {test_mse:.2f}")

# Сохраняем окончательную модель
torch.save(model.state_dict(), "model_trained_final.pth")
print("✅ Final model saved as model_trained_final.pth")