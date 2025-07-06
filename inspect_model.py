# inspect_model.py

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import PriceNet
import pickle

# 1) Загрузка и one-hot (как в train.py)
df = pd.read_csv("data/synthetic_tickets.csv")
df = pd.get_dummies(df, columns=["weekday","train_type","season"], prefix=["wd","tt","se"])
X = df.drop(columns=["price"]).values
y = df["price"].values

# 2) Разбиение на тест так же, как в train.py
_, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3) Масштабирование по сохранённому scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X_test_scaled = scaler.transform(X_test)

# 4) Загрузка модели
device = torch.device("cpu")
n_features = X_test_scaled.shape[1]
model = PriceNet(n_features).to(device)
state = torch.load("model_trained_final.pth", map_location=device)
model.load_state_dict(state)
model.eval()

# Показываем архитектуру
print("Модель:")
print(model)

# 5) Предсказание
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(X_tensor).numpy().flatten()

# 6) Метрики на тесте
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2  = r2_score(y_test, preds)
print(f"\nTest MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R²:  {r2:.3f}")

# 7) График: предсказанное vs фактическое
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Фактическая цена, ₽")
plt.ylabel("Предсказанная цена, ₽")
plt.title("Предсказанное vs Фактическое")
plt.tight_layout()
plt.show()
