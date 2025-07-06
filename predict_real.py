import json
import pickle
import pandas as pd
import torch
from pathlib import Path
from sklearn.impute import SimpleImputer
from model import PriceNet

# 1) Пути к артефактам
MODEL_FILE    = "model_real_final.pth"
SCALER_FILE   = "scaler_real.pkl"
PARAMS_FILE   = "best_params_real.json"
FEATURES_FILE = "feature_names_real.json"

# 2) Загрузка scaler, параметров и списка признаков
scaler        = pickle.load(open(SCALER_FILE, "rb"))
best_params   = json.load(open(PARAMS_FILE,   "r"))
feature_names = json.load(open(FEATURES_FILE,"r"))
hidden1       = best_params["hidden1"]
n_features    = len(feature_names)

# 3) Инициализация и загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = PriceNet(n_features, hidden1).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()
print(f"▶ Loaded {MODEL_FILE} ({n_features} features)")

# 4) Выбор CSV-файла для прогноза
data_root = Path("data/real_data_folfer")
csvs = sorted(data_root.rglob("*.csv"))
if not csvs:
    raise FileNotFoundError(f"No CSV under {data_root}")
print("\nAvailable files:")
for i, p in enumerate(csvs[:10]):
    print(f"  [{i}] {p}")
choice = input(f"\nIndex (0…{len(csvs)-1}) or full path: ").strip()
try:
    idx = int(choice); csv_path = csvs[idx]
except:
    csv_path = Path(choice)
print("▶ Predict on:", csv_path)

# 5) Чтение и предобработка
df = pd.read_csv(csv_path)

# 5.1) Импутация числовых пропусков медианой
num_cols = df.select_dtypes(include=[float, int]).columns
df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

# 5.2) One-hot по тем же колонкам
cat_cols = [
    "weekday",
    "season",
    "is_holyday",
    "is_pre_holyday",
    "cat_pre_holyday",
    "school_holyday",
]
cat_cols = [c for c in cat_cols if c in df.columns and df[c].nunique() < 20]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 5.3) Подгонка под feature_names
for feat in feature_names:
    if feat not in df.columns:
        df[feat] = 0
df = df[feature_names]

# 6) Масштабирование и предсказание
X   = df.values.astype(float)
X_s = scaler.transform(X)
X_t = torch.tensor(X_s, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(X_t).cpu().numpy().flatten()

# 7) Сохранение результатов
orig = pd.read_csv(csv_path)
orig["predicted"] = preds
out = "predictions.csv"
orig.to_csv(out, index=False)
print(f"\n✅ Predictions saved to {out}")
