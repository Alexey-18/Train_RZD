import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer

def load_real(root="data/real_data_folfer"):
    files = list(Path(root).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV under {root}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

# 1) Загрузка всех real-данных
df = load_real("data/real_data_folfer")
print("▶ Loaded real data:", df.shape)

# 2) Импутация всех числовых NaN медианой
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
# Авто-детект target
possible = [c for c in df.columns if c.startswith("target_count_tickets")]
if not possible:
    raise KeyError("No target_count_tickets* column in real data")
target_col = possible[0]
num_features = [c for c in num_cols_all if c != target_col]
df[num_features] = SimpleImputer(strategy="median").fit_transform(df[num_features])
print("✅ Filled numeric NaNs")

# 3) One-hot кодирование ровно тех категорий, что в train_real.py
cat_cols = [
    "weekday",
    "season",
    "is_holyday",
    "is_pre_holyday",
    "cat_pre_holyday",
    "school_holyday",
]
# оставляем только те, что есть и имеют <20 уникальных значений
cat_cols = [c for c in cat_cols if c in df.columns and df[c].nunique() < 20]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("▶ Columns after one-hot:", df.shape[1])

# 4) Сохраняем список признаков (float64 & int64) без target
feature_names = [
    c for c in df.select_dtypes(include=[np.float64, np.int64]).columns.tolist()
    if c != target_col
]
with open("feature_names_real.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=2)
print(f"✅ feature_names_real.json saved ({len(feature_names)} features)")
