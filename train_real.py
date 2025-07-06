import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, json

from model import PriceNet

def find_real_root(data_root="data"):
    p = Path(data_root)
    subs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("real_data")]
    if not subs:
        raise FileNotFoundError(f"В папке {data_root} нет подкаталога real_data*")
    root = subs[0]
    print("▶ Используем папку с реальными данными:", root)
    return root

def load_real(root_dir) -> pd.DataFrame:
    files = list(Path(root_dir).rglob("*.csv"))
    print(f"ℹ️ Найдено {len(files)} CSV файлов в '{root_dir}':")
    for f in files[:5]:
        print("   ", f)
    if not files:
        raise FileNotFoundError(f"No CSV under {root_dir}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

# 1) Load real data
real_root = find_real_root("data")
df = load_real(real_root)
print("▶ Loaded real data:", df.shape)

# 2) Quick EDA & target detection
print("\n— Missing rates —")
print(df.isna().mean().sort_values(ascending=False).head(10))

possible = [c for c in df.columns if c.startswith("target_count_tickets")]
if not possible:
    raise KeyError("Не найдена ни одна колонка target_count_tickets*")
print("\n▶ Найдены потенциальные target-столбцы:", possible)
target_col = possible[0]
print(f"▶ Используем в качестве target: '{target_col}'")

print(f"\n— Top 5 corr with '{target_col}' —")
corr = df.select_dtypes(include=[np.number]) \
         .corrwith(df[target_col]) \
         .abs() \
         .sort_values(ascending=False)
print(corr.head(5))

# 3) Impute numeric missing
num_cols_all = df.select_dtypes(include=[float, int]).columns.tolist()
num_features = [c for c in num_cols_all if c != target_col]
imp = SimpleImputer(strategy="median")
df[num_features] = imp.fit_transform(df[num_features])
print("✅ Заполнили числовые пропуски медианой")
print("Осталось NaN в числовых:", df[num_features].isna().sum().sum())

# 4) One-hot for selected categories
cat_cols = [
    "weekday", "season", "is_holyday",
    "is_pre_holyday", "cat_pre_holyday", "school_holyday"
]
cat_cols = [c for c in cat_cols if c in df.columns and df[c].nunique() < 20]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("▶ Всего колонок после one-hot:", df.shape[1])

# 5) Prepare X and y
num_cols = df.select_dtypes(include=[float, int]).columns.drop(target_col)
X = df[num_cols].values.astype(np.float32)
y = df[target_col].values.reshape(-1, 1).astype(np.float32)

# 6) Scale and save scaler
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
pickle.dump(scaler, open("scaler_real.pkl", "wb"))
print("✅ scaler_real.pkl saved")

# 7) Train/val/test split
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.1, random_state=42
)

# 8) To tensors
dtype = torch.float32
torch.manual_seed(42)
Xtr_t = torch.tensor(X_train, dtype=dtype)
ytr_t = torch.tensor(y_train, dtype=dtype)
Xv_t  = torch.tensor(X_val,   dtype=dtype)
yv_t  = torch.tensor(y_val,   dtype=dtype)
Xte_t = torch.tensor(X_test,  dtype=dtype)
yte_t = torch.tensor(y_test,  dtype=dtype)

# 9) Grid Search
def train_val(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PriceNet(Xtr_t.shape[1], params["hidden1"]).to(device)
    opt = optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(Xtr_t, ytr_t),
                    batch_size=params["batch_size"], shuffle=True)
    model.train()
    for _ in range(params["epochs"]):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xv_t.to(device))
        return loss_fn(pred, yv_t.to(device)).item()

param_grid = {
    "lr": [1e-2, 1e-3],
    "batch_size": [32, 64],
    "epochs": [30, 50],
    "hidden1": [64, 128]
}
best_loss, best_params = float("inf"), None
for p in ParameterGrid(param_grid):
    v = train_val(p)
    print(p, "→ Val MSE:", v)
    if v < best_loss:
        best_loss, best_params = v, p

print("▶ Best params:", best_params)
json.dump(best_params, open("best_params_real.json", "w"))

# 10) Full training with scheduler & early stopping
bs      = best_params["batch_size"]
lr      = best_params["lr"]
hidden1 = best_params["hidden1"]
n_epochs= 200
patience= 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = PriceNet(Xtr_t.shape[1], hidden1).to(device)
opt    = optim.Adam(model.parameters(), lr=lr)
sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=bs, shuffle=True)
val_dl   = DataLoader(TensorDataset(Xv_t, yv_t), batch_size=bs)

best_val, counter = float("inf"), 0
print("\n🚀 Full training on real data…")
for epoch in range(1, n_epochs+1):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        nn.MSELoss()(model(xb), yb).backward()
        opt.step()
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            losses.append(nn.MSELoss()(model(xb), yb).item())
    avg = sum(losses) / len(losses)
    sched.step()
    print(f"Epoch {epoch:03d} | Val MSE: {avg:.2f}")
    if avg < best_val:
        best_val, counter = avg, 0
        torch.save(model.state_dict(), "best_model_real.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"🔒 Early stopping at epoch {epoch}")
            break

# 11) Final evaluation
model.load_state_dict(torch.load("best_model_real.pth", map_location=device))
model.eval()
with torch.no_grad():
    preds = model(Xte_t.to(device)).cpu().numpy().flatten()

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae  = mean_absolute_error(y_test, preds)
r2   = r2_score(y_test, preds)
print(f"\n📊 Test Real → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

torch.save(model.state_dict(), "model_real_final.pth")
print("✅ model_real_final.pth saved")
