import json, pickle
import pandas as pd
import torch
from pathlib import Path
from model import PriceNet

# 1) Выбираем модель и scaler:
if Path('best_model_real_finetuned.pth').exists():
    mp, sp = 'best_model_real_finetuned.pth','scaler_real.pkl'
else:
    mp, sp = 'best_model_synth.pth','scaler_synth.pkl'
bp = json.load(open('best_params_synth.json'))
hid = bp['hidden1']
scaler = pickle.load(open(sp,'rb'))
model = PriceNet(scaler.mean_.shape[0], hid)
model.load_state_dict(torch.load(mp, map_location='cpu'))
model.eval()

# 2) Читаем CSV для прогноза
path = input('CSV for predict: ').strip()
df = pd.read_csv(path)
# Rename и one-hot как в train_real
df = df.rename(columns={'days_to_otp':'days_before','dow_dateotp':'weekday'})
df = pd.get_dummies(df,columns=['weekday','season','is_holyday','is_pre_holyday',
                                 'cat_pre_holyday','school_holyday'], drop_first=True)
X = scaler.transform(df.values)

# 3) Предсказание и сохранение
with torch.no_grad():
    preds = model(torch.tensor(X,dtype=torch.float32)).numpy().flatten()
df['predicted'] = preds
df.to_csv('predictions.csv', index=False)
print('✅ Saved predictions.csv')