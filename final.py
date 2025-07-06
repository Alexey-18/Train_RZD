import pandas as pd
import matplotlib.pyplot as plt

# 1) Подгрузить predictions.csv
df = pd.read_csv("predictions.csv")

# 2) Посмотреть первые строки
print(df.head(10))

# 3) Аггрегировать по дням до отправления (days_to_otp вместо days_before)
profile = df.groupby("days_to_otp")["predicted"].mean().round(2)

print("\nСреднее предсказание билетов по дням до отправления:")
print(profile.head(10))

# 4) Построить график
plt.figure(figsize=(8, 5))
plt.plot(profile.index, profile.values, marker='o')
plt.xlabel("Дней до отправления (days_to_otp)")
plt.ylabel("Среднее предсказание билетов")
plt.title("Прогнозируемые продажи билетов vs. Дни до отправления")
plt.gca().invert_xaxis()   # по желанию: дни от ближнего к дальнему
plt.grid(True)
plt.tight_layout()
plt.show()