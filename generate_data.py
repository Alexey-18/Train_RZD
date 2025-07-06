import pandas as pd
import numpy as np

np.random.seed(42)

def generate_ticket_data(n=50000):
    data = []
    for _ in range(n):
        # Базовые признаки
        days_before = np.random.randint(1, 61)
        weekday     = np.random.randint(0, 7)
        train_type  = np.random.randint(0, 2)
        occupancy   = np.round(np.random.uniform(0.2, 0.95), 2)
        is_event    = np.random.randint(0, 2)
        season      = np.random.randint(0, 4)
        # Дополнительные фичи
        distance    = np.random.randint(100, 2000)
        wagon_class = np.random.choice([0,1,2])
        temp        = np.round(np.random.normal(10, 15), 1)
        trend       = np.round(np.random.uniform(-0.1, 0.1), 3)

        # Формула цены
        price = 1500
        price += (60 - days_before)**1.5 * 5
        price += 30 if weekday >= 4 else -10
        price += 700 if train_type == 1 else 0
        price += occupancy * 600
        price += 800 if is_event == 1 else 0
        price += season * 50
        price += distance * 0.1
        price += (wagon_class + 1) * 100
        price += temp * 2
        price += trend * 1000
        price += np.random.normal(0, 200)

        data.append([
            days_before, weekday, train_type,
            occupancy, is_event, season,
            distance, wagon_class, temp, trend,
            round(price, 2)
        ])

    cols = [
        "days_before","weekday","train_type",
        "occupancy","is_event","season",
        "distance","wagon_class","temp","trend",
        "price"
    ]
    return pd.DataFrame(data, columns=cols)

if __name__ == "__main__":
    df = generate_ticket_data()
    df.to_csv("data/synthetic_tickets.csv", index=False)
    print("✅ Synthetic data generated:", df.shape)