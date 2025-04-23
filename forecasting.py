# Part a: Linear regression + WAPE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("actuals.csv", index_col=0)
df.rename(columns={"x": "volume"}, inplace=True)

# Add time features
df["day"] = np.arange(len(df))
df["dow"] = df["day"] % 7           # Day of week (0=Monday)
df["week"] = df["day"] // 7         # Week number (0 to 259)

# Create train/test split
train = df.iloc[:-70].copy()
test = df.iloc[-70:].copy()

# Try polynomial trend (degree=3 for example)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train = poly.fit_transform(train[["day"]])
X_test = poly.transform(test[["day"]])

model = LinearRegression()
model.fit(X_train, train["volume"])

# Predict
test["y_pred"] = model.predict(X_test)

# WAPE calculation
wape = np.sum(np.abs(test["y_pred"] - test["volume"])) / np.sum(test["volume"])
print(f"WAPE: {wape:.2%}")

# Forecast for week 260 (days 1456 to 1462)
future_days = np.arange(1456, 1463).reshape(-1, 1)
X_future = poly.transform(future_days)
forecast = model.predict(X_future)

print("\nForecast for week 260:")
for i, val in enumerate(forecast, 1):
    print(f"Day {i}: {val:.0f}")

# Optional: plot
plt.figure(figsize=(12, 5))
plt.plot(df["volume"], label="Actual")
plt.plot(test.index, test["y_pred"], label="Test Predictions", color="orange")
plt.plot(range(1456, 1463), forecast, label="Forecast week 260", color="green")
plt.legend()
plt.title("Call Volume Forecasting")
plt.xlabel("Day")
plt.ylabel("Volume")
plt.tight_layout()
plt.show()
