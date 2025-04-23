# Part a: Linear regression + WAPE
# We try to predict the number of daily calls a call center will receive
# based on the past 4 years of data. 

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
df["week_of_year"] = df["day"] % 364 // 7  # 0 to 51

df = pd.get_dummies(df, columns=["dow"], drop_first=True)
df["year_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
df["year_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

# Create train/test split
train = df.iloc[:-70].copy()
test = df.iloc[-70:].copy()

# Try polynomial trend
poly = PolynomialFeatures(degree=4, include_bias=False)

# Polynomial of 'day'
X_poly_train = poly.fit_transform(train[["day"]])
X_poly_test = poly.transform(test[["day"]])

# Combine with seasonality features
seasonal_cols = [col for col in df.columns if col.startswith("dow_")] + ["year_sin", "year_cos"]
X_train = np.hstack([X_poly_train, train[seasonal_cols].values])
X_test = np.hstack([X_poly_test, test[seasonal_cols].values])

model = LinearRegression()
model.fit(X_train, train["volume"])

# Predict
test["y_pred"] = model.predict(X_test)

# WAPE calculation
wape = np.sum(np.abs(test["y_pred"] - test["volume"])) / np.sum(test["volume"])
print(f"WAPE: {wape:.2%}")

# Forecast for week 260 (days 1456 to 1462)
future_days = np.arange(1456, 1463).reshape(-1, 1)

# Add seasonal features for future
future_df = pd.DataFrame({"day": np.arange(1456, 1463)})
future_df["week_of_year"] = future_df["day"] % 364 // 7
future_df["year_sin"] = np.sin(2 * np.pi * future_df["week_of_year"] / 52)
future_df["year_cos"] = np.cos(2 * np.pi * future_df["week_of_year"] / 52)

# Use most recent day-of-week pattern
dow_dummies = pd.get_dummies(future_df["day"] % 7)
dow_dummies = dow_dummies.reindex(columns=range(1, 7), fill_value=0)  # match model dummies

X_future_poly = poly.transform(future_df[["day"]])
X_future = np.hstack([X_future_poly, dow_dummies.values, future_df[["year_sin", "year_cos"]].values])
forecast = model.predict(X_future)

print("\nForecast for week 260:")
for i, val in enumerate(forecast, 1):
    print(f"Day {i}: {val:.0f}")

# Optional: zoomed-in plot on last 100 days + forecast
plt.figure(figsize=(12, 5))
plt.plot(df.index[-100:], df["volume"].iloc[-100:], label="Actual", color="lightblue")
plt.plot(test.index, test["y_pred"], label="Test Predictions", color="orange", linewidth=2)
plt.plot(range(1456, 1463), forecast, label="Forecast week 260", color="green", linewidth=2)

plt.legend()
plt.title("Zoomed-In: Call Volume Forecasting (Last 100 Days + Forecast)")
plt.xlabel("Day")
plt.ylabel("Volume")
plt.tight_layout()
plt.show()
