# Part a: Linear regression + WAPE
# We try to predict the number of daily calls a call center will receive
# based on the past 4 years of data. 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

def forecast_and_evaluate(df, week_num=260, degree=3):
    """
    Forecast week `week_num` using polynomial regression with seasonality.
    Returns forecasted daily volumes and WAPE on last 70 days.
    """
    df = df.copy()
    df["day"] = np.arange(1, len(df)+1)
    df["log_volume"] = np.log(df["volume"])
    df["dow"] = df["day"] % 7
    df["week_of_year"] = df["day"] % 364 // 7
    df = pd.get_dummies(df, columns=["dow"], drop_first=True)
    df["year_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["year_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Split
    train = df.iloc[:-70].copy()
    test = df.iloc[-70:].copy()

    # Polynomial trend
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly.fit_transform(train[["day"]])
    X_poly_test = poly.transform(test[["day"]])
    seasonal_cols = [col for col in df.columns if col.startswith("dow_")] + ["year_sin", "year_cos"]
    X_train = np.hstack([X_poly_train, train[seasonal_cols].values])
    X_test = np.hstack([X_poly_test, test[seasonal_cols].values])

    # Fit model
    model = LinearRegression()
    model.fit(X_train, train["log_volume"])

    # Predict on training set and compute Train WAPE
    train["log_pred"] = model.predict(X_train)
    train["y_pred"] = np.exp(train["log_pred"])
    train_wape = np.sum(np.abs(train["y_pred"] - train["volume"])) / np.sum(train["volume"])

    # Predict test + compute WAPE
    test["log_pred"] = model.predict(X_test)
    test["y_pred"] = np.exp(test["log_pred"])
    wape = np.sum(np.abs(test["y_pred"] - test["volume"])) / np.sum(test["volume"])

    # Forecast week 260 (days 1813–1820)
    # 1456 = last day of week 208 (1-indexed)
    start_day = 1456 + (week_num - 209) * 7
    future_days = np.arange(start_day, start_day + 7)

    future_df = pd.DataFrame({"day": future_days})
    future_df["week_of_year"] = future_df["day"] % 364 // 7
    future_df["year_sin"] = np.sin(2 * np.pi * future_df["week_of_year"] / 52)
    future_df["year_cos"] = np.cos(2 * np.pi * future_df["week_of_year"] / 52)
    future_df["dow"] = future_df["day"] % 7

    # Dummy encoding
    dow_dummies = pd.get_dummies(future_df["dow"], prefix="dow")
    for col in seasonal_cols:
        if col not in dow_dummies:
            dow_dummies[col] = 0
    dow_dummies = dow_dummies[seasonal_cols[:-2]]  # exclude sin/cos already in future_df

    X_future_poly = poly.transform(future_df[["day"]])
    X_future = np.hstack([X_future_poly, dow_dummies.values, future_df[["year_sin", "year_cos"]].values])
    forecast = np.exp(model.predict(X_future))

    return forecast, future_days, wape, train_wape

# Load and clean data
df = pd.read_csv("actuals.csv", index_col=0)
df.rename(columns={"x": "volume"}, inplace=True)

forecast, days, test_wape, train_wape = forecast_and_evaluate(df, week_num=260, degree=3)

print(f"Train WAPE: {train_wape:.2%}")
print(f"Test WAPE:  {test_wape:.2%}\n")
print("Forecast for week 260:")
for day, vol in zip(days, forecast):
    print(f"Day {day}: {vol:.0f}")
