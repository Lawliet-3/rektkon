import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def load_data() -> pd.DataFrame:
    path = os.getenv("TRAINING_DATA_PATH", "data/market_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)

    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 1, 1000))
    returns = np.abs(np.diff(prices, prepend=prices[0]) / np.clip(prices, 1e-12, None))
    return pd.DataFrame(
        {
            "last_price": prices,
            "price_mean": pd.Series(prices).rolling(5, min_periods=1).mean(),
            "price_std": pd.Series(prices).rolling(5, min_periods=1).std().fillna(0.0),
            "volatility": returns,
        }
    )


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("volatility-forecasting")

    data = load_data()
    if "volatility" not in data.columns:
        raise ValueError("Training data must contain a 'volatility' target column")

    x = data[["last_price", "price_mean", "price_std"]]
    y = data["volatility"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
    model = XGBRegressor(**params)

    with mlflow.start_run():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        rmse = root_mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=os.getenv("MLFLOW_MODEL_NAME", "volatility-model"),
        )

        print(f"Logged run with RMSE={rmse:.6f}, MAE={mae:.6f}")


if __name__ == "__main__":
    main()
