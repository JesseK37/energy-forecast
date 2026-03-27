import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

PROJECT_ID = "project-4e4b2ba5-ccce-4673-8e2"
DATASET    = "energy_forecast"

FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "temperature_c", "windspeed_ms", "cloudcover_pct", "relative_humidity",
    "demand_lag_24h", "demand_lag_48h", "demand_lag_168h",
]
TARGET = "demand_mw"


def load_features() -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID)
    query  = f"""
        SELECT timestamp_utc, {', '.join(FEATURES)}, {TARGET}
        FROM `{PROJECT_ID}.{DATASET}.features`
        ORDER BY timestamp_utc
    """
    df = client.query(query).to_dataframe()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df


def walk_forward_splits(df: pd.DataFrame, n_splits: int = 5):
    """
    Walk-forward validation: each fold trains on all data before the test window.
    Test windows are contiguous and non-overlapping, covering the last 30% of data.
    Never shuffles — temporal order is preserved throughout.
    """
    n         = len(df)
    test_size = int(n * 0.30 / n_splits)
    train_end = int(n * 0.70)

    splits = []
    for i in range(n_splits):
        test_start = train_end + i * test_size
        test_end   = test_start + test_size
        if test_end > n:
            break
        splits.append((
            df.iloc[:test_start],
            df.iloc[test_start:test_end],
        ))
    return splits


def train_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict):
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val   = val_df[FEATURES]
    y_val   = val_df[TARGET]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    preds = model.predict(X_val)
    mape  = mean_absolute_percentage_error(y_val, preds) * 100
    rmse  = root_mean_squared_error(y_val, preds)
    return model, mape, rmse


def run():
    mlflow.set_experiment("energy_demand_forecast")

    params = {
        "objective":      "regression",
        "metric":         "rmse",
        "learning_rate":  0.05,
        "num_leaves":     64,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":   5,
        "verbose":        -1,
    }

    print("Loading features from BigQuery...")
    df     = load_features()
    splits = walk_forward_splits(df, n_splits=5)
    print(f"Loaded {len(df)} rows — running {len(splits)}-fold walk-forward CV")

    fold_mapes = []
    fold_rmses = []

    with mlflow.start_run(run_name="lgbm_walkforward"):
        mlflow.log_params(params)
        mlflow.log_param("n_splits",   len(splits))
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("train_rows", len(df))

        for i, (train_df, val_df) in enumerate(splits):
            print(f"\nFold {i+1}: train={len(train_df)} val={len(val_df)}")
            model, mape, rmse = train_fold(train_df, val_df, params)
            fold_mapes.append(mape)
            fold_rmses.append(rmse)
            print(f"  MAPE: {mape:.2f}%  RMSE: {rmse:.1f} MW")
            mlflow.log_metric("mape", mape, step=i)
            mlflow.log_metric("rmse", rmse, step=i)

        mean_mape = np.mean(fold_mapes)
        mean_rmse = np.mean(fold_rmses)
        print(f"\nCV mean MAPE: {mean_mape:.2f}%  CV mean RMSE: {mean_rmse:.1f} MW")
        mlflow.log_metric("cv_mean_mape", mean_mape)
        mlflow.log_metric("cv_mean_rmse", mean_rmse)

        # Retrain on full dataset and register
        print("\nRetraining on full dataset...")
        X_full  = df[FEATURES]
        y_full  = df[TARGET]
        dfull   = lgb.Dataset(X_full, label=y_full)
        final_model = lgb.train(params, dfull, num_boost_round=500)

        mlflow.lightgbm.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="energy_demand_lgbm",
        )
        print("Model registered as 'energy_demand_lgbm'")

        # Feature importances
        importances = pd.Series(
            final_model.feature_importance(importance_type="gain"),
            index=FEATURES,
        ).sort_values(ascending=False)
        print("\nFeature importances (gain):")
        print(importances.to_string())


if __name__ == "__main__":
    run()