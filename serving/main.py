import os
import tempfile
import functions_framework
import lightgbm as lgb
import pandas as pd
from google.cloud import bigquery, storage
from datetime import datetime, timezone

GCS_BUCKET = "project-4e4b2ba5-ccce-4673-8e2"
GCS_MODEL_PATH = "mlflow/energy_demand_lgbm/champion/artifacts/model.lgb"
BQ_PROJECT = "project-4e4b2ba5-ccce-4673-8e2"
BQ_DATASET = "energy_forecast"
BQ_FEATURES_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.features"
BQ_FORECASTS_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.forecasts"
FORECAST_HORIZON_HOURS = 48

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "temperature_c", "windspeed_ms", "cloudcover_pct", "relative_humidity",
    "demand_lag_24h", "demand_lag_48h", "demand_lag_168h"
]

def load_model_from_gcs():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_PATH)
    with tempfile.NamedTemporaryFile(suffix=".lgb", delete=False) as f:
        blob.download_to_file(f)
        tmp_path = f.name
    model = lgb.Booster(model_file=tmp_path)
    os.unlink(tmp_path)
    return model

def load_latest_features():
    client = bigquery.Client(project=BQ_PROJECT)
    query = f"""
        SELECT *
        FROM `{BQ_FEATURES_TABLE}`
        WHERE timestamp_utc >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 72 HOUR)
        ORDER BY timestamp_utc DESC
        LIMIT {FORECAST_HORIZON_HOURS}
    """
    df = client.query(query).to_dataframe()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df

def write_forecasts(df):
    client = bigquery.Client(project=BQ_PROJECT)
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("forecast_mw", "FLOAT64"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_APPEND",
    )
    client.load_table_from_dataframe(df, BQ_FORECASTS_TABLE, job_config=job_config).result()

@functions_framework.http
def run_inference(request):
    try:
        print("Loading model from GCS...")
        model = load_model_from_gcs()

        print("Loading features from BigQuery...")
        features_df = load_latest_features()

        if features_df.empty:
            return ("No feature rows found.", 500)

        print(f"Running inference on {len(features_df)} rows...")
        X = features_df[FEATURE_COLS]
        preds = model.predict(X)

        out_df = pd.DataFrame({
            "timestamp": features_df["timestamp_utc"],
            "forecast_mw": preds,
            "model_version": "champion",
            "created_at": datetime.now(timezone.utc),
        })

        print("Writing forecasts to BigQuery...")
        write_forecasts(out_df)

        msg = f"Inference complete. {len(out_df)} forecasts written."
        print(msg)
        return (msg, 200)

    except Exception as e:
        print(f"ERROR: {e}")
        raise