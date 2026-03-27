import functions_framework
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timezone

BQ_PROJECT = "project-4e4b2ba5-ccce-4673-8e2"
BQ_DATASET = "energy_forecast"
BQ_FORECASTS_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.forecasts"
BQ_DEMAND_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.demand_raw"
BQ_MONITORING_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.monitoring"
MAPE_ALERT_THRESHOLD = 6.0

def compute_metrics():
    client = bigquery.Client(project=BQ_PROJECT)
    query = f"""
        SELECT
            f.timestamp,
            f.forecast_mw,
            f.model_version,
            d.demand_mw AS actual_mw
        FROM `{BQ_FORECASTS_TABLE}` f
        INNER JOIN `{BQ_DEMAND_TABLE}` d
            ON f.timestamp = d.timestamp_utc
        WHERE f.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 48 HOUR)
          AND d.demand_mw IS NOT NULL
    """
    df = client.query(query).to_dataframe()
    return df

def write_metrics(metrics: dict):
    client = bigquery.Client(project=BQ_PROJECT)
    schema = [
        bigquery.SchemaField("evaluated_at", "TIMESTAMP"),
        bigquery.SchemaField("n_hours", "INTEGER"),
        bigquery.SchemaField("mape_pct", "FLOAT64"),
        bigquery.SchemaField("mae_mw", "FLOAT64"),
        bigquery.SchemaField("rmse_mw", "FLOAT64"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("alert", "BOOLEAN"),
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_APPEND",
    )
    out_df = pd.DataFrame([metrics])
    client.load_table_from_dataframe(out_df, BQ_MONITORING_TABLE, job_config=job_config).result()

@functions_framework.http
def run_monitoring(request):
    try:
        print("Loading forecasts and actuals from BigQuery...")
        df = compute_metrics()

        if df.empty:
            msg = "No matched forecast/actual pairs found yet — actuals may not have arrived."
            print(msg)
            return (msg, 200)

        df["abs_pct_error"] = (
            (df["forecast_mw"] - df["actual_mw"]).abs() / df["actual_mw"].abs()
        ) * 100
        df["abs_error"] = (df["forecast_mw"] - df["actual_mw"]).abs()
        df["sq_error"] = (df["forecast_mw"] - df["actual_mw"]) ** 2

        mape = df["abs_pct_error"].mean()
        mae = df["abs_error"].mean()
        rmse = df["sq_error"].mean() ** 0.5
        n = len(df)
        model_version = df["model_version"].iloc[0]
        alert = mape > MAPE_ALERT_THRESHOLD

        metrics = {
            "evaluated_at": datetime.now(timezone.utc),
            "n_hours": n,
            "mape_pct": mape,
            "mae_mw": mae,
            "rmse_mw": rmse,
            "model_version": model_version,
            "alert": alert,
        }

        print(f"Metrics: MAPE={mape:.2f}%, MAE={mae:.0f} MW, RMSE={rmse:.0f} MW, n={n}")
        if alert:
            print(f"ALERT: MAPE {mape:.2f}% exceeds threshold of {MAPE_ALERT_THRESHOLD}%")

        write_metrics(metrics)
        msg = f"Monitoring complete. MAPE={mape:.2f}%, MAE={mae:.0f} MW, RMSE={rmse:.0f} MW over {n} hours. Alert={alert}"
        print(msg)
        return (msg, 200)

    except Exception as e:
        print(f"ERROR: {e}")
        raise