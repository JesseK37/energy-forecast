import os
import requests
import functions_framework
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DATASET    = "energy_forecast"
EIA_KEY    = os.environ["EIA_API_KEY"]
BQ_CLIENT  = bigquery.Client(project=PROJECT_ID)

EIA_URL        = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_LAT = 39.95
WEATHER_LON = -75.16


def fetch_eia_demand(start: str, end: str) -> list[dict]:
    all_rows = []
    offset = 0
    while True:
        params = {
            "api_key": EIA_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "PJM",
            "facets[type][]": "D",
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
            "offset": offset,
        }
        resp = requests.get(EIA_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["response"]
        rows = data["data"]
        if not rows:
            break
        all_rows.extend(rows)
        offset += len(rows)
        if len(rows) < 5000:
            break
    if not all_rows:
        raise ValueError("EIA returned no demand data for the requested window")
    return all_rows


def fetch_weather_historical(start_date: str, end_date: str) -> list[dict]:
    params = {
        "latitude":   WEATHER_LAT,
        "longitude":  WEATHER_LON,
        "hourly":     "temperature_2m,windspeed_10m,cloudcover,relativehumidity_2m",
        "timeformat": "unixtime",
        "timezone":   "UTC",
        "start_date": start_date,
        "end_date":   end_date,
    }
    resp = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    ingested_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, ts in enumerate(hourly["time"]):
        rows.append({
            "timestamp_utc":     datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "temperature_c":     hourly["temperature_2m"][i],
            "windspeed_ms":      hourly["windspeed_10m"][i],
            "cloudcover_pct":    hourly["cloudcover"][i],
            "relative_humidity": hourly["relativehumidity_2m"][i],
            "ingested_at":       ingested_at,
        })
    return rows


def fetch_weather(start: str, end: str) -> list[dict]:
    params = {
        "latitude":   WEATHER_LAT,
        "longitude":  WEATHER_LON,
        "hourly":     "temperature_2m,windspeed_10m,cloudcover,relativehumidity_2m",
        "timeformat": "unixtime",
        "timezone":   "UTC",
        "start_date": start[:10],
        "end_date":   end[:10],
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    ingested_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, ts in enumerate(hourly["time"]):
        rows.append({
            "timestamp_utc":     datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "temperature_c":     hourly["temperature_2m"][i],
            "windspeed_ms":      hourly["windspeed_10m"][i],
            "cloudcover_pct":    hourly["cloudcover"][i],
            "relative_humidity": hourly["relativehumidity_2m"][i],
            "ingested_at":       ingested_at,
        })
    return rows


def merge_into(target_table: str, staging_table: str, key_col: str) -> None:
    query = f"""
        MERGE `{target_table}` T
        USING `{staging_table}` S
        ON T.{key_col} = S.{key_col}
        WHEN NOT MATCHED THEN INSERT ROW
    """
    BQ_CLIENT.query(query).result()


def write_staging_and_merge(rows: list[dict], target: str, staging: str, key_col: str) -> None:
    table_ref = BQ_CLIENT.get_table(f"{PROJECT_ID}.{DATASET}.{target}")
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=table_ref.schema,
    )
    staging_ref = f"{PROJECT_ID}.{DATASET}.{staging}"
    load_job = BQ_CLIENT.load_table_from_json(rows, staging_ref, job_config=job_config)
    load_job.result()
    merge_into(f"{PROJECT_ID}.{DATASET}.{target}", staging_ref, key_col)
    print(f"Merged {len(rows)} rows into {target}")


def prepare_demand_rows(raw_rows: list[dict]) -> list[dict]:
    ingested_at = datetime.now(timezone.utc).isoformat()
    bq_rows = []
    for r in raw_rows:
        raw_ts = r["period"]
        ts = datetime.strptime(raw_ts, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        value = r.get("value")
        if value is None:
            continue
        bq_rows.append({
            "timestamp_utc": ts.isoformat(),
            "region":        "PJM",
            "demand_mw":     float(value),
            "ingested_at":   ingested_at,
        })
    if not bq_rows:
        raise ValueError("No valid demand rows after parsing")
    return bq_rows


@functions_framework.http
def ingest(request):
    now   = datetime.now(timezone.utc)
    start = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H")
    end   = now.strftime("%Y-%m-%dT%H")

    print(f"Fetching EIA demand {start} → {end}")
    demand_rows = fetch_eia_demand(start, end)
    bq_demand   = prepare_demand_rows(demand_rows)
    write_staging_and_merge(bq_demand, "demand_raw", "demand_staging", "timestamp_utc")

    print(f"Fetching weather {start} → {end}")
    weather_rows = fetch_weather(start, end)
    write_staging_and_merge(weather_rows, "weather_raw", "weather_staging", "timestamp_utc")

    return ("Ingest complete", 200)


@functions_framework.http
def backfill(request):
    request_json = request.get_json(silent=True)
    start_date = request_json.get("start_date", "2024-01-01")
    end_date   = request_json.get("end_date",
                   datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    # EIA uses datetime format, Open-Meteo uses date format
    start_dt = f"{start_date}T00"
    end_dt   = f"{end_date}T23"

    print(f"Backfilling EIA demand {start_dt} → {end_dt}")
    demand_rows = fetch_eia_demand(start_dt, end_dt)
    bq_demand   = prepare_demand_rows(demand_rows)
    # Write in chunks of 5000 to avoid payload limits
    for i in range(0, len(bq_demand), 5000):
        chunk = bq_demand[i:i+5000]
        write_staging_and_merge(chunk, "demand_raw", "demand_staging", "timestamp_utc")
        print(f"Demand chunk {i}–{i+len(chunk)} done")

    print(f"Backfilling weather {start_date} → {end_date}")
    weather_rows = fetch_weather_historical(start_date, end_date)
    for i in range(0, len(weather_rows), 5000):
        chunk = weather_rows[i:i+5000]
        write_staging_and_merge(chunk, "weather_raw", "weather_staging", "timestamp_utc")
        print(f"Weather chunk {i}–{i+len(chunk)} done")

    return ("Backfill complete", 200)