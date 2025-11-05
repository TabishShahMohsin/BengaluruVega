import re
import json
import pandas as pd
import joblib, pickle
from datetime import timedelta
import argparse
from geopy.distance import geodesic
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# --------------------------
# Helpers
# (Your helper functions remain unchanged)
# --------------------------
def clean_quoted(x):
    if pd.isna(x):
        return x
    return str(x).strip().strip('"').strip("'")

def parse_vehicle_timestamp(ts_raw):
    if ts_raw is None:
        return None
    s = clean_quoted(ts_raw)
    if s == "":
        return None
    ts = None
    if re.fullmatch(r"\d+", s):
        try:
            ts = pd.to_datetime(int(s), unit="s", utc=True)  # keep UTC
        except Exception:
            ts = pd.to_datetime(s, errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts

def parse_stop_list(stop_list_str):
    if pd.isna(stop_list_str):
        return []
    return [int(n) for n in re.findall(r"\d+", str(stop_list_str))]

def resolve_path(docker_path, local_path):
    return docker_path if os.path.exists(docker_path) else local_path

def safe_read_parquet(parquet_path):
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    alt_path = os.path.join("/app/data", parquet_path)
    if os.path.exists(alt_path):
        print(f"[INFO] Using alternate path: {alt_path}")
        return pd.read_parquet(alt_path)
    base_name = os.path.basename(parquet_path)
    alt_eval_path = os.path.join("/app/data/eval_data", base_name)
    if os.path.exists(alt_eval_path):
        # This print statement already exists and is useful, so we keep it.
        # It's the source of the logs you provided.
        print(f"[INFO] Using alternate eval_data path: {alt_eval_path}")
        return pd.read_parquet(alt_eval_path)
    raise FileNotFoundError(f"Parquet file not found at: {parquet_path}, checked also {alt_path} and {alt_eval_path}")

def compute_trip_avg_speed(df):
    """
    Compute average speed (m/s) for a trip DataFrame with lat/lon + vehicle_timestamp.
    """
    if df.shape[0] < 2:
        return None  # not enough points

    df = df.sort_values("vehicle_timestamp").copy()
    df["vehicle_timestamp"] = pd.to_datetime(df["vehicle_timestamp"], errors="coerce", utc=True)

    total_dist = 0.0
    total_time = 0.0

    prev_lat, prev_lon, prev_time = None, None, None
    for _, row in df.iterrows():
        lat, lon, t = row.get("latitude"), row.get("longitude"), row["vehicle_timestamp"]
        if pd.isna(lat) or pd.isna(lon) or pd.isna(t):
            continue
        if prev_lat is not None and prev_time is not None:
            dist = geodesic((prev_lat, prev_lon), (lat, lon)).meters
            dt = (t - prev_time).total_seconds()
            if dt > 0 and dist > 0:
                total_dist += dist
                total_time += dt
        prev_lat, prev_lon, prev_time = lat, lon, t

    if total_time > 0:
        return total_dist / total_time  # m/s
    return None

# --------------------------
# Load reference data
# (This section remains unchanged)
# --------------------------
STOPS_PATH = resolve_path("/app/refdata/stops_clean.csv", "../refdata/stops_clean.csv")
ROUTE_SEQ_PATH = resolve_path("/app/refdata/route_to_stop_clean.csv", "../refdata/route_to_stop_clean.csv")
MODEL_PATH = resolve_path("/app/refdata/models/eta_model_hypertuned.pkl", "../refdata/models/eta_model_hypertuned.pkl")
ENCODERS_PATH = resolve_path("/app/refdata/models/encoders_hypertuned.pkl", "../refdata/models/encoders_hypertuned.pkl")
SPEED_PATH = resolve_path("/app/refdata/models/route_avg_speed_m_s_hypertuned.pkl", "../refdata/models/route_avg_speed_m_s_hypertuned.pkl")

df_stops = pd.read_csv(STOPS_PATH)
df_route_seq = pd.read_csv(ROUTE_SEQ_PATH, index_col=0)
df_route_seq['route_id'] = df_route_seq['route_id'].astype(str).str.strip()

if 'stop_id' in df_stops.columns:
    try:
        df_stops['stop_id'] = df_stops['stop_id'].astype(int)
    except Exception:
        pass

model = joblib.load(MODEL_PATH)
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

route_avg_speed_m_s = joblib.load(SPEED_PATH)
global_avg_speed_m_s = float(pd.Series(route_avg_speed_m_s).mean())

# --------------------------
# Floor limit function & ETA prediction
# (These functions remain unchanged)
# --------------------------
def apply_floor_limit(eta_time, ping_time, min_gap_minutes=0):
    if eta_time <= ping_time + timedelta(minutes=min_gap_minutes):
        eta_time = ping_time + timedelta(minutes=min_gap_minutes)
    return eta_time

def predict_future_stops(ping, route_id, stop_seq, avg_speed):
    ts = parse_vehicle_timestamp(ping.get('vehicle_timestamp', None))
    if ts is pd.NaT or ts is None:
        raise ValueError("Cannot parse vehicle timestamp for ping.")
    ping_time = ts  # UTC

    coords = df_stops.set_index('stop_id').loc[stop_seq][['stop_lat', 'stop_lon']].reset_index()
    dists = coords.apply(lambda r: geodesic((float(ping['ping_lat']), float(ping['ping_lon'])),
                                            (float(r.stop_lat), float(r.stop_lon))).meters, axis=1).values
    next_idx = int(np.argmin(dists))
    etas = {}
    cumulative = 0.0

    to_stop_id = int(coords.loc[next_idx, 'stop_id'])
    remaining_distance_m = geodesic((float(ping['ping_lat']), float(ping['ping_lon'])),
                                    (float(coords.loc[next_idx, 'stop_lat']), float(coords.loc[next_idx, 'stop_lon']))).meters
    from_stop_id = int(stop_seq[next_idx - 1]) if next_idx > 0 else stop_seq[next_idx]

    try: r_enc = encoders['route_id'].transform([str(route_id)])[0] if str(route_id) in encoders['route_id'].classes_ else 0
    except: r_enc = 0
    try: fs_enc = encoders['from_stop'].transform([str(from_stop_id)])[0] if str(from_stop_id) in encoders['from_stop'].classes_ else 0
    except: fs_enc = 0
    try: ts_enc = encoders['to_stop'].transform([str(to_stop_id)])[0] if str(to_stop_id) in encoders['to_stop'].classes_ else 0
    except: ts_enc = 0

    first_X = pd.DataFrame([{
        'route_id_enc': r_enc,
        'from_stop_enc': fs_enc,
        'to_stop_enc': ts_enc,
        'distance_m': remaining_distance_m,
        'avg_speed_m_s': avg_speed,
        'start_hour': int(ping_time.hour),
        'day_of_week': int(ping_time.dayofweek)
    }])
    cumulative += float(np.expm1(model.predict(first_X)[0]))
    eta_first = ping_time + timedelta(minutes=cumulative)
    eta_first = apply_floor_limit(eta_first, ping_time)
    etas[to_stop_id] = eta_first.strftime("%Y-%m-%d %H:%M:%S")

    for j in range(next_idx + 1, len(stop_seq)):
        f_stop, t_stop = int(stop_seq[j-1]), int(stop_seq[j])
        fcoord = df_stops.set_index('stop_id').loc[f_stop]
        tcoord = df_stops.set_index('stop_id').loc[t_stop]
        segdist = geodesic((float(fcoord.stop_lat), float(fcoord.stop_lon)),
                            (float(tcoord.stop_lat), float(tcoord.stop_lon))).meters
        try: f_enc = encoders['from_stop'].transform([str(f_stop)])[0] if str(f_stop) in encoders['from_stop'].classes_ else 0
        except: f_enc = 0
        try: t_enc = encoders['to_stop'].transform([str(t_stop)])[0] if str(t_stop) in encoders['to_stop'].classes_ else 0
        except: t_enc = 0
        fe = pd.DataFrame([{
            'route_id_enc': r_enc,
            'from_stop_enc': f_enc,
            'to_stop_enc': t_enc,
            'distance_m': segdist,
            'avg_speed_m_s': avg_speed,
            'start_hour': int(ping_time.hour),
            'day_of_week': int(ping_time.dayofweek)
        }])
        cumulative += float(np.expm1(model.predict(fe)[0]))
        eta_next = ping_time + timedelta(minutes=cumulative)
        eta_next = apply_floor_limit(eta_next, ping_time)
        etas[t_stop] = eta_next.strftime("%Y-%m-%d %H:%M:%S")

    return etas



# --------------------------
# CLI
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_json", "--input-json", dest="input_json", required=False, default="/app/data/input.json")
args, unknown = parser.parse_known_args()

input_path = args.input_json
output_path = "/app/out/output.json"
output_dict = {}
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --------------------------
# Handle parquet / JSON inputs
# --------------------------
if input_path.endswith(".parquet"):
    print(f"\n[INFO] Processing single Parquet file: {input_path}")
    df = safe_read_parquet(input_path)
    trip_avg_speed = compute_trip_avg_speed(df)

    latest_rows = df.sort_values("vehicle_timestamp").groupby("trip_id").tail(1)
    for _, row in latest_rows.iterrows():
        if 'latitude' in row and 'longitude' in row:
            row['ping_lat'], row['ping_lon'] = row['latitude'], row['longitude']
        else:
            continue
        route_id = clean_quoted(row.get("route_id", ""))
        if not route_id or route_id.lower() == "nan":
            continue
        matched = df_route_seq[df_route_seq['route_id'].astype(str).str.strip() == route_id]
        if matched.empty: continue
        stop_seq = parse_stop_list(matched['stop_id_list'].iloc[0])
        if not stop_seq: continue

        avg_speed = trip_avg_speed if trip_avg_speed is not None else route_avg_speed_m_s.get(str(route_id), global_avg_speed_m_s)
        
        try:
            
            etas = predict_future_stops(row, route_id, stop_seq, avg_speed=avg_speed)
            if route_id not in output_dict:
                output_dict[route_id] = {}
            output_dict[route_id].update(etas)
        except Exception as e:
            print(f"⚠️ Error predicting {row['trip_id']}: {e}")

else:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_input = json.load(f)

    if isinstance(raw_input, dict) and all(isinstance(v, str) and v.endswith(".parquet") for v in raw_input.values()):
        for team_id, parquet_path in raw_input.items():
            print(f"\n[INFO] Processing input for '{team_id}': {parquet_path}")
            try:
                df = safe_read_parquet(parquet_path)
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                continue
            if "trip_id" not in df.columns or "vehicle_timestamp" not in df.columns:
                print(f"⚠️ Skipping {team_id}: missing columns")
                continue

            trip_avg_speed = compute_trip_avg_speed(df)
            latest_rows = df.sort_values("vehicle_timestamp").groupby("trip_id").tail(1)
            for _, row in latest_rows.iterrows():
                if 'latitude' in row and 'longitude' in row:
                    row['ping_lat'], row['ping_lon'] = row['latitude'], row['longitude']
                else: continue
                route_id = clean_quoted(row.get("route_id", ""))
                if not route_id or route_id.lower() == "nan": continue
                matched = df_route_seq[df_route_seq['route_id'].astype(str).str.strip() == route_id]
                if matched.empty: continue
                stop_seq = parse_stop_list(matched['stop_id_list'].iloc[0])
                if not stop_seq: continue

                avg_speed = trip_avg_speed if trip_avg_speed is not None else route_avg_speed_m_s.get(str(route_id), global_avg_speed_m_s)

                try:
                    
                    etas = predict_future_stops(row, route_id, stop_seq, avg_speed=avg_speed)
                    if route_id not in output_dict:
                        output_dict[route_id] = {}
                    output_dict[route_id].update(etas)
                except Exception as e:
                    print(f"⚠️ Error predicting {row['trip_id']} in {team_id}: {e}")

# --------------------------
# Save output & print
# --------------------------
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_dict, f, ensure_ascii=False)

print("\n✅ Predictions:")
print(json.dumps(output_dict, indent=4, ensure_ascii=False))
print(f"✅ Done — wrote {len(output_dict)} routes to {output_path}")