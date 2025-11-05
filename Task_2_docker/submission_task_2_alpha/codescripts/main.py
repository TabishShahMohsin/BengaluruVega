#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import numpy as np
import math
import h3
import ast
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
from pyproj import CRS, Transformer


# ---------------------------
# Config / Defaults
# ---------------------------
# Set this to True to use a constant 10 minutes for pc, or False to calculate it.
USE_CONSTANT_PC = False 

# --- Heuristic Model Parameters ---
H3_RES = 7
CIRCLE_RADIUS_M = 1840.0
# Best parameters found from testing
PA_ALPHA = 2.0
PA_BETA = 0.0
# pb parameters (can be tuned or replaced with a new model)
PB_ALPHA = 0.2
PB_BETA = 0
# Fallback constant for trip time (pc)
PC_CONST = 10.0
# SDR value below which a ride is considered a "failed attempt"
SDR_FAILURE_THRESHOLD = 0.1 

# ---------------------------
# Utility functions
# ---------------------------

def time_to_slot(timestr):
    try:
        hh, mm, ss = [int(x) for x in timestr.split(":")]
    except Exception:
        return 0
    total_minutes = hh * 60 + mm
    return min(max(total_minutes // 15, 0), 95)

def parse_coordinates(s):
    if pd.isna(s):
        return (None, None)
    # Strip parentheses and quotes before splitting
    ss = str(s).strip().replace("(", "").replace(")", "").replace('"', '')
    parts = [p.strip() for p in ss.split(",")]
    
    if len(parts) != 2:
        return (None, None)
    try:
        # Latitude is first, Longitude is second
        return float(parts[0]), float(parts[1])
    except Exception:
        return (None, None)


def build_aeqd_transformer(lat_center, lon_center):
    aeqd_proj4 = f"+proj=aeqd +lat_0={lat_center} +lon_0={lon_center} +units=m +datum=WGS84 +no_defs"
    aeqd_crs = CRS.from_proj4(aeqd_proj4)
    wgs84 = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84, aeqd_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(aeqd_crs, wgs84, always_xy=True)
    return to_proj.transform, to_wgs84.transform

def convert_h3_to_int64(h3_hex_str):
    """Converts a hexadecimal H3 string to a 64-bit integer, safely handling invalid inputs."""
    if pd.isna(h3_hex_str) or h3_hex_str is None:
        return np.int64(0)
    try:
        # Convert hex string (base 16) to integer (base 10), then cast to int64
        return np.int64(int(str(h3_hex_str), 16))
    except ValueError:
        print(f"CHECKPOINT: H3 Conversion Failed for non-hex string: '{h3_hex_str}'")
        return np.int64(0)

def get_h3_index(lat, lon, res):
    """Robustly calls the correct H3 function and returns the H3 index as a HEX STRING."""
    h3_hex_str = None
    
    # Use the most robust function finding chain
    try:
        h3_hex_str = h3.latlng_to_cell(lat, lon, res)
    except AttributeError:
        try:
            h3_hex_str = h3.h3_from_geo(lat, lon, res)
        except AttributeError:
            h3_hex_str = h3.geo_to_h3(lat, lon, res)
            
    return h3_hex_str # Return the hex string


def compute_hex_circle_weights(lat, lon, radius_m=CIRCLE_RADIUS_M, h3_res=H3_RES, k_ring_k=3):
    
    # 1. Get the starting H3 index as hexadecimal string
    center_h3_hexadecimal = get_h3_index(lat, lon, h3_res)
    
    if center_h3_hexadecimal is None or center_h3_hexadecimal == '0':
        print(f"CHECKPOINT: H3 Index calculation skipped for ({lat}, {lon}). Invalid coordinates.")
        return []
        
    # 2. Convert to int64 for the lookup series
    center_h3_int64 = convert_h3_to_int64(center_h3_hexadecimal)

    if center_h3_int64 == np.int64(0):
        print(f"CHECKPOINT: H3 Index conversion failed for '{center_h3_hexadecimal}'. Skipping weights.")
        return []

    # 3. Use the HEXADECIMAL STRING for h3.grid_disk()
    candidates = h3.grid_disk(center_h3_hexadecimal, k_ring_k)
    print(f"CHECKPOINT: Found {len(candidates)} candidate H3 neighbors using {h3.grid_disk.__name__}.")


    to_proj, _ = build_aeqd_transformer(lat, lon)
    cx, cy = to_proj(lon, lat)
    circle_proj = Point(cx, cy).buffer(radius_m, resolution=64)

    weights = []
    # candidates loop: iterates over hex strings
    for h_hex_str in candidates: 
        
        h3_index_int64 = convert_h3_to_int64(h_hex_str) 
        h3_index_hexadecimal = h_hex_str

        try:
            boundary_func = h3.cell_to_boundary
        except AttributeError:
            try:
                boundary_func = h3.h3_to_coords
            except AttributeError:
                boundary_func = h3.h3_to_geo_boundary

        hex_coords = [(lng, lat) for lat, lng in boundary_func(h3_index_hexadecimal)] 
        hex_poly_wgs = Polygon(hex_coords)
        hex_poly_proj = transform(lambda x, y: to_proj(x, y), hex_poly_wgs)

        inter = hex_poly_proj.intersection(circle_proj)
        inter_area = inter.area if not inter.is_empty else 0.0
        hex_area = hex_poly_proj.area if hex_poly_proj.area > 0 else 1.0
        weight = inter_area / hex_area

        if weight > 0:
            weights.append((h3_index_int64, weight))
            
    print(f"CHECKPOINT: Calculated {len(weights)} non-zero weights.")
    return weights

def lookup_sdr(sdr_series, h3_index, slot, fallback=0.0):
    try:
        sdr_value = float(sdr_series.loc[(h3_index, slot)])
        if sdr_value == fallback:
            pass
        return sdr_value
    except Exception:
        return fallback

# ---------------------------
# Models
# ---------------------------

def predict_pa_robust(SDR_list, area_list, alpha=PA_ALPHA, beta=PA_BETA):
    SDR_list = np.array(SDR_list)
    area_list = np.array(area_list)
    weighted_sdr = np.sum(SDR_list * area_list)
    
    if weighted_sdr < SDR_FAILURE_THRESHOLD:
        return 30.0
    else:
        log_transformed_sdr = math.log(1 + weighted_sdr)
        prob = 1 / (1 + np.exp(-(alpha * log_transformed_sdr + beta)))
        predicted_pa = 9.0 * (1 - prob)
        return predicted_pa

def predict_pb_exponential(SDR_list, area_list, alpha=PB_ALPHA, beta=PB_BETA):
    SDR_list = np.array(SDR_list)
    area_list = np.array(area_list)
    y = np.sum(SDR_list * area_list)
    prob = 1 / (1 + np.exp(-(alpha * y + beta)))
    return 7 * (1 - prob)

# ---------------------------
# PC via OSM route time (NEW DYNAMIC METHOD)
# ---------------------------

def edge_speed_in_df(u, v, speed_df, timeslot_col, G):
    def mean_speed(df):
        return float(df[timeslot_col].mean()) if not df.empty else None

    # 1. EXACT MATCHES (u->v or v->u)
    match = speed_df[((speed_df['u'] == u) & (speed_df['v'] == v)) | ((speed_df['u'] == v) & (speed_df['v'] == u))]
    if not match.empty: return mean_speed(match)

    # 2. OSMID MATCH
    edge_data_dict = G.get_edge_data(u, v)
    osmid_set = set()
    if edge_data_dict:
        for _, attr in edge_data_dict.items():
            osmid = attr.get('osmid')
            if osmid is None: continue
            if isinstance(osmid, list): osmid_set.update(osmid)
            else: osmid_set.add(osmid)
    if osmid_set:
        match = speed_df[speed_df['osmid'].isin(osmid_set)]
        if not match.empty: return mean_speed(match)

    # 3. PARTIAL/INFERRED MATCHES (any edge connected to u or v)
    match = speed_df[(speed_df['u'] == u) | (speed_df['v'] == v) | (speed_df['u'] == v) | (speed_df['v'] == u)]
    if not match.empty: return mean_speed(match)

    return np.nan

def predict_pc_dynamic_graph(start_lat, start_lon, end_lat, end_lon, time_str, speed_df, default_speed_kph=19.489212994167335):
    """
    Calculates route travel time by dynamically creating a graph for the route's bounding box.
    WARNING: This is slow due to network requests for each ride.
    """
    try:
        hh, mm, ss = [int(x) for x in time_str.split(":")]
        seconds_from_midnight = hh * 3600 + mm * 60 + ss
        timeslot = str(seconds_from_midnight // 900)

        buffer_deg = 0.01
        north, south = max(start_lat, end_lat) + buffer_deg, min(start_lat, end_lat) - buffer_deg
        east, west = max(start_lon, end_lon) + buffer_deg, min(start_lon, end_lon) - buffer_deg
        bbox_polygon = box(west, south, east, north)
        G = ox.graph_from_polygon(bbox_polygon, network_type="drive_service", simplify=True)

        u_node, v_node = ox.nearest_nodes(G, (start_lon, end_lon), (start_lat, end_lat))
        route_nodes = nx.shortest_path(G, source=u_node, target=v_node, weight='length')
        
        total_time_sec = 0
        last_speed = default_speed_kph
        
        for i in range(len(route_nodes) - 1):
            u, v = route_nodes[i], route_nodes[i + 1]
            avg_speed = edge_speed_in_df(u, v, speed_df, timeslot, G)
            
            if pd.isna(avg_speed) or avg_speed < 0.5:
                avg_speed = last_speed
            
            last_speed = avg_speed
            
            edge_data = G.get_edge_data(u, v, key=0) 
            edge_length_m = edge_data.get('length', 0)
            
            if avg_speed > 0:
                time_sec = (edge_length_m / 1000) / avg_speed * 3600
                total_time_sec += time_sec
        
        return total_time_sec / 60
    
    except Exception as e:
        print(f"PC WARNING: Dynamic graph calculation failed (Error: {e}). Using fallback PC={PC_CONST}.")
        return PC_CONST

# ---------------------------
# Main
# ---------------------------

def main(input_csv, output_json, sdr_parquet_path, speed_path=None):
    
    try:
        df = pd.read_csv(input_csv, dtype=str)
        print(f"CHECKPOINT 1: SUCCESS. Input CSV ({len(df)} rows) loaded.")
        
        # Print the exact input data received by the script to the logs.
        print("\n--- START: Full Input Data Received by Script ---")
        print(df.to_string())
        print("--- END: Full Input Data Received by Script ---\n")

        sdr_df = pd.read_parquet(sdr_parquet_path)
    except Exception as e:
        print(f"CHECKPOINT 1: FATAL ERROR. Could not load core data files. Error: {e}")
        return
    
    slot_columns = [f'slot_{i}' for i in range(96)]
    sdr_long_df = sdr_df.melt(id_vars=['h3_index'], value_vars=slot_columns, var_name='slot_name', value_name='SDR')
    sdr_long_df['slot'] = sdr_long_df['slot_name'].str.split('_').str[-1].astype(int)
    sdr_long_df['h3_index'] = sdr_long_df['h3_index'].astype(np.int64)
    sdr_series = sdr_long_df.set_index(["h3_index", "slot"])["SDR"]
    print(f"CHECKPOINT 2: Data Melt/Index SUCCESS. SDR lookup series created.")

    speed_df = None
    if speed_path and os.path.exists(speed_path):
        try:
            speed_df = pd.read_parquet(speed_path)
            if 'u' in speed_df.columns: speed_df['u'] = speed_df['u'].astype(np.int64)
            if 'v' in speed_df.columns: speed_df['v'] = speed_df['v'].astype(np.int64)
            print(f"CHECKPOINT 3: SUCCESS. Speed Data loaded from '{speed_path}'.")
        except Exception as e:
            print(f"CHECKPOINT 3: WARNING. Speed Data load FAILED: {e}. PC will use fallback.")
    else:
        print(f"CHECKPOINT 3: WARNING. Speed Data file NOT FOUND. PC will use fallback.")

    results = {}
    
    for index, row in df.iterrows():
        rid = row["rid"]
        slot = time_to_slot(row["ride_request_time"])
        
        lat, lon = parse_coordinates(row["ride_start_location"])
        end_lat, end_lon = parse_coordinates(row["ride_end_location"])

        if lat is None or lon is None or end_lat is None or end_lon is None:
            results[rid] = {"pa": 0.0, "pb": 0.0, "pc": PC_CONST}
            continue

        weights = compute_hex_circle_weights(lat, lon)
        if not weights:
            center_h3_hex = get_h3_index(lat, lon, H3_RES)
            center_h3_int = convert_h3_to_int64(center_h3_hex)
            if center_h3_int != np.int64(0):
                weights = [(center_h3_int, 1.0)]
                print(f"CHECKPOINT 5: RID {rid}: H3 search failed, using center hex fallback.")
            else:
                print(f"CHECKPOINT 5: RID {rid}: H3 index invalid/0. Skipping calculation.")
                results[rid] = {"pa": 0.0, "pb": 0.0, "pc": PC_CONST}
                continue
                
        SDRs, areas = [], []
        for h, w in weights:
            SDRs.append(lookup_sdr(sdr_series, h, slot))
            areas.append(w)

        pa = predict_pa_robust(SDRs, areas)
        pb = predict_pb_exponential(SDRs, areas)
        
        if USE_CONSTANT_PC:
            pc = PC_CONST
            print(f"CHECKPOINT 6: RID {rid}: Using constant PC={PC_CONST} as requested.")
        elif speed_df is not None:
            pc = predict_pc_dynamic_graph(lat, lon, end_lat, end_lon, row["ride_request_time"], speed_df)
            print(f"CHECKPOINT 6: RID {rid}: PC calculated dynamically as {pc:.2f} min.")
        else:
            pc = PC_CONST
            print(f"CHECKPOINT 6: RID {rid}: PC using fallback (Speed DF not available).")

        results[rid] = {"pa": round(pa, 2), "pb": round(pb, 2), "pc": round(pc, 2)}
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

# ---------------------------
# Entry point with auto path switch 
# ---------------------------
if __name__ == "__main__":
    IN_DOCKER = os.path.exists("/app")

    if IN_DOCKER:
        DEFAULT_INPUT_CSV = "/app/data/input.csv"
        DEFAULT_OUTPUT_JSON = "/app/out/output.json"
        DEFAULT_SDR_PATH = "/app/ref_data/processed_hex_avg_SDR.parquet"
        DEFAULT_SPEED_PATH = "/app/ref_data/smoothed_speed_full.parquet"
    else:
        DEFAULT_INPUT_CSV = "data/input.csv"
        DEFAULT_OUTPUT_JSON = "out/output.json"
        DEFAULT_SDR_PATH = "ref_data/processed_hex_avg_SDR.parquet"
        DEFAULT_SPEED_PATH = "ref_data/smoothed_speed_full.parquet"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", dest="input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-json", dest="output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--sdr-path", dest="sdr_path", default=DEFAULT_SDR_PATH, help="Hex SDR parquet file")
    parser.add_argument("--speed-path", dest="speed_path", default=DEFAULT_SPEED_PATH, help="OSM speed parquet file")

    args, _ = parser.parse_known_args()

    sdr_path = args.sdr_path
    speed_path = args.speed_path

    main(args.input_csv, args.output_json, sdr_path, speed_path)