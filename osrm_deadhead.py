import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

METERS_TO_MILES = 0.000621371
DEFAULT_PUBLIC_OSRM = "http://router.project-osrm.org"


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _coord_key(lat: float, lon: float, precision: int = 5) -> Tuple[float, float]:
    return (round(float(lat), precision), round(float(lon), precision))


def _query_osrm_table(
    base_url: str,
    garage_coord: Tuple[float, float],
    stop_coords: List[Tuple[float, float]],
    direction: str,
    timeout: float = 30.0,
    max_retries: int = 4,
) -> List[Tuple[float, float]]:
    """
    Queries OSRM /table/v1/driving/ for 1 garage against N unique stop coordinates.
    direction='pull_out': garage (index 0) -> stops (indices 1..N)
    direction='pull_in' : stops (indices 1..N) -> garage (index 0)
    Returns a list of (miles, minutes) tuples aligned with `stop_coords`.
    """
    # OSRM expects lon,lat order
    all_coords = [garage_coord] + stop_coords
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for (lat, lon) in all_coords)
    stop_indices = ";".join(str(i) for i in range(1, len(all_coords)))

    if direction == "pull_out":
        params = f"sources=0&destinations={stop_indices}&annotations=distance,duration"
    else:
        params = f"sources={stop_indices}&destinations=0&annotations=distance,duration"

    url = f"{base_url.rstrip('/')}/table/v1/driving/{coord_str}?{params}"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CEGE4214-TransitOptimizer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))

            if payload.get("code") != "Ok":
                raise RuntimeError(f"OSRM error code: {payload.get('code')} - {payload.get('message', '')}")

            distances = payload["distances"]
            durations = payload["durations"]
            results = []

            for idx, (s_lat, s_lon) in enumerate(stop_coords):
                if direction == "pull_out":
                    dist_m = distances[0][idx]
                    dur_s = durations[0][idx]
                else:
                    dist_m = distances[idx][0]
                    dur_s = durations[idx][0]

                if dist_m is None or dur_s is None:
                    # Fallback for unroutable points
                    mi = _haversine_miles(garage_coord[0], garage_coord[1], s_lat, s_lon) * 1.25
                    mins = (mi / 25.0) * 60.0
                else:
                    mi = dist_m * METERS_TO_MILES
                    mins = dur_s / 60.0

                results.append((round(mi, 4), round(mins, 2)))

            return results

        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                backoff = (2 ** attempt) * 1.5
                print(f"  [Rate Limit / HTTP {e.code}] Backing off {backoff:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(backoff)
            else:
                raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            backoff = (2 ** attempt) * 1.0
            print(f"  [OSRM Retry] {e} — retrying in {backoff:.1f}s...")
            time.sleep(backoff)

    raise RuntimeError("Exceeded max retries querying OSRM table API.")


def build_osrm_deadhead_matrix(
    endpoints_csv: Union[str, Path] = "block_endpoints_osrm.csv",
    garages_csv: Union[str, Path] = "garage_parameters.csv",
    output_csv: Union[str, Path] = "osrm_deadhead_matrix.csv",
    cache_csv: Union[str, Path] = "osrm_cache/osrm_coordinate_cache.csv",
    osrm_base_url: str = DEFAULT_PUBLIC_OSRM,
    batch_size: int = 75,
    min_request_interval: float = 1.1,
) -> pd.DataFrame:
    """
    Builds `osrm_deadhead_matrix.csv` with zero duplicate OSRM queries, persistent disk caching,
    and rate-limit throttling. Set `osrm_base_url="http://localhost:5000"` and
    `min_request_interval=0.0` when running a local OSRM instance.
    """
    blocks_df = pd.read_csv(endpoints_csv)
    garages_df = pd.read_csv(garages_csv)

    cache_path = Path(cache_csv)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load persistent coordinate-to-coordinate cache
    pair_cache: Dict[Tuple[float, float, float, float], Tuple[float, float]] = {}
    if cache_path.exists():
        cdf = pd.read_csv(cache_path)
        for _, r in cdf.iterrows():
            key = (float(r["orig_lat"]), float(r["orig_lon"]), float(r["dest_lat"]), float(r["dest_lon"]))
            pair_cache[key] = (float(r["miles"]), float(r["minutes"]))

    # 2. Collect unique rounded endpoints across all blocks
    unique_stops = set()
    block_start_keys = {}
    block_end_keys = {}

    for _, row in blocks_df.iterrows():
        b_id = str(row["block_id"])
        s_key = _coord_key(row["start_lat"], row["start_lon"])
        e_key = _coord_key(row["end_lat"], row["end_lon"])
        block_start_keys[b_id] = s_key
        block_end_keys[b_id] = e_key
        unique_stops.add(s_key)
        unique_stops.add(e_key)

    unique_stops_list = sorted(list(unique_stops))
    print(f"OSRM Deduplication: {len(blocks_df)} blocks -> {len(unique_stops_list)} unique stop coordinates.")
    print(f"Loaded {len(pair_cache)} cached directed pairs from '{cache_path}'.")

    # 3. Query missing (garage -> stop) and (stop -> garage) pairs in rate-limited batches
    new_queries_made = 0
    last_req_time = 0.0

    for _, g_row in garages_df.iterrows():
        g_id = str(g_row["garage_id"])
        g_coord = _coord_key(g_row["lat"], g_row["lon"])

        missing_pull_out = [
            s for s in unique_stops_list
            if (g_coord[0], g_coord[1], s[0], s[1]) not in pair_cache
        ]
        missing_pull_in = [
            s for s in unique_stops_list
            if (s[0], s[1], g_coord[0], g_coord[1]) not in pair_cache
        ]

        # Batch pull-out queries
        for i in range(0, len(missing_pull_out), batch_size):
            batch = missing_pull_out[i : i + batch_size]
            elapsed = time.time() - last_req_time
            if elapsed < min_request_interval:
                time.sleep(min_request_interval - elapsed)

            res = _query_osrm_table(osrm_base_url, g_coord, batch, direction="pull_out")
            last_req_time = time.time()
            new_queries_made += 1

            for s_coord, (mi, mins) in zip(batch, res):
                pair_cache[(g_coord[0], g_coord[1], s_coord[0], s_coord[1])] = (mi, mins)

        # Batch pull-in queries
        for i in range(0, len(missing_pull_in), batch_size):
            batch = missing_pull_in[i : i + batch_size]
            elapsed = time.time() - last_req_time
            if elapsed < min_request_interval:
                time.sleep(min_request_interval - elapsed)

            res = _query_osrm_table(osrm_base_url, g_coord, batch, direction="pull_in")
            last_req_time = time.time()
            new_queries_made += 1

            for s_coord, (mi, mins) in zip(batch, res):
                pair_cache[(s_coord[0], s_coord[1], g_coord[0], g_coord[1])] = (mi, mins)

        if missing_pull_out or missing_pull_in:
            print(f"  ✔ Garage '{g_id}': fetched {len(missing_pull_out)} pull-out & {len(missing_pull_in)} pull-in paths.")

    # 4. Update persistent cache if new paths were fetched
    if new_queries_made > 0:
        cache_records = [
            {
                "orig_lat": k[0],
                "orig_lon": k[1],
                "dest_lat": k[2],
                "dest_lon": k[3],
                "miles": v[0],
                "minutes": v[1],
            }
            for k, v in pair_cache.items()
        ]
        pd.DataFrame(cache_records).to_csv(cache_path, index=False)
        print(f"✔ Updated cache '{cache_path}' ({new_queries_made} HTTP batch requests used).")
    else:
        print("✔ 100% cache hit — 0 OSRM HTTP requests needed.")

    # 5. Assemble full (garage_id, block_id) matrix for Gurobi
    matrix_rows = []
    for _, g_row in garages_df.iterrows():
        g_id = str(g_row["garage_id"])
        g_coord = _coord_key(g_row["lat"], g_row["lon"])

        for _, b_row in blocks_df.iterrows():
            b_id = str(b_row["block_id"])
            s_coord = block_start_keys[b_id]
            e_coord = block_end_keys[b_id]

            out_mi, out_min = pair_cache[(g_coord[0], g_coord[1], s_coord[0], s_coord[1])]
            in_mi, in_min = pair_cache[(e_coord[0], e_coord[1], g_coord[0], g_coord[1])]

            matrix_rows.append({
                "garage_id": g_id,
                "block_id": b_id,
                "pull_out_miles": round(out_mi, 3),
                "pull_in_miles": round(in_mi, 3),
                "total_deadhead_miles": round(out_mi + in_mi, 3),
                "pull_out_min": round(out_min, 2),
                "pull_in_min": round(in_min, 2),
            })

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(output_csv, index=False)
    print(f"✔ Saved OSRM deadhead matrix to '{output_csv}' ({len(matrix_df)} garage-block pairs).")
    return matrix_df


if __name__ == "__main__":
    build_osrm_deadhead_matrix()