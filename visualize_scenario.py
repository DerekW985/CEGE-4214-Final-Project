import argparse
import html
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

# High-contrast color palette assigned consistently to garages across map and charts
GARAGE_PALETTE = [
    "#1f77b4",  # Blue
    "#d62728",  # Red
    "#2ca02c",  # Green
    "#ff7f0e",  # Orange
    "#9467bd",  # Purple
    "#17becf",  # Cyan
    "#e377c2",  # Pink
    "#8c564b",  # Brown
    "#bcbd22",  # Olive
    "#393b79",  # Navy
]

# High-contrast color palette assigned consistently to vehicle types across map and charts
VEHICLE_PALETTE = {
    "40ft_diesel": "#1f77b4",       # Blue
    "40ft_brt": "#ff7f0e",          # Orange
    "60ft_artic": "#d62728",        # Red
    "60ft_brt_artic": "#008080",    # Teal
    "40ft_electric": "#2ca02c",     # Green
    "60ft_electric": "#bcbd22",     # Olive Gold
    "60ft_brt_electric": "#9467bd", # Purple
    "coach": "#e377c2",             # Magenta
    "30ft_diesel": "#8c564b",       # Brown
    "cutaway": "#7f7f7f",           # Slate Gray
}


def _natural_route_sort_key(df: pd.DataFrame, col: str = "primary_route") -> pd.DataFrame:
    """Sorts numeric route IDs naturally (2, 3, 10...) followed by named BRT routes (A_Line, Gold_Line...)."""
    out = df.copy()
    out["_num"] = pd.to_numeric(out[col], errors="coerce").fillna(999999)
    out = out.sort_values(by=["_num", col]).drop(columns=["_num"])
    return out


def _format_gtfs_hour(hour_int: int) -> str:
    """Formats an integer hour from midnight (supports >= 24 for post-midnight GTFS service)."""
    if hour_int < 24:
        return f"{hour_int:02d}:00"
    return f"{hour_int:02d}:00 (+1d {hour_int - 24:02d}:00)"


def _resolve_config_path(scenario_path: Path, config_override: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Finds config_used.yaml inside the scenario folder or falls back to config.yaml."""
    if config_override:
        p = Path(config_override)
        if p.exists():
            return p
    for candidate in [
        scenario_path / "config_used.yaml",
        Path("config.yaml"),
        scenario_path.parent.parent / "config.yaml",
    ]:
        if candidate.exists():
            return candidate
    return None


def _extract_gtfs_route_geometries(
    scenario_path: Path,
    assignments_df: pd.DataFrame,
    config_path: Optional[Path] = None,
    max_points_per_shape: int = 250,
) -> Tuple[Dict[Tuple[str, str], List[List[Tuple[float, float]]]], Dict[Tuple[str, str], str]]:
    """
    Extracts route polylines [(lat, lon), ...] and human-readable route names
    from GTFS shapes.txt (or stop sequences if shapes.txt is missing).
    Returns:
      - route_polylines: {(feed_prefix, route_id): [polyline_dir0, polyline_dir1, ...]}
      - route_names: {(feed_prefix, route_id): "Route Short/Long Name"}
    """
    route_polylines: Dict[Tuple[str, str], List[List[Tuple[float, float]]]] = {}
    route_names: Dict[Tuple[str, str], str] = {}

    needed_pairs = set()
    for _, row in assignments_df[["scoped_agency_id", "primary_route"]].drop_duplicates().iterrows():
        pfx = str(row["scoped_agency_id"]).split("_")[0]
        needed_pairs.add((pfx, str(row["primary_route"])))

    cfg_file = _resolve_config_path(scenario_path, config_path)
    if cfg_file is not None:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        for feed_cfg in cfg.get("gtfs_feeds", []):
            prefix = str(feed_cfg.get("feed_prefix", "GTFS"))
            raw_zip = Path(feed_cfg.get("path", ""))

            zip_path = None
            for cand in [raw_zip, scenario_path / raw_zip, scenario_path.parent.parent / raw_zip]:
                if cand.exists() and cand.is_file():
                    zip_path = cand
                    break

            if zip_path is None:
                print(f"[WARN] GTFS archive '{raw_zip}' not found for route shapes; using stop-coordinate fallback.")
                continue

            target_routes = {rt for (pfx, rt) in needed_pairs if pfx == prefix}
            if not target_routes:
                continue

            with zipfile.ZipFile(zip_path, "r") as z:
                namelist = set(z.namelist())
                routes_df = pd.read_csv(z.open("routes.txt"), dtype=str)
                trips_df = pd.read_csv(z.open("trips.txt"), dtype=str)

                # Extract readable route names
                for _, r_row in routes_df[routes_df["route_id"].isin(target_routes)].iterrows():
                    r_id = str(r_row["route_id"])
                    short_n = str(r_row.get("route_short_name", "") or "").strip()
                    long_n = str(r_row.get("route_long_name", "") or "").strip()
                    if short_n and long_n and short_n.lower() not in long_n.lower():
                        route_names[(prefix, r_id)] = f"{short_n} — {long_n}"
                    else:
                        route_names[(prefix, r_id)] = long_n or short_n or r_id

                trips_sub = trips_df[trips_df["route_id"].isin(target_routes)].copy()

                # Preferred: use shapes.txt
                if "shapes.txt" in namelist and "shape_id" in trips_sub.columns:
                    valid_trips = trips_sub.dropna(subset=["shape_id"]).copy()
                    if not valid_trips.empty:
                        grp_cols = ["route_id", "direction_id"] if "direction_id" in valid_trips.columns else ["route_id"]
                        top_shapes = (
                            valid_trips.groupby(grp_cols + ["shape_id"])
                            .size()
                            .reset_index(name="trip_count")
                            .sort_values("trip_count", ascending=False)
                            .drop_duplicates(subset=grp_cols)
                        )
                        needed_shape_ids = set(top_shapes["shape_id"].astype(str).unique())

                        shapes_df = pd.read_csv(z.open("shapes.txt"), dtype=str)
                        shapes_df = shapes_df[shapes_df["shape_id"].isin(needed_shape_ids)].copy()
                        shapes_df["shape_pt_lat"] = shapes_df["shape_pt_lat"].astype(float)
                        shapes_df["shape_pt_lon"] = shapes_df["shape_pt_lon"].astype(float)
                        shapes_df["shape_pt_sequence"] = shapes_df["shape_pt_sequence"].astype(int)
                        shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])

                        shape_coords_map: Dict[str, List[Tuple[float, float]]] = {}
                        for shp_id, grp in shapes_df.groupby("shape_id", sort=False):
                            pts = list(zip(grp["shape_pt_lat"].tolist(), grp["shape_pt_lon"].tolist()))
                            if len(pts) > max_points_per_shape:
                                step = max(1, len(pts) // max_points_per_shape)
                                pts = pts[::step] + ([pts[-1]] if pts[-1] != pts[::step][-1] else [])
                            if len(pts) >= 2:
                                shape_coords_map[str(shp_id)] = pts

                        for _, s_row in top_shapes.iterrows():
                            r_id = str(s_row["route_id"])
                            shp_id = str(s_row["shape_id"])
                            if shp_id in shape_coords_map:
                                route_polylines.setdefault((prefix, r_id), []).append(shape_coords_map[shp_id])

                # Fallback for any routes without shapes.txt: build polyline from stop_times + stops
                missing_routes = [
                    r_id for r_id in target_routes if (prefix, r_id) not in route_polylines
                ]
                if missing_routes and "stop_times.txt" in namelist and "stops.txt" in namelist:
                    rep_trips = (
                        trips_sub[trips_sub["route_id"].isin(missing_routes)]
                        .drop_duplicates(subset=["route_id"])
                    )
                    rep_trip_ids = set(rep_trips["trip_id"].unique())
                    st_df = pd.read_csv(z.open("stop_times.txt"), dtype=str)
                    st_df = st_df[st_df["trip_id"].isin(rep_trip_ids)].copy()
                    stops_df = pd.read_csv(z.open("stops.txt"), dtype=str)[["stop_id", "stop_lat", "stop_lon"]]
                    st_df = st_df.merge(stops_df, on="stop_id", how="left").dropna(subset=["stop_lat", "stop_lon"])
                    st_df["stop_sequence"] = st_df["stop_sequence"].astype(int)
                    st_df["stop_lat"] = st_df["stop_lat"].astype(float)
                    st_df["stop_lon"] = st_df["stop_lon"].astype(float)
                    st_df = st_df.sort_values(["trip_id", "stop_sequence"])

                    trip_to_route = dict(zip(rep_trips["trip_id"], rep_trips["route_id"]))
                    for t_id, grp in st_df.groupby("trip_id", sort=False):
                        r_id = str(trip_to_route[t_id])
                        pts = list(zip(grp["stop_lat"].tolist(), grp["stop_lon"].tolist()))
                        if len(pts) >= 2:
                            route_polylines[(prefix, r_id)] = [pts]

    # Final fallback if GTFS zip was unavailable: use block start/end segments from assignments_df
    for (pfx, r_id) in needed_pairs:
        if (pfx, r_id) not in route_polylines:
            sub = assignments_df[
                (assignments_df["scoped_agency_id"].str.startswith(pfx + "_"))
                & (assignments_df["primary_route"] == r_id)
            ]
            segs = []
            for _, b_row in sub[["start_lat", "start_lon", "end_lat", "end_lon"]].drop_duplicates().iterrows():
                segs.append([(float(b_row["start_lat"]), float(b_row["start_lon"])),
                             (float(b_row["end_lat"]), float(b_row["end_lon"]))])
            if segs:
                route_polylines[(pfx, r_id)] = segs

    return route_polylines, route_names


def _load_scenario_tables(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """Loads scenario CSV outputs and computes summary tables if not already present."""
    req_files = [
        "gurobi_solution_assignments.csv",
        "garage_fleet_summary.csv",
        "garage_parameters.csv",
    ]
    for fname in req_files:
        if not (run_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing '{fname}' in '{run_dir}'. Ensure the scenario solved cleanly first."
            )

    assignments_df = pd.read_csv(run_dir / "gurobi_solution_assignments.csv")
    assignments_df["block_id"] = assignments_df["block_id"].astype(str)
    assignments_df["primary_route"] = assignments_df["primary_route"].astype(str)
    assignments_df["assigned_garage"] = assignments_df["assigned_garage"].astype(str)
    assignments_df["vehicle_type"] = assignments_df["vehicle_type"].astype(str)

    # Ensure start_time and end_time exist (merge from gurobi_block_parameters.csv if needed)
    blocks_param_path = run_dir / "gurobi_block_parameters.csv"
    if (
        ("start_time" not in assignments_df.columns or "end_time" not in assignments_df.columns)
        and blocks_param_path.exists()
    ):
        bp_df = pd.read_csv(blocks_param_path, dtype={"block_id": str})
        merge_cols = [
            c
            for c in ["block_id", "start_time", "end_time"]
            if (c in bp_df.columns and c not in assignments_df.columns) or c == "block_id"
        ]
        if len(merge_cols) > 1:
            assignments_df = assignments_df.merge(bp_df[merge_cols], on="block_id", how="left")

    garage_fleet_df = pd.read_csv(run_dir / "garage_fleet_summary.csv")
    garages_df = pd.read_csv(run_dir / "garage_parameters.csv")

    # Garage capacity utilization table
    cap_path = run_dir / "garage_capacity_utilization.csv"
    if cap_path.exists():
        garage_cap_df = pd.read_csv(cap_path)
    else:
        stationed = (
            garage_fleet_df.groupby("garage_id")
            .agg(
                buses_stationed=("buses_stationed", "sum"),
                blocks_assigned=("blocks_assigned", "sum"),
                revenue_miles=("total_revenue_miles", "sum"),
                deadhead_miles=("total_deadhead_miles", "sum"),
            )
            .reset_index()
        )
        ev_stat = (
            garage_fleet_df[garage_fleet_df["vehicle_type"].str.contains("electric", case=False)]
            .groupby("garage_id")["buses_stationed"]
            .sum()
            .reset_index(name="ev_buses_stationed")
        )
        garage_cap_df = (
            garages_df[["garage_id", "total_capacity_Q_g", "ev_chargers_x_g"]]
            .merge(stationed, on="garage_id", how="left")
            .merge(ev_stat, on="garage_id", how="left")
            .fillna(0)
        )
        garage_cap_df["cap_util_%"] = (
            (garage_cap_df["buses_stationed"] / garage_cap_df["total_capacity_Q_g"].replace(0, 1)) * 100
        ).round(1)

    # Route-to-garage breakdown table
    rg_path = run_dir / "route_garage_breakdown.csv"
    if rg_path.exists():
        route_garage_df = pd.read_csv(rg_path)
        route_garage_df["primary_route"] = route_garage_df["primary_route"].astype(str)
    else:
        route_garage_df = (
            assignments_df.groupby(["scoped_agency_id", "primary_route", "assigned_garage"])
            .agg(
                blocks_assigned=("block_id", "count"),
                deadhead_miles=("deadhead_miles", "sum"),
                vehicle_types=("vehicle_type", lambda s: "|".join(sorted(set(s.astype(str))))),
            )
            .reset_index()
        )
        route_garage_df["route_total_blocks"] = route_garage_df.groupby(
            ["scoped_agency_id", "primary_route"]
        )["blocks_assigned"].transform("sum")
        route_garage_df["route_share_%"] = (
            (route_garage_df["blocks_assigned"] / route_garage_df["route_total_blocks"]) * 100
        ).round(1)

    # Route-to-vehicle-type breakdown table
    route_vehicle_df = (
        assignments_df.groupby(["scoped_agency_id", "primary_route", "vehicle_type"])
        .agg(
            blocks_assigned=("block_id", "count"),
            deadhead_miles=("deadhead_miles", "sum"),
            garages=("assigned_garage", lambda s: "|".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )
    route_vehicle_df["route_total_blocks"] = route_vehicle_df.groupby(
        ["scoped_agency_id", "primary_route"]
    )["blocks_assigned"].transform("sum")
    route_vehicle_df["route_share_%"] = (
        (route_vehicle_df["blocks_assigned"] / route_vehicle_df["route_total_blocks"]) * 100
    ).round(1)

    return {
        "assignments": assignments_df,
        "garage_fleet": garage_fleet_df,
        "garages": garages_df,
        "garage_cap": garage_cap_df,
        "route_garage": route_garage_df,
        "route_vehicle": route_vehicle_df,
    }


def _compute_hourly_dispatch_profile(assignments_df: pd.DataFrame) -> Dict[str, object]:
    """
    Computes hourly active buses in service (by garage and total) plus hourly garage
    pull-out (departures) and pull-in (returns) counts across the operating day.
    """
    if "start_time" not in assignments_df.columns or "end_time" not in assignments_df.columns:
        return {
            "hours": [],
            "hour_labels": [],
            "garages": [],
            "active_by_garage": {},
            "active_total": [],
            "pull_outs_total": [],
            "pull_ins_total": [],
            "po_garage_breakdown": [],
            "pi_garage_breakdown": [],
        }

    df = assignments_df.dropna(subset=["start_time", "end_time"]).copy()
    df["start_time"] = df["start_time"].astype(float)
    df["end_time"] = df["end_time"].astype(float)

    if df.empty:
        return {
            "hours": [],
            "hour_labels": [],
            "garages": [],
            "active_by_garage": {},
            "active_total": [],
            "pull_outs_total": [],
            "pull_ins_total": [],
            "po_garage_breakdown": [],
            "pi_garage_breakdown": [],
        }

    min_hr = int(np.floor(df["start_time"].min() / 60.0))
    max_hr = int(np.ceil(df["end_time"].max() / 60.0))
    hours = list(range(min_hr, max(max_hr, min_hr + 1)))
    hour_labels = [_format_gtfs_hour(h) for h in hours]

    garages = sorted(df["assigned_garage"].unique())
    df["po_hour"] = np.floor(df["start_time"] / 60.0).astype(int)
    df["pi_hour"] = np.floor(df["end_time"] / 60.0).astype(int)

    active_by_garage: Dict[str, List[int]] = {g: [] for g in garages}
    active_total: List[int] = []
    pull_outs_total: List[int] = []
    pull_ins_total: List[int] = []
    po_garage_breakdown: List[str] = []
    pi_garage_breakdown: List[str] = []

    for h in hours:
        h_start = h * 60.0
        h_end = (h + 1) * 60.0

        # Active in service at any point during [h_start, h_end)
        active_mask = (df["start_time"] < h_end) & (df["end_time"] > h_start)
        active_sub = df[active_mask]
        active_total.append(int(len(active_sub)))

        g_active_counts = active_sub["assigned_garage"].value_counts().to_dict()
        for g in garages:
            active_by_garage[g].append(int(g_active_counts.get(g, 0)))

        # Pull-outs (garage departures) during hour h
        po_sub = df[df["po_hour"] == h]
        pull_outs_total.append(int(len(po_sub)))
        if po_sub.empty:
            po_garage_breakdown.append("None")
        else:
            po_counts = po_sub["assigned_garage"].value_counts()
            po_garage_breakdown.append(", ".join(f"{g}: {cnt}" for g, cnt in po_counts.items()))

        # Pull-ins (garage returns) during hour h
        pi_sub = df[df["pi_hour"] == h]
        pull_ins_total.append(int(len(pi_sub)))
        if pi_sub.empty:
            pi_garage_breakdown.append("None")
        else:
            pi_counts = pi_sub["assigned_garage"].value_counts()
            pi_garage_breakdown.append(", ".join(f"{g}: {cnt}" for g, cnt in pi_counts.items()))

    return {
        "hours": hours,
        "hour_labels": hour_labels,
        "garages": garages,
        "active_by_garage": active_by_garage,
        "active_total": active_total,
        "pull_outs_total": pull_outs_total,
        "pull_ins_total": pull_ins_total,
        "po_garage_breakdown": po_garage_breakdown,
        "pi_garage_breakdown": pi_garage_breakdown,
    }


def build_folium_map(
    scenario_path: Path,
    tables: Dict[str, pd.DataFrame],
    garage_colors: Dict[str, str],
    output_html: Path,
    map_mode: str = "routes",
    config_path: Optional[Path] = None,
) -> folium.Map:
    """
    Creates an interactive Folium map supporting:
      - map_mode="routes"    : Draws GTFS routes colored by dominant assigned garage.
      - map_mode="bus_types" : Draws GTFS routes colored by dominant assigned vehicle/bus type.
      - map_mode="deadhead"  : Draws aggregated pull-out/pull-in deadhead lines.
      - map_mode="both"      : Shows both garage-colored routes and pull-out deadhead lines on load.
    All layer groups remain toggleable in the top-right LayerControl.
    """
    assignments_df = tables["assignments"]
    garages_df = tables["garages"]
    garage_cap_df = tables["garage_cap"]
    route_garage_df = tables["route_garage"]
    route_vehicle_df = tables["route_vehicle"]

    show_garage_routes_default = map_mode in ("routes", "both")
    show_bus_type_routes_default = map_mode == "bus_types"
    show_deadhead_default = map_mode in ("deadhead", "both")

    active_garages = set(assignments_df["assigned_garage"].unique())
    active_vehicle_types = sorted(assignments_df["vehicle_type"].unique())

    g_merged = garages_df.merge(
        garage_cap_df[
            [
                "garage_id",
                "buses_stationed",
                "blocks_assigned",
                "ev_buses_stationed",
                "cap_util_%",
                "revenue_miles",
                "deadhead_miles",
            ]
        ],
        on="garage_id",
        how="left",
    ).fillna(0)

    center_df = g_merged[g_merged["garage_id"].isin(active_garages)]
    if center_df.empty:
        center_df = g_merged
    center_lat = float(center_df["lat"].mean())
    center_lon = float(center_df["lon"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    route_polylines, route_names = _extract_gtfs_route_geometries(
        scenario_path=scenario_path,
        assignments_df=assignments_df,
        config_path=config_path,
    )

    # ------------------------------------------------------------------
    # LAYER SET A: GTFS Routes Colored by Dominant Assigned Garage
    # ------------------------------------------------------------------
    route_groups_by_garage: Dict[str, folium.FeatureGroup] = {
        g_id: folium.FeatureGroup(
            name=f"Garage Routes: {g_id}",
            show=show_garage_routes_default,
        )
        for g_id in sorted(active_garages)
    }

    rg_sorted = route_garage_df.sort_values(
        by=["scoped_agency_id", "primary_route", "blocks_assigned"],
        ascending=[True, True, False],
    )
    for (agency_id, r_id), grp in rg_sorted.groupby(["scoped_agency_id", "primary_route"], sort=False):
        pfx = str(agency_id).split("_")[0]
        r_str = str(r_id)
        polylines = route_polylines.get((pfx, r_str), [])
        if not polylines:
            continue

        dominant_row = grp.iloc[0]
        dom_garage = str(dominant_row["assigned_garage"])
        dom_share = float(dominant_row["route_share_%"])
        total_blocks = int(dominant_row["route_total_blocks"])
        color = garage_colors.get(dom_garage, "#333333")
        is_split = len(grp) > 1

        display_name = route_names.get((pfx, r_str), f"Route {r_str}")
        split_lines_html = "".join(
            f"<li><b>{row['assigned_garage']}</b>: {int(row['blocks_assigned'])} blocks "
            f"({row['route_share_%']:.1f}%) — {row['deadhead_miles']:.1f} dh mi</li>"
            for _, row in grp.iterrows()
        )
        v_types_all = "|".join(sorted(set(grp["vehicle_types"].astype(str))))

        tooltip_txt = (
            f"<b>Route {r_str}</b> ({display_name})<br>"
            f"Dominant Garage: <b>{dom_garage}</b> ({dom_share:.1f}% of {total_blocks} blocks)"
            + ("<br><i>Split across multiple garages — click for details</i>" if is_split else "")
        )
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; min-width: 240px;">
            <h4 style="margin: 0 0 4px 0; color: {color};">Route {html.escape(r_str)}</h4>
            <div style="color: #555; margin-bottom: 6px;">{html.escape(display_name)}</div>
            <b>Total Blocks:</b> {total_blocks}<br>
            <b>Vehicle Types:</b> {html.escape(v_types_all)}<br>
            <b>Garage Assignment Breakdown:</b>
            <ul style="margin: 4px 0 0 16px; padding: 0;">{split_lines_html}</ul>
        </div>
        """

        target_fg = route_groups_by_garage.get(dom_garage)
        if target_fg is None:
            continue

        weight = min(2.8 + math.log2(total_blocks + 1) * 0.9, 6.5)
        for pts in polylines:
            folium.PolyLine(
                locations=pts,
                color=color,
                weight=weight,
                opacity=0.82,
                tooltip=tooltip_txt,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(target_fg)

    for g_id in sorted(active_garages):
        route_groups_by_garage[g_id].add_to(m)

    # ------------------------------------------------------------------
    # LAYER SET B: GTFS Routes Colored by Dominant Bus Type
    # ------------------------------------------------------------------
    route_groups_by_vehicle: Dict[str, folium.FeatureGroup] = {
        v_type: folium.FeatureGroup(
            name=f"Bus Type: {v_type}",
            show=show_bus_type_routes_default,
        )
        for v_type in active_vehicle_types
    }

    rv_sorted = route_vehicle_df.sort_values(
        by=["scoped_agency_id", "primary_route", "blocks_assigned"],
        ascending=[True, True, False],
    )
    for (agency_id, r_id), grp in rv_sorted.groupby(["scoped_agency_id", "primary_route"], sort=False):
        pfx = str(agency_id).split("_")[0]
        r_str = str(r_id)
        polylines = route_polylines.get((pfx, r_str), [])
        if not polylines:
            continue

        dominant_row = grp.iloc[0]
        dom_vtype = str(dominant_row["vehicle_type"])
        dom_share = float(dominant_row["route_share_%"])
        total_blocks = int(dominant_row["route_total_blocks"])
        color = VEHICLE_PALETTE.get(dom_vtype, "#636efa")
        is_mixed = len(grp) > 1

        display_name = route_names.get((pfx, r_str), f"Route {r_str}")
        vtype_lines_html = "".join(
            f"<li><b>{row['vehicle_type']}</b>: {int(row['blocks_assigned'])} blocks "
            f"({row['route_share_%']:.1f}%) — Garages: {html.escape(str(row['garages']))}</li>"
            for _, row in grp.iterrows()
        )
        garages_all = "|".join(sorted(set(grp["garages"].astype(str))))

        tooltip_txt = (
            f"<b>Route {r_str}</b> ({display_name})<br>"
            f"Dominant Bus Type: <b>{dom_vtype}</b> ({dom_share:.1f}% of {total_blocks} blocks)"
            + ("<br><i>Mixed bus types — click for breakdown</i>" if is_mixed else "")
        )
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; min-width: 240px;">
            <h4 style="margin: 0 0 4px 0; color: {color};">Route {html.escape(r_str)}</h4>
            <div style="color: #555; margin-bottom: 6px;">{html.escape(display_name)}</div>
            <b>Total Blocks:</b> {total_blocks}<br>
            <b>Assigned Garages:</b> {html.escape(garages_all)}<br>
            <b>Fleet Type Breakdown:</b>
            <ul style="margin: 4px 0 0 16px; padding: 0;">{vtype_lines_html}</ul>
        </div>
        """

        target_fg = route_groups_by_vehicle.get(dom_vtype)
        if target_fg is None:
            continue

        weight = min(2.8 + math.log2(total_blocks + 1) * 0.9, 6.5)
        for pts in polylines:
            folium.PolyLine(
                locations=pts,
                color=color,
                weight=weight,
                opacity=0.85,
                tooltip=tooltip_txt,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(target_fg)

    for v_type in active_vehicle_types:
        route_groups_by_vehicle[v_type].add_to(m)

    # ------------------------------------------------------------------
    # LAYER SET C: Aggregated Deadhead Pull-Out & Pull-In Flows
    # ------------------------------------------------------------------
    garage_coords = {
        str(r["garage_id"]): (float(r["lat"]), float(r["lon"]))
        for _, r in g_merged.iterrows()
    }

    po_df = assignments_df.copy()
    po_df["s_lat_r"] = po_df["start_lat"].round(4)
    po_df["s_lon_r"] = po_df["start_lon"].round(4)
    pull_out_corridors = (
        po_df.groupby(["assigned_garage", "s_lat_r", "s_lon_r"])
        .agg(
            block_count=("block_id", "count"),
            total_po_miles=("pull_out_miles", "sum"),
            avg_po_miles=("pull_out_miles", "mean"),
            routes=("primary_route", lambda s: ", ".join(sorted(set(s.astype(str)))[:10])),
        )
        .reset_index()
    )

    pi_df = assignments_df.copy()
    pi_df["e_lat_r"] = pi_df["end_lat"].round(4)
    pi_df["e_lon_r"] = pi_df["end_lon"].round(4)
    pull_in_corridors = (
        pi_df.groupby(["assigned_garage", "e_lat_r", "e_lon_r"])
        .agg(
            block_count=("block_id", "count"),
            total_pi_miles=("pull_in_miles", "sum"),
            avg_pi_miles=("pull_in_miles", "mean"),
            routes=("primary_route", lambda s: ", ".join(sorted(set(s.astype(str)))[:10])),
        )
        .reset_index()
    )

    for g_id in sorted(active_garages):
        color = garage_colors.get(g_id, "#333333")
        g_lat, g_lon = garage_coords[g_id]

        fg_out = folium.FeatureGroup(name=f"Deadhead Pull-Out: {g_id}", show=show_deadhead_default)
        fg_in = folium.FeatureGroup(name=f"Deadhead Pull-In: {g_id}", show=False)

        sub_out = pull_out_corridors[pull_out_corridors["assigned_garage"] == g_id]
        for _, row in sub_out.iterrows():
            cnt = int(row["block_count"])
            weight = min(1.5 + math.log2(cnt + 1) * 1.4, 8.0)
            tooltip = (
                f"<b>Pull-Out: {g_id}</b><br>"
                f"Blocks: {cnt} | Avg: {row['avg_po_miles']:.2f} mi | Total: {row['total_po_miles']:.1f} mi<br>"
                f"Routes: {row['routes']}"
            )
            folium.PolyLine(
                locations=[[g_lat, g_lon], [row["s_lat_r"], row["s_lon_r"]]],
                color=color,
                weight=weight,
                opacity=0.60,
                tooltip=tooltip,
            ).add_to(fg_out)

            folium.CircleMarker(
                location=[row["s_lat_r"], row["s_lon_r"]],
                radius=min(3 + cnt * 0.4, 9),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=tooltip,
            ).add_to(fg_out)

        sub_in = pull_in_corridors[pull_in_corridors["assigned_garage"] == g_id]
        for _, row in sub_in.iterrows():
            cnt = int(row["block_count"])
            weight = min(1.5 + math.log2(cnt + 1) * 1.4, 8.0)
            tooltip = (
                f"<b>Pull-In: {g_id}</b><br>"
                f"Blocks: {cnt} | Avg: {row['avg_pi_miles']:.2f} mi | Total: {row['total_pi_miles']:.1f} mi<br>"
                f"Routes: {row['routes']}"
            )
            folium.PolyLine(
                locations=[[row["e_lat_r"], row["e_lon_r"]], [g_lat, g_lon]],
                color=color,
                weight=weight,
                opacity=0.60,
                dash_array="6, 6",
                tooltip=tooltip,
            ).add_to(fg_in)

            folium.CircleMarker(
                location=[row["e_lat_r"], row["e_lon_r"]],
                radius=min(3 + cnt * 0.4, 9),
                color=color,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=0.9,
                tooltip=tooltip,
            ).add_to(fg_in)

        fg_out.add_to(m)
        fg_in.add_to(m)

    # ------------------------------------------------------------------
    # LAYER SET D: Garage Hub Markers (Always on top)
    # ------------------------------------------------------------------
    fg_garages = folium.FeatureGroup(name="Garages (Hubs & Capacity)", show=True)
    for _, g_row in g_merged.iterrows():
        g_id = str(g_row["garage_id"])
        is_active = g_id in active_garages
        color = garage_colors.get(g_id, "#7f7f7f") if is_active else "#999999"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; min-width: 210px;">
            <h4 style="margin: 0 0 6px 0; color: {color};">{g_id}</h4>
            <b>Status:</b> {'Active' if is_active else 'Inactive / Excluded'}<br>
            <b>Buses Stationed:</b> {int(g_row['buses_stationed'])} / {int(g_row['total_capacity_Q_g'])} ({g_row['cap_util_%']:.1f}%)<br>
            <b>EV Buses / Chargers:</b> {int(g_row['ev_buses_stationed'])} / {int(g_row['ev_chargers_x_g'])}<br>
            <b>Blocks Assigned:</b> {int(g_row['blocks_assigned'])}<br>
            <b>Revenue Miles:</b> {g_row['revenue_miles']:,.1f} mi<br>
            <b>Deadhead Miles:</b> {g_row['deadhead_miles']:,.1f} mi
        </div>
        """
        folium.Marker(
            location=[float(g_row["lat"]), float(g_row["lon"])],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{g_id}: {int(g_row['buses_stationed'])}/{int(g_row['total_capacity_Q_g'])} buses ({g_row['cap_util_%']:.1f}%)",
            icon=folium.Icon(color="black" if is_active else "lightgray", icon="bus", prefix="fa"),
        ).add_to(fg_garages)

        if is_active:
            folium.CircleMarker(
                location=[float(g_row["lat"]), float(g_row["lon"])],
                radius=14,
                color=color,
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.25,
            ).add_to(fg_garages)

    fg_garages.add_to(m)

    # Floating Legend adapted to initial map_mode
    if map_mode == "bus_types":
        legend_title = "Assigned Bus Types"
        legend_items = "".join(
            f'<div style="margin: 3px 0;"><span style="display:inline-block;width:12px;height:12px;'
            f'background:{VEHICLE_PALETTE.get(v, "#636efa")};margin-right:6px;border-radius:2px;"></span>{v}</div>'
            for v in active_vehicle_types
        )
        mode_notes = [
            "<div><b>Route Color:</b> Dominant bus type</div>",
            '<div style="color:#666; font-size:11px; margin-top:3px;">Switch layers in top-right menu</div>',
        ]
    else:
        legend_title = "Active Garages"
        legend_items = "".join(
            f'<div style="margin: 3px 0;"><span style="display:inline-block;width:12px;height:12px;'
            f'background:{garage_colors[g]};margin-right:6px;border-radius:2px;"></span>{g}</div>'
            for g in sorted(active_garages)
        )
        mode_notes = []
        if show_garage_routes_default:
            mode_notes.append("<div><b>Route Color:</b> Dominant assigned garage</div>")
        if show_deadhead_default:
            mode_notes.append("<div><b>Solid / Dashed:</b> Pull-Out / Pull-In deadhead</div>")
        mode_notes.append('<div style="color:#666; font-size:11px; margin-top:3px;">Switch layers in top-right menu</div>')

    legend_html = f"""
    <div style="
        position: fixed; bottom: 25px; left: 25px; z-index: 9999;
        background: rgba(255, 255, 255, 0.94); padding: 10px 14px;
        border: 1px solid #ccc; border-radius: 6px; font-family: Arial, sans-serif;
        font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    ">
        <div style="font-weight: bold; margin-bottom: 5px;">{legend_title}</div>
        {legend_items}
        <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ddd;">
        {''.join(mode_notes)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(str(output_html))
    return m


def build_plotly_charts(
    tables: Dict[str, pd.DataFrame],
    garage_colors: Dict[str, str],
    scenario_name: str,
    output_html: Path,
) -> go.Figure:
    """
    Builds a 4-panel interactive Plotly figure:
      1. Garage Capacity & Fleet Mix (Stacked Bar + Capacity Limit Markers)
      2. EV Charger Capacity vs. Stationed Electric Buses
      3. Route-to-Garage Assignments (Stacked Bar with Block Count / 100% Share Toggle)
      4. Hourly Active Buses in Service (by Garage) & Pull-Out / Pull-In Dispatch Peaks
    """
    assignments_df = tables["assignments"]
    garage_fleet_df = tables["garage_fleet"]
    garage_cap_df = tables["garage_cap"].copy()
    route_garage_df = _natural_route_sort_key(tables["route_garage"], "primary_route")

    garage_cap_df = garage_cap_df.sort_values(
        by=["buses_stationed", "total_capacity_Q_g"], ascending=[False, False]
    )
    garages_ordered = garage_cap_df["garage_id"].tolist()

    fig = make_subplots(
        rows=4,
        cols=1,
        row_heights=[0.22, 0.16, 0.36, 0.26],
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
        subplot_titles=(
            "1. Garage Fleet Allocation by Vehicle Type vs. Total Bay Capacity",
            "2. EV Charger Availability vs. Stationed Electric Fleet",
            "3. Route-to-Garage Block Assignments (Use Dropdown to Toggle Count vs. % Share)",
            "4. Hourly Active Buses in Service & Garage Pull-Out / Pull-In Peaks Across the Day",
        ),
    )

    # ------------------------------------------------------------------
    # PANEL 1: Garage Capacity Utilization (Stacked by Vehicle Type)
    # ------------------------------------------------------------------
    vehicle_types_present = sorted(garage_fleet_df["vehicle_type"].unique())
    for v_type in vehicle_types_present:
        sub = garage_fleet_df[garage_fleet_df["vehicle_type"] == v_type]
        counts_map = dict(zip(sub["garage_id"], sub["buses_stationed"]))
        y_vals = [counts_map.get(g, 0) for g in garages_ordered]
        fig.add_trace(
            go.Bar(
                name=v_type,
                x=garages_ordered,
                y=y_vals,
                marker_color=VEHICLE_PALETTE.get(v_type, "#636efa"),
                legendgroup="vehicles",
                legendgrouptitle_text="Vehicle Types",
                hovertemplate="<b>%{x}</b><br>" + f"{v_type}: " + "%{y} buses<extra></extra>",
            ),
            row=1,
            col=1,
        )

    cap_labels = [
        f"{int(r['buses_stationed'])}/{int(r['total_capacity_Q_g'])} ({r['cap_util_%']:.0f}%)"
        for _, r in garage_cap_df.iterrows()
    ]
    fig.add_trace(
        go.Scatter(
            name="Max Bay Capacity (Q_g)",
            x=garages_ordered,
            y=garage_cap_df["total_capacity_Q_g"].tolist(),
            mode="markers+text",
            text=cap_labels,
            textposition="top center",
            textfont=dict(size=10, color="#111111"),
            marker=dict(symbol="line-ew-open", size=22, line=dict(width=3, color="#111111")),
            legendgroup="vehicles",
            hovertemplate="<b>%{x}</b><br>Total Capacity: %{y} bays<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # ------------------------------------------------------------------
    # PANEL 2: EV Charger Utilization
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Bar(
            name="EV Buses Stationed",
            x=garages_ordered,
            y=garage_cap_df["ev_buses_stationed"].tolist(),
            marker_color="#2ca02c",
            legendgroup="ev",
            legendgrouptitle_text="EV Infrastructure",
            hovertemplate="<b>%{x}</b><br>EV Buses Stationed: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Installed EV Chargers (x_g)",
            x=garages_ordered,
            y=garage_cap_df["ev_chargers_x_g"].tolist(),
            mode="markers+text",
            text=garage_cap_df["ev_chargers_x_g"].astype(int).astype(str).tolist(),
            textposition="top center",
            marker=dict(symbol="diamond", size=11, color="#d62728"),
            legendgroup="ev",
            hovertemplate="<b>%{x}</b><br>EV Chargers: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # ------------------------------------------------------------------
    # PANEL 3: Route-to-Garage Stacked Bar (Block Count & % Share)
    # ------------------------------------------------------------------
    routes_ordered = list(dict.fromkeys(route_garage_df["primary_route"].tolist()))
    active_garages_rg = sorted(route_garage_df["assigned_garage"].unique())

    blocks_lookup = {
        (r["primary_route"], r["assigned_garage"]): int(r["blocks_assigned"])
        for _, r in route_garage_df.iterrows()
    }
    share_lookup = {
        (r["primary_route"], r["assigned_garage"]): float(r["route_share_%"])
        for _, r in route_garage_df.iterrows()
    }
    dh_lookup = {
        (r["primary_route"], r["assigned_garage"]): float(r["deadhead_miles"])
        for _, r in route_garage_df.iterrows()
    }

    for g_id in active_garages_rg:
        y_blocks = [blocks_lookup.get((rt, g_id), 0) for rt in routes_ordered]
        y_shares = [share_lookup.get((rt, g_id), 0.0) for rt in routes_ordered]
        custom_data = [
            [blocks_lookup.get((rt, g_id), 0), share_lookup.get((rt, g_id), 0.0), dh_lookup.get((rt, g_id), 0.0)]
            for rt in routes_ordered
        ]
        fig.add_trace(
            go.Bar(
                name=f"Garage: {g_id}",
                x=routes_ordered,
                y=y_blocks,
                marker_color=garage_colors.get(g_id, "#333333"),
                legendgroup="garages",
                legendgrouptitle_text="Assigned Garages",
                customdata=custom_data,
                meta={"blocks": y_blocks, "shares": y_shares},
                hovertemplate=(
                    "<b>Route %{x} &rarr; " + g_id + "</b><br>"
                    "Blocks: %{customdata[0]} (%{customdata[1]:.1f}%)<br>"
                    "Deadhead: %{customdata[2]:.1f} mi<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

    rg_trace_indices = [
        idx for idx, tr in enumerate(fig.data) if getattr(tr, "legendgroup", None) == "garages"
    ]
    blocks_arrays = [fig.data[idx].meta["blocks"] for idx in rg_trace_indices]
    shares_arrays = [fig.data[idx].meta["shares"] for idx in rg_trace_indices]

    # ------------------------------------------------------------------
    # PANEL 4: Hourly Active Buses in Service & Pull-Out / Pull-In Peaks
    # ------------------------------------------------------------------
    dispatch = _compute_hourly_dispatch_profile(assignments_df)
    hour_labels = dispatch["hour_labels"]

    if hour_labels:
        # Stacked area traces showing active buses in service per garage
        for g_id in dispatch["garages"]:
            g_col = garage_colors.get(g_id, "#333333")
            fig.add_trace(
                go.Scatter(
                    name=f"Active: {g_id}",
                    x=hour_labels,
                    y=dispatch["active_by_garage"][g_id],
                    mode="lines",
                    stackgroup="active_buses",
                    line=dict(width=0.8, color=g_col),
                    fillcolor=g_col,
                    legendgroup="dispatch",
                    legendgrouptitle_text="Hourly Service & Dispatch",
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        f"Active ({g_id}): " + "%{y} buses<extra></extra>"
                    ),
                ),
                row=4,
                col=1,
                secondary_y=False,
            )

        # Bold total active buses line on top of stacked area
        fig.add_trace(
            go.Scatter(
                name="Total Active Buses in Service",
                x=hour_labels,
                y=dispatch["active_total"],
                mode="lines+markers",
                line=dict(width=2.5, color="#0f172a"),
                marker=dict(size=5, color="#0f172a"),
                legendgroup="dispatch",
                legendgrouptitle_text="Hourly Service & Dispatch",
                hovertemplate="<b>%{x}</b><br>Total Active in Service: <b>%{y} buses</b><extra></extra>",
            ),
            row=4,
            col=1,
            secondary_y=False,
        )

        # Secondary Y-axis: Hourly Pull-Outs (Departures) & Pull-Ins (Returns)
        fig.add_trace(
            go.Scatter(
                name="Pull-Outs (Garage Departures/hr)",
                x=hour_labels,
                y=dispatch["pull_outs_total"],
                mode="lines+markers",
                line=dict(width=2.6, color="#16a34a"),
                marker=dict(symbol="triangle-up", size=8, color="#16a34a"),
                customdata=dispatch["po_garage_breakdown"],
                legendgroup="dispatch",
                hovertemplate=(
                    "<b>%{x} — Pull-Outs</b><br>"
                    "Total Departures: <b>%{y} buses</b><br>"
                    "By Garage: %{customdata}<extra></extra>"
                ),
            ),
            row=4,
            col=1,
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                name="Pull-Ins (Garage Returns/hr)",
                x=hour_labels,
                y=dispatch["pull_ins_total"],
                mode="lines+markers",
                line=dict(width=2.6, color="#dc2626", dash="dash"),
                marker=dict(symbol="triangle-down", size=8, color="#dc2626"),
                customdata=dispatch["pi_garage_breakdown"],
                legendgroup="dispatch",
                hovertemplate=(
                    "<b>%{x} — Pull-Ins</b><br>"
                    "Total Returns: <b>%{y} buses</b><br>"
                    "By Garage: %{customdata}<extra></extra>"
                ),
            ),
            row=4,
            col=1,
            secondary_y=True,
        )

        # Annotate Peak Active Service, Peak Pull-Out, and Peak Pull-In hours
        peak_active_idx = int(np.argmax(dispatch["active_total"]))
        peak_po_idx = int(np.argmax(dispatch["pull_outs_total"]))
        peak_pi_idx = int(np.argmax(dispatch["pull_ins_total"]))

        fig.add_annotation(
            x=hour_labels[peak_active_idx],
            y=dispatch["active_total"][peak_active_idx],
            xref="x4",
            yref="y4",
            text=f"Peak Service: {dispatch['active_total'][peak_active_idx]} buses",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-26,
            font=dict(size=10, color="#0f172a"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#0f172a",
        )
        fig.add_annotation(
            x=hour_labels[peak_po_idx],
            y=dispatch["pull_outs_total"][peak_po_idx],
            xref="x4",
            yref="y5",
            text=f"Peak Pull-Out: {dispatch['pull_outs_total'][peak_po_idx]}/hr",
            showarrow=True,
            arrowhead=2,
            ax=28,
            ay=-22,
            font=dict(size=10, color="#15803d"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#16a34a",
        )
        fig.add_annotation(
            x=hour_labels[peak_pi_idx],
            y=dispatch["pull_ins_total"][peak_pi_idx],
            xref="x4",
            yref="y5",
            text=f"Peak Pull-In: {dispatch['pull_ins_total'][peak_pi_idx]}/hr",
            showarrow=True,
            arrowhead=2,
            ax=-28,
            ay=-22,
            font=dict(size=10, color="#b91c1c"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#dc2626",
        )

    fig.update_layout(
        title=dict(
            text=f"Multi-Agency Transit Block Optimization — Scenario: {scenario_name}",
            x=0.02,
            font=dict(size=20),
        ),
        barmode="stack",
        template="plotly_white",
        height=1600,
        margin=dict(t=110, b=75, l=65, r=65),
        legend=dict(tracegroupgap=12),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.98,
                y=0.56,
                xanchor="right",
                yanchor="bottom",
                bgcolor="#f4f6f8",
                bordercolor="#cbd5e1",
                buttons=[
                    dict(
                        label="Show Block Counts",
                        method="update",
                        args=[
                            {"y": blocks_arrays},
                            {"yaxis3.title.text": "Blocks Assigned"},
                        ],
                    ),
                    dict(
                        label="Show % Share (100% Stacked)",
                        method="update",
                        args=[
                            {"y": shares_arrays},
                            {"yaxis3.title.text": "Route Share (%)"},
                        ],
                    ),
                ],
            )
        ],
    )

    fig.update_yaxes(title_text="Stationed Buses", row=1, col=1)
    fig.update_yaxes(title_text="EV Buses / Chargers", row=2, col=1)
    fig.update_yaxes(title_text="Blocks Assigned", row=3, col=1)
    fig.update_xaxes(title_text="Route ID", tickangle=-45, type="category", row=3, col=1)

    fig.update_yaxes(title_text="Active Buses in Service", row=4, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="Hourly Pull-Outs / Pull-Ins",
        row=4,
        col=1,
        secondary_y=True,
        showgrid=False,
    )
    fig.update_xaxes(title_text="Hour of Service Day", tickangle=-35, type="category", row=4, col=1)

    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return fig


def generate_scenario_dashboard(
    run_dir: Union[str, Path],
    map_mode: str = "routes",
    config_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Generates the interactive Folium map (`route_garage_map.html`), Plotly charts (`scenario_charts.html`),
    and integrated `dashboard.html` inside the given `runs/<scenario_name>` folder.
    """
    scenario_path = Path(run_dir)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario folder not found: '{scenario_path}'")

    scenario_name = scenario_path.name
    tables = _load_scenario_tables(scenario_path)

    all_garages = sorted(tables["garages"]["garage_id"].astype(str).unique())
    garage_colors = {
        g_id: GARAGE_PALETTE[idx % len(GARAGE_PALETTE)]
        for idx, g_id in enumerate(all_garages)
    }

    map_html_path = scenario_path / "route_garage_map.html"
    charts_html_path = scenario_path / "scenario_charts.html"
    dashboard_html_path = scenario_path / "dashboard.html"

    fmap = build_folium_map(
        scenario_path=scenario_path,
        tables=tables,
        garage_colors=garage_colors,
        output_html=map_html_path,
        map_mode=map_mode,
        config_path=Path(config_path) if config_path else None,
    )
    fig = build_plotly_charts(tables, garage_colors, scenario_name, charts_html_path)

    assignments_df = tables["assignments"]
    garage_cap_df = tables["garage_cap"]

    total_blocks = len(assignments_df)
    total_buses = int(garage_cap_df["buses_stationed"].sum())
    total_cap = int(garage_cap_df[garage_cap_df["buses_stationed"] > 0]["total_capacity_Q_g"].sum())
    util_pct = (total_buses / max(total_cap, 1)) * 100
    total_rev = float(assignments_df["revenue_miles"].sum())
    total_dh = float(assignments_df["deadhead_miles"].sum())
    dh_pct = (total_dh / max(total_rev + total_dh, 1e-6)) * 100

    map_title = {
        "routes": "Route Assignments by Garage (Switch Layers in Top-Right Menu)",
        "bus_types": "Route Assignments by Bus Type (Switch Layers in Top-Right Menu)",
        "deadhead": "Deadhead Pull-Out & Pull-In Spatial Flows (Switch Layers in Top-Right Menu)",
        "both": "Garage Route Assignments & Deadhead Spatial Flows",
    }.get(map_mode, "Route & Deadhead Map")

    map_iframe_srcdoc = html.escape(fmap.get_root().render())
    plotly_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Transit Optimization Dashboard — {html.escape(scenario_name)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
        }}
        h1 {{
            margin: 0 0 6px 0;
            font-size: 24px;
        }}
        .subtitle {{
            color: #64748b;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: #475569;
            margin-top: 4px;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card h2 {{
            margin: 0 0 12px 0;
            font-size: 18px;
        }}
        iframe {{
            width: 100%;
            height: 660px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Multi-Agency Transit Garage &amp; Fleet Optimization Dashboard</h1>
    <div class="subtitle">Scenario: <strong>{html.escape(scenario_name)}</strong> | Map Mode: <strong>{html.escape(map_mode)}</strong> | Directory: {html.escape(str(scenario_path.resolve()))}</div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Blocks Covered</div>
            <div class="kpi-value">{total_blocks:,}</div>
            <div class="kpi-sub">Across {assignments_df['primary_route'].nunique()} routes</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Buses Deployed</div>
            <div class="kpi-value">{total_buses:,}</div>
            <div class="kpi-sub">{util_pct:.1f}% of active garage capacity ({total_cap:,} bays)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Deadhead Miles</div>
            <div class="kpi-value">{total_dh:,.1f} mi</div>
            <div class="kpi-sub">{dh_pct:.2f}% of total operating miles</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Revenue Miles</div>
            <div class="kpi-value">{total_rev:,.1f} mi</div>
            <div class="kpi-sub">Avg {total_dh / max(total_blocks, 1):.2f} deadhead mi / block</div>
        </div>
    </div>

    <div class="card">
        <h2>{html.escape(map_title)}</h2>
        <iframe srcdoc="{map_iframe_srcdoc}"></iframe>
    </div>

    <div class="card">
        {plotly_div}
    </div>
</div>
</body>
</html>
"""
    dashboard_html_path.write_text(dashboard_html, encoding="utf-8")

    print("✔ Generated interactive visualizations:")
    print(f"  1. Full Dashboard : {dashboard_html_path.resolve()}")
    print(f"  2. Folium Map     : {map_html_path.resolve()} (mode='{map_mode}')")
    print(f"  3. Plotly Charts  : {charts_html_path.resolve()}")
    return dashboard_html_path


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Visualize a Transit Optimization Scenario under runs/")
    cli.add_argument(
        "-s", "--scenario",
        type=str,
        default="runs/config",
        help="Path to the scenario folder under runs/ (default: runs/config)",
    )
    cli.add_argument(
        "-m", "--map-mode",
        type=str,
        choices=["routes", "bus_types", "deadhead", "both"],
        default="routes",
        help="Initial map layer mode: 'routes' (garage-colored), 'bus_types' (vehicle-type colored), 'deadhead', or 'both'",
    )
    cli.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml if config_used.yaml is not inside the scenario folder",
    )
    args = cli.parse_args()
    generate_scenario_dashboard(args.scenario, map_mode=args.map_mode, config_path=args.config)