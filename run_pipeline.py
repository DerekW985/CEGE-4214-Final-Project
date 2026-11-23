import argparse
import shutil
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from gurobi_solver import solve_block_assignment
from multi_gtfs_parser import MultiAgencyGTFSBlockParser
from osrm_deadhead import DEFAULT_PUBLIC_OSRM, build_osrm_deadhead_matrix


def run_transit_optimization_pipeline(
    config_path: Union[str, Path] = "config.yaml",
    run_name: Optional[str] = None,
    base_output_dir: Union[str, Path] = "runs",
    osrm_cache_csv: Union[str, Path] = "osrm_cache/osrm_coordinate_cache.csv",
    osrm_base_url: str = DEFAULT_PUBLIC_OSRM,
    osrm_rate_limit_sec: float = 1.1,
    skip_osrm: bool = False,
    include_deadhead_in_time_window: bool = False,
    buses_per_charger: int = 1,
    fleet_slack_penalty: Optional[float] = None,
    time_limit: float = 300.0,
    mip_gap: float = 0.005,
    map_mode: str = "routes",
    skip_dashboard: bool = False,
) -> dict:
    """
    Runs the full GTFS parsing -> OSRM deadhead -> Gurobi optimization -> Dashboard
    pipeline for a specified config file and writes isolated outputs for that scenario.
    """
    t0 = time.time()
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: '{cfg_file}'")

    scenario_name = run_name if run_name else cfg_file.stem
    run_dir = Path(base_output_dir) / scenario_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save an exact copy of the config used for this iteration
    shutil.copy2(cfg_file, run_dir / "config_used.yaml")

    print("=================================================================")
    print(f" TRANSIT OPTIMIZATION PIPELINE | Scenario: {scenario_name}")
    print(f" Config File : {cfg_file.resolve()}")
    print(f" Output Dir  : {run_dir.resolve()}")
    print("=================================================================\n")

    # ------------------------------------------------------------------
    # STEP 1: Parse GTFS Feeds, Fleet Counts & Garage Parameters
    # ------------------------------------------------------------------
    print("[STEP 1/4] Parsing GTFS feeds, fleet roster, and garage specs...")
    parser = MultiAgencyGTFSBlockParser(cfg_file)
    blocks_df = parser.process_all_feeds()
    gurobi_data = parser.get_gurobi_dict()

    endpoints_csv = run_dir / "block_endpoints_osrm.csv"
    blocks_csv = run_dir / "gurobi_block_parameters.csv"
    bus_counts_csv = run_dir / "bus_count.csv"
    garages_csv = run_dir / "garage_parameters.csv"

    parser.get_osrm_endpoints().to_csv(endpoints_csv, index=False)
    blocks_df.to_csv(blocks_csv, index=False)
    parser.export_fleet_counts(str(bus_counts_csv))
    parser.export_garage_parameters(str(garages_csv))

    print(f"  ✔ Parsed {len(gurobi_data['B'])} blocks across {len(gurobi_data['R'])} routes and {len(gurobi_data['G'])} garages.\n")

    # ------------------------------------------------------------------
    # STEP 2: Build Deduplicated & Cached OSRM Deadhead Matrix
    # ------------------------------------------------------------------
    osrm_csv = run_dir / "osrm_deadhead_matrix.csv"
    if not skip_osrm:
        print(f"[STEP 2/4] Generating OSRM deadhead matrix ({osrm_base_url})...")
        build_osrm_deadhead_matrix(
            endpoints_csv=endpoints_csv,
            garages_csv=garages_csv,
            output_csv=osrm_csv,
            cache_csv=osrm_cache_csv,
            osrm_base_url=osrm_base_url,
            min_request_interval=osrm_rate_limit_sec,
        )
        print()
    else:
        print("[STEP 2/4] Skipping OSRM (--skip-osrm enabled; solver will use Haversine fallback).\n")

    # ------------------------------------------------------------------
    # STEP 3: Solve Block-to-Garage MILP in Gurobi
    # ------------------------------------------------------------------
    print("[STEP 3/4] Running Gurobi MILP optimization...")
    results = solve_block_assignment(
        gurobi_data=gurobi_data,
        blocks_csv=blocks_csv,
        bus_counts_csv=bus_counts_csv,
        osrm_csv=osrm_csv,
        config_path=cfg_file,
        output_dir=run_dir,
        include_deadhead_in_time_window=include_deadhead_in_time_window,
        buses_per_charger=buses_per_charger,
        fleet_slack_penalty=fleet_slack_penalty,
        time_limit=time_limit,
        mip_gap=mip_gap,
    )

    total_elapsed = round(time.time() - t0, 2)

    if results["status"] == "INFEASIBLE":
        print(f"✖ Scenario '{scenario_name}' was INFEASIBLE (elapsed: {total_elapsed}s).")
        return results

    # ------------------------------------------------------------------
    # STEP 4: Generate Executive Summary Report & HTML Dashboard
    # ------------------------------------------------------------------
    assignments_df = results["assignments_df"].copy()
    assignments_df["primary_route"] = assignments_df["primary_route"].astype(str)
    garage_summary_df = results["garage_summary_df"]
    garages_df = pd.read_csv(garages_csv)

    # 1. Aggregate garage-level utilization (total capacity & EV charger usage)
    stationed_by_garage = (
        garage_summary_df.groupby("garage_id")
        .agg(
            buses_stationed=("buses_stationed", "sum"),
            blocks_assigned=("blocks_assigned", "sum"),
            revenue_miles=("total_revenue_miles", "sum"),
            deadhead_miles=("total_deadhead_miles", "sum"),
        )
        .reset_index()
    )

    ev_stationed = (
        garage_summary_df[garage_summary_df["vehicle_type"].str.contains("electric", case=False)]
        .groupby("garage_id")["buses_stationed"]
        .sum()
        .reset_index(name="ev_buses_stationed")
    )

    garage_report = (
        garages_df[["garage_id", "total_capacity_Q_g", "ev_chargers_x_g"]]
        .merge(stationed_by_garage, on="garage_id", how="left")
        .merge(ev_stationed, on="garage_id", how="left")
        .fillna(0)
    )
    garage_report["buses_stationed"] = garage_report["buses_stationed"].astype(int)
    garage_report["blocks_assigned"] = garage_report["blocks_assigned"].astype(int)
    garage_report["ev_buses_stationed"] = garage_report["ev_buses_stationed"].astype(int)
    garage_report["cap_util_%"] = (
        (garage_report["buses_stationed"] / garage_report["total_capacity_Q_g"].replace(0, 1)) * 100
    ).round(1)
    garage_report["deadhead_miles"] = garage_report["deadhead_miles"].round(1)
    garage_report["revenue_miles"] = garage_report["revenue_miles"].round(1)

    # 2. Fleet deployment by vehicle type
    fleet_by_type = (
        garage_summary_df.groupby("vehicle_type")
        .agg(
            buses_used=("buses_stationed", "sum"),
            blocks_covered=("blocks_assigned", "sum"),
            deadhead_miles=("total_deadhead_miles", "sum"),
        )
        .reset_index()
    )
    fleet_by_type["deadhead_miles"] = fleet_by_type["deadhead_miles"].round(1)

    # 3. Route-to-garage assignment breakdown (blocks & % share per route)
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
    route_garage_df["deadhead_miles"] = route_garage_df["deadhead_miles"].round(1)

    # Natural sort: numeric route IDs first (2, 3, 10...), then named routes (A_Line, Gold_Line...)
    route_garage_df["_sort_num"] = pd.to_numeric(
        route_garage_df["primary_route"], errors="coerce"
    ).fillna(999999)
    route_garage_df = (
        route_garage_df.sort_values(
            by=["scoped_agency_id", "_sort_num", "primary_route", "blocks_assigned"],
            ascending=[True, True, True, False],
        )
        .drop(columns=["_sort_num"])
        .reset_index(drop=True)
    )

    route_garage_df = route_garage_df[
        [
            "scoped_agency_id",
            "primary_route",
            "assigned_garage",
            "blocks_assigned",
            "route_total_blocks",
            "route_share_%",
            "deadhead_miles",
            "vehicle_types",
        ]
    ]

    dh_pct = round(
        (results["total_deadhead_miles"] / max(results["total_revenue_miles"] + results["total_deadhead_miles"], 1e-6)) * 100,
        2,
    )

    report_lines = [
        "=================================================================",
        f" OPTIMIZATION RESULTS SUMMARY | Scenario: {scenario_name}",
        "=================================================================",
        f"  Status               : {results['status']}",
        f"  Total Pipeline Time  : {total_elapsed:.2f}s (Gurobi: {results['runtime_sec']:.2f}s)",
        f"  Total Blocks Covered : {len(assignments_df)}",
        f"  Total Buses Deployed : {results['total_buses_used']}",
        f"  Total Revenue Miles  : {results['total_revenue_miles']:,.2f} mi",
        f"  Total Deadhead Miles : {results['total_deadhead_miles']:,.2f} mi ({dh_pct}% of total miles)",
        "",
        "--- GARAGE CAPACITY & EV CHARGER UTILIZATION ---",
        garage_report.to_string(index=False),
        "",
        "--- FLEET DEPLOYMENT BY VEHICLE TYPE ---",
        fleet_by_type.to_string(index=False),
        "",
        "--- ROUTE-TO-GARAGE ASSIGNMENT BREAKDOWN ---",
        route_garage_df.to_string(index=False),
        "=================================================================",
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text + "\n")

    (run_dir / "run_summary.txt").write_text(report_text, encoding="utf-8")
    garage_report.to_csv(run_dir / "garage_capacity_utilization.csv", index=False)
    route_garage_df.to_csv(run_dir / "route_garage_breakdown.csv", index=False)

    # 4. Generate Interactive HTML Dashboard & Map
    if not skip_dashboard:
        print(f"[STEP 4/4] Generating interactive HTML dashboard (map_mode='{map_mode}')...")
        try:
            from visualize_scenario import generate_scenario_dashboard

            dashboard_path = generate_scenario_dashboard(
                run_dir=run_dir,
                map_mode=map_mode,
                config_path=cfg_file,
            )
            results["dashboard_path"] = str(dashboard_path.resolve())
        except ImportError as exc:
            print(f"  [WARN] Skipping dashboard generation (missing visualization dependency: {exc}).")
            print("         Install with: pip install folium plotly")
    else:
        print("[STEP 4/4] Skipping HTML dashboard generation (--skip-dashboard enabled).")

    print(f"\n✔ All scenario outputs saved to: {run_dir.resolve()}")
    return results


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Run Multi-Agency GTFS + OSRM + Gurobi Optimization Pipeline")
    cli.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file for this run (default: config.yaml)",
    )
    cli.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="Optional scenario name for output folder (defaults to config filename)",
    )
    cli.add_argument(
        "-m", "--map-mode",
        type=str,
        choices=["routes", "bus_types", "deadhead", "both"],
        default="routes",
        help="Initial Folium map layer mode: 'routes' (garage-colored), 'bus_types' (vehicle-type colored), 'deadhead', or 'both' (default: routes)",
    )
    cli.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip generating dashboard.html, route_garage_map.html, and scenario_charts.html",
    )
    cli.add_argument(
        "--osrm-url",
        type=str,
        default=DEFAULT_PUBLIC_OSRM,
        help="OSRM server URL (e.g. http://router.project-osrm.org or http://localhost:5000)",
    )
    cli.add_argument(
        "--rate-limit",
        type=float,
        default=1.1,
        help="Minimum seconds between OSRM batch requests (set to 0 for local OSRM)",
    )
    cli.add_argument(
        "--skip-osrm",
        action="store_true",
        help="Skip OSRM queries and use Haversine * 1.25 distance fallback",
    )
    cli.add_argument(
        "--slack-penalty",
        type=float,
        default=None,
        help="Optional penalty per extra bus if testing soft fleet limits (e.g. 10000)",
    )
    cli.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Gurobi time limit in seconds (default: 300)",
    )
    cli.add_argument(
        "--mip-gap",
        type=float,
        default=0.005,
        help="Gurobi relative MIP gap tolerance (default: 0.005 = 0.5%%)",
    )

    args = cli.parse_args()
    run_transit_optimization_pipeline(
        config_path=args.config,
        run_name=args.name,
        map_mode=args.map_mode,
        skip_dashboard=args.skip_dashboard,
        osrm_base_url=args.osrm_url,
        osrm_rate_limit_sec=args.rate_limit,
        skip_osrm=args.skip_osrm,
        fleet_slack_penalty=args.slack_penalty,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
    )