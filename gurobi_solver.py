import math
from pathlib import Path
from typing import Dict, Optional, Union

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import yaml


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Fallback Great-Circle distance in miles if OSRM matrix is not yet generated."""
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


def _load_deadhead_distances(
    blocks_df: pd.DataFrame,
    garages_cfg: Dict,
    osrm_path: Union[str, Path],
    circuity_fallback: float = 1.25,
) -> Dict[tuple, Dict[str, float]]:
    """
    Loads pull-out and pull-in deadhead miles/minutes from OSRM output CSV.
    Falls back to Haversine * circuity_fallback if the OSRM file is not present.
    """
    osrm_file = Path(osrm_path)
    deadhead_lookup = {}

    if osrm_file.exists():
        osrm_df = pd.read_csv(osrm_file)
        for _, row in osrm_df.iterrows():
            g_id = str(row["garage_id"])
            b_id = str(row["block_id"])
            pull_out_mi = float(row.get("pull_out_miles", 0.0))
            pull_in_mi = float(row.get("pull_in_miles", 0.0))
            total_mi = float(row.get("total_deadhead_miles", pull_out_mi + pull_in_mi))
            pull_out_min = float(row.get("pull_out_min", 0.0))
            pull_in_min = float(row.get("pull_in_min", 0.0))

            deadhead_lookup[(g_id, b_id)] = {
                "pull_out_miles": pull_out_mi,
                "pull_in_miles": pull_in_mi,
                "total_miles": total_mi,
                "pull_out_min": pull_out_min,
                "pull_in_min": pull_in_min,
            }
        print(f"✔ Loaded OSRM deadhead matrix from '{osrm_file}' ({len(deadhead_lookup)} pairs).")
    else:
        print(f"[WARN] OSRM file '{osrm_file}' not found. Using Haversine x {circuity_fallback} fallback.")
        for g_id, g_data in garages_cfg.items():
            g_lat, g_lon = float(g_data["lat"]), float(g_data["lon"])
            for _, b_row in blocks_df.iterrows():
                b_id = str(b_row["block_id"])
                out_mi = _haversine_miles(g_lat, g_lon, b_row["start_lat"], b_row["start_lon"]) * circuity_fallback
                in_mi = _haversine_miles(b_row["end_lat"], b_row["end_lon"], g_lat, g_lon) * circuity_fallback
                deadhead_lookup[(g_id, b_id)] = {
                    "pull_out_miles": round(out_mi, 3),
                    "pull_in_miles": round(in_mi, 3),
                    "total_miles": round(out_mi + in_mi, 3),
                    "pull_out_min": 0.0,
                    "pull_in_min": 0.0,
                }

    return deadhead_lookup


def solve_block_assignment(
    gurobi_data: Optional[Dict] = None,
    blocks_csv: Union[str, Path] = "gurobi_block_parameters.csv",
    bus_counts_csv: Union[str, Path] = "bus_count.csv",
    osrm_csv: Union[str, Path] = "osrm_deadhead_matrix.csv",
    config_path: Union[str, Path] = "config.yaml",
    output_dir: Union[str, Path] = ".",
    include_deadhead_in_time_window: bool = False,
    buses_per_charger: int = 1,
    fleet_tiebreak_weight: float = 0.001,
    fleet_slack_penalty: Optional[float] = None,
    time_limit: float = 300.0,
    mip_gap: float = 0.005,
) -> Dict:
    """
    Solves the multi-agency transit block-to-garage & vehicle-type assignment problem.

    Parameters
    ----------
    fleet_slack_penalty : float, optional
        If provided (e.g., 10000.0), allows borrowing extra buses beyond `bus_count.csv`
        at a high objective penalty and reports the exact deficit instead of failing.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    blocks_df = pd.read_csv(blocks_csv)
    blocks_df["block_id"] = blocks_df["block_id"].astype(str)
    blocks_df["scoped_agency_id"] = blocks_df["scoped_agency_id"].astype(str)

    if gurobi_data is not None:
        set_B = [str(b) for b in gurobi_data["B"]]
        set_T = sorted(gurobi_data["T"])
        set_V = list(gurobi_data["V"])
        set_G = list(gurobi_data["G"])
        a_bv = gurobi_data["a_bv"]
        fleet_counts = gurobi_data["fleet_counts_n_v"]
        garages_cfg = gurobi_data["garages"]
        l_b = gurobi_data["l_b"]
        agency_b = gurobi_data["agency_b"]
    else:
        set_B = blocks_df["block_id"].tolist()
        set_T = sorted(set(blocks_df["start_time"].tolist() + blocks_df["end_time"].tolist()))
        set_V = list(config["vehicle_types"])
        garages_cfg = config.get("garages", {})
        set_G = list(garages_cfg.keys())
        l_b = dict(zip(blocks_df["block_id"], blocks_df["revenue_miles_l_b"]))
        agency_b = dict(zip(blocks_df["block_id"], blocks_df["scoped_agency_id"]))

        a_bv = {}
        for _, row in blocks_df.iterrows():
            b_id = str(row["block_id"])
            compat = set(str(row["compatible_vehicles"]).split("|"))
            for v in set_V:
                a_bv[(b_id, v)] = 1 if v in compat else 0

        bus_df = pd.read_csv(bus_counts_csv)
        fleet_counts = {
            (str(r["scoped_agency_id"]), str(r["vehicle_type"])): int(r["bus_count"])
            for _, r in bus_df.iterrows()
        }

    # Build sparse list of valid (b, g, v) assignment tuples
    valid_triples = []
    triples_by_b = {b: [] for b in set_B}

    for b in set_B:
        b_agency = agency_b[b]
        for g in set_G:
            g_data = garages_cfg[g]
            if b_agency not in g_data.get("operating_agencies", []):
                continue
            g_compat_v = set(g_data.get("compatible_vehicle_types", set_V))
            for v in set_V:
                if v in g_compat_v and a_bv.get((b, v), 0) == 1:
                    triple = (b, g, v)
                    valid_triples.append(triple)
                    triples_by_b[b].append(triple)

    # Pre-solve diagnostic check 1: Garage/vehicle structural compatibility
    unassignable_blocks = [b for b in set_B if not triples_by_b[b]]
    if unassignable_blocks:
        bad_df = blocks_df[blocks_df["block_id"].isin(unassignable_blocks)]
        diag = (
            bad_df.groupby(["scoped_agency_id", "agency_name", "compatible_vehicles"])
            .size()
            .reset_index(name="unassignable_block_count")
        )
        print("\n✖ Pre-Solve Error: Some blocks have zero valid garage/vehicle combinations:")
        print(diag.to_string(index=False))
        raise ValueError(f"{len(unassignable_blocks)} blocks have zero valid garage/vehicle assignments.")

    # Resolve fleet pools (e.g., MT_0 and MT_1 sharing Metro Transit's bus fleet)
    agencies_set = sorted(list(set(agency_b.values())))
    configured_pools = config.get("fleet_settings", {}).get("fleet_pools", {}) or {}
    agency_to_pool = {}
    pool_members = {}

    for pool_name, members in configured_pools.items():
        pool_members[pool_name] = list(members)
        for m in members:
            agency_to_pool[str(m)] = pool_name

    for k in agencies_set:
        if k not in agency_to_pool:
            agency_to_pool[k] = k
            pool_members[k] = [k]

    pool_fleet_counts = {}
    for pool_name, members in pool_members.items():
        for v in set_V:
            pool_fleet_counts[(pool_name, v)] = sum(
                fleet_counts.get((m, v), 0) for m in members
            )

    # Pre-solve diagnostic check 2: Ensure every block has >0 buses in its agency/pool (unless slack enabled)
    if fleet_slack_penalty is None:
        zero_fleet_blocks = []
        for b in set_B:
            p_id = agency_to_pool[agency_b[b]]
            has_any_bus = any(pool_fleet_counts.get((p_id, v), 0) > 0 for (_, _, v) in triples_by_b[b])
            if not has_any_bus:
                zero_fleet_blocks.append(b)

        if zero_fleet_blocks:
            bad_df = blocks_df[blocks_df["block_id"].isin(zero_fleet_blocks)]
            diag = (
                bad_df.groupby(["scoped_agency_id", "agency_name", "compatible_vehicles"])
                .size()
                .reset_index(name="blocks_with_zero_buses")
            )
            print("\n✖ Pre-Solve Error: The following blocks belong to agencies/pools with 0 available buses:")
            print(diag.to_string(index=False))
            raise ValueError(f"{len(zero_fleet_blocks)} blocks have 0 available buses in their agency/pool.")

    # Load OSRM deadhead matrix
    deadhead = _load_deadhead_distances(blocks_df, garages_cfg, osrm_csv)

    # Map block timing & pre-index triples
    start_time_b = dict(zip(blocks_df["block_id"], blocks_df["start_time"]))
    end_time_b = dict(zip(blocks_df["block_id"], blocks_df["end_time"]))
    ev_vehicle_types = [v for v in set_V if "electric" in v.lower()]
    pools_set = sorted(list(set(agency_to_pool.values())))

    triples_by_gv = {(g, v): [] for g in set_G for v in set_V}
    triples_by_pv = {(p, v): [] for p in pools_set for v in set_V}

    for b, g, v in valid_triples:
        triples_by_gv[(g, v)].append((b, g, v))
        triples_by_pv[(agency_to_pool[agency_b[b]], v)].append((b, g, v))

    # ==========================================================================
    # BUILD GUROBI MODEL
    # ==========================================================================
    model = gp.Model("MultiAgency_Block_Garage_Assignment")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    x = model.addVars(valid_triples, vtype=GRB.BINARY, name="x")
    y = model.addVars(set_G, set_V, vtype=GRB.INTEGER, lb=0, name="y")

    s_fleet = None
    if fleet_slack_penalty is not None:
        s_fleet = model.addVars(pools_set, set_V, vtype=GRB.INTEGER, lb=0, name="s_fleet")

    obj_deadhead = gp.quicksum(
        deadhead[(g, b)]["total_miles"] * x[b, g, v]
        for (b, g, v) in valid_triples
    )
    obj_fleet_tiebreak = fleet_tiebreak_weight * gp.quicksum(
        y[g, v] for g in set_G for v in set_V
    )
    obj_slack = (
        fleet_slack_penalty * gp.quicksum(s_fleet[p, v] for p in pools_set for v in set_V)
        if s_fleet is not None
        else 0.0
    )
    model.setObjective(obj_deadhead + obj_fleet_tiebreak + obj_slack, GRB.MINIMIZE)

    # Constraint 1: Every block must be assigned to exactly one (garage, vehicle_type)
    for b in set_B:
        model.addConstr(
            gp.quicksum(x[triple] for triple in triples_by_b[b]) == 1,
            name=f"cover_{b}",
        )

    # Constraint 2: Time-indexed active blocks at (g, v) cannot exceed stationed buses y[g, v]
    for g in set_G:
        for v in set_V:
            gv_triples = triples_by_gv[(g, v)]
            if not gv_triples:
                model.addConstr(y[g, v] == 0, name=f"zero_fleet_{g}_{v}")
                continue

            for t in set_T:
                active_at_t = []
                for (b, _, _) in gv_triples:
                    s_t = start_time_b[b]
                    e_t = end_time_b[b]
                    if include_deadhead_in_time_window:
                        s_t -= deadhead[(g, b)]["pull_out_min"]
                        e_t += deadhead[(g, b)]["pull_in_min"]

                    if (s_t <= t < e_t) or (s_t == e_t == t):
                        active_at_t.append(x[b, g, v])

                if active_at_t:
                    model.addConstr(
                        gp.quicksum(active_at_t) <= y[g, v],
                        name=f"stationed_{g}_{v}_t{int(t)}",
                    )

    # Constraint 3: Garage Total Storage + Maintenance Capacity (Q_g)
    for g in set_G:
        g_data = garages_cfg[g]
        q_g = int(g_data.get("storage_capacity", 0)) + int(g_data.get("maintenance_capacity", 0))
        model.addConstr(
            gp.quicksum(y[g, v] for v in set_V) <= q_g,
            name=f"garage_cap_{g}",
        )

    # Constraint 4: Garage EV Charger Capacity (x_g)
    for g in set_G:
        ev_chargers = int(garages_cfg[g].get("ev_chargers", 0))
        if ev_vehicle_types:
            model.addConstr(
                gp.quicksum(y[g, v] for v in ev_vehicle_types) <= ev_chargers * buses_per_charger,
                name=f"ev_chargers_{g}",
            )

    # Constraint 5: Agency / Pool Fleet Availability (N_{p,v}) across active blocks at time t
    for p in pools_set:
        for v in set_V:
            avail_buses = pool_fleet_counts.get((p, v), 0)
            pv_triples = triples_by_pv[(p, v)]
            if not pv_triples:
                if s_fleet is not None:
                    model.addConstr(s_fleet[p, v] == 0, name=f"zero_slack_{p}_{v}")
                continue

            if avail_buses == 0 and s_fleet is None:
                for triple in pv_triples:
                    x[triple].ub = 0
                continue

            rhs_expr = avail_buses + (s_fleet[p, v] if s_fleet is not None else 0)
            for t in set_T:
                active_pv_t = []
                for (b, g, _) in pv_triples:
                    s_t = start_time_b[b]
                    e_t = end_time_b[b]
                    if include_deadhead_in_time_window:
                        s_t -= deadhead[(g, b)]["pull_out_min"]
                        e_t += deadhead[(g, b)]["pull_in_min"]

                    if (s_t <= t < e_t) or (s_t == e_t == t):
                        active_pv_t.append(x[b, g, v])

                if active_pv_t:
                    model.addConstr(
                        gp.quicksum(active_pv_t) <= rhs_expr,
                        name=f"pool_fleet_{p}_{v}_t{int(t)}",
                    )

    # ==========================================================================
    # OPTIMIZE MODEL
    # ==========================================================================
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("\n✖ Model is INFEASIBLE. Computing Irreducible Inconsistent Subsystem (IIS)...")
        model.computeIIS()
        iis_constrs = [c for c in model.getConstrs() if c.IISConstr]
        iis_vars = [v.VarName for v in model.getVars() if v.IISLB or v.IISUB]

        bottleneck_constrs = [c for c in iis_constrs if not c.ConstrName.startswith("cover_")]
        cover_constrs = [c for c in iis_constrs if c.ConstrName.startswith("cover_")]

        print(f"  Total IIS Constraints: {len(iis_constrs)} ({len(bottleneck_constrs)} capacity/fleet limits, {len(cover_constrs)} block covers)")
        if bottleneck_constrs:
            print("  Bottleneck Constraint(s):")
            for c in bottleneck_constrs:
                print(f"    - {c.ConstrName}  (Sense: '{c.Sense}', RHS: {c.RHS})")

        if cover_constrs:
            conflict_block_ids = [c.ConstrName.replace("cover_", "", 1) for c in cover_constrs]
            conflict_df = blocks_df[blocks_df["block_id"].isin(conflict_block_ids)]
            summary = (
                conflict_df.groupby(["scoped_agency_id", "agency_name", "compatible_vehicles"])
                .agg(
                    conflicting_blocks=("block_id", "count"),
                    earliest_start=("start_time", "min"),
                    latest_end=("end_time", "max"),
                    sample_routes=("primary_route", lambda s: "|".join(sorted(set(s.astype(str)))[:8])),
                )
                .reset_index()
            )
            print("\n  Conflicting Blocks Breakdown:")
            print(summary.to_string(index=False))

        if iis_vars:
            print(f"\n  Conflicting Variable Bounds ({len(iis_vars)} total). First 15:")
            for vname in iis_vars[:15]:
                print(f"    - {vname}")

        return {
            "status": "INFEASIBLE",
            "iis_constraints": [c.ConstrName for c in iis_constrs],
            "bottleneck_constraints": [c.ConstrName for c in bottleneck_constrs],
            "iis_variables": iis_vars,
            "model": model,
        }

    if model.SolCount == 0:
        return {
            "status": f"NO_SOLUTION (Status Code: {model.Status})",
            "model": model,
        }

    # Report any fleet slack used if soft slack was enabled
    if s_fleet is not None:
        slack_used = []
        for p in pools_set:
            for v in set_V:
                val = int(round(s_fleet[p, v].X))
                if val > 0:
                    slack_used.append({
                        "agency_or_pool": p,
                        "vehicle_type": v,
                        "available_in_gtfs": pool_fleet_counts.get((p, v), 0),
                        "extra_buses_needed": val,
                    })
        if slack_used:
            print("\n⚠ Fleet Shortfall Detected (Extra Buses Required Beyond bus_count.csv):")
            print(pd.DataFrame(slack_used).to_string(index=False))

    # ==========================================================================
    # EXTRACT & EXPORT SOLUTION
    # ==========================================================================
    assignment_rows = []
    for (b, g, v) in valid_triples:
        if x[b, g, v].X > 0.5:
            dh = deadhead[(g, b)]
            rev_mi = float(l_b[b])
            assignment_rows.append({
                "block_id": b,
                "assigned_garage": g,
                "vehicle_type": v,
                "pull_out_miles": dh["pull_out_miles"],
                "pull_in_miles": dh["pull_in_miles"],
                "deadhead_miles": dh["total_miles"],
                "revenue_miles": rev_mi,
                "total_miles": round(rev_mi + dh["total_miles"], 2),
            })

    sol_df = pd.DataFrame(assignment_rows)
    assignments_df = blocks_df.merge(sol_df, on="block_id", how="inner")

    garage_rows = []
    for g in set_G:
        g_data = garages_cfg[g]
        q_g = int(g_data.get("storage_capacity", 0)) + int(g_data.get("maintenance_capacity", 0))
        ev_cap = int(g_data.get("ev_chargers", 0))
        for v in set_V:
            stationed = int(round(y[g, v].X))
            sub = assignments_df[(assignments_df["assigned_garage"] == g) & (assignments_df["vehicle_type"] == v)]
            if stationed > 0 or len(sub) > 0:
                garage_rows.append({
                    "garage_id": g,
                    "vehicle_type": v,
                    "buses_stationed": stationed,
                    "blocks_assigned": len(sub),
                    "total_revenue_miles": round(sub["revenue_miles"].sum(), 2),
                    "total_deadhead_miles": round(sub["deadhead_miles"].sum(), 2),
                    "garage_ev_chargers": ev_cap,
                    "garage_total_capacity": q_g,
                })

    garage_summary_df = pd.DataFrame(garage_rows)

    route_summary_df = (
        assignments_df.groupby(["assigned_garage", "scoped_agency_id", "agency_name", "primary_route", "vehicle_type"])
        .agg(
            total_blocks=("block_id", "count"),
            total_revenue_miles=("revenue_miles", "sum"),
            total_deadhead_miles=("deadhead_miles", "sum"),
        )
        .reset_index()
    )
    route_summary_df["total_revenue_miles"] = route_summary_df["total_revenue_miles"].round(2)
    route_summary_df["total_deadhead_miles"] = route_summary_df["total_deadhead_miles"].round(2)

    assignments_df.to_csv(out_path / "gurobi_solution_assignments.csv", index=False)
    garage_summary_df.to_csv(out_path / "garage_fleet_summary.csv", index=False)
    route_summary_df.to_csv(out_path / "route_garage_summary.csv", index=False)

    total_deadhead = round(assignments_df["deadhead_miles"].sum(), 2)
    total_revenue = round(assignments_df["revenue_miles"].sum(), 2)
    total_buses = int(garage_summary_df["buses_stationed"].sum())

    print("\n==================================================")
    print(" Gurobi Optimization Complete ")
    print("==================================================")
    print(f"  • Solver Status       : {'OPTIMAL' if model.Status == GRB.OPTIMAL else model.Status}")
    print(f"  • Solve Runtime       : {model.Runtime:.2f} sec")
    print(f"  • Total Deadhead Miles: {total_deadhead:,.2f} mi")
    print(f"  • Total Revenue Miles : {total_revenue:,.2f} mi")
    print(f"  • Total Buses Used    : {total_buses}")
    print("==================================================\n")

    return {
        "status": "OPTIMAL" if model.Status == GRB.OPTIMAL else str(model.Status),
        "objective_value": model.ObjVal,
        "total_deadhead_miles": total_deadhead,
        "total_revenue_miles": total_revenue,
        "total_buses_used": total_buses,
        "runtime_sec": model.Runtime,
        "assignments_df": assignments_df,
        "garage_summary_df": garage_summary_df,
        "route_summary_df": route_summary_df,
        "model": model,
    }


# ==============================================================================
# STANDALONE TEST RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print(" Testing Gurobi Multi-Agency Block Solver ")
    print("==================================================\n")

    real_config = Path("config.yaml")
    real_blocks = Path("gurobi_block_parameters.csv")
    real_fleet = Path("bus_count.csv")

    if real_config.exists() and real_blocks.exists() and real_fleet.exists():
        print("✔ Found local GTFS parsed files. Running solver on actual network data...\n")
        results = solve_block_assignment(
            gurobi_data=None,
            blocks_csv=real_blocks,
            bus_counts_csv=real_fleet,
            osrm_csv="osrm_deadhead_matrix.csv",
            config_path=real_config,
            output_dir="test_results",
            fleet_slack_penalty=None,  # Set to e.g. 10000.0 to allow soft fleet slack
            time_limit=120.0,
            mip_gap=0.01,
        )
        if results["status"] != "INFEASIBLE":
            print("Garage Fleet Allocation Summary:")
            print(results["garage_summary_df"].to_string(index=False))
            print("\n✔ Saved test outputs to './test_results/'")