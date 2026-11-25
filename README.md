# Multi-Agency Transit Garage & Fleet Optimization Pipeline

A GTFS-to-Gurobi Mixed-Integer Linear Programming (MILP) pipeline that parses multi-agency transit schedules, computes real-world road deadhead distances via OSRM, assigns vehicle blocks to garages and bus types to minimize deadhead and fleet costs, and generates interactive HTML maps and dashboards.

This readme created with the help of Gemini AI

---

## 1. Required Dependencies

Install the Python dependencies into your Conda or virtual environment (note: `gurobipy` requires an active Gurobi license, such as a free academic license):

```bash
pip install pandas numpy pyyaml requests gurobipy folium plotly
```

* **`pandas` & `numpy`**: GTFS schedule parsing, block construction, and time-of-day aggregation.
* **`pyyaml`**: Reading scenario settings from `config.yaml`.
* **`requests`**: Querying the OSRM routing API for pull-out and pull-in deadhead distances.
* **`gurobipy`**: Solving the block-to-garage and fleet-sizing MILP optimization model.
* **`folium` & `plotly`**: Generating the interactive route/garage map and multi-panel analytical charts.

---

## 2. How to Run the Pipeline

### Standard Run (Uses `config.yaml` and saves to `runs/config/`)
```bash
python run_pipeline.py
```

### Run a Named Scenario & Choose Initial Map Coloring
```bash
python run_pipeline.py --config config.yaml --name baseline_mt0 --map-mode routes
```

### Useful CLI Flags for `run_pipeline.py`
* `-c`, `--config <path>`: Path to the YAML configuration file (default: `config.yaml`).
* `-n`, `--name <name>`: Output subfolder name inside `runs/` (defaults to the config filename).
* `-m`, `--map-mode <mode>`: Default layer shown on the map:
  * `routes` *(default)*: GTFS routes colored by dominant assigned garage.
  * `bus_types`: GTFS routes colored by dominant vehicle type (`40ft_diesel`, `60ft_brt_artic`, `40ft_electric`, etc.).
  * `deadhead`: Pull-out and pull-in deadhead lines between garages and route terminals.
  * `both`: Shows garage-colored routes and pull-out deadhead lines together.
* `--skip-osrm`: Bypass OSRM HTTP queries and use a Haversine $\times 1.25$ distance fallback.
* `--skip-dashboard`: Run the solver and export CSVs/summaries without generating HTML visuals.
* `--time-limit <sec>` / `--mip-gap <float>`: Adjust Gurobi solver stopping criteria (defaults: `300` seconds, `0.005` = 0.5% gap).

### Re-render Maps & Charts Without Re-solving Gurobi
```bash
python visualize_scenario.py --scenario runs/baseline_mt0 --map-mode bus_types
```

---

## 3. Project Architecture (Python Modules)

| File | Role | What It Does |
| :--- | :--- | :--- |
| **`run_pipeline.py`** | Orchestrator | Runs the 4-step workflow end-to-end (Parse $\rightarrow$ OSRM $\rightarrow$ Solve $\rightarrow$ Summarize & Visualize) and saves isolated scenario outputs. |
| **`multi_gtfs_parser.py`** | GTFS & Fleet Parser | Reads zipped GTFS feeds (`trips.txt`, `stop_times.txt`, `stops.txt`, `calendar.txt`), builds continuous vehicle blocks, maps compatible vehicle types, and formats garage/fleet parameters. |
| **`osrm_deadhead.py`** | Deadhead Router | Deduplicates block start/end coordinates, caches directed road distances/durations in `osrm_cache/osrm_coordinate_cache.csv`, and builds the garage-to-block deadhead matrix. |
| **`gurobi_solver.py`** | MILP Optimizer | Formulates and solves the Gurobi model enforcing block coverage, vehicle compatibility, agency fleet caps, physical garage bay capacities ($Q_g$), EV charger limits ($x_g$), and time-window overlaps. |
| **`visualize_scenario.py`** | Dashboard & Maps | Generates the interactive Folium map (routes by garage, routes by bus type, and deadhead corridors) and 4-panel Plotly dashboard. |

---

## 4. Where Outputs Are Saved

Every run creates an isolated directory under **`runs/<scenario_name>/`** containing:

* **`dashboard.html`**: All-in-one interactive report with KPI cards, the embedded Folium map, and all 4 Plotly charts.
* **`route_garage_map.html`**: Standalone interactive Folium map with toggleable layers in the top-right menu.
* **`scenario_charts.html`**: Standalone 4-panel Plotly figure (Garage Capacity & Fleet Mix, EV Charger Utilization, Route-to-Garage Split, and Hourly Active Service & Pull-Out/Pull-In Peaks).
* **`run_summary.txt`**: Plain-text executive summary with runtime, total deadhead %, garage utilization, fleet breakdown, and the route-to-garage block/percentage table.
* **`gurobi_solution_assignments.csv`**: Block-by-block optimal assignment (`block_id`, `primary_route`, `assigned_garage`, `vehicle_type`, `pull_out_miles`, `pull_in_miles`, `deadhead_miles`).
* **`garage_fleet_summary.csv`** & **`garage_capacity_utilization.csv`**: Stationed bus counts, bay utilization %, and EV charger usage per garage.
* **`route_garage_breakdown.csv`**: Block counts, route percentage shares, and deadhead miles for every route-garage pair.
* **`config_used.yaml`**: Exact snapshot of the configuration file used for reproducibility.

*(Note: Road distances queried from OSRM are persisted globally in `osrm_cache/osrm_coordinate_cache.csv` so repeat runs execute with zero API wait time.)*

---

## 5. How to Edit `config.yaml`

Open **`config.yaml`** to define new scenarios (such as testing garage expansions, electrification targets, or agency consolidations):

1. **Service Date & GTFS Feeds (`gtfs_feeds`, `service_date`)**:
   * Set `service_date` in `YYYYMMDD` format (e.g., `"20261014"` for a typical Wednesday weekday schedule) and point `path` to each agency's GTFS `.zip` file.
2. **Agency Filtering (`filters.ignore_agency_ids`)**:
   * Add agency IDs (e.g., `"MT_5"`, `"MVTA_0"`) to exclude them from a run, or remove them to include them. **Note:** Any active agency must also have buses defined in the fleet counts section.
3. **Garage Locations, Capacities & EV Chargers (`garages`)**:
   * Edit `total_capacity` ($Q_g$, total physical bus bays), `ev_chargers` ($x_g$, max electric buses supported overnight), `lat`/`lon`, or `allowed_agencies` to test closing, expanding, or sharing garages across agencies.
4. **Fleet Roster (`fleet.manual_agency_fleet_counts`)**:
   * Adjust available bus counts by agency and vehicle type (`40ft_diesel`, `40ft_electric`, `60ft_artic`, `60ft_brt_artic`, `coach`, etc.) to test fleet electrification or vehicle-shortage scenarios.
5. **Route-to-Vehicle Compatibility (`route_vehicle_compatibility`)**:
   * Control which bus types are permitted on specific routes (e.g., restricting BRT lines to `60ft_brt_artic` / `60ft_brt_electric`).