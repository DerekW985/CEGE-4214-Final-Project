import datetime
import math
import tempfile
import zipfile
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, Union


class MultiAgencyGTFSBlockParser:
    """
    Parses GTFS feeds (.zip archives), processes vehicles.txt (with BRT and coach classification, 
    agency reassignments, and additive overrides), and extracts parameters for Gurobi.
    """

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.blocks_df = pd.DataFrame()
        self.vehicles_df = pd.DataFrame()
        self.bus_count_df = pd.DataFrame()
        self.garages_df = pd.DataFrame()
        
        self.set_B = []        # Set of blocks (B)
        self.set_T = []        # Set of time points (T)
        self.set_R = []        # Set of routes (R)
        self.a_bv = {}         # Vehicle compatibility parameter a_{b,v}
        self.fleet_counts = {} # Agency/Vehicle fleet counts

        self._process_garage_config()

    def _process_garage_config(self):
        """Processes garage metadata (lat, lon, storage/maint capacities, EV chargers, vehicle compatibility)."""
        garage_cfg = self.config.get('garages', {})
        records = []

        for g_name, g_data in garage_cfg.items():
            storage_cap = int(g_data.get('storage_capacity', 0))
            maint_cap = int(g_data.get('maintenance_capacity', 0))
            total_cap = storage_cap + maint_cap
            compat_vehicles = g_data.get('compatible_vehicle_types', self.config.get('vehicle_types', []))

            records.append({
                'garage_id': g_name,
                'lat': float(g_data.get('lat', 0.0)),
                'lon': float(g_data.get('lon', 0.0)),
                'ev_chargers_x_g': int(g_data.get('ev_chargers', 0)),
                'storage_capacity': storage_cap,
                'maintenance_capacity': maint_cap,
                'total_capacity_Q_g': total_cap,
                'compatible_vehicle_types': "|".join(compat_vehicles),
                'allowed_agencies': "|".join(g_data.get('operating_agencies', []))
            })

        self.garages_df = pd.DataFrame(records)

    @staticmethod
    def gtfs_time_to_minutes(time_str: str) -> float:
        """Converts GTFS HH:MM:SS format to total minutes from midnight (supports times >= 24:00:00)."""
        if pd.isna(time_str) or not isinstance(time_str, str):
            return np.nan
        parts = time_str.strip().split(':')
        if len(parts) != 3:
            return np.nan
        return float(int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60.0)

    @staticmethod
    def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates Great-Circle distance between two coordinates in miles."""
        R = 3958.8
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _get_active_service_ids(
        self,
        calendar_df: pd.DataFrame,
        calendar_dates_df: pd.DataFrame,
    ) -> Tuple[Set[str], str]:
        """Resolves active GTFS service_ids for a single target date or day-of-week."""
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        filters_cfg = self.config.get('filters', {})
        service_date = filters_cfg.get('service_date', None)
        service_day = str(filters_cfg.get('service_day', 'wednesday')).lower().strip()

        active_ids: Set[str] = set()

        # If no explicit service_date is provided, pick a representative date from the feed matching service_day
        if not service_date:
            if not calendar_df.empty and service_day in calendar_df.columns:
                matching_cal = calendar_df[calendar_df[service_day].astype(str) == "1"]
                if not matching_cal.empty:
                    start_dt = datetime.datetime.strptime(str(matching_cal['start_date'].min()), "%Y%m%d")
                    end_dt = datetime.datetime.strptime(str(matching_cal['end_date'].max()), "%Y%m%d")
                    target_idx = day_names.index(service_day)
                    cur = start_dt + datetime.timedelta(days=(end_dt - start_dt).days // 2)
                    while cur.weekday() != target_idx and cur <= end_dt:
                        cur += datetime.timedelta(days=1)
                    service_date = cur.strftime("%Y%m%d")
            elif not calendar_dates_df.empty:
                cd_add = calendar_dates_df[calendar_dates_df['exception_type'].astype(str) == "1"].copy()
                if not cd_add.empty:
                    cd_add['dow'] = pd.to_datetime(cd_add['date'], format="%Y%m%d").dt.dayofweek.map(lambda i: day_names[i])
                    matching_cd = cd_add[cd_add['dow'] == service_day]
                    if not matching_cd.empty:
                        service_date = str(matching_cd['date'].value_counts().idxmax())

        if service_date:
            date_str = str(service_date)
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            dow_name = day_names[dt.weekday()]

            if not calendar_df.empty and dow_name in calendar_df.columns:
                cal = calendar_df[
                    (calendar_df['start_date'].astype(str) <= date_str)
                    & (calendar_df['end_date'].astype(str) >= date_str)
                    & (calendar_df[dow_name].astype(str) == "1")
                ]
                active_ids.update(cal['service_id'].astype(str).tolist())

            if not calendar_dates_df.empty:
                cd = calendar_dates_df[calendar_dates_df['date'].astype(str) == date_str]
                added = cd[cd['exception_type'].astype(str) == "1"]['service_id'].astype(str).tolist()
                removed = set(cd[cd['exception_type'].astype(str) == "2"]['service_id'].astype(str).tolist())
                active_ids.update(added)
                active_ids -= removed

            return active_ids, f"{date_str} ({dow_name})"

        return active_ids, "ALL (no calendar filter)"

    def _classify_vehicle(self, row: pd.Series) -> str:
        """Classifies a vehicles.txt record into vehicle types including BRT and Coach buses."""
        fuel = str(row.get('fuel', '')).lower()
        desc = str(row.get('vehicle_description', '')).lower()
        label = str(row.get('vehicle_label', '')).lower()
        low_floor_val = str(row.get('low_floor', '')).lower()
        is_high_floor = low_floor_val in ['false', '0', 'no'] or 'hfbus' in desc or 'subhf' in desc

        is_electric = 'electric' in fuel or 'ebrt' in desc or 'lfebus' in desc
        is_artic = 'artic' in desc or '60ft' in desc or '60-foot' in desc
        is_brt = 'brt' in desc or 'brt' in label
        is_coach = is_high_floor or any(kw in desc for kw in ['subhfbus', '40hfbus', 'suburban', 'coach'])

        if is_coach:
            return 'coach'
        
        if is_electric:
            if is_artic and is_brt:
                return '60ft_brt_electric'
            elif is_artic:
                return '60ft_electric'
            elif is_brt:
                return '60ft_brt_electric'
            else:
                return '40ft_electric'

        if is_artic:
            if is_brt:
                return '60ft_brt_artic'
            else:
                return '60ft_artic'

        if is_brt:
            return '40ft_brt'

        return '40ft_diesel'

    def _load_feed(self, feed_cfg: dict) -> tuple:
        """Loads and filters a GTFS zip file, parsing schedules, agencies, and vehicles.txt."""
        prefix = feed_cfg.get('feed_prefix', 'GTFS')
        zip_path = Path(feed_cfg['path'])

        if not zip_path.exists():
            print(f"[WARN] GTFS feed '{zip_path}' (Prefix: {prefix}) not found. Skipping...")
            return pd.DataFrame(), pd.DataFrame()

        with zipfile.ZipFile(zip_path, 'r') as z:
            namelist = set(z.namelist())
            trips = pd.read_csv(z.open('trips.txt'), dtype=str)
            stop_times = pd.read_csv(z.open('stop_times.txt'), dtype=str)
            stops = pd.read_csv(z.open('stops.txt'), dtype=str)
            routes = pd.read_csv(z.open('routes.txt'), dtype=str)

            agency_df = pd.DataFrame()
            if 'agency.txt' in namelist:
                agency_df = pd.read_csv(z.open('agency.txt'), dtype=str)

            feed_vehicles = pd.DataFrame()
            if 'vehicles.txt' in namelist:
                feed_vehicles = pd.read_csv(z.open('vehicles.txt'), dtype=str)

            calendar_df = pd.DataFrame()
            if 'calendar.txt' in namelist:
                calendar_df = pd.read_csv(z.open('calendar.txt'), dtype=str)

            calendar_dates_df = pd.DataFrame()
            if 'calendar_dates.txt' in namelist:
                calendar_dates_df = pd.read_csv(z.open('calendar_dates.txt'), dtype=str)

        if 'agency_id' not in routes.columns and not agency_df.empty:
            routes['agency_id'] = agency_df['agency_id'].iloc[0]
        elif 'agency_id' not in routes.columns:
            routes['agency_id'] = '0'

        if not agency_df.empty and 'agency_id' in agency_df.columns:
            routes = routes.merge(agency_df[['agency_id', 'agency_name']], on='agency_id', how='left')
        else:
            routes['agency_name'] = prefix

        routes['scoped_agency_id'] = prefix + "_" + routes['agency_id'].astype(str)
        routes['agency_name'] = routes['agency_name'].fillna(prefix)

        if not feed_vehicles.empty:
            if 'agency_id' not in feed_vehicles.columns:
                feed_vehicles['agency_id'] = routes['agency_id'].iloc[0]
            feed_vehicles['scoped_agency_id'] = prefix + "_" + feed_vehicles['agency_id'].astype(str)
            feed_vehicles['feed_prefix'] = prefix
            feed_vehicles['vehicle_type'] = feed_vehicles.apply(self._classify_vehicle, axis=1)

            prefix_reassign = self.config.get('fleet_settings', {}).get('label_prefix_agency_reassignment', {})
            if prefix_reassign:
                def reassign_agency(row):
                    v_label = str(row.get('vehicle_label', '')).upper()
                    for pfx, target_agency in prefix_reassign.items():
                        if v_label.startswith(pfx.upper()):
                            return target_agency
                    return row['scoped_agency_id']
                
                feed_vehicles['scoped_agency_id'] = feed_vehicles.apply(reassign_agency, axis=1)

        # Filter by service_id (single day schedule so Weekday/Saturday/Sunday trips do not mix)
        total_raw_trips = len(trips)
        active_service_ids, date_label = self._get_active_service_ids(calendar_df, calendar_dates_df)
        if active_service_ids and 'service_id' in trips.columns:
            trips = trips[trips['service_id'].astype(str).isin(active_service_ids)].copy()
            print(f"  ✔ [{prefix}] Service Date {date_label}: {len(active_service_ids)} active service_ids -> {len(trips)}/{total_raw_trips} trips.")
        elif 'service_id' in trips.columns and (not calendar_df.empty or not calendar_dates_df.empty):
            print(f"  [WARN] [{prefix}] No matching service_ids found for {date_label}; keeping all {total_raw_trips} trips.")

        routes['route_type'] = routes['route_type'].astype(int)
        keep_types = self.config['filters'].get('keep_route_types', [3])
        ignore_route_ids = set(str(r) for r in self.config['filters'].get('ignore_route_ids', []))
        ignore_agency_ids = set(str(a) for a in self.config['filters'].get('ignore_agency_ids', []))

        valid_routes = routes[
            (routes['route_type'].isin(keep_types)) &
            (~routes['route_id'].isin(ignore_route_ids)) &
            (~routes['agency_id'].isin(ignore_agency_ids)) &
            (~routes['scoped_agency_id'].isin(ignore_agency_ids))
        ]['route_id'].unique()

        trips = trips[trips['route_id'].isin(valid_routes)].copy()
        trips = trips.dropna(subset=['block_id']).copy()

        if trips.empty:
            return pd.DataFrame(), feed_vehicles

        trips['block_id'] = prefix + "_" + trips['block_id'].astype(str)
        trips['trip_id'] = prefix + "_" + trips['trip_id'].astype(str)
        stop_times['trip_id'] = prefix + "_" + stop_times['trip_id'].astype(str)
        trips['feed_prefix'] = prefix

        df = trips.merge(routes[['route_id', 'agency_id', 'scoped_agency_id', 'agency_name']], on='route_id', how='left')
        df = df.merge(stop_times, on='trip_id', how='inner')
        df = df.merge(stops[['stop_id', 'stop_lat', 'stop_lon', 'stop_name']], on='stop_id', how='left')

        df['stop_sequence'] = df['stop_sequence'].astype(int)
        df['stop_lat'] = df['stop_lat'].astype(float)
        df['stop_lon'] = df['stop_lon'].astype(float)
        df['arr_min'] = df['arrival_time'].apply(self.gtfs_time_to_minutes)
        df['dep_min'] = df['departure_time'].apply(self.gtfs_time_to_minutes)

        return df, feed_vehicles

    def process_all_feeds(self) -> pd.DataFrame:
        """Processes GTFS feeds, calculates block parameters, and aggregates fleet/garage parameters."""
        all_schedule_frames = []
        all_vehicle_frames = []

        for feed_cfg in self.config['gtfs_feeds']:
            df_sched, df_veh = self._load_feed(feed_cfg)
            if not df_sched.empty:
                all_schedule_frames.append(df_sched)
            if not df_veh.empty:
                all_vehicle_frames.append(df_veh)

        if not all_schedule_frames:
            raise ValueError("No valid GTFS data loaded from configured feeds.")

        full_df = pd.concat(all_schedule_frames, ignore_index=True)
        full_df = full_df.sort_values(by=['block_id', 'dep_min', 'stop_sequence'])

        block_records = []
        vehicle_types = self.config['vehicle_types']
        default_compat = self.config.get('default_vehicle_compatibility', vehicle_types)
        route_overrides = self.config.get('route_vehicle_overrides', {})
        ev_max_miles = self.config.get('ev_max_block_miles', 150.0)
        agencies_cfg = self.config.get('agencies', {})

        for block_id, b_group in full_df.groupby('block_id'):
            routes_in_block = b_group['route_id'].unique().tolist()
            scoped_agency_id = b_group['scoped_agency_id'].iloc[0]
            raw_agency_id = b_group['agency_id'].iloc[0]
            agency_name = b_group['agency_name'].iloc[0]
            feed_prefix = b_group['feed_prefix'].iloc[0]
            
            if scoped_agency_id in agencies_cfg:
                agency_name = agencies_cfg[scoped_agency_id].get('name', agency_name)

            first_row = b_group.iloc[0]
            last_row = b_group.iloc[-1]

            start_time = first_row['dep_min']
            end_time = last_row['arr_min']

            revenue_miles = 0.0
            stops_seq = b_group[['stop_lat', 'stop_lon']].values
            for i in range(len(stops_seq) - 1):
                revenue_miles += self.haversine_miles(
                    stops_seq[i][0], stops_seq[i][1],
                    stops_seq[i+1][0], stops_seq[i+1][1]
                )
            revenue_miles = round(revenue_miles, 2)

            compat_vehicles = set(default_compat)
            for r in routes_in_block:
                if str(r) in route_overrides:
                    compat_vehicles = set(route_overrides[str(r)])

            for v in list(compat_vehicles):
                if 'electric' in v and revenue_miles > ev_max_miles:
                    compat_vehicles.remove(v)

            for v in vehicle_types:
                self.a_bv[(str(block_id), v)] = 1 if v in compat_vehicles else 0

            block_records.append({
                'block_id': str(block_id),
                'feed_prefix': feed_prefix,
                'raw_agency_id': str(raw_agency_id),
                'scoped_agency_id': scoped_agency_id,
                'agency_name': agency_name,
                'primary_route': str(routes_in_block[0]),
                'all_routes': "|".join(str(r) for r in routes_in_block),
                'start_time': start_time,
                'end_time': end_time,
                'start_stop_id': first_row['stop_id'],
                'start_lat': first_row['stop_lat'],
                'start_lon': first_row['stop_lon'],
                'end_stop_id': last_row['stop_id'],
                'end_lat': last_row['stop_lat'],
                'end_lon': last_row['stop_lon'],
                'revenue_miles_l_b': revenue_miles,
                'compatible_vehicles': "|".join(sorted(list(compat_vehicles)))
            })

        self.blocks_df = pd.DataFrame(block_records)
        self.set_B = self.blocks_df['block_id'].tolist()
        self.set_R = sorted(list(set(r for routes in full_df['route_id'] for r in [routes])))
        
        time_points = set(self.blocks_df['start_time'].tolist() + self.blocks_df['end_time'].tolist())
        self.set_T = sorted(list(time_points))

        # --- PROCESS FLEET COUNTS ---
        fleet_settings = self.config.get('fleet_settings', {})
        use_gtfs_fleet = fleet_settings.get('use_gtfs_fleet_counts', True)
        manual_counts = fleet_settings.get('manual_agency_fleet_counts', {}) or {}
        agencies_with_gtfs_vehicles = set()

        if use_gtfs_fleet and all_vehicle_frames:
            self.vehicles_df = pd.concat(all_vehicle_frames, ignore_index=True)
            grouped = self.vehicles_df.groupby(['scoped_agency_id', 'vehicle_type']).size().to_dict()
            for (agency_id, v_type), count in grouped.items():
                self.fleet_counts[(agency_id, v_type)] = int(count)
                agencies_with_gtfs_vehicles.add(agency_id)

        # Fallback to manual_agency_fleet_counts if disabled or agency missing from vehicles.txt
        for agency_id, counts in manual_counts.items():
            if not use_gtfs_fleet or agency_id not in agencies_with_gtfs_vehicles:
                for v_type, count in counts.items():
                    self.fleet_counts[(agency_id, v_type)] = int(count)

        # Exact per-type overrides (works regardless of use_gtfs_fleet_counts)
        exact_overrides = fleet_settings.get('fleet_count_overrides', {}) or {}
        for agency_id, counts in exact_overrides.items():
            if counts:
                for v_type, exact_count in counts.items():
                    self.fleet_counts[(agency_id, v_type)] = int(exact_count)

        reassignments = fleet_settings.get('reassign_agency_buses', []) or []
        for r in reassignments:
            src = (r.get('from_agency'), r.get('vehicle_type'))
            dst = (r.get('to_agency'), r.get('vehicle_type'))
            count_to_move = int(r.get('count', 0))
            if count_to_move > 0:
                actual_move = min(count_to_move, self.fleet_counts.get(src, 0))
                self.fleet_counts[src] = self.fleet_counts.get(src, 0) - actual_move
                self.fleet_counts[dst] = self.fleet_counts.get(dst, 0) + actual_move

        additive = fleet_settings.get('additive_fleet_overrides', {}) or {}
        for agency_id, counts in additive.items():
            if counts:
                for v_type, extra_count in counts.items():
                    key = (agency_id, v_type)
                    self.fleet_counts[key] = self.fleet_counts.get(key, 0) + int(extra_count)

        fleet_records = []
        for (agency_id, v_type), count in self.fleet_counts.items():
            if count > 0:
                prefix = agency_id.split('_')[0] if '_' in agency_id else 'GTFS'
                a_name = agencies_cfg.get(agency_id, {}).get('name', agency_id)
                fleet_records.append({
                    'feed_prefix': prefix,
                    'scoped_agency_id': agency_id,
                    'agency_name': a_name,
                    'vehicle_type': v_type,
                    'bus_count': count
                })

        self.bus_count_df = pd.DataFrame(fleet_records)
        return self.blocks_df

    def export_fleet_counts(self, filename: str = "bus_count.csv") -> pd.DataFrame:
        """Exports the bus count breakdown per agency and vehicle type to a CSV file."""
        if not self.bus_count_df.empty:
            self.bus_count_df.to_csv(filename, index=False)
        return self.bus_count_df

    def export_garage_parameters(self, filename: str = "garage_parameters.csv") -> pd.DataFrame:
        """Exports garage parameters (lat/lon, total capacities, EV chargers, vehicle filters) to CSV."""
        if not self.garages_df.empty:
            self.garages_df.to_csv(filename, index=False)
        return self.garages_df

    def get_osrm_endpoints(self) -> pd.DataFrame:
        """Returns block start and end coordinates for OSRM matrix calculations."""
        return self.blocks_df[['block_id', 'feed_prefix', 'scoped_agency_id', 'agency_name', 'start_lat', 'start_lon', 'end_lat', 'end_lon', 'start_time', 'end_time']].copy()

    def get_gurobi_dict(self) -> Dict:
        """Exports extracted sets and parameters formatted for Gurobi optimization models."""
        return {
            'B': self.set_B,
            'T': self.set_T,
            'R': self.set_R,
            'V': self.config['vehicle_types'],
            'G': list(self.config.get('garages', {}).keys()),
            'a_bv': self.a_bv,
            'fleet_counts_n_v': self.fleet_counts,
            'garages': self.config.get('garages', {}),
            'l_b': dict(zip(self.blocks_df['block_id'], self.blocks_df['revenue_miles_l_b'])),
            'agency_b': dict(zip(self.blocks_df['block_id'], self.blocks_df['scoped_agency_id'])),
            'agency_name_b': dict(zip(self.blocks_df['block_id'], self.blocks_df['agency_name'])),
            'routes_b': dict(zip(self.blocks_df['block_id'], self.blocks_df['all_routes']))
        }


# ==============================================================================
# EXECUTION & TEST RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print(" Running Multi-Agency GTFS & Garage Parser ")
    print("==================================================\n")

    try:
        parser = MultiAgencyGTFSBlockParser("config.yaml")
        blocks_df = parser.process_all_feeds()
        gurobi_data = parser.get_gurobi_dict()
        osrm_df = parser.get_osrm_endpoints()
        bus_count_df = parser.export_fleet_counts("bus_count.csv")
        garages_df = parser.export_garage_parameters("garage_parameters.csv")

        # Save exports
        osrm_df.to_csv("block_endpoints_osrm.csv", index=False)
        blocks_df.to_csv("gurobi_block_parameters.csv", index=False)

        print("✔ Successfully parsed GTFS schedule, vehicle, and garage data:")
        print(f"  • Total Combined Blocks (|B|): {len(gurobi_data['B'])}")
        print(f"  • Total Time Points (|T|): {len(gurobi_data['T'])}")
        print(f"  • Total Routes (|R|): {len(gurobi_data['R'])}")
        print(f"  • Total Garages (|G|): {len(gurobi_data['G'])}")
        print(f"  • Vehicle Types (|V|): {gurobi_data['V']}\n")

        print("Garage Configuration Summary (garage_parameters.csv):")
        if not garages_df.empty:
            print(garages_df[['garage_id', 'storage_capacity', 'maintenance_capacity', 'total_capacity_Q_g', 'compatible_vehicle_types']].to_string(index=False))

        print(f"\n✔ Saved exports:")
        print("  1. block_endpoints_osrm.csv")
        print("  2. gurobi_block_parameters.csv")
        print("  3. bus_count.csv")
        print("  4. garage_parameters.csv\n")

    except Exception as e:
        print(f"✖ Execution note: {e}")