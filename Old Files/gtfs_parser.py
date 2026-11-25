import math
import tempfile
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union


class GTFSBlockParser:
    """
    Parses GTFS data feeds (.zip archive or directory) and builds data structures
    and matrices for the Bus-Garage-Route Assignment Optimization Model.
    """

    def __init__(self, gtfs_path: Union[str, Path]):
        self.gtfs_path = Path(gtfs_path)
        self.trips_df = pd.DataFrame()
        self.stop_times_df = pd.DataFrame()
        self.stops_df = pd.DataFrame()
        self.calendar_df = pd.DataFrame()

        self.blocks_df = pd.DataFrame()
        self.set_B = []  # Set of blocks (B)
        self.set_T = []  # Set of time points (T)
        self.set_R = []  # Set of routes (R)
        
        self.s_bt = {}  # Pull-out parameter s_{b,t}
        self.e_bt = {}  # Pull-in parameter e_{b,t}
        
        self._load_gtfs_files()

    def _load_gtfs_files(self):
        """Loads required GTFS text files from a ZIP archive or standard folder."""
        required_files = ['trips.txt', 'stop_times.txt', 'stops.txt']
        dfs = {}

        if self.gtfs_path.is_file() and self.gtfs_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(self.gtfs_path, 'r') as z:
                for fname in required_files:
                    if fname in z.namelist():
                        dfs[fname] = pd.read_csv(z.open(fname), dtype=str)
                if 'calendar.txt' in z.namelist():
                    dfs['calendar.txt'] = pd.read_csv(z.open('calendar.txt'), dtype=str)
        elif self.gtfs_path.is_dir():
            for fname in required_files:
                fpath = self.gtfs_path / fname
                if fpath.exists():
                    dfs[fname] = pd.read_csv(fpath, dtype=str)
            cal_path = self.gtfs_path / 'calendar.txt'
            if cal_path.exists():
                dfs['calendar.txt'] = pd.read_csv(cal_path, dtype=str)
        else:
            raise FileNotFoundError(f"GTFS source not found at: {self.gtfs_path}")

        self.trips_df = dfs.get('trips.txt', pd.DataFrame())
        self.stop_times_df = dfs.get('stop_times.txt', pd.DataFrame())
        self.stops_df = dfs.get('stops.txt', pd.DataFrame())
        self.calendar_df = dfs.get('calendar.txt', pd.DataFrame())

        if 'block_id' not in self.trips_df.columns:
            raise ValueError("GTFS trips.txt must contain a 'block_id' field for block optimization.")

    @staticmethod
    def gtfs_time_to_minutes(time_str: str) -> float:
        """Converts GTFS HH:MM:SS format to total minutes from midnight (supports times >= 24:00:00)."""
        if pd.isna(time_str) or not isinstance(time_str, str):
            return np.nan
        parts = time_str.strip().split(':')
        if len(parts) != 3:
            return np.nan
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return hours * 60.0 + minutes + (seconds / 60.0)

    @staticmethod
    def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates Great-Circle distance between two coordinates in miles."""
        R = 3958.8  # Earth radius in miles
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def process_data(self, filter_service_id: str = None) -> pd.DataFrame:
        """Processes GTFS data to aggregate trips into block schedules and model parameters."""
        trips = self.trips_df.dropna(subset=['block_id']).copy()
        if filter_service_id:
            trips = trips[trips['service_id'] == filter_service_id]

        df = trips.merge(self.stop_times_df, on='trip_id', how='inner')
        df = df.merge(self.stops_df[['stop_id', 'stop_lat', 'stop_lon', 'stop_name']], on='stop_id', how='left')

        df['stop_sequence'] = df['stop_sequence'].astype(int)
        df['stop_lat'] = df['stop_lat'].astype(float)
        df['stop_lon'] = df['stop_lon'].astype(float)
        df['arr_min'] = df['arrival_time'].apply(self.gtfs_time_to_minutes)
        df['dep_min'] = df['departure_time'].apply(self.gtfs_time_to_minutes)

        df = df.sort_values(by=['block_id', 'dep_min', 'stop_sequence'])

        block_records = []
        for block_id, b_group in df.groupby('block_id'):
            routes_in_block = b_group['route_id'].unique().tolist()
            service_ids = b_group['service_id'].unique().tolist()
            
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

            block_records.append({
                'block_id': block_id,
                'service_id': service_ids[0] if service_ids else 'DEFAULT',
                'routes': routes_in_block,
                'start_time': start_time,
                'end_time': end_time,
                'start_stop_id': first_row['stop_id'],
                'start_lat': first_row['stop_lat'],
                'start_lon': first_row['stop_lon'],
                'end_stop_id': last_row['stop_id'],
                'end_lat': last_row['stop_lat'],
                'end_lon': last_row['stop_lon'],
                'revenue_miles_l_b': round(revenue_miles, 2)
            })

        self.blocks_df = pd.DataFrame(block_records)
        self.set_B = self.blocks_df['block_id'].tolist()
        self.set_R = sorted(list(set(r for routes in self.blocks_df['routes'] for r in routes)))
        
        time_points = set(self.blocks_df['start_time'].tolist() + self.blocks_df['end_time'].tolist())
        self.set_T = sorted(list(time_points))

        for _, row in self.blocks_df.iterrows():
            b = row['block_id']
            p_out = row['start_time']
            p_in = row['end_time']

            for t in self.set_T:
                self.s_bt[(b, t)] = 1 if t == p_out else 0
                self.e_bt[(b, t)] = 1 if t == p_in else 0

        return self.blocks_df

    def get_osrm_endpoints(self) -> pd.DataFrame:
        """Returns block start and end coordinates for OSRM deadhead distance/time calculations."""
        return self.blocks_df[['block_id', 'start_lat', 'start_lon', 'end_lat', 'end_lon', 'start_time', 'end_time']].copy()

    def get_gurobi_dict(self) -> Dict:
        """Exports extracted sets and parameters formatted for Gurobi optimization models."""
        return {
            'B': self.set_B,
            'T': self.set_T,
            'R': self.set_R,
            's_bt': self.s_bt,
            'e_bt': self.e_bt,
            'l_b': dict(zip(self.blocks_df['block_id'], self.blocks_df['revenue_miles_l_b'])),
            'sigma_b': dict(zip(self.blocks_df['block_id'], self.blocks_df['service_id']))
        }


# ==============================================================================
# TEST SUITE
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print(" Running GTFSBlockParser Synthetic Unit Tests ")
    print("==================================================\n")

    # --------------------------------------------------------------------------
    # 1. Synthetic Data Test (Isolated in Temporary Directory)
    # --------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        trips_csv = """route_id,service_id,trip_id,block_id
RTE_1,WEEKDAY,TRIP_101,BLK_1
RTE_1,WEEKDAY,TRIP_102,BLK_1
RTE_2,WEEKDAY,TRIP_201,BLK_2
"""
        stop_times_csv = """trip_id,arrival_time,departure_time,stop_id,stop_sequence
TRIP_101,06:00:00,06:00:00,STOP_A,1
TRIP_101,06:30:00,06:30:00,STOP_B,2
TRIP_102,07:00:00,07:00:00,STOP_B,1
TRIP_102,07:45:00,07:45:00,STOP_C,2
TRIP_201,08:00:00,08:00:00,STOP_A,1
TRIP_201,25:15:00,25:15:00,STOP_C,2
"""
        stops_csv = """stop_id,stop_name,stop_lat,stop_lon
STOP_A,Downtown Transit Center,44.9778,-93.2650
STOP_B,Midtown Station,44.9537,-93.0900
STOP_C,Uptown Garage Terminal,44.9480,-93.2980
"""
        calendar_csv = """service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WEEKDAY,1,1,1,1,1,0,0,20260101,20261231
"""

        (dir_path / "trips.txt").write_text(trips_csv)
        (dir_path / "stop_times.txt").write_text(stop_times_csv)
        (dir_path / "stops.txt").write_text(stops_csv)
        (dir_path / "calendar.txt").write_text(calendar_csv)

        parser = GTFSBlockParser(dir_path)
        blocks_df = parser.process_data(filter_service_id="WEEKDAY")
        osrm_df = parser.get_osrm_endpoints()
        gurobi_data = parser.get_gurobi_dict()

        # Validations
        assert GTFSBlockParser.gtfs_time_to_minutes("06:00:00") == 360.0
        assert GTFSBlockParser.gtfs_time_to_minutes("25:15:00") == 1515.0
        assert set(gurobi_data['B']) == {'BLK_1', 'BLK_2'}
        assert set(gurobi_data['R']) == {'RTE_1', 'RTE_2'}
        assert gurobi_data['T'] == [360.0, 465.0, 480.0, 1515.0]
        assert gurobi_data['s_bt'][('BLK_1', 360.0)] == 1
        assert gurobi_data['e_bt'][('BLK_1', 465.0)] == 1
        assert gurobi_data['s_bt'][('BLK_1', 465.0)] == 0
        assert len(osrm_df) == 2

        print("✔ Synthetic unit tests passed successfully.\n")

    # --------------------------------------------------------------------------
    # 2. Integration Test for Real Feed: MTgtfs10-8-2026.zip
    # --------------------------------------------------------------------------
    target_zip = Path("MTgtfs10-8-2026.zip")

    print("==================================================")
    print(f" Running Integration Test: {target_zip.name} ")
    print("==================================================")

    if target_zip.exists():
        try:
            mt_parser = GTFSBlockParser(target_zip)
            mt_blocks = mt_parser.process_data()
            mt_gurobi = mt_parser.get_gurobi_dict()
            mt_osrm = mt_parser.get_osrm_endpoints()

            print(f"✔ File loaded successfully: {target_zip.name}")
            print(f"  • Total Blocks (|B|): {len(mt_gurobi['B'])}")
            print(f"  • Total Time Points (|T|): {len(mt_gurobi['T'])}")
            print(f"  • Total Routes (|R|): {len(mt_gurobi['R'])}")
            print(f"  • OSRM Coordinates Export Rows: {len(mt_osrm)}")
            print(f"  • Total Pull-out (s_bt) Entries: {len(mt_gurobi['s_bt'])}")
            
            print("\nFirst 5 Parsed Blocks:")
            print(mt_blocks[['block_id', 'service_id', 'start_time', 'end_time', 'revenue_miles_l_b']].head().to_string(index=False))

            # 1. Export extracted endpoints for OSRM matrix calculations
            osrm_output_csv = "MT_block_endpoints_osrm.csv"
            mt_osrm.to_csv(osrm_output_csv, index=False)
            print(f"\n✔ OSRM block endpoints saved to: {osrm_output_csv}")

            # 2. Export get_gurobi_dict parameters directly to CSV for Gurobi
            gurobi_df = pd.DataFrame([
                {
                    'block_id': b,
                    'service_id': mt_gurobi['sigma_b'].get(b),
                    'revenue_miles_l_b': mt_gurobi['l_b'].get(b),
                    'start_time': mt_blocks.loc[mt_blocks['block_id'] == b, 'start_time'].values[0],
                    'end_time': mt_blocks.loc[mt_blocks['block_id'] == b, 'end_time'].values[0]
                }
                for b in mt_gurobi['B']
            ])
            gurobi_output_csv = "gurobi_block_parameters.csv"
            gurobi_df.to_csv(gurobi_output_csv, index=False)
            print(f"✔ Gurobi parameter payload saved to: {gurobi_output_csv}")

        except Exception as e:
            print(f"✖ Error parsing {target_zip.name}: {e}")
    else:
        print(f"[INFO] '{target_zip.name}' was not found in the current working directory.")
        print("  Place the ZIP file in the root directory to run the full integration test.")