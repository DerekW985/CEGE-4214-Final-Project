import requests
from typing import Tuple, Optional, Dict, Any

def get_routing_info(
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    profile: str = 'driving',
    osrm_url: str = 'http://router.project-osrm.org/route/v1'
) -> Optional[Dict[str, Any]]:
    """
    Calculate route distance and duration between two points using OSRM API.
    
    Args:
        start_coords: Tuple of (longitude, latitude) for starting point
        end_coords: Tuple of (longitude, latitude) for ending point
        profile: Routing mode ('driving', 'walking', 'cycling')
        osrm_url: Base URL for OSRM server (public demo or self-hosted)
    
    Returns:
        Dictionary containing distance (km), duration (minutes), and route geometry
        None if request fails
    
    Note:
        OSRM expects coordinates in [longitude, latitude] order
        Public demo server has rate limits (~60 req/min); use self-hosted for production
    """
    
    # Validate coordinate ranges
    lon_start, lat_start = start_coords
    lon_end, lat_end = end_coords
    
    if not (-180 <= lon_start <= 180) or not (-90 <= lat_start <= 90):
        raise ValueError("Invalid start coordinates")
    if not (-180 <= lon_end <= 180) or not (-90 <= lat_end <= 90):
        raise ValueError("Invalid end coordinates")
    
    # Build query string (OSRM uses lon,lat format)
    coords_string = f"{lon_start},{lat_start};{lon_end},{lat_end}"
    url = f"{osrm_url}/{profile}/{coords_string}"
    
    params = {
        'overview': 'false',  # Return only summary stats, not full geometry
        'alternatives': 'false'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') != 'Ok':
            print(f"OSRM API returned error: {data.get('message', 'Unknown error')}")
            return None
        
        routes = data.get('routes', [])
        if not routes:
            return None
        
        route = routes[0]
        
        return {
            'distance_km': round(route['distance'] / 1000, 2),  # meters to km
            'distance_miles': round(route['distance'] / 1609.34, 2),
            'duration_minutes': round(route['duration'] / 60, 2),
            'duration_hours': round(route['duration'] / 3600, 3),
            'speed_kmh': round(route['distance'] / 1000 / (route['duration'] / 3600), 1)
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except (KeyError, TypeError) as e:
        print(f"Failed to parse OSRM response: {e}")
        return None


def get_transit_info(
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    departure_time: int = None,
    osrm_url: str = 'https://transit.project-osrm.org/transit/v1'
) -> Optional[Dict[str, Any]]:
    """
    Get public transit routing information (if using a transit-enabled OSRM instance).
    
    Args:
        start_coords: Tuple of (longitude, latitude) for starting point
        end_coords: Tuple of (longitude, latitude) for ending point  
        departure_time: Unix timestamp for planned departure (optional)
        osrm_url: Transit-specific OSRM endpoint
    
    Returns:
        Dictionary with transit time, distance, and route details
    """
    
    coords_string = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    url = f"{osrm_url}/transit/{coords_string}"
    
    params = {}
    if departure_time:
        params['departure_time'] = departure_time
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') != 'Ok':
            return None
        
        routes = data.get('routes', [])
        if not routes:
            return None
        
        route = routes[0]
        
        return {
            'distance_km': round(route.get('distance', 0) / 1000, 2),
            'duration_minutes': round(route.get('duration', 0) / 60, 2),
            'walk_distance_km': round(route.get('walk_distance', 0) / 1000, 2),
            'num_transfers': route.get('transfers', 0)
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Transit request failed: {e}")
        return None


# Example usage with placeholder input functions
def example_integration():
    """Demonstrates how to integrate this with other functions"""
    
    # These would come from your other functions
    def get_start_location():
        # Replace with your actual location retrieval logic
        return (-93.2678, 44.6508)  # Example: Minneapolis area
    
    def get_end_location():
        # Replace with your actual destination retrieval logic
        return (-93.2314, 44.9761)  # Example: Another Minneapolis location
    
    # Get coordinates from your functions
    start = get_start_location()
    end = get_end_location()
    
    # Get routing info
    result = get_routing_info(start, end, profile='driving')
    
    if result:
        print(f"Distance: {result['distance_km']} km ({result['distance_miles']} miles)")
        print(f"Estimated Travel Time: {result['duration_minutes']} minutes")
        print(f"Average Speed: {result['speed_kmh']} km/h")
    else:
        print("Could not calculate route")


if __name__ == '__main__':
    example_integration()