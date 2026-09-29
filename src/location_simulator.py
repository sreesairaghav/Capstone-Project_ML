import random
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class LocationSource(Enum):
    SIMULATED = "SIMULATED"
    GPS = "GPS"
    MANUAL = "MANUAL"


@dataclass
class VehicleLocation:
    latitude: float
    longitude: float
    altitude: float = 0.0
    speed_kmh: float = 0.0
    heading_deg: float = 0.0
    accuracy_m: float = 5.0
    timestamp: float = 0.0
    source: LocationSource = LocationSource.SIMULATED

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "speed_kmh": self.speed_kmh,
            "heading_deg": self.heading_deg,
            "accuracy_m": self.accuracy_m,
            "timestamp": self.timestamp,
            "source": self.source.value,
        }

    @staticmethod
    def distance_to(loc1: 'VehicleLocation', loc2: 'VehicleLocation') -> float:
        """Calculate distance between two locations in meters (Haversine)."""
        R = 6371000
        lat1, lon1 = math.radians(loc1.latitude), math.radians(loc1.longitude)
        lat2, lon2 = math.radians(loc2.latitude), math.radians(loc2.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


class LocationSimulator:
    """Simulates vehicle GPS location for demonstration purposes.

    IMPORTANT: This is a SOFTWARE SIMULATION ONLY. It does not use real GPS hardware.
    """

    def __init__(
        self,
        start_lat: float = 12.9716,
        start_lon: float = 77.5946,
        start_speed_kmh: float = 50.0,
        start_heading: float = 45.0,
    ):
        self.current_location = VehicleLocation(
            latitude=start_lat,
            longitude=start_lon,
            speed_kmh=start_speed_kmh,
            heading_deg=start_heading,
            timestamp=time.time(),
        )
        self.route = []
        self.route_index = 0
        self.crashed = False
        self.crash_location: Optional[VehicleLocation] = None
        self.movement_noise_m = 2.0

    def set_location(self, lat: float, lon: float, speed_kmh: float = 0.0, heading: float = 0.0):
        """Manually set location (for testing)."""
        self.current_location = VehicleLocation(
            latitude=lat,
            longitude=lon,
            speed_kmh=speed_kmh,
            heading_deg=heading,
            timestamp=time.time(),
            source=LocationSource.MANUAL,
        )

    def update_position(self, dt_seconds: float):
        """Update position based on speed and heading."""
        if self.crashed:
            return

        distance_m = self.current_location.speed_kmh / 3.6 * dt_seconds

        lat_rad = math.radians(self.current_location.latitude)
        lon_rad = math.radians(self.current_location.longitude)
        heading_rad = math.radians(self.current_location.heading_deg)

        dlat = distance_m * math.cos(heading_rad) / 111320
        dlon = distance_m * math.sin(heading_rad) / (111320 * math.cos(lat_rad))

        noise_lat = random.uniform(-self.movement_noise_m, self.movement_noise_m) / 111320
        noise_lon = random.uniform(-self.movement_noise_m, self.movement_noise_m) / (111320 * math.cos(lat_rad))

        self.current_location.latitude += dlat + noise_lat
        self.current_location.longitude += dlon + noise_lon
        self.current_location.timestamp = time.time()

    def simulate_crash(self) -> VehicleLocation:
        """Mark current location as crash site and freeze it."""
        self.crashed = True
        self.crash_location = VehicleLocation(
            latitude=self.current_location.latitude,
            longitude=self.current_location.longitude,
            speed_kmh=0.0,
            heading_deg=self.current_location.heading_deg,
            timestamp=time.time(),
            source=LocationSource.SIMULATED,
        )
        self.current_location.speed_kmh = 0.0
        return self.crash_location

    def get_current_location(self) -> VehicleLocation:
        """Get current (or crash) location."""
        if self.crashed and self.crash_location:
            return self.crash_location
        return self.current_location

    def get_crash_location(self) -> Optional[VehicleLocation]:
        """Get crash location if crash occurred."""
        return self.crash_location

    def reset(self, lat: float = None, lon: float = None):
        """Reset simulator to initial state."""
        if lat is not None and lon is not None:
            self.current_location.latitude = lat
            self.current_location.longitude = lon
        self.current_location.speed_kmh = 50.0
        self.current_location.heading_deg = 45.0
        self.crashed = False
        self.crash_location = None
        self.current_location.timestamp = time.time()


def demo():
    """Demo the location simulator."""
    print("=" * 70)
    print("LOCATION SIMULATOR DEMO")
    print("=" * 70)

    sim = LocationSimulator(start_lat=12.9716, start_lon=77.5946, start_speed_kmh=60.0)

    print("\n--- Initial Location ---")
    loc = sim.get_current_location()
    print(f"  Lat: {loc.latitude:.6f}, Lon: {loc.longitude:.6f}")
    print(f"  Speed: {loc.speed_kmh:.1f} km/h, Heading: {loc.heading_deg:.1f}°")

    print("\n--- Simulating Movement (5 updates) ---")
    for i in range(5):
        sim.update_position(1.0)
        loc = sim.get_current_location()
        print(f"  Step {i+1}: Lat={loc.latitude:.6f}, Lon={loc.longitude:.6f}, Speed={loc.speed_kmh:.1f} km/h")

    print("\n--- Simulating Crash ---")
    crash_loc = sim.simulate_crash()
    print(f"  CRASH LOCATION: Lat={crash_loc.latitude:.6f}, Lon={crash_loc.longitude:.6f}")
    print(f"  (Location frozen at crash site)")

    print("\n--- Post-Crash Location Updates ---")
    for i in range(3):
        sim.update_position(1.0)
        loc = sim.get_current_location()
        print(f"  Step {i+1}: Lat={loc.latitude:.6f}, Lon={loc.longitude:.6f}, Speed={loc.speed_kmh:.1f} km/h")

    print("\n--- Distance from Crash ---")
    test_loc = VehicleLocation(latitude=12.9800, longitude=77.6000)
    dist = VehicleLocation.distance_to(crash_loc, test_loc)
    print(f"  Distance to (12.9800, 77.6000): {dist:.1f} m")

    print("\n" + "=" * 70)
    print("NOTE: This is SIMULATED location data. No real GPS hardware used.")
    print("=" * 70)


if __name__ == "__main__":
    demo()