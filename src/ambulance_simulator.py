import random
import math
import time
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
from location_simulator import VehicleLocation, LocationSource


class AmbulanceStatus(Enum):
    AVAILABLE = "AVAILABLE"
    DISPATCHED = "DISPATCHED"
    EN_ROUTE = "EN_ROUTE"
    ON_SCENE = "ON_SCENE"
    TRANSPORTING = "TRANSPORTING"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class Ambulance:
    id: str
    location: VehicleLocation
    status: AmbulanceStatus = AmbulanceStatus.AVAILABLE
    assigned_accident: Optional[str] = None
    speed_kmh: float = 80.0
    eta_minutes: float = 0.0

    def distance_to(self, target: VehicleLocation) -> float:
        return VehicleLocation.distance_to(self.location, target)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "location": self.location.to_dict(),
            "status": self.status.value,
            "assigned_accident": self.assigned_accident,
            "speed_kmh": self.speed_kmh,
            "eta_minutes": self.eta_minutes,
        }


class AmbulanceSimulator:
    """Simulates nearby ambulances for emergency dispatch demonstration.

    IMPORTANT: This is a SOFTWARE SIMULATION ONLY. No real emergency services integration.
    """

    def __init__(
        self,
        center_lat: float = 12.9716,
        center_lon: float = 77.5946,
        radius_km: float = 10.0,
        num_ambulances: int = 5,
    ):
        self.ambulances: List[Ambulance] = []
        self.center_lat = center_lat
        self.center_lon = center_lon
        self._generate_ambulances(num_ambulances, radius_km)
        self.dispatch_log = []

    def _generate_ambulances(self, count: int, radius_km: float):
        """Generate ambulances randomly distributed around center."""
        for i in range(count):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.5, radius_km) * 1000
            lat_offset = distance * math.cos(angle) / 111320
            lon_offset = distance * math.sin(angle) / (111320 * math.cos(math.radians(self.center_lat)))

            lat = self.center_lat + lat_offset
            lon = self.center_lon + lon_offset

            status = random.choices(
                [AmbulanceStatus.AVAILABLE, AmbulanceStatus.UNAVAILABLE],
                weights=[0.7, 0.3]
            )[0]

            amb = Ambulance(
                id=f"AMB-{i+1:02d}",
location=VehicleLocation(
                latitude=lat,
                longitude=lon,
                speed_kmh=0.0,
                heading_deg=random.uniform(0, 360),
                source=LocationSource.SIMULATED,
            ),
                status=status,
                speed_kmh=random.uniform(60, 100),
            )
            self.ambulances.append(amb)

    def get_available_ambulances(self) -> List[Ambulance]:
        """Get list of available ambulances."""
        return [a for a in self.ambulances if a.status == AmbulanceStatus.AVAILABLE]

    def find_nearest_available(self, accident_location: VehicleLocation) -> Optional[Ambulance]:
        """Find nearest available ambulance to accident location."""
        available = self.get_available_ambulances()
        if not available:
            return None

        nearest = min(available, key=lambda a: a.distance_to(accident_location))
        return nearest

    def dispatch_ambulance(self, ambulance_id: str, accident_location: VehicleLocation,
                           accident_id: str) -> bool:
        """Dispatch ambulance to accident."""
        ambulance = next((a for a in self.ambulances if a.id == ambulance_id), None)
        if not ambulance or ambulance.status != AmbulanceStatus.AVAILABLE:
            return False

        distance_m = ambulance.distance_to(accident_location)
        eta_min = (distance_m / 1000) / ambulance.speed_kmh * 60

        ambulance.status = AmbulanceStatus.DISPATCHED
        ambulance.assigned_accident = accident_id
        ambulance.eta_minutes = eta_min

        self.dispatch_log.append({
            "ambulance_id": ambulance_id,
            "accident_id": accident_id,
            "dispatch_time": time.time(),
            "distance_m": distance_m,
            "eta_minutes": eta_min,
        })

        return True

    def update_ambulance_positions(self, dt_seconds: float, accident_locations: dict = None):
        """Update ambulance positions (simple simulation)."""
        if accident_locations is None:
            accident_locations = {}

        for amb in self.ambulances:
            if amb.status == AmbulanceStatus.DISPATCHED and amb.assigned_accident:
                target_loc = accident_locations.get(amb.assigned_accident)
                if target_loc:
                    amb.status = AmbulanceStatus.EN_ROUTE
                    self._move_towards(amb, target_loc, dt_seconds)
                    if amb.distance_to(target_loc) < 50:
                        amb.status = AmbulanceStatus.ON_SCENE

    def _move_towards(self, ambulance: Ambulance, target: VehicleLocation, dt_seconds: float):
        """Move ambulance towards target location."""
        distance = ambulance.distance_to(target)
        if distance < 10:
            return

        speed_ms = ambulance.speed_kmh / 3.6
        distance_to_move = speed_ms * dt_seconds

        lat1, lon1 = math.radians(ambulance.location.latitude), math.radians(ambulance.location.longitude)
        lat2, lon2 = math.radians(target.latitude), math.radians(target.longitude)

        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(y, x)

        distance_rad = distance_to_move / 6371000
        new_lat = math.asin(
            math.sin(lat1) * math.cos(distance_rad) +
            math.cos(lat1) * math.sin(distance_rad) * math.cos(bearing)
        )
        new_lon = lon1 + math.atan2(
            math.sin(bearing) * math.sin(distance_rad) * math.cos(lat1),
            math.cos(distance_rad) - math.sin(lat1) * math.sin(new_lat)
        )

        ambulance.location.latitude = math.degrees(new_lat)
        ambulance.location.longitude = math.degrees(new_lon)
        ambulance.location.timestamp = time.time()

    def get_status_display(self) -> str:
        """Get formatted status display."""
        lines = ["=" * 70, "AMBULANCE STATUS", "=" * 70]
        for amb in self.ambulances:
            lines.append(
                f"  {amb.id}: {amb.status.value:12} | "
                f"Lat={amb.location.latitude:.6f}, Lon={amb.location.longitude:.6f} | "
                f"Speed={amb.speed_kmh:.0f} km/h | ETA={amb.eta_minutes:.1f} min"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


def demo():
    """Demo the ambulance simulator."""
    print("=" * 70)
    print("AMBULANCE SIMULATOR DEMO")
    print("=" * 70)

    sim = AmbulanceSimulator(center_lat=12.9716, center_lon=77.5946, num_ambulances=6)

    print("\n--- Initial Ambulance Status ---")
    print(sim.get_status_display())

    print("\n--- Simulating Accident ---")
    accident_loc = VehicleLocation(latitude=12.9750, longitude=77.5980)

    nearest = sim.find_nearest_available(accident_loc)
    if nearest:
        print(f"\nNearest available: {nearest.id}")
        print(f"  Distance: {nearest.distance_to(accident_loc):.0f} m")
        print(f"  ETA: {nearest.eta_minutes:.1f} min")

        print("\n--- Dispatching Ambulance ---")
        success = sim.dispatch_ambulance(nearest.id, accident_loc, "ACC-001")
        print(f"  Dispatch: {'SUCCESS' if success else 'FAILED'}")

    print("\n--- Updated Status ---")
    print(sim.get_status_display())

    print("\n" + "=" * 70)
    print("NOTE: This is SIMULATED ambulance data. No real emergency services.")
    print("=" * 70)


if __name__ == "__main__":
    demo()