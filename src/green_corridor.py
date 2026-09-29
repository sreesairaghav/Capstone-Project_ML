import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class SignalState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class CorridorMode(Enum):
    NORMAL = "NORMAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class TrafficSignal:
    id: str
    location: Dict[str, float]  # lat, lon
    state: SignalState = SignalState.RED
    normal_cycle: List[SignalState] = None
    cycle_index: int = 0
    last_change: float = 0.0

    def __post_init__(self):
        if self.normal_cycle is None:
            self.normal_cycle = [
                SignalState.GREEN, SignalState.YELLOW, SignalState.RED, SignalState.RED
            ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "location": self.location,
            "state": self.state.value,
        }


class GreenCorridorSimulator:
    """Simulates traffic signal coordination for emergency vehicle corridor.

    Creates a 'green wave' for approaching ambulance by coordinating
    signals along its route.

    IMPORTANT: This is a SOFTWARE SIMULATION ONLY. Does not control real traffic signals.
    """

    def __init__(
        self,
        center_lat: float = 12.9716,
        center_lon: float = 77.5946,
        num_signals: int = 4,
        spacing_m: float = 500.0,
    ):
        self.signals: List[TrafficSignal] = []
        self.mode = CorridorMode.NORMAL
        self.ambulance_route: List[Dict[str, float]] = []
        self.activation_time: Optional[float] = None
        self.center_lat = center_lat
        self.center_lon = center_lon

        self._generate_signals(num_signals, spacing_m)

    def _generate_signals(self, count: int, spacing_m: float):
        """Generate traffic signals along a corridor."""
        for i in range(count):
            # Place signals along a line (e.g., North-South corridor)
            distance = i * spacing_m
            lat_offset = distance / 111320
            lon_offset = 0  # Same longitude for straight corridor

            signal = TrafficSignal(
                id=f"SIG-{i+1:02d}",
                location={
                    "lat": self.center_lat + lat_offset,
                    "lon": self.center_lon + lon_offset,
                },
                state=random.choice(list(SignalState)),
            )
            self.signals.append(signal)

    def set_ambulance_route(self, route: List[Dict[str, float]]):
        """Set the ambulance route for corridor planning."""
        self.ambulance_route = route

    def activate_emergency_corridor(self, ambulance_location: Dict[str, float]):
        """Activate green corridor for approaching ambulance."""
        self.mode = CorridorMode.EMERGENCY
        self.activation_time = time.time()

        for signal in self.signals:
            signal.state = SignalState.GREEN

        print(f"\n{'='*60}")
        print("GREEN CORRIDOR ACTIVATED")
        print(f"{'='*60}")
        print(f"Emergency vehicle at: {ambulance_location}")
        print(f"All {len(self.signals)} signals set to GREEN")
        print(f"{'='*60}")

    def deactivate_emergency_corridor(self):
        """Return to normal signal operation."""
        self.mode = CorridorMode.NORMAL
        self.activation_time = None

        for signal in self.signals:
            signal.state = SignalState.RED
            signal.cycle_index = 0

        print(f"\n{'='*60}")
        print("GREEN CORRIDOR DEACTIVATED")
        print(f"Returned to normal signal operation")
        print(f"{'='*60}")

    def update_signals(self, dt_seconds: float):
        """Update signal states based on current mode."""
        if self.mode == CorridorMode.NORMAL:
            self._update_normal_cycle(dt_seconds)

    def _update_normal_cycle(self, dt_seconds: float):
        """Normal traffic signal cycle."""
        cycle_time = 30.0
        for signal in self.signals:
            if time.time() - signal.last_change >= cycle_time:
                signal.cycle_index = (signal.cycle_index + 1) % len(signal.normal_cycle)
                signal.state = signal.normal_cycle[signal.cycle_index]
                signal.last_change = time.time()

    def get_signal_status(self) -> str:
        """Get formatted signal status display."""
        lines = ["=" * 60, "TRAFFIC SIGNAL STATUS", "=" * 60]
        lines.append(f"Mode: {self.mode.value}")
        if self.mode == CorridorMode.EMERGENCY and self.activation_time:
            elapsed = time.time() - self.activation_time
            lines.append(f"Emergency Active: {elapsed:.1f}s")
        lines.append("-" * 60)

        for signal in self.signals:
            dist_str = ""
            if self.ambulance_route:
                # Find nearest route point
                min_dist = min(
                    ((signal.location["lat"] - p["lat"])**2 +
                     (signal.location["lon"] - p["lon"])**2)**0.5 * 111320
                    for p in self.ambulance_route
                )
                dist_str = f" | Dist to route: {min_dist:.0f}m"

            lines.append(
                f"  {signal.id}: {signal.state.value:6} "
                f"@ ({signal.location['lat']:.6f}, {signal.location['lon']:.6f}){dist_str}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


def demo():
    """Demo the green corridor simulator."""
    print("=" * 70)
    print("GREEN CORRIDOR SIMULATOR DEMO")
    print("=" * 70)

    corridor = GreenCorridorSimulator(center_lat=12.9716, center_lon=77.5946, num_signals=4)

    print("\n--- Normal Signal Operation ---")
    print(corridor.get_signal_status())

    print("\n--- Simulating Ambulance Route ---")
    route = [
        {"lat": 12.9716, "lon": 77.5946},
        {"lat": 12.9766, "lon": 77.5946},
        {"lat": 12.9816, "lon": 77.5946},
        {"lat": 12.9866, "lon": 77.5946},
    ]
    corridor.set_ambulance_route(route)

    print("\n--- Activating Emergency Corridor ---")
    ambulance_loc = {"lat": 12.9600, "lon": 77.5946}
    corridor.activate_emergency_corridor(ambulance_loc)
    print(corridor.get_signal_status())

    print("\n--- Simulating Time Pass (Normal Mode) ---")
    corridor.deactivate_emergency_corridor()
    print("Waiting for signal cycle...")
    time.sleep(2)
    corridor.update_signals(2.0)
    print(corridor.get_signal_status())

    print("\n" + "=" * 70)
    print("NOTE: This is SIMULATED traffic signal coordination. No real signals controlled.")
    print("=" * 70)


if __name__ == "__main__":
    demo()