import random
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from location_simulator import VehicleLocation


class VehicleType(Enum):
    AMBULANCE = "AMBULANCE"
    PASSENGER_CAR = "PASSENGER_CAR"
    TRUCK = "TRUCK"
    MOTORCYCLE = "MOTORCYCLE"


class V2VMessageType(Enum):
    EMERGENCY_APPROACHING = "EMERGENCY_APPROACHING"
    CLEAR_ROUTE = "CLEAR_ROUTE"
    ACCIDENT_AHEAD = "ACCIDENT_AHEAD"
    GREEN_CORRIDOR_ACTIVE = "GREEN_CORRIDOR_ACTIVE"


@dataclass
class V2VMessage:
    message_id: str
    message_type: V2VMessageType
    sender_id: str
    sender_type: VehicleType
    sender_location: VehicleLocation
    timestamp: float
    content: str
    warning_zone_m: float = 500.0


@dataclass
class SimulatedVehicle:
    id: str
    vehicle_type: VehicleType
    location: VehicleLocation
    speed_kmh: float = 50.0
    heading_deg: float = 0.0
    received_messages: List[V2VMessage] = field(default_factory=list)
    alert_active: bool = False

    def distance_to(self, target: VehicleLocation) -> float:
        return VehicleLocation.distance_to(self.location, target)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.vehicle_type.value,
            "location": self.location.to_dict(),
            "speed_kmh": self.speed_kmh,
            "heading_deg": self.heading_deg,
            "alert_active": self.alert_active,
            "messages_received": len(self.received_messages),
        }


class V2VSimulator:
    """Simulates V2V (Vehicle-to-Vehicle) communication for emergency scenarios.

    Features:
    - 500m warning zone around ambulance
    - Emergency message broadcasting
    - Vehicle alert status tracking

    IMPORTANT: This is a SOFTWARE SIMULATION ONLY. No real V2V hardware.
    """

    WARNING_ZONE_METERS = 500.0
    COMMUNICATION_RANGE_METERS = 1000.0

    def __init__(self, center_lat: float = 12.9716, center_lon: float = 77.5946):
        self.vehicles: List[SimulatedVehicle] = []
        self.ambulance: Optional[SimulatedVehicle] = None
        self.accident_location: Optional[VehicleLocation] = None
        self.message_log: List[V2VMessage] = []
        self.message_counter = 0
        self.center_lat = center_lat
        self.center_lon = center_lon

    def add_ambulance(self, ambulance_id: str, location: VehicleLocation,
                      speed_kmh: float = 80.0, heading: float = 0.0) -> SimulatedVehicle:
        """Add ambulance to simulation."""
        self.ambulance = SimulatedVehicle(
            id=ambulance_id,
            vehicle_type=VehicleType.AMBULANCE,
            location=location,
            speed_kmh=speed_kmh,
            heading_deg=heading,
        )
        self.vehicles.append(self.ambulance)
        return self.ambulance

    def add_vehicle(self, vehicle_id: str, vehicle_type: VehicleType,
                    location: VehicleLocation, speed_kmh: float = 50.0,
                    heading: float = 0.0) -> SimulatedVehicle:
        """Add regular vehicle to simulation."""
        vehicle = SimulatedVehicle(
            id=vehicle_id,
            vehicle_type=vehicle_type,
            location=location,
            speed_kmh=speed_kmh,
            heading_deg=heading,
        )
        self.vehicles.append(vehicle)
        return vehicle

    def set_accident_location(self, location: VehicleLocation):
        """Set accident location for ACCIDENT_AHEAD messages."""
        self.accident_location = location

    def _create_message(self, msg_type: V2VMessageType, sender: SimulatedVehicle,
                        content: str) -> V2VMessage:
        self.message_counter += 1
        return V2VMessage(
            message_id=f"V2V-{self.message_counter:04d}",
            message_type=msg_type,
            sender_id=sender.id,
            sender_type=sender.vehicle_type,
            sender_location=sender.location,
            timestamp=time.time(),
            content=content,
            warning_zone_m=self.WARNING_ZONE_METERS,
        )

    def broadcast_emergency_approaching(self) -> List[V2VMessage]:
        """Broadcast EMERGENCY_APPROACHING from ambulance to vehicles in range."""
        if not self.ambulance:
            return []

        messages = []
        for vehicle in self.vehicles:
            if vehicle.id == self.ambulance.id:
                continue

            distance = VehicleLocation.distance_to(self.ambulance.location, vehicle.location)
            if distance <= self.COMMUNICATION_RANGE_METERS:
                msg = self._create_message(
                    V2VMessageType.EMERGENCY_APPROACHING,
                    self.ambulance,
                    f"Emergency ambulance approaching. Distance: {distance:.0f}m. Please clear route."
                )
                vehicle.received_messages.append(msg)
                vehicle.alert_active = distance <= self.WARNING_ZONE_METERS
                messages.append(msg)
                self.message_log.append(msg)

        return messages

    def broadcast_accident_ahead(self) -> List[V2VMessage]:
        """Broadcast ACCIDENT_AHEAD warning from accident location."""
        if not self.accident_location:
            return []

        messages = []
        for vehicle in self.vehicles:
            if vehicle.vehicle_type == VehicleType.AMBULANCE:
                continue

            distance = VehicleLocation.distance_to(vehicle.location, self.accident_location)
            if distance <= self.COMMUNICATION_RANGE_METERS:
                msg = self._create_message(
                    V2VMessageType.ACCIDENT_AHEAD,
                    SimulatedVehicle(
                        id="ACCIDENT-SITE",
                        vehicle_type=VehicleType.PASSENGER_CAR,
                        location=self.accident_location
                    ),
                    f"Accident ahead at {distance:.0f}m. Proceed with caution."
                )
                vehicle.received_messages.append(msg)
                vehicle.alert_active = True
                messages.append(msg)
                self.message_log.append(msg)

        return messages

    def broadcast_green_corridor(self) -> List[V2VMessage]:
        """Broadcast GREEN_CORRIDOR_ACTIVE message."""
        if not self.ambulance:
            return []

        messages = []
        for vehicle in self.vehicles:
            if vehicle.id == self.ambulance.id:
                continue

            distance = VehicleLocation.distance_to(self.ambulance.location, vehicle.location)
            if distance <= self.COMMUNICATION_RANGE_METERS:
                msg = self._create_message(
                    V2VMessageType.GREEN_CORRIDOR_ACTIVE,
                    self.ambulance,
                    f"Green corridor active. Traffic signals coordinated for emergency vehicle."
                )
                vehicle.received_messages.append(msg)
                messages.append(msg)
                self.message_log.append(msg)

        return messages

    def update_vehicle_positions(self, dt_seconds: float):
        """Update all vehicle positions."""
        for vehicle in self.vehicles:
            if vehicle.vehicle_type == VehicleType.AMBULANCE and self.accident_location:
                target = self.accident_location
            else:
                target = None

            self._move_vehicle(vehicle, dt_seconds, target)

    def _move_vehicle(self, vehicle: SimulatedVehicle, dt_seconds: float,
                      target: Optional[VehicleLocation] = None):
        """Move vehicle towards target or continue on heading."""
        if target:
            current = vehicle.location
            distance = VehicleLocation.distance_to(current, target)
            if distance < 10:
                return

            lat1, lon1 = math.radians(current.latitude), math.radians(current.longitude)
            lat2, lon2 = math.radians(target.latitude), math.radians(target.longitude)
            dlon = lon2 - lon1
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            bearing = math.atan2(y, x)
            vehicle.heading_deg = math.degrees(bearing)

        speed_ms = vehicle.speed_kmh / 3.6
        distance_to_move = speed_ms * dt_seconds

        lat1 = math.radians(vehicle.location.latitude)
        lon1 = math.radians(vehicle.location.longitude)
        heading_rad = math.radians(vehicle.heading_deg)

        distance_rad = distance_to_move / 6371000
        new_lat = math.asin(
            math.sin(lat1) * math.cos(distance_rad) +
            math.cos(lat1) * math.sin(distance_rad) * math.cos(heading_rad)
        )
        new_lon = lon1 + math.atan2(
            math.sin(heading_rad) * math.sin(distance_rad) * math.cos(lat1),
            math.cos(distance_rad) - math.sin(lat1) * math.sin(new_lat)
        )

        vehicle.location.latitude = math.degrees(new_lat)
        vehicle.location.longitude = math.degrees(new_lon)
        vehicle.location.timestamp = time.time()

    def get_status_display(self) -> str:
        """Get formatted V2V status display."""
        lines = ["=" * 70, "V2V COMMUNICATION STATUS", "=" * 70]

        if self.ambulance:
            amb = self.ambulance
            lines.append(f"AMBULANCE: {amb.id} @ ({amb.location.latitude:.6f}, {amb.location.longitude:.6f})")
            lines.append(f"  Speed: {amb.speed_kmh:.0f} km/h, Heading: {amb.heading_deg:.1f}°")
            lines.append(f"  Warning Zone: {self.WARNING_ZONE_METERS}m")
            lines.append("-" * 70)

        for vehicle in self.vehicles:
            if vehicle.vehicle_type == VehicleType.AMBULANCE:
                continue
            dist_to_amb = ""
            if self.ambulance:
                d = vehicle.distance_to(self.ambulance.location)
                dist_to_amb = f" | Dist to Amb: {d:.0f}m"
                if d <= self.WARNING_ZONE_METERS:
                    dist_to_amb += " *** IN WARNING ZONE ***"

            alert_status = "ALERT" if vehicle.alert_active else "Normal"
            lines.append(
                f"  {vehicle.id} ({vehicle.vehicle_type.value:14}) | "
                f"Lat={vehicle.location.latitude:.6f}, Lon={vehicle.location.longitude:.6f} | "
                f"Speed={vehicle.speed_kmh:.0f} km/h | {alert_status}{dist_to_amb}"
            )

        lines.append("-" * 70)
        lines.append(f"Total Messages Sent: {len(self.message_log)}")
        lines.append("=" * 70)
        return "\n".join(lines)


def demo():
    """Demo the V2V simulator."""
    print("=" * 70)
    print("V2V SIMULATOR DEMO")
    print("=" * 70)

    sim = V2VSimulator(center_lat=12.9716, center_lon=77.5946)

    print("\n--- Adding Ambulance ---")
    amb_loc = VehicleLocation(latitude=12.9600, longitude=77.5900, speed_kmh=80, heading_deg=45)
    sim.add_ambulance("AMB-01", amb_loc, speed_kmh=80, heading=45)

    print("\n--- Adding Other Vehicles ---")
    vehicle_types = [VehicleType.PASSENGER_CAR, VehicleType.TRUCK, VehicleType.MOTORCYCLE]
    for i in range(8):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(100, 800)
        lat_off = distance * math.cos(angle) / 111320
        lon_off = distance * math.sin(angle) / (111320 * math.cos(math.radians(12.9716)))

        vloc = VehicleLocation(
            latitude=12.9716 + lat_off,
            longitude=77.5946 + lon_off,
            speed_kmh=random.uniform(30, 60),
            heading_deg=random.uniform(0, 360),
        )
        vtype = random.choice(vehicle_types)
        sim.add_vehicle(f"V-{i+1:02d}", vtype, vloc)

    print("\n--- Initial Status ---")
    print(sim.get_status_display())

    print("\n--- Broadcasting EMERGENCY_APPROACHING ---")
    messages = sim.broadcast_emergency_approaching()
    print(f"Messages sent: {len(messages)}")
    for msg in messages:
        print(f"  {msg.message_id}: {msg.content[:60]}...")

    print("\n--- Status After Broadcast ---")
    print(sim.get_status_display())

    print("\n" + "=" * 70)
    print("NOTE: This is SIMULATED V2V communication. No real hardware.")
    print("=" * 70)


if __name__ == "__main__":
    demo()