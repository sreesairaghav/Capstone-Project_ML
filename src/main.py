import time
import random
import os
import csv
import math
from datetime import datetime
from pathlib import Path

from live_sensor_simulator import LiveSensorSimulator, SensorMode, SensorReading, CrashEventSimulator
from crash_detector import CrashDetector, CrashStatus, CrashDetectionResult
from predict_severity import SeverityPredictor, CrashFeatures
from emergency_decision import EmergencyDecisionEngine, SeverityLevel, EmergencyAction, EmergencyDecision
from location_simulator import LocationSimulator, VehicleLocation, LocationSource
from ambulance_simulator import AmbulanceSimulator, Ambulance, AmbulanceStatus
from v2v_simulator import V2VSimulator, SimulatedVehicle, VehicleType, V2VMessageType
from green_corridor import GreenCorridorSimulator, CorridorMode


class AccidentDetectionSystem:
    """Complete accident detection and emergency response pipeline.

    Pipeline:
    LIVE SENSOR DATA -> CRASH DETECTION -> SEVERITY PREDICTION
    -> EMERGENCY DECISION -> AMBULANCE DISPATCH -> V2V ALERTS -> GREEN CORRIDOR
    """

    def __init__(
        self,
        sample_rate_hz: float = 20.0,
        log_dir: str = "data/live",
        enable_logging: bool = True,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.log_dir = Path(log_dir)
        self.enable_logging = enable_logging
        self.running = False
        self.step_count = 0

        # Initialize components
        self.sensor_simulator = LiveSensorSimulator(sample_rate_hz=sample_rate_hz, crash_probability=0.0)
        self.crash_detector = CrashDetector()
        self.severity_predictor = SeverityPredictor()
        self.decision_engine = EmergencyDecisionEngine()
        self.location_sim = LocationSimulator(start_speed_kmh=60.0)
        self.ambulance_sim = AmbulanceSimulator(center_lat=12.9716, center_lon=77.5946, num_ambulances=6)
        self.v2v_sim = V2VSimulator(center_lat=12.9716, center_lon=77.5946)
        self.green_corridor = GreenCorridorSimulator(center_lat=12.9716, center_lon=77.5946, num_signals=4)

        # State tracking
        self.crash_detected = False
        self.crash_confidence = 0.0
        self.crash_features = None
        self.severity_result = None
        self.emergency_decision = None
        self.accident_id = None

        # Setup logging
        if self.enable_logging:
            self._setup_logging()

        # Demo scenario setup
        self._setup_demo_scenario()

    def _setup_logging(self):
        """Setup CSV logging for live sensor data."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"live_sensor_data_{timestamp}.csv"
        self.log_handle = open(self.log_file, 'w', newline='')
        self.log_writer = csv.writer(self.log_handle)
        self.log_writer.writerow([
            "Timestamp", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
            "AccMagnitude", "GyroMagnitude", "DrivingBehaviour",
            "CrashStatus", "CrashConfidence", "Severity",
            "Latitude", "Longitude"
        ])
        print(f"[LOG] Logging to: {self.log_file}")

    def _log_sensor_data(self, reading: SensorReading, crash_result: CrashDetectionResult,
                         severity: str = "", lat: float = 0.0, lon: float = 0.0):
        """Log sensor data to CSV."""
        if not self.enable_logging:
            return

        self.log_writer.writerow([
            datetime.fromtimestamp(reading.timestamp).strftime("%H:%M:%S.%f")[:-3],
            f"{reading.acc_x:.6f}", f"{reading.acc_y:.6f}", f"{reading.acc_z:.6f}",
            f"{reading.gyro_x:.6f}", f"{reading.gyro_y:.6f}", f"{reading.gyro_z:.6f}",
            f"{reading.acc_magnitude:.6f}", f"{reading.gyro_magnitude:.6f}",
            reading.mode.value,
            crash_result.status.value,
            f"{crash_result.confidence:.4f}",
            severity,
            f"{lat:.6f}", f"{lon:.6f}",
        ])
        self.log_handle.flush()

    def _setup_demo_scenario(self):
        """Setup demo scenario with ambulance, vehicles, and traffic signals."""
        # Add ambulance
        amb_loc = VehicleLocation(latitude=12.9600, longitude=77.5900, speed_kmh=0, heading_deg=45)
        self.ambulance = self.v2v_sim.add_ambulance("AMB-01", amb_loc, speed_kmh=80, heading=45)

        # Add other vehicles near the corridor
        vehicle_types = [VehicleType.PASSENGER_CAR, VehicleType.TRUCK, VehicleType.MOTORCYCLE]
        for i in range(8):
            angle = random.uniform(-0.5, 0.5)
            distance = random.uniform(200, 800)
            lat_off = distance * math.cos(angle) / 111320
            lon_off = distance * math.sin(angle) / (111320 * math.cos(math.radians(12.9716)))

            vloc = VehicleLocation(
                latitude=12.9716 + lat_off,
                longitude=77.5946 + lon_off,
                speed_kmh=random.uniform(30, 60),
                heading_deg=random.uniform(0, 360),
            )
            vtype = random.choice(vehicle_types)
            self.v2v_sim.add_vehicle(f"V-{i+1:02d}", vtype, vloc)

        # Set ambulance route for green corridor
        route = [
            {"lat": 12.9600, "lon": 77.5900},
            {"lat": 12.9716, "lon": 77.5946},
            {"lat": 12.9766, "lon": 77.5946},
            {"lat": 12.9816, "lon": 77.5946},
            {"lat": 12.9866, "lon": 77.5946},
        ]
        self.green_corridor.set_ambulance_route(route)

    def run_demo_scenario(self, duration_seconds: float = 60.0):
        """Run the complete demo scenario."""
        print("=" * 80)
        print("VEHICLE ACCIDENT DETECTION SYSTEM - COMPLETE PIPELINE DEMO")
        print("=" * 80)
        print(f"Sample Rate: {self.sample_rate_hz} Hz")
        print(f"Duration: {duration_seconds}s")
        print(f"Logging: {'ENABLED' if self.enable_logging else 'DISABLED'}")
        print("=" * 80)

        self.running = True
        start_time = time.time()
        last_status_print = 0
        status_interval = 2.0

        # Scenario phases
        phase = "normal"
        phase_start = start_time
        crash_triggered = False
        phase_printed = {"normal": False, "aggressive": False, "crash": False}

        print("\n[SYSTEM] Starting simulation...")
        print("[SYSTEM] Phase 1: NORMAL DRIVING (0-15s)")
        print("[SYSTEM] Phase 2: AGGRESSIVE DRIVING (15-30s)")
        print("[SYSTEM] Phase 3: CRASH EVENT (30s+)")
        print()

        try:
            while self.running and (time.time() - start_time) < duration_seconds:
                loop_start = time.time()

                # Update scenario phase
                elapsed = time.time() - start_time

                if elapsed < 15:
                    if not phase_printed["normal"]:
                        phase = "normal"
                        self.sensor_simulator.set_mode(SensorMode.NORMAL)
                        self.location_sim.current_location.speed_kmh = 60.0
                        print(f"\n{'='*60}")
                        print(f"PHASE: NORMAL DRIVING")
                        print(f"{'='*60}")
                        phase_printed["normal"] = True

                elif elapsed < 30:
                    if not phase_printed["aggressive"]:
                        phase = "aggressive"
                        self.sensor_simulator.set_mode(SensorMode.AGGRESSIVE)
                        self.location_sim.current_location.speed_kmh = 80.0
                        print(f"\n{'='*60}")
                        print(f"PHASE: AGGRESSIVE DRIVING")
                        print(f"{'='*60}")
                        phase_printed["aggressive"] = True

                elif not crash_triggered:
                    phase = "crash"
                    crash_triggered = True
                    self.sensor_simulator.set_mode(SensorMode.CRASH)
                    self.location_sim.current_location.speed_kmh = 0.0
                    print(f"\n{'='*60}")
                    print(f"PHASE: CRASH EVENT INITIATED")
                    print(f"{'='*60}")
                    phase_printed["crash"] = True

                # Get sensor reading
                reading = self.sensor_simulator.next_reading()

                # Update location
                self.location_sim.update_position(self.dt)
                current_loc = self.location_sim.get_current_location()

                # Crash detection
                crash_result = self.crash_detector.update(
                    reading.acc_x, reading.acc_y, reading.acc_z,
                    reading.gyro_x, reading.gyro_y, reading.gyro_z
                )

                # Check for crash detection
                if crash_result.status == CrashStatus.CRASH_DETECTED and not self.crash_detected:
                    self._handle_crash_detected(reading, crash_result, current_loc)

                # Update V2V and corridor if emergency active
                if self.emergency_decision and self.emergency_decision.requires_v2v:
                    self._update_emergency_response(current_loc)

                # Log data
                severity_str = self.severity_result['severity'] if self.severity_result else ""
                self._log_sensor_data(reading, crash_result, severity_str,
                                      current_loc.latitude, current_loc.longitude)

                # Print status periodically
                if time.time() - last_status_print >= status_interval:
                    self._print_status(reading, crash_result, current_loc)
                    last_status_print = time.time()

                # Maintain sample rate
                elapsed_loop = time.time() - loop_start
                sleep_time = max(0, self.dt - elapsed_loop)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                self.step_count += 1

        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted by user")

        finally:
            self._cleanup()

    def _handle_crash_detected(self, reading: SensorReading, crash_result: CrashDetectionResult,
                                location: VehicleLocation):
        """Handle crash detection event."""
        self.crash_detected = True
        self.crash_confidence = crash_result.confidence
        self.crash_features = crash_result.features
        self.accident_id = f"ACC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Freeze location at crash site
        crash_location = self.location_sim.simulate_crash()

        print(f"\n{'='*60}")
        print(f"!!! CRASH DETECTED !!!")
        print(f"{'='*60}")
        print(f"Accident ID: {self.accident_id}")
        print(f"Confidence: {crash_result.confidence:.1%}")
        print(f"Trigger: {crash_result.trigger_reason}")
        print(f"Location: {crash_location.latitude:.6f}, {crash_location.longitude:.6f}")
        print(f"{'='*60}")

        # Severity prediction
        print(f"\n[SEVERITY] Analyzing crash severity...")
        self.severity_result = self.severity_predictor.predict_from_window(
            acc_mag_buffer=list(self.crash_detector.acc_magnitude_buffer),
            gyro_mag_buffer=list(self.crash_detector.gyro_magnitude_buffer),
            acc_x_buffer=list(self.crash_detector.acc_x_buffer),
            acc_y_buffer=list(self.crash_detector.acc_y_buffer),
            acc_z_buffer=list(self.crash_detector.acc_z_buffer),
            sample_rate_hz=self.sample_rate_hz,
        )
        print(f"[SEVERITY] Predicted: {self.severity_result['severity']}")
        print(f"[SEVERITY] Confidence: {self.severity_result['confidence']:.1%}")

        # Emergency decision
        severity_enum = SeverityLevel(self.severity_result['severity'])
        self.emergency_decision = self.decision_engine.decide(
            severity_enum,
            self.severity_result['confidence'],
            {"lat": crash_location.latitude, "lon": crash_location.longitude}
        )
        print(self.decision_engine.get_decision_summary(self.emergency_decision))

        # If CRITICAL, activate full emergency response
        if self.emergency_decision.requires_ambulance:
            self._activate_emergency_response(crash_location)

    def _activate_emergency_response(self, crash_location: VehicleLocation):
        """Activate full emergency response for CRITICAL accidents."""
        print(f"\n{'='*60}")
        print(f"EMERGENCY RESPONSE ACTIVATED")
        print(f"{'='*60}")

        # Dispatch ambulance
        nearest = self.ambulance_sim.find_nearest_available(crash_location)
        if nearest:
            print(f"\n[AMBULANCE] Nearest available: {nearest.id}")
            print(f"[AMBULANCE] Distance: {nearest.distance_to(crash_location):.0f} m")
            print(f"[AMBULANCE] ETA: {nearest.eta_minutes:.1f} min")

            success = self.ambulance_sim.dispatch_ambulance(nearest.id, crash_location, self.accident_id)
            if success:
                print(f"[AMBULANCE] DISPATCHED: {nearest.id}")

                # Update V2V ambulance position
                amb_loc = VehicleLocation(
                    latitude=nearest.location.latitude,
                    longitude=nearest.location.longitude,
                    speed_kmh=nearest.speed_kmh,
                )
                self.v2v_sim.ambulance.location = amb_loc
                self.v2v_sim.ambulance.speed_kmh = nearest.speed_kmh

                # Set accident location for V2V
                self.v2v_sim.set_accident_location(crash_location)

                # Activate green corridor
                print(f"\n[GREEN CORRIDOR] Activating...")
                self.green_corridor.activate_emergency_corridor({
                    "lat": crash_location.latitude,
                    "lon": crash_location.longitude,
                })

                # Broadcast V2V messages
                self._broadcast_v2v_alerts()
            else:
                print(f"[AMBULANCE] Dispatch failed")
        else:
            print(f"[AMBULANCE] No available ambulances!")

    def _update_emergency_response(self, current_loc: VehicleLocation):
        """Update emergency response components."""
        # Update ambulance position
        self.ambulance_sim.update_ambulance_positions(
            self.dt,
            {self.accident_id: current_loc}
        )

        # Update V2V vehicles
        self.v2v_sim.update_vehicle_positions(self.dt)

        # Broadcast periodic V2V messages
        if self.step_count % 40 == 0:  # Every 2 seconds at 20Hz
            self._broadcast_v2v_alerts()

        # Update green corridor
        self.green_corridor.update_signals(self.dt)

    def _broadcast_v2v_alerts(self):
        """Broadcast V2V emergency messages."""
        # Emergency approaching
        msgs1 = self.v2v_sim.broadcast_emergency_approaching()
        # Accident ahead
        msgs2 = self.v2v_sim.broadcast_accident_ahead()
        # Green corridor
        msgs3 = self.v2v_sim.broadcast_green_corridor()

        total = len(msgs1) + len(msgs2) + len(msgs3)
        if total > 0:
            print(f"\n[V2V] Broadcasting {total} messages (Approaching: {len(msgs1)}, "
                  f"Accident: {len(msgs2)}, Corridor: {len(msgs3)})")

    def _print_status(self, reading: SensorReading, crash_result: CrashDetectionResult,
                      location: VehicleLocation):
        """Print current system status."""
        driving_mode = reading.mode.value
        crash_status = crash_result.status.value
        conf = f"{crash_result.confidence:.1%}"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"Mode: {driving_mode:10} | "
              f"AccMag: {reading.acc_magnitude:5.2f}g | "
              f"GyroMag: {reading.gyro_magnitude:5.2f}rad/s | "
              f"Crash: {crash_status:15} (conf: {conf}) | "
              f"Loc: ({location.latitude:.6f}, {location.longitude:.6f})")

    def _cleanup(self):
        """Cleanup resources."""
        self.running = False
        if self.enable_logging and hasattr(self, 'log_handle'):
            self.log_handle.close()
            print(f"\n[LOG] Data saved to: {self.log_file}")

        print("\n[SYSTEM] Simulation complete")
        print("[SYSTEM] Final Summary:")
        print(f"  Total Steps: {self.step_count}")
        print(f"  Crash Detected: {self.crash_detected}")
        if self.severity_result:
            print(f"  Severity: {self.severity_result['severity']}")
        if self.emergency_decision:
            print(f"  Action: {self.emergency_decision.action.value}")


def main():
    """Main entry point."""

    print("=" * 80)
    print("  VEHICLE ACCIDENT DETECTION + SEVERITY PREDICTION SYSTEM")
    print("  Software Simulation Demo")
    print("=" * 80)
    print()
    print("This demo simulates:")
    print("  1. Live MPU6050 sensor data (accelerometer + gyroscope)")
    print("  2. ML-based crash detection with sliding window")
    print("  3. CIREN-based severity prediction (prototype mapping)")
    print("  4. Emergency decision logic (MINOR/SERIOUS/CRITICAL)")
    print("  5. Simulated GPS location")
    print("  6. Nearby ambulance search & dispatch")
    print("  7. V2V communication (500m warning zone)")
    print("  8. Green corridor traffic signal coordination")
    print()
    print("IMPORTANT: This is a SOFTWARE SIMULATION for demonstration only.")
    print("No real hardware, GPS, V2V, or emergency services are used.")
    print("=" * 80)

    system = AccidentDetectionSystem(
        sample_rate_hz=20.0,
        log_dir="data/live",
        enable_logging=True,
    )

    system.run_demo_scenario(duration_seconds=45.0)


if __name__ == "__main__":
    main()