import time
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Generator, Optional, List
from enum import Enum


class SensorMode(Enum):
    NORMAL = "NORMAL"
    SLOW = "SLOW"
    AGGRESSIVE = "AGGRESSIVE"
    CRASH = "CRASH"


@dataclass
class SensorReading:
    timestamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mode: SensorMode

    @property
    def acc_magnitude(self) -> float:
        return np.sqrt(self.acc_x**2 + self.acc_y**2 + self.acc_z**2)

    @property
    def gyro_magnitude(self) -> float:
        return np.sqrt(self.gyro_x**2 + self.gyro_y**2 + self.gyro_z**2)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "AccX": self.acc_x,
            "AccY": self.acc_y,
            "AccZ": self.acc_z,
            "GyroX": self.gyro_x,
            "GyroY": self.gyro_y,
            "GyroZ": self.gyro_z,
            "AccMagnitude": self.acc_magnitude,
            "GyroMagnitude": self.gyro_magnitude,
            "Mode": self.mode.value,
        }


class CrashEventSimulator:
    """Simulates a realistic crash event with physical impact pattern."""

    PHASE_PRE_IMPACT = "pre_impact"
    PHASE_IMPACT = "impact"
    PHASE_POST_IMPACT = "post_impact"
    PHASE_STABILIZATION = "stabilization"

    def __init__(
        self,
        pre_impact_duration: float = 1.0,
        impact_duration: float = 0.5,
        post_impact_duration: float = 1.5,
        stabilization_duration: float = 2.0,
        sample_rate_hz: float = 20.0,
    ):
        self.pre_impact_duration = pre_impact_duration
        self.impact_duration = impact_duration
        self.post_impact_duration = post_impact_duration
        self.stabilization_duration = stabilization_duration
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz

        self.total_samples = int(
            (pre_impact_duration + impact_duration + post_impact_duration + stabilization_duration)
            * sample_rate_hz
        )

        self.phase_samples = {
            self.PHASE_PRE_IMPACT: int(pre_impact_duration * sample_rate_hz),
            self.PHASE_IMPACT: int(impact_duration * sample_rate_hz),
            self.PHASE_POST_IMPACT: int(post_impact_duration * sample_rate_hz),
            self.PHASE_STABILIZATION: int(stabilization_duration * sample_rate_hz),
        }

        self.current_sample = 0
        self.current_phase = self.PHASE_PRE_IMPACT
        self.phase_sample_count = 0
        self.crash_direction = np.random.choice(
            ["frontal", "side", "rear", "rollover"], p=[0.4, 0.3, 0.2, 0.1]
        )
        self._initialize_crash_profile()

    def _initialize_crash_profile(self):
        """Initialize crash-specific parameters based on crash type."""
        if self.crash_direction == "frontal":
            self.peak_acc_g = np.random.uniform(30, 60)
            self.impact_vector = np.array([1.0, 0.0, 0.0])
            self.gyro_peak = np.random.uniform(5, 15)
        elif self.crash_direction == "side":
            self.peak_acc_g = np.random.uniform(20, 40)
            self.impact_vector = np.array([0.0, 1.0, 0.0])
            self.gyro_peak = np.random.uniform(10, 25)
        elif self.crash_direction == "rear":
            self.peak_acc_g = np.random.uniform(15, 30)
            self.impact_vector = np.array([-1.0, 0.0, 0.0])
            self.gyro_peak = np.random.uniform(3, 10)
        else:
            self.peak_acc_g = np.random.uniform(25, 50)
            self.impact_vector = np.array([0.5, 0.5, 0.7])
            self.gyro_peak = np.random.uniform(15, 35)

        self.impact_vector = self.impact_vector / np.linalg.norm(self.impact_vector)

    def _get_phase_for_sample(self, sample_idx: int) -> str:
        cumulative = 0
        for phase, count in self.phase_samples.items():
            if sample_idx < cumulative + count:
                return phase
            cumulative += count
        return self.PHASE_STABILIZATION

    def _generate_pre_impact(self) -> np.ndarray:
        """Pre-impact: aggressive driving leading to crash."""
        base_acc = np.random.normal(0, 2.5, 3)
        base_gyro = np.random.normal(0, 0.5, 3)
        return base_acc, base_gyro

    def _generate_impact(self, progress: float) -> np.ndarray:
        """Impact phase: high acceleration spike."""
        pulse = np.sin(progress * np.pi)
        acc_magnitude = self.peak_acc_g * pulse
        acc = self.impact_vector * acc_magnitude + np.random.normal(0, 2.0, 3)
        gyro_magnitude = self.gyro_peak * pulse
        gyro_dir = np.random.normal(0, 1, 3)
        gyro_dir = gyro_dir / (np.linalg.norm(gyro_dir) + 1e-8)
        gyro = gyro_dir * gyro_magnitude
        return acc, gyro

    def _generate_post_impact(self, progress: float) -> np.ndarray:
        """Post-impact: decaying oscillations."""
        decay = np.exp(-progress * 4)
        acc_magnitude = self.peak_acc_g * 0.3 * decay
        acc = self.impact_vector * acc_magnitude + np.random.normal(0, 1.5 * decay, 3)
        gyro_magnitude = self.gyro_peak * 0.4 * decay
        gyro_dir = np.random.normal(0, 1, 3)
        gyro_dir = gyro_dir / (np.linalg.norm(gyro_dir) + 1e-8)
        gyro = gyro_dir * gyro_magnitude
        return acc, gyro

    def _generate_stabilization(self, progress: float) -> np.ndarray:
        """Stabilization: returning to normal."""
        decay = np.exp(-progress * 2)
        acc = np.random.normal(0, 0.8 * decay, 3)
        gyro = np.random.normal(0, 0.1 * decay, 3)
        return acc, gyro

    def next_reading(self) -> Optional[SensorReading]:
        if self.current_sample >= self.total_samples:
            return None

        phase = self._get_phase_for_sample(self.current_sample)
        phase_progress = self.phase_sample_count / max(self.phase_samples[phase], 1)

        if phase == self.PHASE_PRE_IMPACT:
            acc, gyro = self._generate_pre_impact()
        elif phase == self.PHASE_IMPACT:
            acc, gyro = self._generate_impact(phase_progress)
        elif phase == self.PHASE_POST_IMPACT:
            acc, gyro = self._generate_post_impact(phase_progress)
        else:
            acc, gyro = self._generate_stabilization(phase_progress)

        reading = SensorReading(
            timestamp=time.time(),
            acc_x=float(acc[0]),
            acc_y=float(acc[1]),
            acc_z=float(acc[2]),
            gyro_x=float(gyro[0]),
            gyro_y=float(gyro[1]),
            gyro_z=float(gyro[2]),
            mode=SensorMode.CRASH,
        )

        self.current_sample += 1
        self.phase_sample_count += 1

        if phase != self._get_phase_for_sample(self.current_sample):
            self.phase_sample_count = 0

        return reading

    def is_complete(self) -> bool:
        return self.current_sample >= self.total_samples


class LiveSensorSimulator:
    """Generates realistic vehicle sensor data from learned driving patterns."""

    DRIVING_PROFILES = {
        SensorMode.NORMAL: {
            "acc_mean": np.array([-0.018, -0.028, 0.034]),
            "acc_std": np.array([0.866, 0.814, 0.965]),
            "gyro_mean": np.array([0.002, -0.001, 0.010]),
            "gyro_std": np.array([0.061, 0.118, 0.114]),
            "correlation": 0.1,
        },
        SensorMode.SLOW: {
            "acc_mean": np.array([0.047, -0.034, 0.009]),
            "acc_std": np.array([0.859, 0.736, 0.867]),
            "gyro_mean": np.array([0.003, 0.000, 0.004]),
            "gyro_std": np.array([0.068, 0.125, 0.103]),
            "correlation": 0.05,
        },
        SensorMode.AGGRESSIVE: {
            "acc_mean": np.array([0.096, -0.171, -0.021]),
            "acc_std": np.array([1.218, 1.139, 1.129]),
            "gyro_mean": np.array([0.000, -0.003, 0.011]),
            "gyro_std": np.array([0.072, 0.136, 0.131]),
            "correlation": 0.15,
        },
    }

    def __init__(
        self,
        sample_rate_hz: float = 20.0,
        mode: SensorMode = SensorMode.NORMAL,
        crash_probability: float = 0.001,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.current_mode = mode
        self.crash_probability = crash_probability
        self.crash_simulator: Optional[CrashEventSimulator] = None
        self.in_crash = False
        self.reading_count = 0

    def set_mode(self, mode: SensorMode):
        """Change the driving mode."""
        if mode == SensorMode.CRASH and not self.in_crash:
            self._start_crash()
        else:
            self.current_mode = mode

    def _start_crash(self):
        """Initiate a crash event."""
        self.in_crash = True
        self.crash_simulator = CrashEventSimulator(sample_rate_hz=self.sample_rate_hz)

    def _generate_normal_reading(self, mode: SensorMode) -> SensorReading:
        profile = self.DRIVING_PROFILES[mode]
        acc = np.random.normal(profile["acc_mean"], profile["acc_std"])
        gyro = np.random.normal(profile["gyro_mean"], profile["gyro_std"])

        if profile["correlation"] > 0:
            noise = np.random.normal(0, profile["correlation"], 3)
            acc += noise
            gyro += noise * 0.1

        return SensorReading(
            timestamp=time.time(),
            acc_x=float(acc[0]),
            acc_y=float(acc[1]),
            acc_z=float(acc[2]),
            gyro_x=float(gyro[0]),
            gyro_y=float(gyro[1]),
            gyro_z=float(gyro[2]),
            mode=mode,
        )

    def next_reading(self) -> SensorReading:
        """Generate the next sensor reading."""
        if self.in_crash and self.crash_simulator:
            reading = self.crash_simulator.next_reading()
            if reading is None:
                self.in_crash = False
                self.crash_simulator = None
                self.current_mode = SensorMode.NORMAL
                return self._generate_normal_reading(SensorMode.NORMAL)
            return reading

        if self.current_mode != SensorMode.CRASH and random.random() < self.crash_probability:
            self._start_crash()
            return self.next_reading()

        reading = self._generate_normal_reading(self.current_mode)
        self.reading_count += 1
        return reading

    def stream(self, duration_seconds: Optional[float] = None) -> Generator[SensorReading, None, None]:
        """Continuously stream sensor readings."""
        start_time = time.time()
        while True:
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                break
            yield self.next_reading()
            time.sleep(self.dt)


def demo():
    """Demo the sensor simulator with different modes."""
    print("=" * 70)
    print("LIVE SENSOR SIMULATOR DEMO")
    print("=" * 70)

    simulator = LiveSensorSimulator(sample_rate_hz=20.0, mode=SensorMode.NORMAL)

    print("\n--- NORMAL DRIVING (5 samples) ---")
    for _ in range(5):
        reading = simulator.next_reading()
        print(f"AccX={reading.acc_x:7.3f} AccY={reading.acc_y:7.3f} AccZ={reading.acc_z:7.3f} | "
              f"GyroX={reading.gyro_x:7.3f} GyroY={reading.gyro_y:7.3f} GyroZ={reading.gyro_z:7.3f} | "
              f"AccMag={reading.acc_magnitude:.3f} GyroMag={reading.gyro_magnitude:.3f} | {reading.mode.value}")

    print("\n--- AGGRESSIVE DRIVING (5 samples) ---")
    simulator.set_mode(SensorMode.AGGRESSIVE)
    for _ in range(5):
        reading = simulator.next_reading()
        print(f"AccX={reading.acc_x:7.3f} AccY={reading.acc_y:7.3f} AccZ={reading.acc_z:7.3f} | "
              f"GyroX={reading.gyro_x:7.3f} GyroY={reading.gyro_y:7.3f} GyroZ={reading.gyro_z:7.3f} | "
              f"AccMag={reading.acc_magnitude:.3f} GyroMag={reading.gyro_magnitude:.3f} | {reading.mode.value}")

    print("\n--- CRASH EVENT (full sequence) ---")
    simulator.set_mode(SensorMode.CRASH)
    crash_readings = []
    while not simulator.crash_simulator.is_complete() if simulator.crash_simulator else False:
        reading = simulator.next_reading()
        crash_readings.append(reading)
        if len(crash_readings) % 10 == 1:
            phase = simulator.crash_simulator._get_phase_for_sample(simulator.crash_simulator.current_sample - 1)
            print(f"  [{phase}] AccMag={reading.acc_magnitude:6.2f} GyroMag={reading.gyro_magnitude:6.2f}")

    print(f"\nTotal crash samples: {len(crash_readings)}")
    print(f"Peak AccMagnitude: {max(r.acc_magnitude for r in crash_readings):.2f} g")
    print(f"Peak GyroMagnitude: {max(r.gyro_magnitude for r in crash_readings):.2f} rad/s")


if __name__ == "__main__":
    demo()