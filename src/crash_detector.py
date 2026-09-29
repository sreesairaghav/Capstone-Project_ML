import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class CrashStatus(Enum):
    NO_CRASH = "NO_CRASH"
    POSSIBLE_CRASH = "POSSIBLE_CRASH"
    CRASH_DETECTED = "CRASH_DETECTED"


@dataclass
class CrashDetectionResult:
    status: CrashStatus
    confidence: float
    features: dict
    trigger_reason: str = ""


class CrashDetector:
    """Robust crash detection using sliding window and multiple sensor features."""

    def __init__(
        self,
        window_size: int = 20,
        min_consecutive_abnormal: int = 5,
        acc_magnitude_threshold: float = 8.0,
        acc_change_threshold: float = 5.0,
        gyro_magnitude_threshold: float = 3.0,
        impact_duration_min: int = 3,
        impact_duration_max: int = 15,
        confidence_threshold: float = 0.7,
    ):
        self.window_size = window_size
        self.min_consecutive_abnormal = min_consecutive_abnormal
        self.acc_magnitude_threshold = acc_magnitude_threshold
        self.acc_change_threshold = acc_change_threshold
        self.gyro_magnitude_threshold = gyro_magnitude_threshold
        self.impact_duration_min = impact_duration_min
        self.impact_duration_max = impact_duration_max
        self.confidence_threshold = confidence_threshold

        self.acc_magnitude_buffer = deque(maxlen=window_size)
        self.gyro_magnitude_buffer = deque(maxlen=window_size)
        self.acc_x_buffer = deque(maxlen=window_size)
        self.acc_y_buffer = deque(maxlen=window_size)
        self.acc_z_buffer = deque(maxlen=window_size)
        self.gyro_x_buffer = deque(maxlen=window_size)
        self.gyro_y_buffer = deque(maxlen=window_size)
        self.gyro_z_buffer = deque(maxlen=window_size)

        self.consecutive_abnormal = 0
        self.crash_start_idx = -1
        self.in_crash_sequence = False

    def _extract_window_features(self) -> dict:
        """Extract features from current sliding window."""
        if len(self.acc_magnitude_buffer) < self.window_size:
            return {}

        acc_mag = np.array(self.acc_magnitude_buffer)
        gyro_mag = np.array(self.gyro_magnitude_buffer)
        acc_x = np.array(self.acc_x_buffer)
        acc_y = np.array(self.acc_y_buffer)
        acc_z = np.array(self.acc_z_buffer)

        acc_change = np.diff(acc_mag)
        gyro_change = np.diff(gyro_mag)

        features = {
            "max_acc_mag": float(np.max(acc_mag)),
            "mean_acc_mag": float(np.mean(acc_mag)),
            "std_acc_mag": float(np.std(acc_mag)),
            "min_acc_mag": float(np.min(acc_mag)),
            "max_gyro_mag": float(np.max(gyro_mag)),
            "mean_gyro_mag": float(np.mean(gyro_mag)),
            "std_gyro_mag": float(np.std(gyro_mag)),
            "max_acc_change": float(np.max(np.abs(acc_change))),
            "mean_acc_change": float(np.mean(np.abs(acc_change))),
            "max_gyro_change": float(np.max(np.abs(gyro_change))),
            "max_acc_x": float(np.max(np.abs(acc_x))),
            "max_acc_y": float(np.max(np.abs(acc_y))),
            "max_acc_z": float(np.max(np.abs(acc_z))),
            "impact_duration_estimate": 0,
        }

        return features

    def _check_abnormal_reading(self, acc_mag: float, gyro_mag: float, acc_change: float) -> bool:
        """Check if a single reading is abnormal."""
        checks = [
            acc_mag > self.acc_magnitude_threshold,
            abs(acc_change) > self.acc_change_threshold,
            gyro_mag > self.gyro_magnitude_threshold,
        ]
        return any(checks)

    def _compute_confidence(self, features: dict, consecutive_abnormal: int) -> float:
        """Compute crash confidence score based on features."""
        if not features or "max_acc_mag" not in features:
            return 0.0

        confidence = 0.0

        if features["max_acc_mag"] > self.acc_magnitude_threshold:
            confidence += 0.3 * min(features["max_acc_mag"] / 30.0, 1.0)

        if features["max_acc_change"] > self.acc_change_threshold:
            confidence += 0.25 * min(features["max_acc_change"] / 15.0, 1.0)

        if features["max_gyro_mag"] > self.gyro_magnitude_threshold:
            confidence += 0.2 * min(features["max_gyro_mag"] / 10.0, 1.0)

        if features["max_acc_mag"] > 15.0:
            confidence += 0.15

        if consecutive_abnormal >= self.min_consecutive_abnormal:
            confidence += 0.1 * min(consecutive_abnormal / self.min_consecutive_abnormal, 1.0)

        return min(confidence, 1.0)

    def _estimate_impact_duration(self) -> int:
        """Estimate impact duration from acceleration spikes."""
        if len(self.acc_magnitude_buffer) < self.window_size:
            return 0

        acc_mag = np.array(self.acc_magnitude_buffer)
        above_threshold = acc_mag > self.acc_magnitude_threshold

        if not np.any(above_threshold):
            return 0

        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if above_threshold[0]:
            starts = np.concatenate(([0], starts))
        if above_threshold[-1]:
            ends = np.concatenate((ends, [len(above_threshold)]))

        if len(starts) > 0 and len(ends) > 0:
            durations = ends - starts
            return int(np.max(durations))

        return 0

    def update(self, acc_x: float, acc_y: float, acc_z: float,
               gyro_x: float, gyro_y: float, gyro_z: float) -> CrashDetectionResult:
        """Update detector with new sensor reading."""

        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        self.acc_x_buffer.append(acc_x)
        self.acc_y_buffer.append(acc_y)
        self.acc_z_buffer.append(acc_z)
        self.gyro_x_buffer.append(gyro_x)
        self.gyro_y_buffer.append(gyro_y)
        self.gyro_z_buffer.append(gyro_z)
        self.acc_magnitude_buffer.append(acc_mag)
        self.gyro_magnitude_buffer.append(gyro_mag)

        acc_change = 0.0
        if len(self.acc_magnitude_buffer) >= 2:
            acc_change = self.acc_magnitude_buffer[-1] - self.acc_magnitude_buffer[-2]

        is_abnormal = self._check_abnormal_reading(acc_mag, gyro_mag, acc_change)

        features = self._extract_window_features()

        if is_abnormal:
            self.consecutive_abnormal += 1
            if not self.in_crash_sequence:
                self.in_crash_sequence = True
                self.crash_start_idx = len(self.acc_magnitude_buffer) - 1
        else:
            if self.in_crash_sequence:
                self.in_crash_sequence = False
            self.consecutive_abnormal = 0

        impact_duration = self._estimate_impact_duration()
        if features:
            features["impact_duration_estimate"] = impact_duration
        else:
            features = {"impact_duration_estimate": impact_duration}

        confidence = self._compute_confidence(features, self.consecutive_abnormal)

        if confidence >= self.confidence_threshold and self.consecutive_abnormal >= self.min_consecutive_abnormal:
            if impact_duration >= self.impact_duration_min and impact_duration <= self.impact_duration_max:
                status = CrashStatus.CRASH_DETECTED
                trigger_reason = (
                    f"High acc ({features['max_acc_mag']:.1f}g), "
                    f"acc change ({features['max_acc_change']:.1f}g), "
                    f"gyro ({features['max_gyro_mag']:.1f}rad/s), "
                    f"duration ({impact_duration} samples)"
                )
            else:
                status = CrashStatus.POSSIBLE_CRASH
                trigger_reason = f"Abnormal readings but duration {impact_duration} outside expected range"
        elif self.consecutive_abnormal >= self.min_consecutive_abnormal // 2:
            status = CrashStatus.POSSIBLE_CRASH
            trigger_reason = f"Consecutive abnormal readings: {self.consecutive_abnormal}"
        else:
            status = CrashStatus.NO_CRASH
            trigger_reason = "Normal driving"

        return CrashDetectionResult(
            status=status,
            confidence=confidence,
            features=features,
            trigger_reason=trigger_reason,
        )

    def reset(self):
        """Reset detector state."""
        self.acc_magnitude_buffer.clear()
        self.gyro_magnitude_buffer.clear()
        self.acc_x_buffer.clear()
        self.acc_y_buffer.clear()
        self.acc_z_buffer.clear()
        self.gyro_x_buffer.clear()
        self.gyro_y_buffer.clear()
        self.gyro_z_buffer.clear()
        self.consecutive_abnormal = 0
        self.crash_start_idx = -1
        self.in_crash_sequence = False


def test_crash_detector():
    """Test crash detector with simulated data."""
    print("=" * 70)
    print("CRASH DETECTOR TEST")
    print("=" * 70)

    detector = CrashDetector()

    print("\n--- Testing NORMAL driving ---")
    normal_acc_std = 0.9
    normal_gyro_std = 0.1
    for i in range(30):
        acc = np.random.normal(0, normal_acc_std, 3)
        gyro = np.random.normal(0, normal_gyro_std, 3)
        result = detector.update(*acc, *gyro)
        if i % 10 == 0:
            print(f"  Sample {i}: {result.status.value} (conf: {result.confidence:.3f})")

    detector.reset()

    print("\n--- Testing AGGRESSIVE driving ---")
    agg_acc_std = 1.2
    agg_gyro_std = 0.13
    for i in range(30):
        acc = np.random.normal(0, agg_acc_std, 3)
        gyro = np.random.normal(0, agg_gyro_std, 3)
        result = detector.update(*acc, *gyro)
        if i % 10 == 0:
            print(f"  Sample {i}: {result.status.value} (conf: {result.confidence:.3f})")

    detector.reset()

    print("\n--- Testing CRASH EVENT ---")
    crash_sim = __import__('live_sensor_simulator', fromlist=['CrashEventSimulator']).CrashEventSimulator(sample_rate_hz=20.0)
    crash_detected = False
    while not crash_sim.is_complete():
        reading = crash_sim.next_reading()
        if reading:
            result = detector.update(
                reading.acc_x, reading.acc_y, reading.acc_z,
                reading.gyro_x, reading.gyro_y, reading.gyro_z
            )
            if result.status == CrashStatus.CRASH_DETECTED and not crash_detected:
                print(f"  *** CRASH DETECTED at sample {crash_sim.current_sample} ***")
                print(f"      Confidence: {result.confidence:.3f}")
                print(f"      Reason: {result.trigger_reason}")
                print(f"      Features: max_acc={result.features['max_acc_mag']:.1f}, "
                      f"max_gyro={result.features['max_gyro_mag']:.1f}, "
                      f"duration={result.features['impact_duration_estimate']}")
                crash_detected = True

    if not crash_detected:
        print("  No crash detected (may need threshold tuning)")


if __name__ == "__main__":
    test_crash_detector()