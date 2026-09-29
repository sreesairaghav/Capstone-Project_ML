import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Any
from pathlib import Path
import warnings


@dataclass
class CrashFeatures:
    """Features extracted from crash sensor data."""
    peak_acc_magnitude: float
    peak_gyro_magnitude: float
    impact_duration_samples: int
    max_acc_change: float
    mean_acc_magnitude: float
    max_acc_x: float
    max_acc_y: float
    max_acc_z: float
    delta_v_estimate: float
    energy_estimate: float


class SeverityPredictor:
    """CIREN-based severity prediction with sensor-to-CIREN feature mapping.

    IMPORTANT: This is a PROTOTYPE mapping. The CIREN model was trained on
    crash investigation data (Delta-V, vehicle specs, road conditions, etc.)
    which are NOT directly available from an MPU6050 sensor.

    This mapping derives what it can from sensor data and uses population
    defaults for the rest. The predictions should be treated as DEMONSTRATION
    ONLY and not used for real emergency decisions.
    """

    CIREN_FEATURES = [
        'DVEST', 'DVTOTAL', 'DVLAT', 'DVLONG', 'ENERGY', 'ACCTYPE', 'BODYTYPE',
        'CURBWGT', 'MANEUVER', 'PREEVENT', 'PREMOVE', 'PREILOC', 'PREISTAB',
        'ROLLOVER', 'ROLINTYP', 'LANES', 'LGTCOND', 'SURCOND', 'SURTYPE',
        'SPLIMIT', 'TRAVELSP', 'RELINTER', 'TRAFCONT', 'TRAFFLOW', 'DVC1',
        'DVC2', 'DVC3', 'DVC4', 'DVC5', 'DVC6', 'DVD', 'DVL', 'EXTENT1',
        'EXTENT2', 'PDOF1', 'PDOF2', 'FIRE', 'ABS_DVLAT', 'ABS_DVLONG',
        'DV_COMPONENT_MAG'
    ]

    DEFAULT_VALUES = {
        'DVEST': 25.0,
        'DVTOTAL': 30.0,
        'DVLAT': 0.0,
        'DVLONG': -20.0,
        'ENERGY': 500.0,
        'ACCTYPE': 1.0,
        'BODYTYPE': 4.0,
        'CURBWGT': 1500.0,
        'MANEUVER': 1.0,
        'PREEVENT': 1.0,
        'PREMOVE': 1.0,
        'PREILOC': 1.0,
        'PREISTAB': 1.0,
        'ROLLOVER': 0.0,
        'ROLINTYP': 0.0,
        'LANES': 2.0,
        'LGTCOND': 1.0,
        'SURCOND': 1.0,
        'SURTYPE': 1.0,
        'SPLIMIT': 60.0,
        'TRAVELSP': 50.0,
        'RELINTER': 0.0,
        'TRAFCONT': 0.0,
        'TRAFFLOW': 1.0,
        'DVC1': 0.0,
        'DVC2': 0.0,
        'DVC3': 0.0,
        'DVC4': 0.0,
        'DVC5': 0.0,
        'DVC6': 0.0,
        'DVD': 100.0,
        'DVL': 150.0,
        'EXTENT1': 2.0,
        'EXTENT2': 2.0,
        'PDOF1': 12.0,
        'PDOF2': 0.0,
        'FIRE': 0.0,
        'ABS_DVLAT': 0.0,
        'ABS_DVLONG': 20.0,
        'DV_COMPONENT_MAG': 20.0,
    }

    SEVERITY_LABELS = ['MINOR', 'SERIOUS', 'CRITICAL']

    def __init__(self, model_path: str = "models/ciren_severity_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained CIREN severity model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"Loaded CIREN severity model from {self.model_path}")

    def _estimate_delta_v(self, peak_acc: float, duration_samples: int, sample_rate_hz: float = 20.0) -> float:
        """Estimate Delta-V from peak acceleration and duration.
        Delta-V ≈ integral of acceleration over time.
        Rough approximation: peak_acc * duration * conversion_factor
        """
        duration_seconds = duration_samples / sample_rate_hz
        delta_v_kmh = peak_acc * 9.81 * duration_seconds * 3.6
        return min(delta_v_kmh, 100.0)

    def _estimate_energy(self, delta_v: float, curb_weight: float = 1500.0) -> float:
        """Estimate crash energy: 0.5 * mass * v^2 (in kJ)."""
        v_ms = delta_v / 3.6
        energy_j = 0.5 * curb_weight * v_ms**2
        return energy_j / 1000.0

    def _map_sensor_to_ciren(self, crash_features: CrashFeatures,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Map sensor-derived crash features to CIREN feature space.

        Args:
            crash_features: Features extracted from sensor data during crash
            context: Optional context (vehicle_type, speed_limit, etc.)

        Returns:
            Dictionary with all 40 CIREN features
        """
        features = self.DEFAULT_VALUES.copy()

        dv_estimate = crash_features.delta_v_estimate
        if dv_estimate <= 0:
            dv_estimate = self._estimate_delta_v(
                crash_features.peak_acc_magnitude,
                crash_features.impact_duration_samples
            )
            crash_features.delta_v_estimate = dv_estimate

        features['DVEST'] = dv_estimate
        features['DVTOTAL'] = dv_estimate * 1.2
        features['DVLAT'] = crash_features.max_acc_y * 9.81 * (crash_features.impact_duration_samples / 20.0) * 3.6
        features['DVLONG'] = -crash_features.max_acc_x * 9.81 * (crash_features.impact_duration_samples / 20.0) * 3.6

        features['ABS_DVLAT'] = abs(features['DVLAT'])
        features['ABS_DVLONG'] = abs(features['DVLONG'])
        features['DV_COMPONENT_MAG'] = np.sqrt(features['DVLAT']**2 + features['DVLONG']**2)

        features['ENERGY'] = crash_features.energy_estimate
        if features['ENERGY'] <= 0:
            features['ENERGY'] = self._estimate_energy(dv_estimate)

        features['IMPACTSP'] = dv_estimate

        if crash_features.peak_acc_magnitude > 40:
            features['ACCTYPE'] = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        elif crash_features.peak_acc_magnitude > 20:
            features['ACCTYPE'] = np.random.choice([1, 2, 4], p=[0.3, 0.4, 0.3])
        else:
            features['ACCTYPE'] = np.random.choice([1, 4, 5], p=[0.2, 0.5, 0.3])

        if crash_features.peak_gyro_magnitude > 10:
            features['ROLLOVER'] = 1.0
            features['ROLINTYP'] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            features['ROLLDIST'] = crash_features.peak_gyro_magnitude * 2

        max_crush = crash_features.peak_acc_magnitude / 10.0
        features['DVC1'] = max_crush * np.random.uniform(0.8, 1.2)
        features['DVC2'] = max_crush * np.random.uniform(0.6, 1.0)
        features['DVC3'] = max_crush * np.random.uniform(0.4, 0.8)
        features['DVC4'] = max_crush * np.random.uniform(0.2, 0.6)
        features['DVC5'] = max_crush * np.random.uniform(0.1, 0.4)
        features['DVC6'] = max_crush * np.random.uniform(0.0, 0.2)
        features['DVD'] = max_crush * 50
        features['DVL'] = max_crush * 80
        features['EXTENT1'] = min(int(max_crush / 5) + 1, 6)
        features['EXTENT2'] = min(int(max_crush / 8) + 1, 6)

        if features['DVLONG'] < -10:
            features['PDOF1'] = 12.0
        elif features['DVLAT'] > 10:
            features['PDOF1'] = 3.0
        elif features['DVLAT'] < -10:
            features['PDOF1'] = 9.0
        else:
            features['PDOF1'] = 6.0

        if context:
            if 'vehicle_type' in context:
                features['BODYTYPE'] = context['vehicle_type']
            if 'speed_limit' in context:
                features['SPLIMIT'] = context['speed_limit']
                features['TRAVELSP'] = min(context['speed_limit'] * 0.9, features['TRAVELSP'])
            if 'road_type' in context:
                features['SURTYPE'] = context['road_type']
            if 'lighting' in context:
                features['LGTCOND'] = context['lighting']
            if 'weather' in context:
                features['SURCOND'] = context['weather']

        return features

    def predict(self, crash_features: CrashFeatures,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predict severity from crash sensor features.

        Args:
            crash_features: Features extracted from sensor data
            context: Optional contextual information

        Returns:
            Dictionary with severity prediction and probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        ciren_features = self._map_sensor_to_ciren(crash_features, context)

        feature_vector = np.array([[ciren_features[f] for f in self.CIREN_FEATURES]])
        feature_df = pd.DataFrame(feature_vector, columns=self.CIREN_FEATURES)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = self.model.predict(feature_df)[0]
            probabilities = self.model.predict_proba(feature_df)[0]

        prob_dict = dict(zip(self.model.classes_, probabilities))

        return {
            "severity": prediction,
            "probabilities": prob_dict,
            "confidence": float(np.max(probabilities)),
            "ciren_features_used": ciren_features,
            "note": "PROTOTYPE: Features mapped from sensor data with defaults. Not for real use."
        }

    def predict_from_window(self, acc_mag_buffer: list, gyro_mag_buffer: list,
                            acc_x_buffer: list, acc_y_buffer: list, acc_z_buffer: list,
                            sample_rate_hz: float = 20.0,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convenience method to predict from raw sensor buffers."""
        acc_mag = np.array(acc_mag_buffer)
        gyro_mag = np.array(gyro_mag_buffer)

        crash_features = CrashFeatures(
            peak_acc_magnitude=float(np.max(acc_mag)),
            peak_gyro_magnitude=float(np.max(gyro_mag)),
            impact_duration_samples=len(acc_mag),
            max_acc_change=float(np.max(np.abs(np.diff(acc_mag)))) if len(acc_mag) > 1 else 0.0,
            mean_acc_magnitude=float(np.mean(acc_mag)),
            max_acc_x=float(np.max(np.abs(acc_x_buffer))),
            max_acc_y=float(np.max(np.abs(acc_y_buffer))),
            max_acc_z=float(np.max(np.abs(acc_z_buffer))),
            delta_v_estimate=0.0,
            energy_estimate=0.0,
        )

        return self.predict(crash_features, context)


def demo():
    """Demo the severity predictor with simulated crash features."""
    print("=" * 70)
    print("SEVERITY PREDICTOR DEMO")
    print("=" * 70)

    predictor = SeverityPredictor()

    test_cases = [
        ("Low Impact", CrashFeatures(
            peak_acc_magnitude=15.0, peak_gyro_magnitude=3.0,
            impact_duration_samples=8, max_acc_change=5.0,
            mean_acc_magnitude=8.0, max_acc_x=10.0, max_acc_y=5.0, max_acc_z=8.0,
            delta_v_estimate=15.0, energy_estimate=100.0
        )),
        ("Medium Impact", CrashFeatures(
            peak_acc_magnitude=30.0, peak_gyro_magnitude=8.0,
            impact_duration_samples=10, max_acc_change=12.0,
            mean_acc_magnitude=18.0, max_acc_x=25.0, max_acc_y=15.0, max_acc_z=10.0,
            delta_v_estimate=35.0, energy_estimate=500.0
        )),
        ("High Impact", CrashFeatures(
            peak_acc_magnitude=50.0, peak_gyro_magnitude=20.0,
            impact_duration_samples=12, max_acc_change=25.0,
            mean_acc_magnitude=30.0, max_acc_x=40.0, max_acc_y=25.0, max_acc_z=15.0,
            delta_v_estimate=60.0, energy_estimate=1500.0
        )),
    ]

    for name, features in test_cases:
        print(f"\n--- {name} Crash ---")
        result = predictor.predict(features)
        print(f"  Predicted Severity: {result['severity']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probabilities:")
        for sev, prob in result['probabilities'].items():
            print(f"    {sev}: {prob:.3f}")

    print("\n" + "=" * 70)
    print("IMPORTANT: This is a PROTOTYPE mapping for demonstration only.")
    print("The CIREN model expects crash investigation data not available from MPU6050.")
    print("Predictions are illustrative and NOT reliable for real emergencies.")
    print("=" * 70)


if __name__ == "__main__":
    demo()