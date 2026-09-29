from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime


class SeverityLevel(Enum):
    MINOR = "MINOR"
    SERIOUS = "SERIOUS"
    CRITICAL = "CRITICAL"


class EmergencyAction(Enum):
    LOG_ONLY = "LOG_ONLY"
    ALERT_AUTHORITIES = "ALERT_AUTHORITIES"
    DISPATCH_AMBULANCE = "DISPATCH_AMBULANCE"
    FULL_EMERGENCY_RESPONSE = "FULL_EMERGENCY_RESPONSE"


@dataclass
class EmergencyDecision:
    severity: SeverityLevel
    action: EmergencyAction
    requires_ambulance: bool
    requires_v2v: bool
    requires_green_corridor: bool
    message: str
    timestamp: datetime
    accident_location: Optional[Dict[str, float]] = None


class EmergencyDecisionEngine:
    """Makes emergency response decisions based on predicted severity."""

    def __init__(self):
        self.decision_log = []

    def decide(self, severity: SeverityLevel, confidence: float,
               location: Optional[Dict[str, float]] = None) -> EmergencyDecision:
        """Make emergency decision based on severity and confidence."""

        if severity == SeverityLevel.MINOR:
            action = EmergencyAction.LOG_ONLY
            message = "Minor accident detected. Logging event only."
            requires_ambulance = False
            requires_v2v = False
            requires_green_corridor = False

        elif severity == SeverityLevel.SERIOUS:
            action = EmergencyAction.ALERT_AUTHORITIES
            message = "Serious accident detected. Alerting authorities."
            requires_ambulance = False
            requires_v2v = False
            requires_green_corridor = False

        elif severity == SeverityLevel.CRITICAL:
            action = EmergencyAction.FULL_EMERGENCY_RESPONSE
            message = "CRITICAL: Life-threatening accident. Initiating full emergency response."
            requires_ambulance = True
            requires_v2v = True
            requires_green_corridor = True

        else:
            action = EmergencyAction.LOG_ONLY
            message = "Unknown severity. Logging event."
            requires_ambulance = False
            requires_v2v = False
            requires_green_corridor = False

        decision = EmergencyDecision(
            severity=severity,
            action=action,
            requires_ambulance=requires_ambulance,
            requires_v2v=requires_v2v,
            requires_green_corridor=requires_green_corridor,
            message=message,
            timestamp=datetime.now(),
            accident_location=location,
        )

        self.decision_log.append(decision)
        return decision

    def get_decision_summary(self, decision: EmergencyDecision) -> str:
        """Format decision for display."""
        lines = [
            "=" * 60,
            "EMERGENCY DECISION",
            "=" * 60,
            f"Severity         : {decision.severity.value}",
            f"Action           : {decision.action.value}",
            f"Ambulance Required: {'YES' if decision.requires_ambulance else 'NO'}",
            f"V2V Alert Required: {'YES' if decision.requires_v2v else 'NO'}",
            f"Green Corridor   : {'YES' if decision.requires_green_corridor else 'NO'}",
            f"Message          : {decision.message}",
            f"Timestamp        : {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if decision.accident_location:
            lines.append(f"Accident Location: {decision.accident_location['lat']:.6f}, {decision.accident_location['lon']:.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def demo():
    """Demo the emergency decision engine."""
    print("=" * 70)
    print("EMERGENCY DECISION ENGINE DEMO")
    print("=" * 70)

    engine = EmergencyDecisionEngine()

    test_cases = [
        (SeverityLevel.MINOR, 0.95),
        (SeverityLevel.SERIOUS, 0.85),
        (SeverityLevel.CRITICAL, 0.92),
    ]

    for severity, confidence in test_cases:
        print(f"\n--- Testing {severity.value} (confidence: {confidence}) ---")
        decision = engine.decide(severity, confidence, {"lat": 12.9716, "lon": 77.5946})
        print(engine.get_decision_summary(decision))


if __name__ == "__main__":
    demo()