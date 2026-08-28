"""Pure telemetry validation shared by ROS adapters and unit tests."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class BatteryReading:
    voltage: float
    valid: bool
    fresh: bool
    reason: str


def validate_battery_sample(
    raw_millivolts: object,
    received_at: float | None,
    now: float,
    minimum_voltage: float,
    maximum_voltage: float,
    timeout_seconds: float,
) -> BatteryReading:
    if raw_millivolts is None or received_at is None:
        return BatteryReading(math.nan, False, False, "no_sample")
    try:
        voltage = float(raw_millivolts) / 1000.0
    except (TypeError, ValueError, OverflowError):
        return BatteryReading(math.nan, False, False, "parse_failure")
    if not math.isfinite(voltage):
        return BatteryReading(math.nan, False, False, "non_finite")
    fresh = now - float(received_at) <= timeout_seconds
    if not minimum_voltage <= voltage <= maximum_voltage:
        return BatteryReading(math.nan, False, fresh, "out_of_range")
    if not fresh:
        return BatteryReading(math.nan, False, False, "stale")
    return BatteryReading(voltage, True, True, "valid")
