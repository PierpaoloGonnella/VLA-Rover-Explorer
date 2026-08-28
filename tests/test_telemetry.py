import math

from rover_explorer.telemetry import validate_battery_sample


def sample(value, received=10.0, now=10.5):
    return validate_battery_sample(value, received, now, 3.0, 9.0, 2.0)


def test_battery_startup_and_malformed_samples_are_unknown():
    assert sample(None, None).reason == "no_sample"
    assert sample("broken").reason == "parse_failure"
    assert math.isnan(sample(None, None).voltage)


def test_battery_rejects_non_finite_out_of_range_and_stale_values():
    assert sample(math.nan).reason == "non_finite"
    assert sample(math.inf).reason == "non_finite"
    assert sample(12000).reason == "out_of_range"
    assert sample(5000, received=1.0, now=10.0).reason == "stale"


def test_battery_accepts_first_valid_sample_and_recovers_after_invalid_data():
    first = sample(7420)
    assert first.valid and first.voltage == 7.42
    assert not sample(math.nan).valid
    recovered = sample(7300, received=20.0, now=20.1)
    assert recovered.valid and recovered.reason == "valid"
