from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class BleConfig(BaseModel):
    device_name: str = "BT05"
    characteristic_uuid: str = "0000ffe1-0000-1000-8000-00805f9b34fb"
    reconnect_attempts: int = 4
    backoff_seconds: float = 0.5


class MotionConfig(BaseModel):
    speed: int = Field(150, ge=60, le=255)
    translation_ms: int = 400
    turn_ms: int = 250
    settle_ms: int = 250
    watchdog_seconds: float = 2.0


class CameraConfig(BaseModel):
    index: int = 0
    width: int = 640
    height: int = 480


class LocalizationConfig(BaseModel):
    aruco_marker_id: int = 0
    min_confidence: float = 0.25
    color_hsv_low: tuple[int, int, int] = (35, 80, 80)
    color_hsv_high: tuple[int, int, int] = (90, 255, 255)
    vlm_grid_cols: int = 6
    vlm_grid_rows: int = 4


class CalibrationConfig(BaseModel):
    repetitions: int = 3
    noise_threshold_px: float = 2.0
    minimum_margin_frac: float = 0.2
    localization_timeout_seconds: float = 10.0
    post_motion_timeout_seconds: float = 2.0
    return_to_start_each_sample: bool = False
    minimum_valid_samples: int = 2
    sample_retries: int = 1
    angular_noise_threshold_degrees: float = 1.0
    lost_recovery_pulses: int = 1
    retry_duration_scale: float = 0.65
    minimum_retry_pulse_ms: int = 150


class GuardConfig(BaseModel):
    margin_frac: float = 0.12


class UltrasonicConfig(BaseModel):
    enabled: bool = True
    map_cols: int = Field(12, ge=4, le=40)
    map_rows: int = Field(8, ge=3, le=30)
    cm_per_translation_pulse: float = Field(10.0, gt=0)
    rover_radius_cm: float = Field(12.0, ge=0)
    scan_angle_degrees: float = Field(50.0, ge=10, le=80)
    maximum_mapping_distance_cm: int = Field(150, ge=30, le=300)
    obstacle_ttl_cycles: int = Field(40, ge=1)


class CoverageConfig(BaseModel):
    cols: int = 6
    rows: int = 4
    target: float = 0.9


class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    model: str = "qwen2.5vl:3b"
    timeout_seconds: float = 20.0


class RunnerConfig(BaseModel):
    cycles: int = 100
    target_cycle_seconds: float = 1.0
    session_dir: str = "sessions"


class SimulatorConfig(BaseModel):
    wheel_slip: float = 0.03
    ble_latency_ms: int = 10
    px_per_second_at_full_speed: float = 180.0
    radians_per_second_at_full_speed: float = 3.0
    seed: int = 7


class AppConfig(BaseModel):
    ble: BleConfig = BleConfig()
    motion: MotionConfig = MotionConfig()
    camera: CameraConfig = CameraConfig()
    localization: LocalizationConfig = LocalizationConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    guard: GuardConfig = GuardConfig()
    ultrasonic: UltrasonicConfig = UltrasonicConfig()
    coverage: CoverageConfig = CoverageConfig()
    ollama: OllamaConfig = OllamaConfig()
    runner: RunnerConfig = RunnerConfig()
    simulator: SimulatorConfig = SimulatorConfig()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load the JSON-compatible YAML configuration without an extra YAML dependency."""
    return AppConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
