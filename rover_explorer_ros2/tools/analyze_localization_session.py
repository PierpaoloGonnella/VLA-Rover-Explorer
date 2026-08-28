"""Analyze localization continuity from a ROS 2 JSONL session.

The historical logger recorded image receive time but not the Image header.  Motion
classification for images is therefore explicitly labelled as receive-time based;
new sessions record the complete structured localization diagnostic instead.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import cv2
import numpy as np


def percentile(values: Iterable[float], percentage: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(samples),
        "min": min(samples, default=None),
        "p50": percentile(samples, 50),
        "p90": percentile(samples, 90),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "max": max(samples, default=None),
        "mean": statistics.fmean(samples) if samples else None,
    }


def stamp(message: dict[str, Any], fallback: float) -> float:
    value = message.get("header", {}).get("stamp", {})
    seconds = value.get("sec")
    nanoseconds = value.get("nanosec")
    if seconds is None or nanoseconds is None:
        return fallback
    result = float(seconds) + float(nanoseconds) * 1e-9
    return result if result > 0 else fallback


class MotionTimeline:
    def __init__(self, events: list[dict[str, Any]], settle_seconds: float) -> None:
        commands = [event for event in events if event.get("topic") == "/cmd_vel"]
        self.times = [float(event["timestamp"]) for event in commands]
        self.values: list[tuple[float, float]] = []
        self.stop_transitions: list[float] = []
        was_moving = False
        for event in commands:
            message = event["message"]
            linear = float(message.get("linear", {}).get("x", 0.0))
            angular = float(message.get("angular", {}).get("z", 0.0))
            moving = abs(linear) > 1e-6 or abs(angular) > 1e-6
            if was_moving and not moving:
                self.stop_transitions.append(float(event["timestamp"]))
            self.values.append((linear, angular))
            was_moving = moving
        self.settle_seconds = settle_seconds

    def classify(self, timestamp: float) -> str:
        index = bisect.bisect_right(self.times, timestamp) - 1
        if index >= 0:
            linear, angular = self.values[index]
            if abs(linear) > 1e-6 and abs(angular) <= 1e-6:
                return "translation"
            if abs(angular) > 1e-6 and abs(linear) <= 1e-6:
                return "rotation"
            if abs(linear) > 1e-6 or abs(angular) > 1e-6:
                return "mixed_motion"
        stop_index = bisect.bisect_right(self.stop_transitions, timestamp) - 1
        if stop_index >= 0 and timestamp < self.stop_transitions[stop_index] + self.settle_seconds:
            return "settling"
        return "stationary"


def visual_metrics(path: Path, detector: cv2.aruco.ArucoDetector, marker_id: int) -> dict[str, Any]:
    frame = cv2.imread(str(path))
    if frame is None:
        return {"outcome": "invalid_image", "path": str(path)}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    started = cv2.getTickCount()
    corners, ids, rejected = detector.detectMarkers(frame)
    latency_ms = (cv2.getTickCount() - started) * 1000.0 / cv2.getTickFrequency()
    detected_ids = [] if ids is None else [int(value) for value in ids.ravel()]
    result: dict[str, Any] = {
        "path": str(path),
        "detector_latency_ms": latency_ms,
        "brightness_mean": float(gray.mean()),
        "contrast_stddev": float(gray.std()),
        "sharpness_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_fraction": float(np.count_nonzero(gray <= 15) / gray.size),
        "saturated_fraction": float(np.count_nonzero(gray >= 240) / gray.size),
        "candidate_count": len(corners),
        "rejected_candidate_count": len(rejected),
        "detected_ids": detected_ids,
    }
    if marker_id not in detected_ids:
        result["outcome"] = (
            "wrong_marker_id" if detected_ids else "decode_failed" if rejected else "no_candidates"
        )
        return result
    points = np.asarray(corners[detected_ids.index(marker_id)], dtype=np.float32).reshape(4, 2)
    sides = [float(np.linalg.norm(points[(index + 1) % 4] - points[index])) for index in range(4)]
    perimeter = float(cv2.arcLength(points, True))
    height, width = gray.shape
    result.update(
        outcome="valid",
        marker_side_px=statistics.fmean(sides),
        marker_min_side_px=min(sides),
        marker_max_side_px=max(sides),
        marker_perimeter_px=perimeter,
        marker_area_ratio=float(abs(cv2.contourArea(points)) / (width * height)),
        marker_boundary_distance_px=float(
            min(points[:, 0].min(), points[:, 1].min(), width - 1 - points[:, 0].max(), height - 1 - points[:, 1].max())
        ),
        confidence=min(1.0, perimeter / max(32.0, 0.2 * min(height, width))),
    )
    return result


def analyze(events_path: Path, settle_seconds: float, images: bool, marker_id: int) -> dict[str, Any]:
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    events.sort(key=lambda event: float(event["timestamp"]))
    timeline = MotionTimeline(events, settle_seconds)
    image_events = [event for event in events if event.get("topic") == "/rover/image_raw"]
    pose_events = [event for event in events if event.get("topic") == "/rover/pose"]
    image_states = [timeline.classify(float(event["timestamp"])) for event in image_events]
    pose_times = [stamp(event["message"], float(event["timestamp"])) for event in pose_events]
    pose_states = [timeline.classify(value) for value in pose_times]
    states = ["stationary", "translation", "rotation", "mixed_motion", "settling"]
    by_state: dict[str, Any] = {}
    for state in states:
        image_count = image_states.count(state)
        pose_count = pose_states.count(state)
        by_state[state] = {
            "images": image_count,
            "valid_poses": pose_count,
            "detection_rate": pose_count / image_count if image_count else None,
        }

    gaps = [newer - older for older, newer in zip(pose_times, pose_times[1:])]
    reacquisition = []
    for stop_time in timeline.stop_transitions:
        eligible = stop_time + settle_seconds
        index = bisect.bisect_left(pose_times, eligible)
        if index < len(pose_times):
            reacquisition.append(max(0.0, pose_times[index] - eligible))

    legal = [event for event in events if event.get("topic") == "/rover/legal_actions"]
    decisions = [event for event in events if event.get("topic") == "/rover/policy/classic_decision"]
    batteries = [event for event in events if event.get("topic") == "/rover/battery"]
    finite_battery = [
        event for event in batteries
        if math.isfinite(float(event.get("message", {}).get("voltage", math.nan)))
    ]
    result: dict[str, Any] = {
        "source": str(events_path),
        "limitations": [
            "Historical image events contain logger receive time, not Image.header.stamp.",
            "Per-state image rates are receive-time estimates around command transitions.",
            "Historical logs contain no per-frame detector outcome or callback latency.",
        ],
        "timeline": {
            "start": float(events[0]["timestamp"]),
            "end": float(events[-1]["timestamp"]),
            "duration_seconds": float(events[-1]["timestamp"]) - float(events[0]["timestamp"]),
            "stop_transitions": len(timeline.stop_transitions),
            "final_cmd_vel_zero": bool(timeline.values) and all(abs(value) <= 1e-6 for value in timeline.values[-1]),
        },
        "counts": {
            "images": len(image_events),
            "valid_poses": len(pose_events),
            "cmd_vel": len(timeline.times),
            "policy_decisions": len(decisions),
            "stale_pose_legal_actions": sum(
                "stale/lost" in str(event.get("message", {}).get("reason", "")) for event in legal
            ),
            "sonar_blocked_legal_actions": sum(
                bool(event.get("message", {}).get("sonar_blocked")) for event in legal
            ),
            "policy_stop_decisions": sum(
                event.get("message", {}).get("action") == "stop" for event in decisions
            ),
        },
        "rates_hz": {
            "images": len(image_events) / (float(events[-1]["timestamp"]) - float(events[0]["timestamp"])),
            "valid_poses": len(pose_events) / (float(events[-1]["timestamp"]) - float(events[0]["timestamp"])),
        },
        "overall_detection_rate": len(pose_events) / len(image_events) if image_events else None,
        "receive_time_detection_by_motion_state": by_state,
        "valid_pose_gap_seconds": distribution(gaps),
        "post_settle_reacquisition_seconds": distribution(reacquisition),
        "battery": {
            "messages": len(batteries),
            "unknown_before_first_valid": (
                len(batteries) if not finite_battery else batteries.index(finite_battery[0])
            ),
            "first_valid_voltage": (
                None if not finite_battery else float(finite_battery[0]["message"]["voltage"])
            ),
            "first_valid_delay_seconds": (
                None if not finite_battery else float(finite_battery[0]["timestamp"]) - float(batteries[0]["timestamp"])
            ),
        },
    }

    if images:
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
        )
        metrics = []
        for event, state in zip(image_events, image_states):
            path = Path(str(event["message"].get("path", "")))
            if not path.is_absolute():
                path = events_path.parents[3] / path
            item = visual_metrics(path, detector, marker_id)
            item["motion_state"] = state
            metrics.append(item)
        valid = [item for item in metrics if item["outcome"] == "valid"]
        result["offline_image_analysis"] = {
            "note": "Detector latency is offline compute time, not historical callback latency.",
            "outcomes": {
                outcome: sum(item["outcome"] == outcome for item in metrics)
                for outcome in sorted({item["outcome"] for item in metrics})
            },
            "target_detection_by_motion_state": {
                state: {
                    "frames": sum(item["motion_state"] == state for item in metrics),
                    "target_detected": sum(
                        item["motion_state"] == state and item["outcome"] == "valid" for item in metrics
                    ),
                    "detection_rate": (
                        sum(item["motion_state"] == state and item["outcome"] == "valid" for item in metrics)
                        / sum(item["motion_state"] == state for item in metrics)
                        if any(item["motion_state"] == state for item in metrics)
                        else None
                    ),
                }
                for state in states
            },
            "quality_by_outcome": {
                outcome: {
                    "sharpness": distribution(
                        item["sharpness_laplacian_variance"]
                        for item in metrics
                        if item["outcome"] == outcome
                    ),
                    "brightness": distribution(
                        item["brightness_mean"] for item in metrics if item["outcome"] == outcome
                    ),
                    "contrast": distribution(
                        item["contrast_stddev"] for item in metrics if item["outcome"] == outcome
                    ),
                }
                for outcome in sorted({item["outcome"] for item in metrics})
            },
            "marker_side_px": distribution(item["marker_side_px"] for item in valid),
            "marker_boundary_distance_px": distribution(item["marker_boundary_distance_px"] for item in valid),
            "sharpness": distribution(item["sharpness_laplacian_variance"] for item in metrics),
            "brightness": distribution(item["brightness_mean"] for item in metrics),
            "contrast": distribution(item["contrast_stddev"] for item in metrics),
            "dark_fraction": distribution(item["dark_fraction"] for item in metrics),
            "saturated_fraction": distribution(item["saturated_fraction"] for item in metrics),
            "offline_detector_latency_ms": distribution(item["detector_latency_ms"] for item in metrics),
            "false_positive_count": 0,
            "false_positive_note": "No labelled negative-control corpus is present in this hardware session.",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("--settle-seconds", type=float, default=0.4)
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--analyze-images", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = analyze(arguments.events, arguments.settle_seconds, arguments.analyze_images, arguments.marker_id)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
