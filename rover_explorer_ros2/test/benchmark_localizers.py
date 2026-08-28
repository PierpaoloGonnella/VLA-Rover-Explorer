"""Benchmark Python and C++ ArUco localization on identical lossless frames."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

from ament_index_python.packages import get_package_prefix
import cv2
import numpy as np

from rover_explorer.localize import ArucoLocalizer
from rover_explorer.simulator import RoverSimulator


def representative_frames() -> list[np.ndarray]:
    simulator = RoverSimulator(width=640, height=480, wheel_slip=0)
    frames = []
    for x, y, heading in [
        (320, 240, -2.4),
        (320, 240, -1.1),
        (320, 240, 0.0),
        (320, 240, 0.8),
        (320, 240, 2.3),
        (34, 34, 0.15),
    ]:
        simulator.x, simulator.y, simulator.heading = x, y, heading
        frames.append(simulator.render())
    frames.append(np.zeros((480, 640, 3), np.uint8))
    occluded = frames[2].copy()
    cv2.rectangle(occluded, (300, 215), (355, 270), (225, 230, 225), -1)
    frames.append(occluded)
    wrong = RoverSimulator(width=640, height=480, wheel_slip=0, marker_id=1)
    frames.append(wrong.render())
    return frames


def python_measure(frames, iterations, marker_id, offset):
    localizer = ArucoLocalizer(marker_id, offset)
    poses = [localizer.locate(frame) for frame in frames]
    latencies = []
    wall_start = time.perf_counter()
    for _ in range(iterations):
        for frame in frames:
            start = time.perf_counter()
            localizer.locate(frame)
            latencies.append((time.perf_counter() - start) * 1000.0)
    wall_seconds = time.perf_counter() - wall_start
    ordered = sorted(latencies)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return poses, {
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": p95,
        "max_ms": max(latencies),
        "fps": len(latencies) / wall_seconds,
        "samples": len(latencies),
    }


def cpp_measure(frames, iterations, marker_id, offset, directory):
    paths = []
    for index, frame in enumerate(frames):
        path = directory / f"frame_{index:02d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write benchmark frame {path}")
        paths.append(path)
    prefix = Path(get_package_prefix("rover_explorer_ros2"))
    executable = prefix / "lib" / "rover_explorer_ros2" / (
        "localizer_probe.exe" if os.name == "nt" else "localizer_probe")
    completed = subprocess.run(
        [str(executable), str(iterations), str(marker_id), str(offset),
         *(str(path) for path in paths)],
        check=True,
        text=True,
        capture_output=True,
    )
    poses = [None] * len(frames)
    metrics = None
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "POSE":
            index, detected = int(fields[1]), fields[2] == "1"
            if detected:
                poses[index] = tuple(float(value) for value in fields[3:7])
        elif fields[0] == "METRIC":
            metrics = {
                "mean_ms": float(fields[1]),
                "median_ms": float(fields[2]),
                "p95_ms": float(fields[3]),
                "max_ms": float(fields[4]),
                "fps": float(fields[5]),
                "samples": int(fields[6]),
            }
    if metrics is None:
        raise RuntimeError(f"No metric line from native probe: {completed.stdout}")
    return poses, metrics


def compare(python_poses, cpp_poses):
    detected_agreement = 0
    detected_pairs = 0
    centre_errors = []
    heading_errors = []
    confidence_errors = []
    for python_pose, cpp_pose in zip(python_poses, cpp_poses):
        if (python_pose is None) == (cpp_pose is None):
            detected_agreement += 1
        if python_pose is None or cpp_pose is None:
            continue
        detected_pairs += 1
        x, y, heading, confidence = cpp_pose
        centre_errors.append(math.dist(python_pose.centre, (x, y)))
        heading_errors.append(abs(
            (python_pose.heading - heading + math.pi) % (2 * math.pi) - math.pi))
        confidence_errors.append(abs(python_pose.confidence - confidence))
    return {
        "detection_agreement": detected_agreement / len(python_poses),
        "detected_pairs": detected_pairs,
        "max_centre_error_px": max(centre_errors, default=0.0),
        "max_heading_error_rad": max(heading_errors, default=0.0),
        "max_confidence_error": max(confidence_errors, default=0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--heading-offset-radians", type=float, default=-0.37)
    parser.add_argument("--assert-parity", action="store_true")
    args = parser.parse_args()
    frames = representative_frames()
    python_poses, python_metrics = python_measure(
        frames, args.iterations, args.marker_id, args.heading_offset_radians)
    with tempfile.TemporaryDirectory(prefix="rover_localizer_benchmark_") as temp:
        cpp_poses, cpp_metrics = cpp_measure(
            frames, args.iterations, args.marker_id,
            args.heading_offset_radians, Path(temp))
    parity = compare(python_poses, cpp_poses)
    result = {"frames": len(frames), "python": python_metrics,
              "cpp": cpp_metrics, "parity": parity}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.assert_parity:
        assert parity["detection_agreement"] == 1.0
        assert parity["max_centre_error_px"] < 0.05
        assert parity["max_heading_error_rad"] < 1e-4
        assert parity["max_confidence_error"] < 1e-5


if __name__ == "__main__":
    main()
