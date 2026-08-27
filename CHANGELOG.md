# Changelog

All notable changes to this project are documented in this file. Versions use
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-08-28

### Added

- Native ROS 2 package with custom pose, guard, policy, and VLM advisory
  messages; Windows-compatible node launchers; hardware and simulation launch
  files; and structured session logging.
- Asynchronous VLM scene reasoning with semantic waypoints, temporal context,
  hazard memory, stall detection, and deterministic A* execution.
- `bottom_center` closed-loop policy with live image-space waypoint updates,
  transform prediction, Schmitt-trigger HOLD behaviour, and stale-data fallback.
- Pose-verified turn bursts that rearm only after measured progress and stop on
  wrong-direction motion, physical stall, or excessive cumulative rotation.
- Immediate policy STOP preemption for active translation pulses.
- ROS 2 safety and launch integration tests in addition to the retained Python
  core suite.

### Changed

- Hardware translation pulses were shortened and recalibrated from physical
  session logs to reduce waypoint overshoot.
- The VLM is advisory and asynchronous; fast deterministic control, ultrasonic
  vetoes, frame guards, and the BLE bridge remain in the hard motion path.
- Windows setup and build documentation now uses the correct PowerShell setup
  syntax and the ROS 2 Pixi environment.

### Safety

- Lost or stale localization is fail-closed.
- Forward motion is vetoed by fresh guard and ultrasonic state.
- Stairs and drop-offs can be classified semantically but are not protected by
  a dedicated depth or cliff sensor; supervised testing remains mandatory.

## [1.0.0] - 2026-08-25

- Initial Python VLA rover release with BLE motion, ArUco localization,
  simulator, classical and VLM policies, coverage tracking, logging, and
  ultrasonic safety firmware.

[2.0.0]: https://github.com/PierpaoloGonnella/VLA-Rover-Explorer/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/PierpaoloGonnella/VLA-Rover-Explorer/releases/tag/v1.0.0
