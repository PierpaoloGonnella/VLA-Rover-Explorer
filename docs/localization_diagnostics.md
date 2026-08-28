# ArUco localization diagnostics and controlled experiments

## What the recorded hardware session establishes

`sessions/ros2/20260828-152615/events.jsonl` spans 321.28 s. It contains 2,856
logged images (8.89 Hz), 1,306 published poses (4.07 Hz), and a 45.73% aggregate
pose/image ratio. The longest gap between published pose capture stamps is
26.71 s (p95 0.51 s, p99 1.71 s). Final `/cmd_vel` is zero.

The old image records contain logger receive time but not `Image.header.stamp`.
Consequently, historic stationary/moving/settling rates are estimates around
command transitions, not proof about capture state. The estimates are:

| Receive-time state | Images | Published poses | Ratio |
|---|---:|---:|---:|
| stationary | 1,142 | 185 | 16.20% |
| translation | 292 | 167 | 57.19% |
| rotation | 285 | 212 | 74.39% |
| settling (400 ms) | 1,137 | 742 | 65.26% |

The unexpectedly poor “stationary” bucket includes long periods after marker
loss and demonstrates why the aggregate 45.7% cannot be interpreted as motion
blur. It does not justify relaxing confidence or pose freshness limits.

Re-running the current detector over the saved JPEGs finds target ID 0 in 1,207
frames, 1,517 frames with rejected square candidates (`decode_failed`), 131
with no candidates, and one decoded non-target ID. JPEG recompression explains
why this offline count need not equal the 1,306 live poses. Detected marker side
length is 19.04–75.12 px (mean 46.95, p95 64.13). Valid frames have much higher
Laplacian sharpness than decode failures (44.94 versus 29.84 mean), while mean
brightness is similar (119.67 versus 117.47). Blur/focus is therefore the best
supported software-visible cause; gross exposure is not. Offline detector latency is
7.22 ms mean and 12.81 ms p95. This is not evidence of Python camera scheduling
or callback backlog, so migrating `camera_node` to C++ is not currently
justified.

The reproducible analysis command is:

```powershell
python rover_explorer_ros2/tools/analyze_localization_session.py `
  sessions/ros2/20260828-152615/events.jsonl --analyze-images `
  --output sessions/ros2/20260828-152615/localization_analysis.json
```

## Public diagnostic interfaces

- `/rover/localization/diagnostics` (`diagnostic_msgs/DiagnosticArray`) publishes
  one structured status per processed frame. Outcomes include `valid`,
  `moving_frame`, `pre_settle_frame`, `no_candidates`, `decode_failed`,
  `wrong_marker_id`, `below_confidence`, `too_small`, `near_boundary`,
  `invalid_image`, and `opencv_error`.
- `/rover/localization/failure_image` (`sensor_msgs/Image`) carries an optional
  annotated and rate-limited failure frame. `logger_node` writes it through a
  bounded worker queue to
  `sessions/ros2/<session>/localization_failures/`.
- `/rover/camera/diagnostics` records backend plus requested, accepted, and
  effective camera properties. A successful `VideoCapture.set()` is never
  assumed when the backend returns false.
- `/rover/battery/diagnostics` distinguishes `no_sample`, `parse_failure`,
  `non_finite`, `out_of_range`, `stale`, and `valid`.

Failure capture is off by default. A conservative collection run can use:

```yaml
capture_failure_images: true
failure_image_min_interval_seconds: 2.0
failure_image_max_per_session: 50
failure_image_annotated: true
failure_image_queue_size: 4
failure_image_jpeg_quality: 85
```

## Camera and detector experiments

Unset camera-control strings preserve backend behavior. On Windows, test
`camera_backend: dshow` and `camera_backend: msmf` separately. Change only one
of these strings per run and verify its `.accepted` and `.effective` values on
the diagnostic topic:

```yaml
camera_autofocus: "0"
camera_focus: "<measured fixed-focus value>"
camera_auto_exposure: "<backend-specific manual value>"
camera_exposure: "<shorter exposure value>"
camera_gain: ""
camera_brightness: ""
camera_contrast: ""
camera_white_balance: ""
```

OpenCV exposure units are backend-specific. Never copy a DirectShow value to
MSMF without checking the effective value. Prefer repeatable manual focus and a
short exposure with added diffuse illumination over a higher nominal FPS.

All exposed ArUco values retain the pre-instrumentation behavior, including
`aruco_corner_refinement: subpix`. Run one-factor experiments; do not make the
detector more permissive until the negative controls remain at zero false target
poses. The effective configuration is embedded in every localization diagnostic.

## Physical experiment procedure

An operator must be present and ready to assert emergency STOP. Do not run
autonomous physical motion merely to collect a report.

1. Stationary: keep ID 0 fully visible for at least 120 s. Record valid rate,
   side pixels, boundary distance, brightness, contrast, sharpness, frame age,
   and latency. Acceptance is at least 95% valid.
2. Rotation only: command the existing bounded left/right pulses. Report moving
   frames separately from post-settle frames and measure time from the end of
   settling to the first valid pose.
3. Translation only: in an enclosed level area, repeat bounded forward/backward
   pulses with the same measurements.
4. Position grid: physically place the rover at centre, sides, corners, and the
   farthest allowed positions. Record side-pixel and detection distributions.
5. Lighting: compare current light, diffuse added light, then shorter exposure
   plus added light. Record all effective controls.
6. Negative controls while stationary: blank scene, ID 1, partially clipped ID
   0, strong blur, edge placement, glare, and low contrast. Required false valid
   pose count is zero.

The localizer processes moving/pre-settle frames for diagnostics but never
publishes their pose. `motion_node` holds explicit STOP until it receives both a
pose whose capture stamp is after STOP plus settling and legal actions computed
after that pose. A timeout remains STOP; it never refreshes or synthesizes pose.

## Initial physical recommendation

The saved frames show a 19 px minimum and 47 px mean marker side, with rejected
square candidates dominating failures. First use a matte, flat, rigid 60 mm
marker with its full white quiet border, mounted above chassis occlusions and
aligned with rover forward. Keep the camera rigid and near perpendicular, remove
LED/gloss glare, and add diffuse light before shortening exposure. Move to 70–80
mm or reduce the operating envelope only if the 60 mm grid experiment still
misses the 95% stationary target or the marker approaches the image boundary.
This is a staged recommendation, not a hard-coded pose dimension.
