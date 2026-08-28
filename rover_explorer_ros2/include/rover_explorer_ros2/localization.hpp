#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace rover_explorer_ros2::localization
{

struct Pose
{
  cv::Point2d centre;
  std::optional<double> heading;
  float confidence;
};

class Localizer
{
public:
  virtual ~Localizer() = default;
  virtual std::optional<Pose> locate(const cv::Mat & frame) = 0;
};

class ArucoLocalizer final : public Localizer
{
public:
  explicit ArucoLocalizer(std::int64_t marker_id = 0, double heading_offset_radians = 0.0);

  std::optional<Pose> locate(const cv::Mat & frame) override;
  const std::vector<int> & last_detected_ids() const noexcept;

private:
  std::int64_t marker_id_;
  double heading_offset_radians_;
  std::vector<int> last_detected_ids_;
  cv::aruco::ArucoDetector detector_;
};

class ColorBlobLocalizer final : public Localizer
{
public:
  ColorBlobLocalizer(
    cv::Scalar hsv_low = cv::Scalar(35, 80, 80),
    cv::Scalar hsv_high = cv::Scalar(90, 255, 255),
    double min_area = 30.0);

  std::optional<Pose> locate(const cv::Mat & frame) override;

private:
  cv::Scalar hsv_low_;
  cv::Scalar hsv_high_;
  double min_area_;
};

/// A validated view over a ROS image. The view must not outlive the message.
/// The repository's Python fallback supports packed/padded bgr8 and 8UC3 only;
/// the native path deliberately preserves that contract.
cv::Mat validated_bgr_view(const sensor_msgs::msg::Image & message);

double normalize_heading(double radians) noexcept;

std::optional<float> aruco_projection_error(
  const cv::Point2d & custom_centre,
  const cv::Point3d & external_position,
  double fx, double fy, double cx, double cy) noexcept;

}  // namespace rover_explorer_ros2::localization
