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

struct ArucoDetectorConfig
{
  int adaptive_threshold_window_min{3};
  int adaptive_threshold_window_max{23};
  int adaptive_threshold_window_step{10};
  double adaptive_threshold_constant{7.0};
  double minimum_marker_perimeter_rate{0.03};
  double maximum_marker_perimeter_rate{4.0};
  double polygonal_approximation_accuracy_rate{0.03};
  double minimum_corner_distance_rate{0.05};
  int minimum_distance_to_border{3};
  int marker_border_bits{1};
  int perspective_remove_pixels_per_cell{4};
  double error_correction_rate{0.6};
  int corner_refinement_method{cv::aruco::CORNER_REFINE_SUBPIX};
  int corner_refinement_window{5};
  int corner_refinement_iterations{30};
  double corner_refinement_accuracy{0.1};

  void validate() const;
  cv::aruco::DetectorParameters opencv_parameters() const;
  std::string describe() const;
};

struct FrameDiagnostics
{
  std::string outcome{"no_candidates"};
  std::vector<int> detected_ids;
  std::vector<std::vector<cv::Point2f>> detected_corners;
  std::vector<std::vector<cv::Point2f>> rejected_corners;
  std::vector<cv::Point2f> target_corners;
  std::size_t candidate_count{0U};
  std::size_t rejected_candidate_count{0U};
  double marker_mean_side_px{0.0};
  double marker_min_side_px{0.0};
  double marker_max_side_px{0.0};
  double marker_perimeter_px{0.0};
  double marker_area_ratio{0.0};
  double marker_boundary_distance_px{0.0};
  double grayscale_mean{0.0};
  double grayscale_stddev{0.0};
  double dark_fraction{0.0};
  double saturated_fraction{0.0};
  double sharpness{0.0};
  float confidence{0.0F};
  std::optional<Pose> pose;
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
  explicit ArucoLocalizer(
    std::int64_t marker_id = 0, double heading_offset_radians = 0.0,
    const ArucoDetectorConfig & config = ArucoDetectorConfig{});

  std::optional<Pose> locate(const cv::Mat & frame) override;
  FrameDiagnostics analyze(const cv::Mat & frame, double minimum_confidence = 0.0);
  const std::vector<int> & last_detected_ids() const noexcept;
  const ArucoDetectorConfig & config() const noexcept;

private:
  std::int64_t marker_id_;
  double heading_offset_radians_;
  ArucoDetectorConfig config_;
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
