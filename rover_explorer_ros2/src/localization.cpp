#include "rover_explorer_ros2/localization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

namespace rover_explorer_ros2::localization
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kBgrChannels = 3U;

void measure_marker_geometry(
  const std::vector<cv::Point2f> & points, const cv::Size & image_size,
  FrameDiagnostics & result)
{
  if (points.size() != 4U) {
    return;
  }
  std::vector<double> sides;
  sides.reserve(4U);
  for (std::size_t index = 0; index < 4U; ++index) {
    sides.push_back(cv::norm(points[(index + 1U) % 4U] - points[index]));
  }
  result.marker_mean_side_px =
    std::accumulate(sides.begin(), sides.end(), 0.0) / sides.size();
  result.marker_min_side_px = *std::min_element(sides.begin(), sides.end());
  result.marker_max_side_px = *std::max_element(sides.begin(), sides.end());
  result.marker_perimeter_px = cv::arcLength(points, true);
  result.marker_area_ratio = std::abs(cv::contourArea(points)) /
    std::max(1.0, static_cast<double>(image_size.area()));
  auto boundary = std::numeric_limits<double>::infinity();
  for (const auto & point : points) {
    boundary = std::min({
        boundary, static_cast<double>(point.x), static_cast<double>(point.y),
        static_cast<double>(image_size.width - 1) - point.x,
        static_cast<double>(image_size.height - 1) - point.y});
  }
  result.marker_boundary_distance_px = boundary;
}
}  // namespace

void ArucoDetectorConfig::validate() const
{
  if (adaptive_threshold_window_min < 3 || adaptive_threshold_window_max < adaptive_threshold_window_min ||
    adaptive_threshold_window_step <= 0)
  {
    throw std::invalid_argument("invalid adaptive threshold window configuration");
  }
  if (minimum_marker_perimeter_rate <= 0.0 ||
    maximum_marker_perimeter_rate <= minimum_marker_perimeter_rate)
  {
    throw std::invalid_argument("marker perimeter rates must be positive and ordered");
  }
  if (polygonal_approximation_accuracy_rate <= 0.0 ||
    minimum_corner_distance_rate < 0.0 || minimum_distance_to_border < 0)
  {
    throw std::invalid_argument("invalid marker geometry detector configuration");
  }
  if (marker_border_bits <= 0 || perspective_remove_pixels_per_cell <= 0 ||
    error_correction_rate < 0.0 || error_correction_rate > 1.0)
  {
    throw std::invalid_argument("invalid marker decoding detector configuration");
  }
  if (corner_refinement_method < cv::aruco::CORNER_REFINE_NONE ||
    corner_refinement_method > cv::aruco::CORNER_REFINE_APRILTAG ||
    corner_refinement_window <= 0 || corner_refinement_iterations <= 0 ||
    corner_refinement_accuracy <= 0.0)
  {
    throw std::invalid_argument("invalid corner refinement detector configuration");
  }
}

cv::aruco::DetectorParameters ArucoDetectorConfig::opencv_parameters() const
{
  validate();
  cv::aruco::DetectorParameters parameters;
  parameters.adaptiveThreshWinSizeMin = adaptive_threshold_window_min;
  parameters.adaptiveThreshWinSizeMax = adaptive_threshold_window_max;
  parameters.adaptiveThreshWinSizeStep = adaptive_threshold_window_step;
  parameters.adaptiveThreshConstant = adaptive_threshold_constant;
  parameters.minMarkerPerimeterRate = minimum_marker_perimeter_rate;
  parameters.maxMarkerPerimeterRate = maximum_marker_perimeter_rate;
  parameters.polygonalApproxAccuracyRate = polygonal_approximation_accuracy_rate;
  parameters.minCornerDistanceRate = minimum_corner_distance_rate;
  parameters.minDistanceToBorder = minimum_distance_to_border;
  parameters.markerBorderBits = marker_border_bits;
  parameters.perspectiveRemovePixelPerCell = perspective_remove_pixels_per_cell;
  parameters.errorCorrectionRate = error_correction_rate;
  parameters.cornerRefinementMethod = corner_refinement_method;
  parameters.cornerRefinementWinSize = corner_refinement_window;
  parameters.cornerRefinementMaxIterations = corner_refinement_iterations;
  parameters.cornerRefinementMinAccuracy = corner_refinement_accuracy;
  return parameters;
}

std::string ArucoDetectorConfig::describe() const
{
  std::ostringstream output;
  output << "adaptive_window=" << adaptive_threshold_window_min << ':' <<
    adaptive_threshold_window_max << ':' << adaptive_threshold_window_step <<
    ", adaptive_constant=" << adaptive_threshold_constant <<
    ", perimeter_rate=" << minimum_marker_perimeter_rate << ':' <<
    maximum_marker_perimeter_rate << ", polygon_accuracy=" <<
    polygonal_approximation_accuracy_rate << ", corner_distance=" <<
    minimum_corner_distance_rate << ", border_px=" << minimum_distance_to_border <<
    ", border_bits=" << marker_border_bits << ", perspective_px=" <<
    perspective_remove_pixels_per_cell << ", error_correction=" << error_correction_rate <<
    ", refinement=" << corner_refinement_method << ':' << corner_refinement_window << ':' <<
    corner_refinement_iterations << ':' << corner_refinement_accuracy;
  return output.str();
}

double normalize_heading(const double radians) noexcept
{
  auto wrapped = std::fmod(radians + kPi, 2.0 * kPi);
  if (wrapped < 0.0) {
    wrapped += 2.0 * kPi;
  }
  return wrapped - kPi;
}

std::optional<float> aruco_projection_error(
  const cv::Point2d & custom_centre,
  const cv::Point3d & external_position,
  const double fx, const double fy, const double cx, const double cy) noexcept
{
  if (fx <= 0.0 || fy <= 0.0 || std::abs(external_position.z) < 1e-9) {
    return std::nullopt;
  }
  const cv::Point2d projected(
    fx * external_position.x / external_position.z + cx,
    fy * external_position.y / external_position.z + cy);
  return static_cast<float>(cv::norm(custom_centre - projected));
}

ArucoLocalizer::ArucoLocalizer(
  const std::int64_t marker_id, const double heading_offset_radians,
  const ArucoDetectorConfig & config)
: marker_id_(marker_id),
  heading_offset_radians_(heading_offset_radians),
  config_(config),
  detector_(
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
    config_.opencv_parameters())
{
}

std::optional<Pose> ArucoLocalizer::locate(const cv::Mat & frame)
{
  return analyze(frame).pose;
}

FrameDiagnostics ArucoLocalizer::analyze(
  const cv::Mat & frame, const double minimum_confidence)
{
  FrameDiagnostics result;
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::Scalar mean;
  cv::Scalar deviation;
  cv::meanStdDev(gray, mean, deviation);
  result.grayscale_mean = mean[0];
  result.grayscale_stddev = deviation[0];
  result.dark_fraction = static_cast<double>(cv::countNonZero(gray <= 15)) / gray.total();
  result.saturated_fraction = static_cast<double>(cv::countNonZero(gray >= 240)) / gray.total();
  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar laplacian_mean;
  cv::Scalar laplacian_deviation;
  cv::meanStdDev(laplacian, laplacian_mean, laplacian_deviation);
  result.sharpness = laplacian_deviation[0] * laplacian_deviation[0];

  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  detector_.detectMarkers(frame, corners, ids, result.rejected_corners);
  last_detected_ids_ = ids;
  result.detected_ids = ids;
  result.detected_corners = corners;
  result.candidate_count = ids.size();
  result.rejected_candidate_count = result.rejected_corners.size();

  const auto match = std::find(ids.begin(), ids.end(), marker_id_);
  if (match == ids.end()) {
    if (!ids.empty()) {
      result.outcome = "wrong_marker_id";
    } else if (!result.rejected_corners.empty()) {
      const auto largest = std::max_element(
        result.rejected_corners.begin(), result.rejected_corners.end(),
        [](const auto & left, const auto & right) {
          return cv::arcLength(left, true) < cv::arcLength(right, true);
        });
      measure_marker_geometry(*largest, frame.size(), result);
      const auto minimum_side =
        config_.minimum_marker_perimeter_rate * std::max(frame.rows, frame.cols) / 4.0;
      if (result.marker_mean_side_px < minimum_side) {
        result.outcome = "too_small";
      } else if (result.marker_boundary_distance_px < config_.minimum_distance_to_border) {
        result.outcome = "near_boundary";
      } else {
        result.outcome = "decode_failed";
      }
    }
    return result;
  }
  const auto index = static_cast<std::size_t>(std::distance(ids.begin(), match));
  if (index >= corners.size() || corners[index].size() != 4U) {
    result.outcome = "decode_failed";
    return result;
  }

  const auto & points = corners[index];
  result.target_corners = points;
  measure_marker_geometry(points, frame.size(), result);
  cv::Point2d centre(0.0, 0.0);
  for (const auto & point : points) {
    centre.x += static_cast<double>(point.x);
    centre.y += static_cast<double>(point.y);
  }
  centre.x /= 4.0;
  centre.y /= 4.0;

  const auto edge = points[1] - points[0];
  const auto heading = normalize_heading(
    std::atan2(static_cast<double>(edge.y), static_cast<double>(edge.x)) +
    heading_offset_radians_);
  const auto denominator = std::max(
    32.0, 0.2 * static_cast<double>(std::min(frame.rows, frame.cols)));
  result.confidence = static_cast<float>(
    std::min(1.0, result.marker_perimeter_px / denominator));
  if (result.confidence < minimum_confidence) {
    const auto confidence_side = minimum_confidence * denominator / 4.0;
    result.outcome = result.marker_mean_side_px < confidence_side ?
      "too_small" : "below_confidence";
    return result;
  }
  result.outcome = "valid";
  result.pose = Pose{centre, heading, result.confidence};
  return result;
}

const std::vector<int> & ArucoLocalizer::last_detected_ids() const noexcept
{
  return last_detected_ids_;
}

const ArucoDetectorConfig & ArucoLocalizer::config() const noexcept
{
  return config_;
}

ColorBlobLocalizer::ColorBlobLocalizer(
  const cv::Scalar hsv_low, const cv::Scalar hsv_high, const double min_area)
: hsv_low_(hsv_low), hsv_high_(hsv_high), min_area_(min_area)
{
}

std::optional<Pose> ColorBlobLocalizer::locate(const cv::Mat & frame)
{
  cv::Mat hsv;
  cv::Mat mask;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, hsv_low_, hsv_high_, mask);
  cv::morphologyEx(
    mask, mask, cv::MORPH_OPEN,
    cv::Mat::ones(3, 3, CV_8U));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return std::nullopt;
  }
  const auto largest = std::max_element(
    contours.begin(), contours.end(),
    [](const auto & left, const auto & right) {
      return cv::contourArea(left) < cv::contourArea(right);
    });
  const auto area = cv::contourArea(*largest);
  const auto moments = cv::moments(*largest);
  if (area < min_area_ || moments.m00 == 0.0) {
    return std::nullopt;
  }

  const cv::Point2d centre(moments.m10 / moments.m00, moments.m01 / moments.m00);
  const auto denominator = static_cast<double>(frame.rows) * frame.cols * 0.01;
  const auto confidence = static_cast<float>(std::min(1.0, area / denominator));
  return Pose{centre, std::nullopt, confidence};
}

cv::Mat validated_bgr_view(const sensor_msgs::msg::Image & message)
{
  if (message.width == 0U || message.height == 0U) {
    throw std::invalid_argument("image width and height must be positive");
  }
  if (message.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
    message.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
  {
    throw std::invalid_argument("image dimensions exceed OpenCV limits");
  }
  if (message.encoding != "bgr8" && message.encoding != "8UC3") {
    throw std::invalid_argument(
            "unsupported image encoding '" + message.encoding + "'; expected bgr8 or 8UC3");
  }

  const auto width = static_cast<std::size_t>(message.width);
  if (width > std::numeric_limits<std::size_t>::max() / kBgrChannels) {
    throw std::invalid_argument("image row size overflows size_t");
  }
  const auto packed_row_bytes = width * kBgrChannels;
  const auto row_bytes = message.step == 0U ? packed_row_bytes :
    static_cast<std::size_t>(message.step);
  if (row_bytes < packed_row_bytes) {
    throw std::invalid_argument("image row stride is shorter than width * 3");
  }
  const auto height = static_cast<std::size_t>(message.height);
  if (height > std::numeric_limits<std::size_t>::max() / row_bytes) {
    throw std::invalid_argument("image data size overflows size_t");
  }
  const auto expected_size = height * row_bytes;
  if (message.data.size() != expected_size) {
    throw std::invalid_argument(
            "image data length does not match height * row stride");
  }

  return cv::Mat(
    static_cast<int>(message.height), static_cast<int>(message.width), CV_8UC3,
    const_cast<unsigned char *>(message.data.data()), row_bytes);
}

}  // namespace rover_explorer_ros2::localization
