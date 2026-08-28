#include "rover_explorer_ros2/localization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

namespace rover_explorer_ros2::localization
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kBgrChannels = 3U;
}  // namespace

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
  const std::int64_t marker_id, const double heading_offset_radians)
: marker_id_(marker_id),
  heading_offset_radians_(heading_offset_radians),
  detector_(
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
    [] {
      cv::aruco::DetectorParameters parameters;
      parameters.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
      return parameters;
    }())
{
}

std::optional<Pose> ArucoLocalizer::locate(const cv::Mat & frame)
{
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  detector_.detectMarkers(frame, corners, ids);
  last_detected_ids_ = ids;

  const auto match = std::find(ids.begin(), ids.end(), marker_id_);
  if (match == ids.end()) {
    return std::nullopt;
  }
  const auto index = static_cast<std::size_t>(std::distance(ids.begin(), match));
  if (index >= corners.size() || corners[index].size() != 4U) {
    return std::nullopt;
  }

  const auto & points = corners[index];
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
  const auto perimeter = cv::arcLength(points, true);
  const auto denominator = std::max(
    32.0, 0.2 * static_cast<double>(std::min(frame.rows, frame.cols)));
  const auto confidence = static_cast<float>(std::min(1.0, perimeter / denominator));
  return Pose{centre, heading, confidence};
}

const std::vector<int> & ArucoLocalizer::last_detected_ids() const noexcept
{
  return last_detected_ids_;
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
