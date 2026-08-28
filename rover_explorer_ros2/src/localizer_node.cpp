#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>

#include "rover_explorer_ros2/localization.hpp"
#include "rover_explorer_ros2/msg/rover_pose.hpp"

#ifdef ROVER_HAS_ROS2_ARUCO
#include <ros2_aruco_interfaces/msg/aruco_markers.hpp>
#endif

namespace rover_explorer_ros2
{
namespace
{
constexpr double kDegreesToRadians = 0.01745329251994329577;
constexpr auto kWarningPeriod = std::chrono::seconds(2);
}  // namespace

class LocalizerNode final : public rclcpp::Node
{
public:
  LocalizerNode()
  : Node("localizer_node")
  {
    const auto marker_id = declare_parameter<int>("aruco_marker_id", 0);
    const auto heading_offset_degrees =
      declare_parameter<double>("aruco_heading_offset_degrees", 0.0);
    const auto backend = declare_parameter<std::string>("localization_backend", "aruco_custom");
    declare_parameter<double>("min_confidence", 0.25);
    const auto compare_topic =
      declare_parameter<std::string>("aruco_compare_topic", "/aruco/markers");
    declare_parameter<double>("camera_fx", 0.0);
    declare_parameter<double>("camera_fy", 0.0);
    declare_parameter<double>("camera_cx", 0.0);
    declare_parameter<double>("camera_cy", 0.0);

    if (backend == "color") {
      localizer_ = std::make_unique<localization::ColorBlobLocalizer>();
    } else {
      localizer_ = std::make_unique<localization::ArucoLocalizer>(
        marker_id, heading_offset_degrees * kDegreesToRadians);
    }

    pose_publisher_ = create_publisher<msg::RoverPose>("/rover/pose", 10);
    comparison_publisher_ =
      create_publisher<std_msgs::msg::Float32>("/rover/localization/aruco_error_px", 10);
    image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      "/rover/image_raw", 10,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr message) {on_image(message);});

#ifdef ROVER_HAS_ROS2_ARUCO
    comparison_subscription_ =
      create_subscription<ros2_aruco_interfaces::msg::ArucoMarkers>(
      compare_topic, 10,
      [this](const ros2_aruco_interfaces::msg::ArucoMarkers::ConstSharedPtr message) {
        on_ros2_aruco(*message);
      });
#else
    (void)compare_topic;
    RCLCPP_INFO(
      get_logger(), "ros2_aruco interfaces unavailable; custom ArUco remains active");
#endif
  }

private:
  bool warning_due(std::chrono::steady_clock::time_point & last_warning)
  {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(warning_mutex_);
    if (last_warning.time_since_epoch().count() != 0 && now - last_warning < kWarningPeriod) {
      return false;
    }
    last_warning = now;
    return true;
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr & message)
  {
    cv::Mat frame;
    try {
      frame = localization::validated_bgr_view(*message);
    } catch (const std::exception & error) {
      if (warning_due(last_invalid_frame_warning_)) {
        RCLCPP_WARN(get_logger(), "Rejecting invalid image: %s", error.what());
      }
      return;
    }

    std::optional<localization::Pose> pose;
    try {
      std::lock_guard<std::mutex> lock(localizer_mutex_);
      pose = localizer_->locate(frame);
    } catch (const cv::Exception & error) {
      if (warning_due(last_invalid_frame_warning_)) {
        RCLCPP_WARN(get_logger(), "OpenCV localization failed: %s", error.what());
      }
      return;
    }

    const auto minimum_confidence = get_parameter("min_confidence").as_double();
    if (!pose || pose->confidence < minimum_confidence) {
      if (warning_due(last_lost_marker_warning_)) {
        RCLCPP_WARN(get_logger(), "No valid localization in the latest image");
      }
      return;
    }

    msg::RoverPose result;
    result.header = message->header;
    result.centre.x = pose->centre.x;
    result.centre.y = pose->centre.y;
    result.centre.z = 0.0;
    result.has_heading = pose->heading.has_value();
    result.heading = pose->heading.value_or(0.0);
    result.confidence = pose->confidence;
    {
      std::lock_guard<std::mutex> lock(last_pose_mutex_);
      last_custom_pose_ = pose;
    }
    pose_publisher_->publish(result);
  }

#ifdef ROVER_HAS_ROS2_ARUCO
  void on_ros2_aruco(const ros2_aruco_interfaces::msg::ArucoMarkers & message)
  {
    std::optional<localization::Pose> custom;
    {
      std::lock_guard<std::mutex> lock(last_pose_mutex_);
      custom = last_custom_pose_;
    }
    if (!custom || message.poses.empty()) {
      return;
    }

    const auto marker_id = get_parameter("aruco_marker_id").as_int();
    const auto match = std::find(message.marker_ids.begin(), message.marker_ids.end(), marker_id);
    if (match == message.marker_ids.end()) {
      return;
    }
    const auto index = static_cast<std::size_t>(
      std::distance(message.marker_ids.begin(), match));
    if (index >= message.poses.size()) {
      return;
    }

    const auto & external = message.poses[index].position;
    const auto error = localization::aruco_projection_error(
      custom->centre, cv::Point3d(external.x, external.y, external.z),
      get_parameter("camera_fx").as_double(),
      get_parameter("camera_fy").as_double(),
      get_parameter("camera_cx").as_double(),
      get_parameter("camera_cy").as_double());
    if (!error) {
      return;
    }
    std_msgs::msg::Float32 output;
    output.data = *error;
    comparison_publisher_->publish(output);
  }
#endif

  std::unique_ptr<localization::Localizer> localizer_;
  std::mutex localizer_mutex_;
  std::mutex last_pose_mutex_;
  std::mutex warning_mutex_;
  std::optional<localization::Pose> last_custom_pose_;
  std::chrono::steady_clock::time_point last_invalid_frame_warning_{};
  std::chrono::steady_clock::time_point last_lost_marker_warning_{};
  rclcpp::Publisher<msg::RoverPose>::SharedPtr pose_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr comparison_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
#ifdef ROVER_HAS_ROS2_ARUCO
  rclcpp::Subscription<ros2_aruco_interfaces::msg::ArucoMarkers>::SharedPtr
    comparison_subscription_;
#endif
};

}  // namespace rover_explorer_ros2

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<rover_explorer_ros2::LocalizerNode>());
  rclcpp::shutdown();
  return 0;
}
