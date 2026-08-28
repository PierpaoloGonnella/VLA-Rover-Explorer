#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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

std::string number(const double value)
{
  std::ostringstream output;
  output << std::fixed << std::setprecision(6) << value;
  return output.str();
}

diagnostic_msgs::msg::KeyValue key(const std::string & name, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue result;
  result.key = name;
  result.value = value;
  return result;
}

std::string join_ids(const std::vector<int> & ids)
{
  std::ostringstream output;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    output << (index == 0U ? "" : ",") << ids[index];
  }
  return output.str();
}
}  // namespace

class LocalizerNode final : public rclcpp::Node
{
public:
  LocalizerNode()
  : Node("localizer_node")
  {
    const auto marker_id = declare_parameter<int>("aruco_marker_id", 0);
    const auto heading_offset = declare_parameter<double>("aruco_heading_offset_degrees", 0.0);
    const auto backend = declare_parameter<std::string>("localization_backend", "aruco_custom");
    const auto minimum_confidence = declare_parameter<double>("min_confidence", 0.25);
    if (minimum_confidence < 0.0 || minimum_confidence > 1.0) {
      throw std::invalid_argument("min_confidence must be in [0, 1]");
    }
    declare_parameter<int>("localization_settle_ms", 400);
    declare_parameter<bool>("capture_failure_images", false);
    declare_parameter<double>("failure_image_min_interval_seconds", 2.0);
    declare_parameter<int>("failure_image_max_per_session", 50);
    declare_parameter<bool>("failure_image_annotated", true);
    const auto compare_topic = declare_parameter<std::string>("aruco_compare_topic", "/aruco/markers");
    declare_parameter<double>("camera_fx", 0.0);
    declare_parameter<double>("camera_fy", 0.0);
    declare_parameter<double>("camera_cx", 0.0);
    declare_parameter<double>("camera_cy", 0.0);
    declare_detector_parameters();

    if (backend == "color") {
      localizer_ = std::make_unique<localization::ColorBlobLocalizer>();
    } else if (backend == "aruco_custom") {
      auto aruco = std::make_unique<localization::ArucoLocalizer>(
        marker_id, heading_offset * kDegreesToRadians, detector_config());
      aruco_localizer_ = aruco.get();
      RCLCPP_INFO(get_logger(), "Effective ArUco detector: %s", aruco->config().describe().c_str());
      localizer_ = std::move(aruco);
    } else {
      throw std::invalid_argument("localization_backend must be 'aruco_custom' or 'color'");
    }

    pose_publisher_ = create_publisher<msg::RoverPose>("/rover/pose", 10);
    comparison_publisher_ = create_publisher<std_msgs::msg::Float32>(
      "/rover/localization/aruco_error_px", 10);
    diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/rover/localization/diagnostics", 10);
    failure_image_publisher_ = create_publisher<sensor_msgs::msg::Image>(
      "/rover/localization/failure_image", rclcpp::QoS(rclcpp::KeepLast(2)));
    image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      "/rover/image_raw", rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
      [this](const sensor_msgs::msg::Image::ConstSharedPtr message) {on_image(message);});
    command_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::QoS(rclcpp::KeepLast(10)),
      [this](const geometry_msgs::msg::Twist::ConstSharedPtr message) {on_command(*message);});

#ifdef ROVER_HAS_ROS2_ARUCO
    comparison_subscription_ = create_subscription<ros2_aruco_interfaces::msg::ArucoMarkers>(
      compare_topic, 10,
      [this](const ros2_aruco_interfaces::msg::ArucoMarkers::ConstSharedPtr message) {
        on_ros2_aruco(*message);
      });
#else
    (void)compare_topic;
    RCLCPP_INFO(get_logger(), "ros2_aruco interfaces unavailable; custom ArUco remains active");
#endif
  }

private:
  void declare_detector_parameters()
  {
    declare_parameter<int>("aruco_adaptive_threshold_window_min", 3);
    declare_parameter<int>("aruco_adaptive_threshold_window_max", 23);
    declare_parameter<int>("aruco_adaptive_threshold_window_step", 10);
    declare_parameter<double>("aruco_adaptive_threshold_constant", 7.0);
    declare_parameter<double>("aruco_minimum_marker_perimeter_rate", 0.03);
    declare_parameter<double>("aruco_maximum_marker_perimeter_rate", 4.0);
    declare_parameter<double>("aruco_polygonal_approximation_accuracy_rate", 0.03);
    declare_parameter<double>("aruco_minimum_corner_distance_rate", 0.05);
    declare_parameter<int>("aruco_minimum_distance_to_border", 3);
    declare_parameter<int>("aruco_marker_border_bits", 1);
    declare_parameter<int>("aruco_perspective_remove_pixels_per_cell", 4);
    declare_parameter<double>("aruco_error_correction_rate", 0.6);
    declare_parameter<std::string>("aruco_corner_refinement", "subpix");
    declare_parameter<int>("aruco_corner_refinement_window", 5);
    declare_parameter<int>("aruco_corner_refinement_iterations", 30);
    declare_parameter<double>("aruco_corner_refinement_accuracy", 0.1);
  }

  localization::ArucoDetectorConfig detector_config() const
  {
    localization::ArucoDetectorConfig config;
    config.adaptive_threshold_window_min = static_cast<int>(
      get_parameter("aruco_adaptive_threshold_window_min").as_int());
    config.adaptive_threshold_window_max = static_cast<int>(
      get_parameter("aruco_adaptive_threshold_window_max").as_int());
    config.adaptive_threshold_window_step = static_cast<int>(
      get_parameter("aruco_adaptive_threshold_window_step").as_int());
    config.adaptive_threshold_constant = get_parameter("aruco_adaptive_threshold_constant").as_double();
    config.minimum_marker_perimeter_rate = get_parameter("aruco_minimum_marker_perimeter_rate").as_double();
    config.maximum_marker_perimeter_rate = get_parameter("aruco_maximum_marker_perimeter_rate").as_double();
    config.polygonal_approximation_accuracy_rate =
      get_parameter("aruco_polygonal_approximation_accuracy_rate").as_double();
    config.minimum_corner_distance_rate = get_parameter("aruco_minimum_corner_distance_rate").as_double();
    config.minimum_distance_to_border = static_cast<int>(
      get_parameter("aruco_minimum_distance_to_border").as_int());
    config.marker_border_bits = static_cast<int>(get_parameter("aruco_marker_border_bits").as_int());
    config.perspective_remove_pixels_per_cell = static_cast<int>(
      get_parameter("aruco_perspective_remove_pixels_per_cell").as_int());
    config.error_correction_rate = get_parameter("aruco_error_correction_rate").as_double();
    const auto refinement = get_parameter("aruco_corner_refinement").as_string();
    if (refinement == "none") {
      config.corner_refinement_method = cv::aruco::CORNER_REFINE_NONE;
    } else if (refinement == "subpix") {
      config.corner_refinement_method = cv::aruco::CORNER_REFINE_SUBPIX;
    } else if (refinement == "contour") {
      config.corner_refinement_method = cv::aruco::CORNER_REFINE_CONTOUR;
    } else if (refinement == "apriltag") {
      config.corner_refinement_method = cv::aruco::CORNER_REFINE_APRILTAG;
    } else {
      throw std::invalid_argument("aruco_corner_refinement must be none, subpix, contour, or apriltag");
    }
    config.corner_refinement_window = static_cast<int>(
      get_parameter("aruco_corner_refinement_window").as_int());
    config.corner_refinement_iterations = static_cast<int>(
      get_parameter("aruco_corner_refinement_iterations").as_int());
    config.corner_refinement_accuracy = get_parameter("aruco_corner_refinement_accuracy").as_double();
    config.validate();
    return config;
  }

  bool warning_due(std::chrono::steady_clock::time_point & last_warning)
  {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(warning_mutex_);
    if (last_warning.time_since_epoch().count() && now - last_warning < kWarningPeriod) {
      return false;
    }
    last_warning = now;
    return true;
  }

  void on_command(const geometry_msgs::msg::Twist & message)
  {
    const bool now_moving = std::abs(message.linear.x) > 1e-6 || std::abs(message.angular.z) > 1e-6;
    std::lock_guard<std::mutex> lock(motion_mutex_);
    if (moving_ && !now_moving) {
      last_stop_time_ = get_clock()->now();
    }
    moving_ = now_moving;
    linear_velocity_ = message.linear.x;
    angular_velocity_ = message.angular.z;
  }

  void publish_diagnostics(
    const sensor_msgs::msg::Image & message, const localization::FrameDiagnostics & frame,
    const std::string & outcome, const std::string & detector_outcome, const bool moving,
    const double linear, const double angular, const double since_stop,
    const double frame_age_ms, const double latency_ms)
  {
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = get_clock()->now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "rover/localization/frame";
    status.hardware_id = message.header.frame_id;
    status.level = outcome == "valid" || outcome == "moving_frame" || outcome == "pre_settle_frame" ?
      diagnostic_msgs::msg::DiagnosticStatus::OK : diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = outcome;
    status.values = {
      key("outcome", outcome), key("detector_outcome", detector_outcome),
      key("frame_sequence", std::to_string(frame_sequence_)),
      key("image_stamp", number(rclcpp::Time(message.header.stamp).seconds())),
      key("encoding", message.encoding), key("width", std::to_string(message.width)),
      key("height", std::to_string(message.height)), key("stride", std::to_string(message.step)),
      key("frame_age_ms", number(frame_age_ms)), key("processing_latency_ms", number(latency_ms)),
      key("moving", moving ? "true" : "false"), key("linear_x", number(linear)),
      key("angular_z", number(angular)), key("seconds_since_stop", number(since_stop)),
      key("candidate_count", std::to_string(frame.candidate_count)),
      key("rejected_candidate_count", std::to_string(frame.rejected_candidate_count)),
      key("detected_ids", join_ids(frame.detected_ids)),
      key("target_present", frame.target_corners.empty() ? "false" : "true"),
      key("marker_mean_side_px", number(frame.marker_mean_side_px)),
      key("marker_min_side_px", number(frame.marker_min_side_px)),
      key("marker_max_side_px", number(frame.marker_max_side_px)),
      key("marker_perimeter_px", number(frame.marker_perimeter_px)),
      key("marker_area_ratio", number(frame.marker_area_ratio)),
      key("marker_boundary_distance_px", number(frame.marker_boundary_distance_px)),
      key("grayscale_mean", number(frame.grayscale_mean)),
      key("grayscale_stddev", number(frame.grayscale_stddev)),
      key("dark_fraction", number(frame.dark_fraction)),
      key("saturated_fraction", number(frame.saturated_fraction)),
      key("sharpness_laplacian_variance", number(frame.sharpness)),
      key("confidence", number(frame.confidence)),
      key("detector_configuration", aruco_localizer_ ? aruco_localizer_->config().describe() : "color")};
    if (!frame.target_corners.empty()) {
      std::ostringstream corners;
      for (const auto & point : frame.target_corners) {
        corners << point.x << ',' << point.y << ';';
      }
      status.values.push_back(key("target_corners", corners.str()));
    }
    array.status.push_back(std::move(status));
    diagnostics_publisher_->publish(array);
  }

  void maybe_publish_failure_image(
    const sensor_msgs::msg::Image & message, const cv::Mat & frame,
    const localization::FrameDiagnostics & diagnostic, const std::string & outcome,
    const bool moving, const double frame_age_ms, const double latency_ms)
  {
    if (!get_parameter("capture_failure_images").as_bool() || outcome == "valid" ||
      outcome == "moving_frame" || outcome == "pre_settle_frame")
    {
      return;
    }
    const auto maximum = std::max<std::int64_t>(0, get_parameter("failure_image_max_per_session").as_int());
    const auto now = std::chrono::steady_clock::now();
    const auto interval = std::chrono::duration<double>(
      std::max(0.0, get_parameter("failure_image_min_interval_seconds").as_double()));
    if (failure_image_count_ >= static_cast<std::size_t>(maximum) ||
      (last_failure_image_.time_since_epoch().count() && now - last_failure_image_ < interval))
    {
      return;
    }
    cv::Mat output = frame.clone();
    if (get_parameter("failure_image_annotated").as_bool()) {
      const auto draw_contours = [&output](
        const std::vector<std::vector<cv::Point2f>> & contours,
        const cv::Scalar & color, const int thickness)
        {
          for (const auto & contour : contours) {
            std::vector<cv::Point> integer_contour;
            integer_contour.reserve(contour.size());
            for (const auto & point : contour) {
              integer_contour.emplace_back(cvRound(point.x), cvRound(point.y));
            }
            if (integer_contour.size() >= 2U) {
              cv::polylines(output, integer_contour, true, color, thickness);
            }
          }
        };
      draw_contours(diagnostic.detected_corners, cv::Scalar(0, 255, 0), 2);
      draw_contours(diagnostic.rejected_corners, cv::Scalar(0, 165, 255), 1);
      const std::vector<std::string> lines{
        outcome + " ids=" + join_ids(diagnostic.detected_ids),
        "moving=" + std::string(moving ? "true" : "false") + " age_ms=" + number(frame_age_ms),
        "latency_ms=" + number(latency_ms) + " side_px=" + number(diagnostic.marker_mean_side_px),
        "sharp=" + number(diagnostic.sharpness) + " mean/std=" +
          number(diagnostic.grayscale_mean) + "/" + number(diagnostic.grayscale_stddev),
        "confidence=" + number(diagnostic.confidence)};
      for (std::size_t index = 0; index < lines.size(); ++index) {
        const cv::Point origin(10, 24 + static_cast<int>(index) * 24);
        cv::putText(output, lines[index], origin, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3);
        cv::putText(output, lines[index], origin, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
      }
    }
    sensor_msgs::msg::Image result;
    result.header = message.header;
    result.header.frame_id = outcome;
    result.height = static_cast<std::uint32_t>(output.rows);
    result.width = static_cast<std::uint32_t>(output.cols);
    result.encoding = "bgr8";
    result.step = static_cast<std::uint32_t>(output.cols * output.elemSize());
    result.data.assign(output.data, output.data + output.total() * output.elemSize());
    failure_image_publisher_->publish(result);
    last_failure_image_ = now;
    ++failure_image_count_;
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr & message)
  {
    ++frame_sequence_;
    const auto callback_started = std::chrono::steady_clock::now();
    const auto callback_ros = get_clock()->now();
    const rclcpp::Time capture_time(message->header.stamp, get_clock()->get_clock_type());
    const auto frame_age_ms = capture_time.nanoseconds() > 0 ?
      std::max(0.0, (callback_ros - capture_time).seconds() * 1000.0) :
      std::numeric_limits<double>::infinity();
    cv::Mat frame;
    try {
      frame = localization::validated_bgr_view(*message);
    } catch (const std::exception & error) {
      localization::FrameDiagnostics invalid;
      invalid.outcome = "invalid_image";
      const auto latency = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - callback_started).count();
      publish_diagnostics(*message, invalid, "invalid_image", error.what(), false, 0.0, 0.0,
        std::numeric_limits<double>::infinity(), frame_age_ms, latency);
      if (warning_due(last_invalid_frame_warning_)) {
        RCLCPP_WARN(get_logger(), "Rejecting invalid image: %s", error.what());
      }
      return;
    }

    localization::FrameDiagnostics diagnostic;
    try {
      std::lock_guard<std::mutex> lock(localizer_mutex_);
      if (aruco_localizer_) {
        diagnostic = aruco_localizer_->analyze(frame, get_parameter("min_confidence").as_double());
      } else {
        diagnostic.pose = localizer_->locate(frame);
        diagnostic.outcome = diagnostic.pose ? "valid" : "no_candidates";
        diagnostic.confidence = diagnostic.pose ? diagnostic.pose->confidence : 0.0F;
        if (diagnostic.pose && diagnostic.pose->confidence < get_parameter("min_confidence").as_double()) {
          diagnostic.pose.reset();
          diagnostic.outcome = "below_confidence";
        }
      }
    } catch (const cv::Exception & error) {
      diagnostic.outcome = "opencv_error";
      if (warning_due(last_invalid_frame_warning_)) {
        RCLCPP_WARN(get_logger(), "OpenCV localization failed: %s", error.what());
      }
    }

    bool moving;
    double linear;
    double angular;
    double since_stop = std::numeric_limits<double>::infinity();
    bool post_settle = true;
    {
      std::lock_guard<std::mutex> lock(motion_mutex_);
      moving = moving_;
      linear = linear_velocity_;
      angular = angular_velocity_;
      if (last_stop_time_) {
        since_stop = (callback_ros - *last_stop_time_).seconds();
        const auto settle = rclcpp::Duration::from_seconds(
          std::max<std::int64_t>(0, get_parameter("localization_settle_ms").as_int()) / 1000.0);
        post_settle = capture_time.nanoseconds() > 0 && capture_time >= *last_stop_time_ + settle;
      }
    }
    const auto detector_outcome = diagnostic.outcome;
    auto outcome = detector_outcome;
    if (moving) {
      outcome = "moving_frame";
      diagnostic.pose.reset();
    } else if (!post_settle) {
      outcome = "pre_settle_frame";
      diagnostic.pose.reset();
    }

    const auto latency_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - callback_started).count();
    publish_diagnostics(*message, diagnostic, outcome, detector_outcome, moving, linear, angular,
      since_stop, frame_age_ms, latency_ms);
    maybe_publish_failure_image(*message, frame, diagnostic, outcome, moving, frame_age_ms, latency_ms);

    if (!diagnostic.pose) {
      if (outcome != "moving_frame" && outcome != "pre_settle_frame" &&
        warning_due(last_lost_marker_warning_))
      {
        RCLCPP_WARN(get_logger(), "Localization rejected: %s", outcome.c_str());
      }
      return;
    }
    msg::RoverPose result;
    result.header = message->header;
    result.centre.x = diagnostic.pose->centre.x;
    result.centre.y = diagnostic.pose->centre.y;
    result.centre.z = 0.0;
    result.has_heading = diagnostic.pose->heading.has_value();
    result.heading = diagnostic.pose->heading.value_or(0.0);
    result.confidence = diagnostic.pose->confidence;
    {
      std::lock_guard<std::mutex> lock(last_pose_mutex_);
      last_custom_pose_ = diagnostic.pose;
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
    const auto index = static_cast<std::size_t>(std::distance(message.marker_ids.begin(), match));
    if (index >= message.poses.size()) {
      return;
    }
    const auto & external = message.poses[index].position;
    const auto error = localization::aruco_projection_error(
      custom->centre, cv::Point3d(external.x, external.y, external.z),
      get_parameter("camera_fx").as_double(), get_parameter("camera_fy").as_double(),
      get_parameter("camera_cx").as_double(), get_parameter("camera_cy").as_double());
    if (error) {
      std_msgs::msg::Float32 output;
      output.data = *error;
      comparison_publisher_->publish(output);
    }
  }
#endif

  std::unique_ptr<localization::Localizer> localizer_;
  localization::ArucoLocalizer * aruco_localizer_{nullptr};
  std::mutex localizer_mutex_;
  std::mutex last_pose_mutex_;
  std::mutex warning_mutex_;
  std::mutex motion_mutex_;
  std::optional<localization::Pose> last_custom_pose_;
  std::optional<rclcpp::Time> last_stop_time_;
  bool moving_{false};
  double linear_velocity_{0.0};
  double angular_velocity_{0.0};
  std::size_t failure_image_count_{0U};
  std::uint64_t frame_sequence_{0U};
  std::chrono::steady_clock::time_point last_failure_image_{};
  std::chrono::steady_clock::time_point last_invalid_frame_warning_{};
  std::chrono::steady_clock::time_point last_lost_marker_warning_{};
  rclcpp::Publisher<msg::RoverPose>::SharedPtr pose_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr comparison_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr failure_image_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_subscription_;
#ifdef ROVER_HAS_ROS2_ARUCO
  rclcpp::Subscription<ros2_aruco_interfaces::msg::ArucoMarkers>::SharedPtr comparison_subscription_;
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
