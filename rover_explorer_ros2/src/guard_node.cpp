#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/bool.hpp"

#include "rover_explorer_ros2/deterministic_logic.hpp"
#include "rover_explorer_ros2/msg/legal_actions.hpp"
#include "rover_explorer_ros2/msg/rover_pose.hpp"

namespace rover_explorer_ros2
{

class GuardNode : public rclcpp::Node
{
public:
  GuardNode()
  : Node("guard_node")
  {
    declare_parameter("px_per_forward_pulse", 35.0);
    declare_parameter("radians_per_turn_pulse", 0.35);
    declare_parameter("forward_axis_x", 1.0);
    declare_parameter("forward_axis_y", 0.0);
    declare_parameter("camera_width", 640);
    declare_parameter("camera_height", 480);
    declare_parameter("margin_frac", 0.12);
    declare_parameter("pose_timeout_seconds", 1.0);
    declare_parameter("sonar_timeout_seconds", 1.0);
    declare_parameter("sonar_stop_distance_m", 0.25);

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    publisher_ = create_publisher<msg::LegalActions>("/rover/legal_actions", qos);
    pose_subscription_ = create_subscription<msg::RoverPose>(
      "/rover/pose", qos,
      [this](const msg::RoverPose::SharedPtr message) {on_pose(*message);});
    sonar_subscription_ = create_subscription<sensor_msgs::msg::Range>(
      "/rover/sonar", qos,
      [this](const sensor_msgs::msg::Range::SharedPtr message) {on_sonar(*message);});
    emergency_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/rover/emergency_stop", qos,
      [this](const std_msgs::msg::Bool::SharedPtr message) {
        emergency_ = message->data;
        recalculate();
      });
    timer_ = create_wall_timer(std::chrono::milliseconds(50), [this]() {recalculate();});
  }

private:
  using SteadyClock = std::chrono::steady_clock;

  static bool ends_with(const std::string & value, const std::string & suffix)
  {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
  }

  void on_pose(const msg::RoverPose & message)
  {
    pose_ = Pose2D{
      message.centre.x,
      message.centre.y,
      message.has_heading ? std::optional<double>(message.heading) : std::nullopt,
      message.confidence};
    pose_received_ = SteadyClock::now();
    pose_seen_ = true;
  }

  void on_sonar(const sensor_msgs::msg::Range & message)
  {
    if (!message.header.frame_id.empty() && !ends_with(message.header.frame_id, "front")) {
      return;
    }
    front_range_ = message.range;
    sonar_received_ = SteadyClock::now();
    sonar_seen_ = true;
  }

  BodyToImage transform() const
  {
    return BodyToImage{
      get_parameter("px_per_forward_pulse").as_double(),
      get_parameter("radians_per_turn_pulse").as_double(),
      get_parameter("forward_axis_x").as_double(),
      get_parameter("forward_axis_y").as_double()};
  }

  void recalculate()
  {
    const auto now = SteadyClock::now();
    const auto pose_timeout = std::chrono::duration<double>(
      get_parameter("pose_timeout_seconds").as_double());
    const auto sonar_timeout = std::chrono::duration<double>(
      get_parameter("sonar_timeout_seconds").as_double());
    const bool pose_fresh = pose_seen_ && now - pose_received_ <= pose_timeout;
    const bool sonar_fresh = sonar_seen_ && now - sonar_received_ <= sonar_timeout;

    msg::LegalActions output;
    output.header.stamp = get_clock()->now();
    output.emergency_stop = emergency_;
    output.sonar_blocked = !sonar_fresh ||
      front_range_ <= get_parameter("sonar_stop_distance_m").as_double();
    if (emergency_) {
      output.actions = {"stop"};
      output.reason = "Emergency stop is active.";
    } else {
      const auto actions = apply_ultrasonic_guard(
        allowed_actions(
          pose_fresh ? pose_ : std::nullopt,
          transform(),
          static_cast<int>(get_parameter("camera_width").as_int()),
          static_cast<int>(get_parameter("camera_height").as_int()),
          get_parameter("margin_frac").as_double()),
        output.sonar_blocked);
      output.actions.reserve(actions.size());
      for (const auto action : actions) {
        output.actions.push_back(action_name(action));
      }
      output.reason = pose_fresh ?
        "fresh guard calculation" : "pose stale/lost; conservative recovery";
    }
    publisher_->publish(output);
  }

  std::optional<Pose2D> pose_;
  SteadyClock::time_point pose_received_{};
  SteadyClock::time_point sonar_received_{};
  bool pose_seen_{false};
  bool sonar_seen_{false};
  bool emergency_{false};
  double front_range_{INFINITY};
  rclcpp::Publisher<msg::LegalActions>::SharedPtr publisher_;
  rclcpp::Subscription<msg::RoverPose>::SharedPtr pose_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr sonar_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace rover_explorer_ros2

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<rover_explorer_ros2::GuardNode>());
  rclcpp::shutdown();
  return 0;
}
