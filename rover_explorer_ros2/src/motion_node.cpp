#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/u_int32.hpp"

#include "rover_explorer_ros2/msg/legal_actions.hpp"
#include "rover_explorer_ros2/msg/policy_decision.hpp"
#include "rover_explorer_ros2/msg/rover_pose.hpp"

namespace rover_explorer_ros2
{

class MotionNode : public rclcpp::Node
{
public:
  MotionNode()
  : Node("motion_node")
  {
    declare_parameter("policy", "sweep");
    declare_parameter("decision_timeout_seconds", 1.0);
    declare_parameter("legal_actions_timeout_seconds", 0.25);
    declare_parameter("translation_ms", 250);
    declare_parameter("turn_ms", 180);
    declare_parameter("settle_ms", 500);
    declare_parameter("turn_scale", 0.55);
    declare_parameter("recovery_scan_timeout_seconds", 2.5);
    declare_parameter("recovery_cooldown_ms", 250);
    declare_parameter("max_consecutive_turn_pulses", 3);
    declare_parameter("turn_burst_recheck_ms", 150);
    declare_parameter("turn_rearm_min_progress_degrees", 8.0);
    declare_parameter("max_continuous_turn_degrees", 220.0);
    declare_parameter("turn_pose_timeout_seconds", 1.0);
    declare_parameter("radians_per_turn_pulse", -0.48);

    legal_.insert("stop");
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    publisher_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", qos);
    legal_subscription_ = create_subscription<msg::LegalActions>(
      "/rover/legal_actions", qos,
      [this](const msg::LegalActions::SharedPtr message) {on_legal(*message);});
    pose_subscription_ = create_subscription<msg::RoverPose>(
      "/rover/pose", qos,
      [this](const msg::RoverPose::SharedPtr message) {on_pose(*message);});
    left_range_subscription_ = create_subscription<sensor_msgs::msg::Range>(
      "/rover/sonar/left", qos,
      [this](const sensor_msgs::msg::Range::SharedPtr message) {on_side_range(*message);});
    right_range_subscription_ = create_subscription<sensor_msgs::msg::Range>(
      "/rover/sonar/right", qos,
      [this](const sensor_msgs::msg::Range::SharedPtr message) {on_side_range(*message);});
    scan_subscription_ = create_subscription<std_msgs::msg::UInt32>(
      "/rover/sonar/scan_sequence", qos,
      [this](const std_msgs::msg::UInt32::SharedPtr message) {scan_sequence_ = message->data;});
    decision_subscription_ = create_subscription<msg::PolicyDecision>(
      "/rover/policy/classic_decision", qos,
      [this](const msg::PolicyDecision::SharedPtr message) {on_decision(*message);});
    timer_ = create_wall_timer(std::chrono::milliseconds(50), [this]() {publish_command();});
  }

  ~MotionNode() override
  {
    if (publisher_) {
      try {
        publisher_->publish(geometry_msgs::msg::Twist{});
      } catch (const std::exception & error) {
        RCLCPP_ERROR(get_logger(), "Unable to publish shutdown STOP: %s", error.what());
      }
    }
  }

private:
  using SteadyClock = std::chrono::steady_clock;
  using TimePoint = SteadyClock::time_point;
  static constexpr double kPi = 3.14159265358979323846;

  static bool ends_with(const std::string & value, const std::string & suffix)
  {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
  }

  static bool is_turn(const std::string & action)
  {
    return action == "turn_left" || action == "turn_right";
  }

  static double seconds_since(TimePoint newer, TimePoint older)
  {
    return std::chrono::duration<double>(newer - older).count();
  }

  static double wrapped_angle(double value)
  {
    auto wrapped = std::fmod(value + kPi, 2.0 * kPi);
    if (wrapped < 0.0) {
      wrapped += 2.0 * kPi;
    }
    return wrapped - kPi;
  }

  void on_legal(const msg::LegalActions & message)
  {
    const bool was_blocked = sonar_blocked_;
    legal_.clear();
    if (message.emergency_stop) {
      legal_.insert("stop");
    } else {
      legal_.insert(message.actions.begin(), message.actions.end());
    }
    legal_received_ = SteadyClock::now();
    legal_seen_ = true;
    emergency_stop_ = message.emergency_stop;
    sonar_blocked_ = message.sonar_blocked;
    if (emergency_stop_) {
      recovery_stage_.reset();
    } else if (
      get_parameter("policy").as_string() != "vlm" && sonar_blocked_ &&
      !was_blocked && !recovery_stage_.has_value())
    {
      start_recovery(legal_received_);
    }
  }

  void on_side_range(const sensor_msgs::msg::Range & message)
  {
    if (ends_with(message.header.frame_id, "left")) {
      left_range_ = message.range;
    } else if (ends_with(message.header.frame_id, "right")) {
      right_range_ = message.range;
    }
  }

  void on_decision(const msg::PolicyDecision & message)
  {
    decision_action_ = message.action;
    decision_received_ = SteadyClock::now();
    decision_seen_ = true;
  }

  void on_pose(const msg::RoverPose & message)
  {
    if (!message.has_heading || !std::isfinite(message.heading)) {
      return;
    }
    pose_heading_ = message.heading;
    pose_received_ = SteadyClock::now();
    pose_seen_ = true;
    ++pose_sequence_;
  }

  void start_recovery(TimePoint now)
  {
    recovery_stage_ = "wait_scan";
    recovery_started_ = now;
    scan_baseline_ = scan_sequence_;
    RCLCPP_WARN(get_logger(), "Front obstacle: starting scan/reverse/turn recovery");
  }

  std::optional<std::string> recovery_request(TimePoint now)
  {
    if (!recovery_stage_.has_value() || emergency_stop_) {
      return std::nullopt;
    }
    if (*recovery_stage_ == "wait_scan") {
      const bool scan_ready = scan_sequence_ > scan_baseline_;
      const bool scan_timed_out = seconds_since(now, recovery_started_) >=
        get_parameter("recovery_scan_timeout_seconds").as_double();
      if (!scan_ready && !scan_timed_out) {
        return "stop";
      }
      recovery_turn_ = left_range_ >= right_range_ ? "turn_left" : "turn_right";
      recovery_stage_ = legal_.count("backward") > 0 ? "backward" : "turning";
    }
    if (*recovery_stage_ == "backward") {
      return legal_.count("backward") > 0 ? "backward" : "stop";
    }
    if (*recovery_stage_ == "turning") {
      return legal_.count(recovery_turn_) > 0 ? recovery_turn_ : "stop";
    }
    if (*recovery_stage_ == "cooldown") {
      if (now < recovery_cooldown_until_) {
        return "stop";
      }
      recovery_stage_.reset();
      if (sonar_blocked_) {
        start_recovery(now);
        return "stop";
      }
      return std::nullopt;
    }
    return "stop";
  }

  void recovery_pulse_started(const std::string & action)
  {
    if (recovery_stage_ == "backward" && action == "backward") {
      recovery_stage_ = "turning";
    } else if (recovery_stage_ == "turning" && action == recovery_turn_) {
      recovery_stage_ = "cooldown";
      recovery_cooldown_until_ = settle_until_ + std::chrono::milliseconds(
        std::max<std::int64_t>(0, get_parameter("recovery_cooldown_ms").as_int()));
    }
  }

  void reset_turn_guard()
  {
    consecutive_turn_pulses_ = 0;
    turn_burst_action_.reset();
    turn_burst_start_heading_.reset();
    turn_verification_pending_ = false;
    turn_verification_after_ = TimePoint{};
    turn_verification_pose_sequence_ = pose_sequence_;
    continuous_turn_radians_ = 0.0;
    turn_progress_blocked_ = false;
  }

  void begin_turn_burst(const std::string & action, TimePoint now)
  {
    turn_burst_action_ = action;
    const bool pose_fresh = pose_seen_ && seconds_since(now, pose_received_) <=
      get_parameter("turn_pose_timeout_seconds").as_double();
    turn_burst_start_heading_ = pose_fresh ? pose_heading_ : std::nullopt;
  }

  void arm_turn_verification(TimePoint now)
  {
    turn_verification_pending_ = true;
    turn_verification_after_ = now + std::chrono::milliseconds(
      std::max<std::int64_t>(0, get_parameter("turn_burst_recheck_ms").as_int()));
    turn_verification_pose_sequence_ = pose_sequence_;
    RCLCPP_INFO(
      get_logger(), "Turn burst complete; holding STOP while fresh pose verifies progress.");
  }

  bool turn_burst_can_rearm(TimePoint now)
  {
    if (turn_progress_blocked_) {
      return false;
    }
    if (!turn_verification_pending_) {
      arm_turn_verification(now);
      return false;
    }
    const bool pose_fresh = pose_seen_ && seconds_since(now, pose_received_) <=
      get_parameter("turn_pose_timeout_seconds").as_double();
    if (
      now < turn_verification_after_ || !pose_fresh ||
      pose_sequence_ <= turn_verification_pose_sequence_)
    {
      return false;
    }
    if (!turn_burst_start_heading_.has_value() || !turn_burst_action_.has_value()) {
      turn_progress_blocked_ = true;
      RCLCPP_ERROR(
        get_logger(),
        "Turn burst cannot be verified from a fresh starting pose; forcing STOP until policy changes.");
      return false;
    }

    const auto observed_delta = wrapped_angle(*pose_heading_ - *turn_burst_start_heading_);
    auto expected_sign = get_parameter("radians_per_turn_pulse").as_double() >= 0.0 ? 1.0 : -1.0;
    if (*turn_burst_action_ == "turn_right") {
      expected_sign *= -1.0;
    }
    const auto directed_progress = observed_delta * expected_sign;
    const auto minimum_progress = std::max(
      0.0, get_parameter("turn_rearm_min_progress_degrees").as_double()) * kPi / 180.0;
    const auto maximum_continuous = std::max(
      1.0, get_parameter("max_continuous_turn_degrees").as_double()) * kPi / 180.0;
    if (directed_progress < minimum_progress) {
      turn_progress_blocked_ = true;
      RCLCPP_ERROR(
        get_logger(),
        "Turn burst made insufficient or wrong-direction progress (%.1f deg); forcing STOP until policy changes.",
        directed_progress * 180.0 / kPi);
      return false;
    }
    if (continuous_turn_radians_ + directed_progress > maximum_continuous) {
      turn_progress_blocked_ = true;
      RCLCPP_ERROR(
        get_logger(), "Continuous verified turn limit reached; forcing STOP until policy changes.");
      return false;
    }

    continuous_turn_radians_ += directed_progress;
    consecutive_turn_pulses_ = 0;
    turn_burst_start_heading_.reset();
    turn_verification_pending_ = false;
    RCLCPP_INFO(
      get_logger(), "Turn burst rearmed after %.1f deg of verified progress.",
      directed_progress * 180.0 / kPi);
    return true;
  }

  geometry_msgs::msg::Twist twist(const std::string & action) const
  {
    geometry_msgs::msg::Twist output;
    const auto turn_scale = std::clamp(get_parameter("turn_scale").as_double(), 0.1, 1.0);
    if (action == "forward") {
      output.linear.x = 1.0;
    } else if (action == "backward") {
      output.linear.x = -1.0;
    } else if (action == "turn_left") {
      output.angular.z = turn_scale;
    } else if (action == "turn_right") {
      output.angular.z = -turn_scale;
    } else if (action == "arc_left") {
      output.linear.x = 0.65;
      output.angular.z = 0.35;
    } else if (action == "arc_right") {
      output.linear.x = 0.65;
      output.angular.z = -0.35;
    }
    return output;
  }

  void force_stop(TimePoint now)
  {
    active_action_ = "stop";
    pulse_until_ = now;
    settle_until_ = now;
    reset_turn_guard();
    publisher_->publish(twist("stop"));
  }

  void publish_command()
  {
    const auto now = SteadyClock::now();
    const bool legal_fresh = legal_seen_ && seconds_since(now, legal_received_) <=
      get_parameter("legal_actions_timeout_seconds").as_double();
    const bool decision_fresh = decision_seen_ && seconds_since(now, decision_received_) <=
      get_parameter("decision_timeout_seconds").as_double();
    const auto recovery = legal_fresh && get_parameter("policy").as_string() != "vlm" ?
      recovery_request(now) : std::nullopt;
    const auto requested = recovery.has_value() ? *recovery : decision_action_;

    if (
      !legal_fresh || (!recovery.has_value() && !decision_fresh) ||
      legal_.count(requested) == 0)
    {
      force_stop(now);
      return;
    }
    if (legal_.count(active_action_) == 0) {
      force_stop(now);
      return;
    }
    if (requested == "stop") {
      force_stop(now);
      return;
    }
    if (now < pulse_until_) {
      publisher_->publish(twist(active_action_));
      return;
    }
    if (now < settle_until_) {
      publisher_->publish(twist("stop"));
      return;
    }

    const bool requested_is_turn = is_turn(requested);
    if (
      requested_is_turn && turn_burst_action_.has_value() &&
      *turn_burst_action_ != requested)
    {
      reset_turn_guard();
    }
    const auto maximum_turns = std::max<std::int64_t>(
      1, get_parameter("max_consecutive_turn_pulses").as_int());
    if (requested_is_turn && consecutive_turn_pulses_ >= maximum_turns) {
      active_action_ = "stop";
      publisher_->publish(twist("stop"));
      turn_burst_can_rearm(now);
      return;
    }
    if (!requested_is_turn) {
      reset_turn_guard();
    }
    if (requested_is_turn && consecutive_turn_pulses_ == 0) {
      begin_turn_burst(requested, now);
    }

    active_action_ = requested;
    const auto duration_ms = is_turn(requested) ?
      get_parameter("turn_ms").as_int() : get_parameter("translation_ms").as_int();
    pulse_until_ = now + std::chrono::milliseconds(std::max<std::int64_t>(0, duration_ms));
    settle_until_ = pulse_until_ + std::chrono::milliseconds(
      std::max<std::int64_t>(0, get_parameter("settle_ms").as_int()));
    recovery_pulse_started(active_action_);
    if (requested_is_turn) {
      ++consecutive_turn_pulses_;
    }
    publisher_->publish(twist(active_action_));
  }

  std::unordered_set<std::string> legal_;
  TimePoint legal_received_{};
  TimePoint decision_received_{};
  TimePoint pose_received_{};
  TimePoint recovery_started_{};
  TimePoint recovery_cooldown_until_{};
  TimePoint pulse_until_{};
  TimePoint settle_until_{};
  TimePoint turn_verification_after_{};
  bool legal_seen_{false};
  bool decision_seen_{false};
  bool pose_seen_{false};
  bool emergency_stop_{false};
  bool sonar_blocked_{false};
  bool turn_verification_pending_{false};
  bool turn_progress_blocked_{false};
  std::uint32_t scan_sequence_{0};
  std::uint32_t scan_baseline_{0};
  std::uint64_t pose_sequence_{0};
  std::uint64_t turn_verification_pose_sequence_{0};
  std::int64_t consecutive_turn_pulses_{0};
  double left_range_{INFINITY};
  double right_range_{INFINITY};
  double continuous_turn_radians_{0.0};
  std::optional<double> pose_heading_;
  std::optional<double> turn_burst_start_heading_;
  std::optional<std::string> recovery_stage_;
  std::optional<std::string> turn_burst_action_;
  std::string recovery_turn_{"turn_left"};
  std::string decision_action_{"stop"};
  std::string active_action_{"stop"};
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
  rclcpp::Subscription<msg::LegalActions>::SharedPtr legal_subscription_;
  rclcpp::Subscription<msg::RoverPose>::SharedPtr pose_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr left_range_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr right_range_subscription_;
  rclcpp::Subscription<std_msgs::msg::UInt32>::SharedPtr scan_subscription_;
  rclcpp::Subscription<msg::PolicyDecision>::SharedPtr decision_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace rover_explorer_ros2

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<rover_explorer_ros2::MotionNode>());
  rclcpp::shutdown();
  return 0;
}
