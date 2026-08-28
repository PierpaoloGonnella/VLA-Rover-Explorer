#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/range.hpp"

#include "rover_explorer_ros2/deterministic_logic.hpp"
#include "rover_explorer_ros2/msg/rover_pose.hpp"

namespace rover_explorer_ros2
{

class ObstacleGridNode : public rclcpp::Node
{
public:
  ObstacleGridNode()
  : Node("obstacle_grid_node")
  {
    declare_parameter("px_per_forward_pulse", 35.0);
    declare_parameter("radians_per_turn_pulse", 0.35);
    declare_parameter("forward_axis_x", 1.0);
    declare_parameter("forward_axis_y", 0.0);
    declare_parameter("camera_width", 640);
    declare_parameter("camera_height", 480);
    declare_parameter("map_cols", 12);
    declare_parameter("map_rows", 8);
    declare_parameter("obstacle_ttl_cycles", 40);
    declare_parameter("cm_per_translation_pulse", 10.0);
    declare_parameter("maximum_mapping_distance_cm", 150);
    declare_parameter("map_frame_id", "map");
    grid_ = std::make_unique<ObstacleGrid>(
      static_cast<int>(get_parameter("camera_width").as_int()),
      static_cast<int>(get_parameter("camera_height").as_int()),
      static_cast<int>(get_parameter("map_cols").as_int()),
      static_cast<int>(get_parameter("map_rows").as_int()),
      static_cast<int>(get_parameter("obstacle_ttl_cycles").as_int()));

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/rover/occupancy_grid", qos);
    pose_subscription_ = create_subscription<msg::RoverPose>(
      "/rover/pose", qos,
      [this](const msg::RoverPose::SharedPtr message) {on_pose(*message);});
    sonar_subscription_ = create_subscription<sensor_msgs::msg::Range>(
      "/rover/sonar", qos,
      [this](const sensor_msgs::msg::Range::SharedPtr message) {on_sonar(*message);});
  }

private:
  static constexpr double kPi = 3.14159265358979323846;

  void on_pose(const msg::RoverPose & message)
  {
    pose_ = Pose2D{
      message.centre.x,
      message.centre.y,
      message.has_heading ? std::optional<double>(message.heading) : std::nullopt,
      message.confidence};
  }

  BodyToImage transform() const
  {
    return BodyToImage{
      get_parameter("px_per_forward_pulse").as_double(),
      get_parameter("radians_per_turn_pulse").as_double(),
      get_parameter("forward_axis_x").as_double(),
      get_parameter("forward_axis_y").as_double()};
  }

  void on_sonar(const sensor_msgs::msg::Range & message)
  {
    if (!pose_.has_value() || !std::isfinite(message.range)) {
      return;
    }
    double offset = 0.0;
    if (message.header.frame_id == "sonar_left") {
      offset = 50.0 * kPi / 180.0;
    } else if (message.header.frame_id == "sonar_right") {
      offset = -50.0 * kPi / 180.0;
    } else if (message.header.frame_id != "sonar_front") {
      return;
    }

    ++cycle_;
    const auto body_transform = transform();
    auto heading = pose_->heading;
    if (!heading.has_value()) {
      const auto [dx, dy] = body_transform.direction(*pose_);
      heading = std::atan2(dy, dx);
    }
    const auto distance_cm = static_cast<double>(message.range) * 100.0;
    const auto maximum = static_cast<double>(
      get_parameter("maximum_mapping_distance_cm").as_int());
    const auto cm_per_pulse = get_parameter("cm_per_translation_pulse").as_double();
    const auto distance_px = std::min(distance_cm, maximum) *
      body_transform.px_per_forward_pulse / cm_per_pulse;
    grid_->observe_ray(
      pose_->x, pose_->y, *heading + offset, distance_px,
      distance_cm < maximum, cycle_);
    publish_grid();
  }

  void publish_grid()
  {
    nav_msgs::msg::OccupancyGrid output;
    output.header.stamp = get_clock()->now();
    output.header.frame_id = get_parameter("map_frame_id").as_string();
    output.info.width = static_cast<std::uint32_t>(grid_->cols());
    output.info.height = static_cast<std::uint32_t>(grid_->rows());
    output.info.resolution = 1.0F;
    output.info.origin.orientation.w = 1.0;
    const auto occupied = grid_->occupied();
    output.data.reserve(static_cast<std::size_t>(grid_->cols() * grid_->rows()));
    for (int row = 0; row < grid_->rows(); ++row) {
      for (int col = 0; col < grid_->cols(); ++col) {
        output.data.push_back(
          occupied.count({col, row}) > 0 ? static_cast<std::int8_t>(100) : 0);
      }
    }
    publisher_->publish(output);
  }

  std::unique_ptr<ObstacleGrid> grid_;
  std::optional<Pose2D> pose_;
  std::int64_t cycle_{0};
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  rclcpp::Subscription<msg::RoverPose>::SharedPtr pose_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr sonar_subscription_;
};

}  // namespace rover_explorer_ros2

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<rover_explorer_ros2::ObstacleGridNode>());
  rclcpp::shutdown();
  return 0;
}
