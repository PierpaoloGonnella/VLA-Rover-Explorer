#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"

#include "rover_explorer_ros2/deterministic_logic.hpp"
#include "rover_explorer_ros2/msg/rover_pose.hpp"

namespace rover_explorer_ros2
{

class CoverageNode : public rclcpp::Node
{
public:
  CoverageNode()
  : Node("coverage_node")
  {
    declare_parameter("camera_width", 640);
    declare_parameter("camera_height", 480);
    declare_parameter("coverage_cols", 6);
    declare_parameter("coverage_rows", 4);
    declare_parameter("map_frame_id", "map");
    grid_ = std::make_unique<CoverageGrid>(
      static_cast<int>(get_parameter("camera_width").as_int()),
      static_cast<int>(get_parameter("camera_height").as_int()),
      static_cast<int>(get_parameter("coverage_cols").as_int()),
      static_cast<int>(get_parameter("coverage_rows").as_int()));

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/rover/coverage_map", qos);
    pose_subscription_ = create_subscription<msg::RoverPose>(
      "/rover/pose", qos,
      [this](const msg::RoverPose::SharedPtr message) {on_pose(*message);});
  }

private:
  void on_pose(const msg::RoverPose & message)
  {
    grid_->update(Pose2D{
      message.centre.x,
      message.centre.y,
      message.has_heading ? std::optional<double>(message.heading) : std::nullopt,
      message.confidence});

    nav_msgs::msg::OccupancyGrid output;
    output.header.stamp = get_clock()->now();
    output.header.frame_id = get_parameter("map_frame_id").as_string();
    output.info.width = static_cast<std::uint32_t>(grid_->cols());
    output.info.height = static_cast<std::uint32_t>(grid_->rows());
    output.info.resolution = 1.0F;
    output.info.origin.orientation.w = 1.0;
    output.data.reserve(static_cast<std::size_t>(grid_->cols() * grid_->rows()));
    for (int row = 0; row < grid_->rows(); ++row) {
      for (int col = 0; col < grid_->cols(); ++col) {
        output.data.push_back(
          grid_->visited().count({col, row}) > 0 ? static_cast<std::int8_t>(100) : 0);
      }
    }
    publisher_->publish(output);
  }

  std::unique_ptr<CoverageGrid> grid_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  rclcpp::Subscription<msg::RoverPose>::SharedPtr pose_subscription_;
};

}  // namespace rover_explorer_ros2

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<rover_explorer_ros2::CoverageNode>());
  rclcpp::shutdown();
  return 0;
}
