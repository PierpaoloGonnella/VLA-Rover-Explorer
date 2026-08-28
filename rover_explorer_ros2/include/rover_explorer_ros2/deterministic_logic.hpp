#ifndef ROVER_EXPLORER_ROS2__DETERMINISTIC_LOGIC_HPP_
#define ROVER_EXPLORER_ROS2__DETERMINISTIC_LOGIC_HPP_

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rover_explorer_ros2
{

enum class Action
{
  kForward,
  kBackward,
  kTurnLeft,
  kTurnRight,
  kArcLeft,
  kArcRight,
  kStop,
};

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  std::optional<double> heading;
  double confidence{0.0};
};

struct BodyToImage
{
  double px_per_forward_pulse{35.0};
  double radians_per_turn_pulse{0.35};
  double forward_axis_x{1.0};
  double forward_axis_y{0.0};

  std::pair<double, double> direction(const Pose2D & pose) const;
  std::pair<double, double> predict(const Pose2D & pose, Action action) const;
};

std::string action_name(Action action);
std::vector<Action> allowed_actions(
  const std::optional<Pose2D> & pose,
  const BodyToImage & transform,
  int frame_width,
  int frame_height,
  double margin_fraction);
std::vector<Action> apply_ultrasonic_guard(
  const std::vector<Action> & actions, bool blocked);

class CoverageGrid
{
public:
  CoverageGrid(int frame_width, int frame_height, int cols, int rows);

  std::pair<int, int> cell_for(double x, double y) const;
  void update(const Pose2D & pose);
  const std::set<std::pair<int, int>> & visited() const {return visited_;}
  int cols() const {return cols_;}
  int rows() const {return rows_;}

private:
  int frame_width_;
  int frame_height_;
  int cols_;
  int rows_;
  std::set<std::pair<int, int>> visited_;
};

class ObstacleGrid
{
public:
  using Cell = std::pair<int, int>;

  ObstacleGrid(int frame_width, int frame_height, int cols, int rows, int ttl_cycles);

  Cell cell_for(double x, double y) const;
  std::vector<Cell> ray_cells(double origin_x, double origin_y, double angle, double distance_px) const;
  std::optional<Cell> observe_ray(
    double origin_x, double origin_y, double angle, double distance_px, bool hit, std::int64_t cycle);
  void prune(std::int64_t cycle);
  std::set<Cell> occupied(int inflation_cells = 0) const;
  int cols() const {return cols_;}
  int rows() const {return rows_;}

private:
  int frame_width_;
  int frame_height_;
  int cols_;
  int rows_;
  int ttl_cycles_;
  std::map<Cell, std::int64_t> hits_;
};

}  // namespace rover_explorer_ros2

#endif  // ROVER_EXPLORER_ROS2__DETERMINISTIC_LOGIC_HPP_
