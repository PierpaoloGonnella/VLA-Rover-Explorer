#include "rover_explorer_ros2/deterministic_logic.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rover_explorer_ros2
{

bool post_motion_observation_ready(
  const double image_capture_seconds,
  const double earliest_capture_seconds,
  const std::uint64_t pose_sequence,
  const std::uint64_t stop_pose_sequence,
  const double pose_received_seconds,
  const double legal_actions_received_seconds) noexcept
{
  return std::isfinite(image_capture_seconds) && std::isfinite(earliest_capture_seconds) &&
         image_capture_seconds >= earliest_capture_seconds &&
         pose_sequence > stop_pose_sequence &&
         legal_actions_received_seconds >= pose_received_seconds;
}

std::pair<double, double> BodyToImage::direction(const Pose2D & pose) const
{
  if (pose.heading.has_value()) {
    return {std::cos(*pose.heading), std::sin(*pose.heading)};
  }
  const auto norm = std::hypot(forward_axis_x, forward_axis_y);
  if (norm == 0.0) {
    return {1.0, 0.0};
  }
  return {forward_axis_x / norm, forward_axis_y / norm};
}

std::pair<double, double> BodyToImage::predict(const Pose2D & pose, Action action) const
{
  const auto [dx, dy] = direction(pose);
  if (action == Action::kForward) {
    return {pose.x + dx * px_per_forward_pulse, pose.y + dy * px_per_forward_pulse};
  }
  if (action == Action::kBackward) {
    return {pose.x - dx * px_per_forward_pulse, pose.y - dy * px_per_forward_pulse};
  }
  if (action == Action::kArcLeft || action == Action::kArcRight) {
    const auto sign = action == Action::kArcLeft ? 1.0 : -1.0;
    const auto angle = sign * radians_per_turn_pulse * 0.5;
    const auto adx = std::cos(angle) * dx - std::sin(angle) * dy;
    const auto ady = std::sin(angle) * dx + std::cos(angle) * dy;
    return {
      pose.x + adx * px_per_forward_pulse * 0.75,
      pose.y + ady * px_per_forward_pulse * 0.75};
  }
  return {pose.x, pose.y};
}

std::string action_name(Action action)
{
  switch (action) {
    case Action::kForward: return "forward";
    case Action::kBackward: return "backward";
    case Action::kTurnLeft: return "turn_left";
    case Action::kTurnRight: return "turn_right";
    case Action::kArcLeft: return "arc_left";
    case Action::kArcRight: return "arc_right";
    case Action::kStop: return "stop";
  }
  return "stop";
}

std::vector<Action> allowed_actions(
  const std::optional<Pose2D> & pose,
  const BodyToImage & transform,
  int frame_width,
  int frame_height,
  double margin_fraction)
{
  if (!pose.has_value()) {
    return {Action::kBackward, Action::kStop};
  }
  const auto margin_x = frame_width * margin_fraction;
  const auto margin_y = frame_height * margin_fraction;
  const auto uncertainty = std::max(2.0, transform.px_per_forward_pulse);
  const Action translations[] = {
    Action::kForward, Action::kBackward, Action::kArcLeft, Action::kArcRight};
  std::vector<Action> result;
  for (const auto action : translations) {
    const auto [x, y] = transform.predict(*pose, action);
    if (
      margin_x + uncertainty <= x && x <= frame_width - margin_x - uncertainty &&
      margin_y + uncertainty <= y && y <= frame_height - margin_y - uncertainty)
    {
      result.push_back(action);
    }
  }
  result.push_back(Action::kTurnLeft);
  result.push_back(Action::kTurnRight);
  result.push_back(Action::kStop);
  return result;
}

std::vector<Action> apply_ultrasonic_guard(
  const std::vector<Action> & actions, bool blocked)
{
  if (!blocked) {
    return actions;
  }
  std::vector<Action> result;
  for (const auto action : actions) {
    if (
      action != Action::kForward && action != Action::kArcLeft &&
      action != Action::kArcRight)
    {
      result.push_back(action);
    }
  }
  return result;
}

CoverageGrid::CoverageGrid(int frame_width, int frame_height, int cols, int rows)
: frame_width_(frame_width), frame_height_(frame_height), cols_(cols), rows_(rows)
{
  if (frame_width_ <= 0 || frame_height_ <= 0 || cols_ <= 0 || rows_ <= 0) {
    throw std::invalid_argument("coverage dimensions must be positive");
  }
}

std::pair<int, int> CoverageGrid::cell_for(double x, double y) const
{
  const auto col = std::clamp(static_cast<int>(x * cols_ / frame_width_), 0, cols_ - 1);
  const auto row = std::clamp(static_cast<int>(y * rows_ / frame_height_), 0, rows_ - 1);
  return {col, row};
}

void CoverageGrid::update(const Pose2D & pose)
{
  visited_.insert(cell_for(pose.x, pose.y));
}

ObstacleGrid::ObstacleGrid(
  int frame_width, int frame_height, int cols, int rows, int ttl_cycles)
: frame_width_(frame_width), frame_height_(frame_height), cols_(cols), rows_(rows),
  ttl_cycles_(ttl_cycles)
{
  if (frame_width_ <= 0 || frame_height_ <= 0 || cols_ <= 0 || rows_ <= 0) {
    throw std::invalid_argument("obstacle-grid dimensions must be positive");
  }
}

ObstacleGrid::Cell ObstacleGrid::cell_for(double x, double y) const
{
  const auto col = std::clamp(static_cast<int>(x * cols_ / frame_width_), 0, cols_ - 1);
  const auto row = std::clamp(static_cast<int>(y * rows_ / frame_height_), 0, rows_ - 1);
  return {col, row};
}

std::vector<ObstacleGrid::Cell> ObstacleGrid::ray_cells(
  double origin_x, double origin_y, double angle, double distance_px) const
{
  const auto step = std::max(
    2.0, std::min(
      static_cast<double>(frame_width_) / cols_,
      static_cast<double>(frame_height_) / rows_) / 3.0);
  const auto count = std::max(1, static_cast<int>(std::ceil(std::max(0.0, distance_px) / step)));
  std::vector<Cell> cells;
  for (int index = 1; index <= count; ++index) {
    const auto distance = distance_px * index / count;
    const auto x = origin_x + std::cos(angle) * distance;
    const auto y = origin_y + std::sin(angle) * distance;
    if (x < 0.0 || x >= frame_width_ || y < 0.0 || y >= frame_height_) {
      break;
    }
    const auto cell = cell_for(x, y);
    if (cells.empty() || cells.back() != cell) {
      cells.push_back(cell);
    }
  }
  return cells;
}

std::optional<ObstacleGrid::Cell> ObstacleGrid::observe_ray(
  double origin_x, double origin_y, double angle, double distance_px, bool hit,
  std::int64_t cycle)
{
  const auto cells = ray_cells(origin_x, origin_y, angle, distance_px);
  const auto clear_count = hit && !cells.empty() ? cells.size() - 1 : cells.size();
  for (std::size_t index = 0; index < clear_count; ++index) {
    hits_.erase(cells[index]);
  }
  std::optional<Cell> hit_cell;
  if (hit && !cells.empty()) {
    hit_cell = cells.back();
    hits_[*hit_cell] = cycle;
  }
  prune(cycle);
  return hit_cell;
}

void ObstacleGrid::prune(std::int64_t cycle)
{
  const auto oldest = cycle - std::max(1, ttl_cycles_);
  for (auto it = hits_.begin(); it != hits_.end();) {
    if (it->second < oldest) {
      it = hits_.erase(it);
    } else {
      ++it;
    }
  }
}

std::set<ObstacleGrid::Cell> ObstacleGrid::occupied(int inflation_cells) const
{
  std::set<Cell> result;
  const auto radius = std::max(0, inflation_cells);
  for (const auto & [cell, unused_seen] : hits_) {
    (void)unused_seen;
    for (int dc = -radius; dc <= radius; ++dc) {
      for (int dr = -radius; dr <= radius; ++dr) {
        if (dc * dc + dr * dr > radius * radius) {
          continue;
        }
        const Cell candidate{cell.first + dc, cell.second + dr};
        if (
          candidate.first >= 0 && candidate.first < cols_ &&
          candidate.second >= 0 && candidate.second < rows_)
        {
          result.insert(candidate);
        }
      }
    }
  }
  return result;
}

}  // namespace rover_explorer_ros2
