#include <optional>
#include <vector>

#include "gtest/gtest.h"

#include "rover_explorer_ros2/deterministic_logic.hpp"

TEST(PostMotionObservation, RequiresNewPostSettleFrameAndSubsequentLegalActions)
{
  EXPECT_FALSE(rover_explorer_ros2::post_motion_observation_ready(
      10.3, 10.4, 2, 1, 20.0, 20.1));
  EXPECT_FALSE(rover_explorer_ros2::post_motion_observation_ready(
      10.5, 10.4, 1, 1, 20.0, 20.1));
  EXPECT_FALSE(rover_explorer_ros2::post_motion_observation_ready(
      10.5, 10.4, 2, 1, 20.1, 20.0));
  EXPECT_TRUE(rover_explorer_ros2::post_motion_observation_ready(
      10.5, 10.4, 2, 1, 20.0, 20.1));
}

namespace
{
using rover_explorer_ros2::Action;
using rover_explorer_ros2::BodyToImage;
using rover_explorer_ros2::CoverageGrid;
using rover_explorer_ros2::ObstacleGrid;
using rover_explorer_ros2::Pose2D;

TEST(GuardLogic, LostPoseAndSonarAreFailClosed)
{
  const BodyToImage transform{30.0, 0.3, 1.0, 0.0};
  const auto lost = rover_explorer_ros2::allowed_actions(
    std::nullopt, transform, 640, 480, 0.12);
  ASSERT_EQ(lost, (std::vector<Action>{Action::kBackward, Action::kStop}));

  const auto blocked = rover_explorer_ros2::apply_ultrasonic_guard(
    {Action::kForward, Action::kBackward, Action::kArcLeft, Action::kTurnLeft, Action::kStop},
    true);
  EXPECT_EQ(
    blocked, (std::vector<Action>{Action::kBackward, Action::kTurnLeft, Action::kStop}));
}

TEST(CoverageLogic, MatchesImageSpaceClamping)
{
  CoverageGrid grid(640, 480, 6, 4);
  grid.update(Pose2D{320.0, 240.0, 0.0, 1.0});
  grid.update(Pose2D{999.0, -10.0, std::nullopt, 1.0});
  EXPECT_TRUE(grid.visited().count({3, 2}));
  EXPECT_TRUE(grid.visited().count({5, 0}));
}

TEST(ObstacleLogic, RayHitClearingAndTtlMatchPythonReference)
{
  ObstacleGrid grid(640, 480, 12, 8, 2);
  const auto hit = grid.observe_ray(320.0, 240.0, 0.0, 100.0, true, 1);
  ASSERT_TRUE(hit.has_value());
  EXPECT_TRUE(grid.occupied().count(*hit));

  grid.observe_ray(320.0, 240.0, 0.0, 100.0, false, 2);
  EXPECT_TRUE(grid.occupied().empty());

  const auto old_hit = grid.observe_ray(320.0, 240.0, 0.0, 60.0, true, 3);
  ASSERT_TRUE(old_hit.has_value());
  grid.prune(6);
  EXPECT_TRUE(grid.occupied().empty());
}

TEST(TransformLogic, ArcPredictionUsesCalibratedImageSign)
{
  const BodyToImage transform{40.0, -0.4, 1.0, 0.0};
  const Pose2D pose{100.0, 100.0, 0.0, 1.0};
  const auto left = transform.predict(pose, Action::kArcLeft);
  EXPECT_GT(left.first, 100.0);
  EXPECT_LT(left.second, 100.0);
}
}  // namespace
