#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "rover_explorer_ros2/localization.hpp"

namespace localization = rover_explorer_ros2::localization;

namespace
{
constexpr double kPi = 3.14159265358979323846;

double angular_error(const double left, const double right)
{
  return std::abs(localization::normalize_heading(left - right));
}

cv::Mat marker_frame(
  const cv::Point2f centre = cv::Point2f(320.0F, 240.0F),
  const double angle = 0.0, const int marker_id = 0, const float size = 68.0F)
{
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(225, 230, 225));
  const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  cv::Mat marker;
  cv::aruco::generateImageMarker(dictionary, marker_id, 100, marker, 1);
  cv::Mat marker_bgr;
  cv::cvtColor(marker, marker_bgr, cv::COLOR_GRAY2BGR);

  const auto cosine = static_cast<float>(std::cos(angle));
  const auto sine = static_cast<float>(std::sin(angle));
  const auto half = size / 2.0F;
  const std::vector<cv::Point2f> local{
    {-half, -half}, {half, -half}, {half, half}, {-half, half}};
  std::vector<cv::Point2f> destination;
  destination.reserve(4U);
  for (const auto & point : local) {
    destination.emplace_back(
      centre.x + cosine * point.x - sine * point.y,
      centre.y + sine * point.x + cosine * point.y);
  }
  const std::vector<cv::Point2f> source{
    {0.0F, 0.0F}, {99.0F, 0.0F}, {99.0F, 99.0F}, {0.0F, 99.0F}};
  const auto transform = cv::getPerspectiveTransform(source, destination);
  cv::Mat warped;
  cv::Mat mask;
  cv::warpPerspective(
    marker_bgr, warped, transform, frame.size(), cv::INTER_LINEAR,
    cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  cv::warpPerspective(
    cv::Mat(100, 100, CV_8U, cv::Scalar(255)), mask, transform, frame.size());
  warped.copyTo(frame, mask);
  return frame;
}
}  // namespace

TEST(ArucoLocalization, MatchesSyntheticCentreRotationAndOffset)
{
  constexpr double offset = -0.37;
  localization::ArucoLocalizer localizer(0, offset);
  for (const auto angle : {-2.2, -0.8, 0.0, 0.65, 2.4}) {
    const auto pose = localizer.locate(marker_frame(cv::Point2f(320.0F, 240.0F), angle));
    ASSERT_TRUE(pose.has_value());
    EXPECT_LT(cv::norm(pose->centre - cv::Point2d(320.0, 240.0)), 1.0);
    ASSERT_TRUE(pose->heading.has_value());
    EXPECT_LT(angular_error(*pose->heading, angle + offset), 0.025);
    EXPECT_GE(pose->confidence, 0.0F);
    EXPECT_LE(pose->confidence, 1.0F);
  }
}

TEST(ArucoLocalization, DetectsFullyVisibleMarkerNearBoundary)
{
  localization::ArucoLocalizer localizer;
  const auto pose = localizer.locate(marker_frame(cv::Point2f(42.0F, 42.0F), 0.2));
  ASSERT_TRUE(pose.has_value());
  EXPECT_LT(cv::norm(pose->centre - cv::Point2d(42.0, 42.0)), 1.0);
}

TEST(ArucoLocalization, RejectsBlankWrongAndOccludedMarkersWithoutCaching)
{
  localization::ArucoLocalizer localizer;
  ASSERT_TRUE(localizer.locate(marker_frame()).has_value());
  EXPECT_FALSE(localizer.locate(cv::Mat::zeros(480, 640, CV_8UC3)).has_value());
  EXPECT_FALSE(localizer.locate(marker_frame({}, 0.0, 1)).has_value());
  auto occluded = marker_frame();
  cv::rectangle(occluded, cv::Rect(300, 205, 55, 70), cv::Scalar(225, 230, 225), -1);
  EXPECT_FALSE(localizer.locate(occluded).has_value());
}

TEST(ArucoDiagnostics, ClassifiesFailuresAndCollectsGeometryAndImageQuality)
{
  localization::ArucoLocalizer localizer;
  const auto valid = localizer.analyze(marker_frame(), 0.25);
  EXPECT_EQ(valid.outcome, "valid");
  ASSERT_TRUE(valid.pose.has_value());
  EXPECT_EQ(valid.candidate_count, 1U);
  EXPECT_EQ(valid.detected_ids, std::vector<int>({0}));
  EXPECT_GT(valid.marker_mean_side_px, 60.0);
  EXPECT_GT(valid.marker_boundary_distance_px, 0.0);
  EXPECT_GT(valid.grayscale_mean, 0.0);
  EXPECT_GT(valid.grayscale_stddev, 0.0);
  EXPECT_GT(valid.sharpness, 0.0);

  const auto blank = localizer.analyze(cv::Mat::zeros(480, 640, CV_8UC3), 0.25);
  EXPECT_EQ(blank.outcome, "no_candidates");
  EXPECT_FALSE(blank.pose.has_value());

  const auto wrong = localizer.analyze(
    marker_frame(cv::Point2f(320.0F, 240.0F), 0.0, 1), 0.25);
  EXPECT_EQ(wrong.outcome, "wrong_marker_id");
  EXPECT_FALSE(wrong.pose.has_value());

  const auto low_confidence = localizer.analyze(marker_frame(), 1.1);
  EXPECT_EQ(low_confidence.outcome, "below_confidence");
  EXPECT_FALSE(low_confidence.pose.has_value());

  const auto small = localizer.analyze(
    marker_frame(cv::Point2f(320.0F, 240.0F), 0.0, 0, 32.0F), 1.5);
  EXPECT_EQ(small.outcome, "too_small");
  EXPECT_FALSE(small.pose.has_value());
}

TEST(ArucoConfiguration, ValidatesRangesAndSupportsExplicitCornerRefinement)
{
  localization::ArucoDetectorConfig defaults;
  EXPECT_EQ(defaults.corner_refinement_method, cv::aruco::CORNER_REFINE_SUBPIX);
  EXPECT_NO_THROW(localization::ArucoLocalizer(0, 0.0, defaults));

  auto none = defaults;
  none.corner_refinement_method = cv::aruco::CORNER_REFINE_NONE;
  EXPECT_TRUE(localization::ArucoLocalizer(0, 0.0, none).locate(marker_frame()).has_value());

  auto invalid = defaults;
  invalid.adaptive_threshold_window_step = 0;
  EXPECT_THROW(localization::ArucoLocalizer(0, 0.0, invalid), std::invalid_argument);
  invalid = defaults;
  invalid.error_correction_rate = 1.5;
  EXPECT_THROW(localization::ArucoLocalizer(0, 0.0, invalid), std::invalid_argument);
}

TEST(ArucoNegativeControls, NeverPublishesPoseForBlurPartialGlareOrLowContrast)
{
  localization::ArucoLocalizer localizer;
  auto blurred = marker_frame();
  cv::GaussianBlur(blurred, blurred, cv::Size(61, 61), 20.0);
  EXPECT_FALSE(localizer.analyze(blurred, 0.25).pose.has_value());

  auto partial = marker_frame(cv::Point2f(12.0F, 240.0F));
  EXPECT_FALSE(localizer.analyze(partial, 0.25).pose.has_value());

  auto glare = marker_frame();
  cv::rectangle(glare, cv::Rect(285, 205, 70, 70), cv::Scalar(255, 255, 255), -1);
  EXPECT_FALSE(localizer.analyze(glare, 0.25).pose.has_value());

  auto low_contrast = marker_frame();
  low_contrast.convertTo(low_contrast, -1, 0.05, 120.0);
  EXPECT_FALSE(localizer.analyze(low_contrast, 0.25).pose.has_value());
}

TEST(ColorLocalization, PreservesPositionOnlyAndConfidenceSemantics)
{
  cv::Mat frame = cv::Mat::zeros(200, 300, CV_8UC3);
  cv::rectangle(frame, cv::Rect(100, 60, 40, 30), cv::Scalar(0, 255, 0), -1);
  localization::ColorBlobLocalizer localizer;
  const auto pose = localizer.locate(frame);
  ASSERT_TRUE(pose.has_value());
  EXPECT_NEAR(pose->centre.x, 119.5, 0.01);
  EXPECT_NEAR(pose->centre.y, 74.5, 0.01);
  EXPECT_FALSE(pose->heading.has_value());
  EXPECT_NEAR(pose->confidence, 1.0, 1e-6);
}

TEST(ImageValidation, AcceptsPackedZeroStepAndPaddedBgrRows)
{
  sensor_msgs::msg::Image packed;
  packed.width = 2;
  packed.height = 2;
  packed.encoding = "bgr8";
  packed.step = 0;
  packed.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const auto packed_view = localization::validated_bgr_view(packed);
  EXPECT_EQ(packed_view.rows, 2);
  EXPECT_EQ(packed_view.cols, 2);
  EXPECT_EQ(packed_view.at<cv::Vec3b>(1, 1)[2], 12);

  auto padded = packed;
  padded.encoding = "8UC3";
  padded.step = 8;
  padded.data = {1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 99, 99};
  const auto padded_view = localization::validated_bgr_view(padded);
  EXPECT_EQ(padded_view.step, 8U);
  EXPECT_EQ(padded_view.at<cv::Vec3b>(1, 1)[2], 12);
}

TEST(ImageValidation, RejectsMalformedDimensionsStrideLengthAndEncoding)
{
  sensor_msgs::msg::Image image;
  image.width = 2;
  image.height = 2;
  image.encoding = "bgr8";
  image.step = 6;
  image.data.resize(12);

  auto invalid = image;
  invalid.width = 0;
  EXPECT_THROW(localization::validated_bgr_view(invalid), std::invalid_argument);
  invalid = image;
  invalid.step = 5;
  invalid.data.resize(10);
  EXPECT_THROW(localization::validated_bgr_view(invalid), std::invalid_argument);
  invalid = image;
  invalid.data.pop_back();
  EXPECT_THROW(localization::validated_bgr_view(invalid), std::invalid_argument);
  invalid = image;
  invalid.encoding = "rgb8";
  EXPECT_THROW(localization::validated_bgr_view(invalid), std::invalid_argument);
}

TEST(HeadingNormalization, UsesPythonMinusPiInclusiveConvention)
{
  EXPECT_NEAR(localization::normalize_heading(3.0 * kPi), -kPi, 1e-12);
  EXPECT_NEAR(localization::normalize_heading(-3.0 * kPi), -kPi, 1e-12);
  EXPECT_NEAR(localization::normalize_heading(0.25), 0.25, 1e-12);
}

TEST(ArucoComparison, MatchesPythonPixelProjectionAndRejectsInvalidCalibration)
{
  const auto error = localization::aruco_projection_error(
    cv::Point2d(330.0, 250.0), cv::Point3d(0.1, 0.2, 2.0),
    200.0, 100.0, 300.0, 200.0);
  ASSERT_TRUE(error.has_value());
  EXPECT_NEAR(*error, std::sqrt(2000.0), 1e-5);
  EXPECT_FALSE(localization::aruco_projection_error(
      cv::Point2d(), cv::Point3d(1.0, 1.0, 0.0), 200.0, 100.0, 0.0, 0.0));
  EXPECT_FALSE(localization::aruco_projection_error(
      cv::Point2d(), cv::Point3d(1.0, 1.0, 1.0), 0.0, 100.0, 0.0, 0.0));
}
