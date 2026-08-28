#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "rover_explorer_ros2/localization.hpp"

namespace localization = rover_explorer_ros2::localization;

int main(int argc, char ** argv)
{
  if (argc < 5) {
    std::cerr << "usage: localizer_probe ITERATIONS MARKER_ID OFFSET_RADIANS IMAGE...\n";
    return 2;
  }
  const auto iterations = std::max(1, std::stoi(argv[1]));
  const auto marker_id = std::stoll(argv[2]);
  const auto offset = std::stod(argv[3]);
  std::vector<cv::Mat> frames;
  for (int index = 4; index < argc; ++index) {
    auto frame = cv::imread(argv[index], cv::IMREAD_COLOR);
    if (frame.empty()) {
      std::cerr << "could not read " << argv[index] << '\n';
      return 3;
    }
    frames.push_back(std::move(frame));
  }

  localization::ArucoLocalizer localizer(marker_id, offset);
  std::cout << std::setprecision(17);
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const auto pose = localizer.locate(frames[index]);
    std::cout << "POSE\t" << index << '\t' << (pose ? 1 : 0);
    if (pose) {
      std::cout << '\t' << pose->centre.x << '\t' << pose->centre.y << '\t'
                << pose->heading.value_or(0.0) << '\t' << pose->confidence;
    }
    std::cout << '\n';
  }

  std::vector<double> latency_ms;
  latency_ms.reserve(static_cast<std::size_t>(iterations) * frames.size());
  const auto wall_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (const auto & frame : frames) {
      const auto start = std::chrono::steady_clock::now();
      static_cast<void>(localizer.locate(frame));
      const auto finish = std::chrono::steady_clock::now();
      latency_ms.push_back(
        std::chrono::duration<double, std::milli>(finish - start).count());
    }
  }
  const auto wall_finish = std::chrono::steady_clock::now();
  std::sort(latency_ms.begin(), latency_ms.end());
  const auto mean = std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0) /
    static_cast<double>(latency_ms.size());
  const auto percentile = [&latency_ms](const double fraction) {
      const auto rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(latency_ms.size())));
      const auto bounded_rank = std::min(
        latency_ms.size(), std::max<std::size_t>(1U, rank));
      return latency_ms[bounded_rank - 1U];
    };
  const auto wall_seconds =
    std::chrono::duration<double>(wall_finish - wall_start).count();
  std::cout << "METRIC\t" << mean << '\t' << percentile(0.5) << '\t'
            << percentile(0.95) << '\t' << latency_ms.back() << '\t'
            << static_cast<double>(latency_ms.size()) / wall_seconds << '\t'
            << latency_ms.size() << '\n';
  return 0;
}
