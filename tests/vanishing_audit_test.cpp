#include "pixelaudit/tests/vanishing_point_test.hpp"

#include <filesystem>
#include <iostream>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

cv::Mat MakePerspectiveScene(const int w, const int h) {
  cv::Mat img(h, w, CV_8UC3, cv::Scalar(30, 30, 30));
  const cv::Point vp(w / 2, h / 3);

  for (int i = 0; i < 16; ++i) {
    const int x = static_cast<int>((i + 1) * (w / 17.0));
    cv::line(img, cv::Point(x, h - 1), vp, cv::Scalar(230, 230, 230), 2, cv::LINE_AA);
  }

  for (int i = 0; i < 12; ++i) {
    const int y = h - 20 - i * 26;
    const float t = static_cast<float>(h - y) / static_cast<float>(h);
    const int half = static_cast<int>((1.0f - 0.72f * t) * (w / 2.0f));
    cv::line(img, cv::Point(w / 2 - half, y), cv::Point(w / 2 + half, y),
             cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
  }

  cv::GaussianBlur(img, img, cv::Size(0, 0), 0.8);
  return img;
}

cv::Mat MakeInconsistentScene(const int w, const int h) {
  cv::Mat img(h, w, CV_8UC3, cv::Scalar(30, 30, 30));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> xdist(0, w - 1);
  std::uniform_int_distribution<int> ydist(0, h - 1);

  for (int i = 0; i < 45; ++i) {
    const cv::Point p0(xdist(rng), ydist(rng));
    const cv::Point p1(xdist(rng), ydist(rng));
    cv::line(img, p0, p1, cv::Scalar(230, 230, 230), 2, cv::LINE_AA);
  }

  cv::GaussianBlur(img, img, cv::Size(0, 0), 0.8);
  return img;
}

}  // namespace

int main() {
  pixelaudit::VanishingPointGeometryTest test;

  const auto tmp_root = std::filesystem::temp_directory_path() / "pixelaudit_vanishing_test";
  std::filesystem::create_directories(tmp_root);

  const cv::Mat perspective = MakePerspectiveScene(640, 480);
  const cv::Mat inconsistent = MakeInconsistentScene(640, 480);

  const pixelaudit::TestContext perspective_ctx{(tmp_root / "perspective").string()};
  const pixelaudit::TestContext inconsistent_ctx{(tmp_root / "inconsistent").string()};

  const auto perspective_result = test.Run(perspective, perspective_ctx);
  const auto inconsistent_result = test.Run(inconsistent, inconsistent_ctx);

  std::cout << "vanishing perspective score=" << perspective_result.score_percent
            << ", inconsistent score=" << inconsistent_result.score_percent << "\n";

  if (inconsistent_result.score_percent < perspective_result.score_percent + 8.0) {
    std::cerr << "Vanishing audit failed to penalize inconsistent geometry strongly enough.\n";
    return 1;
  }

  if (inconsistent_result.raw_metrics.find("dominant_vp_support_ratio") ==
      inconsistent_result.raw_metrics.end()) {
    std::cerr << "Missing expected dominant VP support metric output.\n";
    return 1;
  }

  std::cout << "Vanishing audit behavior test passed.\n";
  return 0;
}
