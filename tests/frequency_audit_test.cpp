#include "pixelaudit/tests/frequency_spectrum_test.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

cv::Mat MakeCheckerboardPattern(const int size) {
  cv::Mat img(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const bool checker = ((x / 8) + (y / 8)) % 2 == 0;
      const double stripe = 30.0 * std::sin(2.0 * CV_PI * static_cast<double>(x) / 16.0);
      const int v = static_cast<int>(std::clamp((checker ? 210.0 : 40.0) + stripe, 0.0, 255.0));
      img.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
    }
  }
  return img;
}

cv::Mat MakeNaturalLikePattern(const int size) {
  cv::Mat noise(size, size, CV_32F);
  cv::randn(noise, 0.0, 1.0);
  cv::GaussianBlur(noise, noise, cv::Size(0, 0), 2.4);

  cv::Mat gradient(size, size, CV_32F);
  for (int y = 0; y < size; ++y) {
    float* row = gradient.ptr<float>(y);
    for (int x = 0; x < size; ++x) {
      row[x] = static_cast<float>(0.35 * x / std::max(1, size - 1));
    }
  }

  cv::Mat mixed = 0.6 * noise + gradient;
  cv::normalize(mixed, mixed, 0.0, 255.0, cv::NORM_MINMAX);
  cv::Mat gray_u8;
  mixed.convertTo(gray_u8, CV_8U);

  cv::Mat bgr;
  cv::cvtColor(gray_u8, bgr, cv::COLOR_GRAY2BGR);
  return bgr;
}

}  // namespace

int main() {
  pixelaudit::FrequencySpectrumTest test;

  const auto tmp_root = std::filesystem::temp_directory_path() / "pixelaudit_frequency_test";
  std::filesystem::create_directories(tmp_root);

  const cv::Mat synthetic_periodic = MakeCheckerboardPattern(512);
  const cv::Mat natural_like = MakeNaturalLikePattern(512);

  const pixelaudit::TestContext periodic_ctx{(tmp_root / "periodic").string()};
  const pixelaudit::TestContext natural_ctx{(tmp_root / "natural").string()};

  const auto periodic_result = test.Run(synthetic_periodic, periodic_ctx);
  const auto natural_result = test.Run(natural_like, natural_ctx);

  std::cout << "frequency periodic score=" << periodic_result.score_percent
            << ", natural_like score=" << natural_result.score_percent << "\n";

  if (periodic_result.score_percent < natural_result.score_percent + 8.0) {
    std::cerr << "Frequency audit failed to penalize periodic spectral artifacts strongly enough.\n";
    return 1;
  }

  if (periodic_result.raw_metrics.find("decay_fit_r2") == periodic_result.raw_metrics.end()) {
    std::cerr << "Missing expected decay-fit metric output.\n";
    return 1;
  }

  std::cout << "Frequency audit behavior test passed.\n";
  return 0;
}
