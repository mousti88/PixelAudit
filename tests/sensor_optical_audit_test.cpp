#include "pixelaudit/tests/optical_photometric_test.hpp"
#include "pixelaudit/tests/sensor_noise_prnu_test.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

cv::Mat MakeSensorLikePhoto(const int w, const int h) {
  cv::Mat base(h, w, CV_32FC3);
  cv::randu(base, 0.0f, 1.0f);
  cv::GaussianBlur(base, base, cv::Size(0, 0), 2.0);

  cv::Mat shared_noise(h, w, CV_32F);
  cv::randn(shared_noise, 0.0, 0.010);
  std::vector<cv::Mat> ch(3);
  cv::split(base, ch);
  ch[0] += shared_noise;
  ch[1] += shared_noise;
  ch[2] += shared_noise;
  cv::merge(ch, base);

  cv::Mat out_u8;
  cv::Mat clipped;
  cv::min(cv::max(base, 0.0), 1.0, clipped);
  clipped.convertTo(out_u8, CV_8UC3, 255.0);
  return out_u8;
}

cv::Mat MakeSyntheticLikePhoto(const int w, const int h) {
  cv::Mat base = MakeSensorLikePhoto(w, h);

  std::vector<cv::Mat> ch;
  cv::split(base, ch);

  cv::Mat f0;
  cv::Mat f1;
  cv::Mat f2;
  ch[0].convertTo(f0, CV_32F, 1.0 / 255.0);
  ch[1].convertTo(f1, CV_32F, 1.0 / 255.0);
  ch[2].convertTo(f2, CV_32F, 1.0 / 255.0);

  cv::GaussianBlur(f0, f0, cv::Size(0, 0), 0.6);
  cv::GaussianBlur(f1, f1, cv::Size(0, 0), 2.2);
  cv::GaussianBlur(f2, f2, cv::Size(0, 0), 1.1);

  cv::Mat n0(h, w, CV_32F);
  cv::Mat n1(h, w, CV_32F);
  cv::Mat n2(h, w, CV_32F);
  cv::randn(n0, 0.0, 0.018);
  cv::randn(n1, 0.0, 0.010);
  cv::randn(n2, 0.0, 0.022);
  f0 += n0;
  f1 += n1;
  f2 += n2;

  for (int y = 0; y < h; y += 24) {
    for (int x = 0; x < w; x += 24) {
      const cv::Rect roi(x, y, std::min(24, w - x), std::min(24, h - y));
      const float g = (((x / 24 + y / 24) & 1) == 0) ? 1.16f : 0.84f;
      f2(roi) *= g;
    }
  }

  cv::min(cv::max(f0, 0.0), 1.0, f0);
  cv::min(cv::max(f1, 0.0), 1.0, f1);
  cv::min(cv::max(f2, 0.0), 1.0, f2);
  f0.convertTo(ch[0], CV_8U, 255.0);
  f1.convertTo(ch[1], CV_8U, 255.0);
  f2.convertTo(ch[2], CV_8U, 255.0);

  cv::Mat aff = (cv::Mat_<double>(2, 3) << 1, 0, 1.4, 0, 1, 0.6);
  cv::warpAffine(ch[2], ch[2], aff, ch[2].size(), cv::INTER_LINEAR,
                 cv::BORDER_REFLECT101);

  cv::Mat green_f;
  ch[1].convertTo(green_f, CV_32F, 1.0 / 255.0);
  for (int y = 0; y < green_f.rows; ++y) {
    float* row = green_f.ptr<float>(y);
    for (int x = 0; x < green_f.cols; ++x) {
      const float cfa = ((x + y) & 1) ? 0.013f : -0.013f;
      row[x] = std::clamp(row[x] + cfa, 0.0f, 1.0f);
    }
  }
  green_f.convertTo(ch[1], CV_8U, 255.0);

  cv::merge(ch, base);
  return base;
}

}  // namespace

int main() {
  pixelaudit::SensorNoisePrnuCfaTest sensor_test;
  pixelaudit::OpticalPhotometricPlausibilityTest optical_test;

  const auto tmp_root = std::filesystem::temp_directory_path() / "pixelaudit_sensor_optical_test";
  std::filesystem::create_directories(tmp_root);

  const cv::Mat sensor_like = MakeSensorLikePhoto(640, 480);
  const cv::Mat synthetic_like = MakeSyntheticLikePhoto(640, 480);

  const pixelaudit::TestContext sensor_ctx{(tmp_root / "sensor").string()};
  const pixelaudit::TestContext synth_ctx{(tmp_root / "synthetic").string()};

  const auto sensor_real = sensor_test.Run(sensor_like, sensor_ctx);
  const auto sensor_fake = sensor_test.Run(synthetic_like, synth_ctx);
  const auto optical_real = optical_test.Run(sensor_like, sensor_ctx);
  const auto optical_fake = optical_test.Run(synthetic_like, synth_ctx);

  std::cout << "sensor real=" << sensor_real.score_percent
            << " fake=" << sensor_fake.score_percent
            << " | optical real=" << optical_real.score_percent
            << " fake=" << optical_fake.score_percent << "\n";

  const double real_stationarity =
      sensor_real.raw_metrics.at("stationarity_cv");
  const double fake_stationarity =
      sensor_fake.raw_metrics.at("stationarity_cv");
  const double real_corr =
      sensor_real.raw_metrics.at("mean_abs_channel_corr");
  const double fake_corr =
      sensor_fake.raw_metrics.at("mean_abs_channel_corr");

  if (!(fake_stationarity > real_stationarity || fake_corr < real_corr - 0.02)) {
    std::cerr << "Sensor audit features did not react to synthetic-like residual inconsistency.\n";
    return 1;
  }

  if (sensor_fake.score_percent < sensor_real.score_percent - 8.0) {
    std::cerr << "Sensor audit score dropped too much on synthetic-like sample.\n";
    return 1;
  }

  if (optical_fake.score_percent < optical_real.score_percent + 5.0) {
    std::cerr << "Optical audit did not sufficiently penalize synthetic-like chromatic shifts.\n";
    return 1;
  }

  return 0;
}
