#include "pixelaudit/tests/sensor_noise_prnu_test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pixelaudit {
namespace {

double Clamp(const double value, const double lo, const double hi) {
  return std::max(lo, std::min(hi, value));
}

double Correlation(const cv::Mat& a, const cv::Mat& b) {
  CV_Assert(a.type() == CV_32F && b.type() == CV_32F);
  CV_Assert(a.size() == b.size());

  double sum_a = 0.0;
  double sum_b = 0.0;
  const int total = a.rows * a.cols;

  for (int y = 0; y < a.rows; ++y) {
    const float* pa = a.ptr<float>(y);
    const float* pb = b.ptr<float>(y);
    for (int x = 0; x < a.cols; ++x) {
      sum_a += pa[x];
      sum_b += pb[x];
    }
  }

  const double mean_a = sum_a / std::max(1, total);
  const double mean_b = sum_b / std::max(1, total);

  double cov = 0.0;
  double var_a = 0.0;
  double var_b = 0.0;
  for (int y = 0; y < a.rows; ++y) {
    const float* pa = a.ptr<float>(y);
    const float* pb = b.ptr<float>(y);
    for (int x = 0; x < a.cols; ++x) {
      const double da = static_cast<double>(pa[x]) - mean_a;
      const double db = static_cast<double>(pb[x]) - mean_b;
      cov += da * db;
      var_a += da * da;
      var_b += db * db;
    }
  }

  if (var_a <= 1e-12 || var_b <= 1e-12) {
    return 0.0;
  }

  return cov / std::sqrt(var_a * var_b + 1e-12);
}

cv::Mat BuildCfaPeriodicityPlot(const std::array<double, 4>& phase_energy) {
  const int cell = 120;
  const int margin = 36;
  const int width = 2 * cell + 2 * margin;
  const int height = 2 * cell + 2 * margin + 30;

  cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(248, 248, 248));

  const double max_v = std::max(1e-9, *std::max_element(phase_energy.begin(), phase_energy.end()));
  const char* labels[4] = {"(0,0)", "(1,0)", "(0,1)", "(1,1)"};

  for (int py = 0; py < 2; ++py) {
    for (int px = 0; px < 2; ++px) {
      const int idx = py * 2 + px;
      const double ratio = phase_energy[idx] / max_v;
      const int intensity = static_cast<int>(Clamp(ratio, 0.0, 1.0) * 255.0);

      const cv::Rect r(margin + px * cell, margin + py * cell, cell, cell);
      cv::Mat tile(1, 1, CV_8UC1, cv::Scalar(intensity));
      cv::Mat tile_color;
      cv::applyColorMap(tile, tile_color, cv::COLORMAP_VIRIDIS);
      cv::rectangle(canvas, r, tile_color.at<cv::Vec3b>(0, 0), cv::FILLED);
      cv::rectangle(canvas, r, cv::Scalar(230, 230, 230), 1);

      std::ostringstream val;
      val.precision(3);
      val << std::fixed << phase_energy[idx];
      cv::putText(canvas, labels[idx], cv::Point(r.x + 10, r.y + 22),
                  cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(20, 20, 20), 1,
                  cv::LINE_AA);
      cv::putText(canvas, val.str(), cv::Point(r.x + 10, r.y + 48),
                  cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(20, 20, 20), 1,
                  cv::LINE_AA);
    }
  }

  cv::putText(canvas, "CFA 2x2 Phase Residual Energy", cv::Point(40, height - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(50, 50, 50), 1,
              cv::LINE_AA);
  return canvas;
}

}  // namespace

std::string SensorNoisePrnuCfaTest::Id() const { return "sensor_noise_prnu_cfa"; }

std::string SensorNoisePrnuCfaTest::Name() const {
  return "Sensor Noise and PRNU/CFA Consistency Checks";
}

std::string SensorNoisePrnuCfaTest::Description() const {
  return "Extracts high-frequency residual noise and evaluates stationarity, "
         "cross-channel residual correlation, and 2x2 CFA-phase periodicity as "
         "camera-pipeline consistency evidence.";
}

TestResult SensorNoisePrnuCfaTest::Run(const cv::Mat& bgr_image,
                                       const TestContext& context) {
  TestResult result;
  result.test_id = Id();
  result.name = Name();
  result.description = Description();

  if (bgr_image.empty()) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.0;
    result.evidence_summary = "Input image is empty; sensor-noise audit skipped.";
    return result;
  }

  if (bgr_image.channels() != 3) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary =
        "Expected 3-channel color image for PRNU/CFA consistency audit.";
    return result;
  }

  cv::Mat bgr_f;
  bgr_image.convertTo(bgr_f, CV_32FC3, 1.0 / 255.0);

  cv::Mat denoised;
  cv::GaussianBlur(bgr_f, denoised, cv::Size(0, 0), 1.4, 1.4);

  cv::Mat residual = bgr_f - denoised;

  std::vector<cv::Mat> channels;
  cv::split(residual, channels);

  cv::Mat residual_abs;
  cv::Mat abs_b = cv::abs(channels[0]);
  cv::Mat abs_g = cv::abs(channels[1]);
  cv::Mat abs_r = cv::abs(channels[2]);
  residual_abs = (abs_b + abs_g + abs_r) / 3.0f;

  cv::Scalar res_mean;
  cv::Scalar res_std;
  cv::meanStdDev(residual_abs, res_mean, res_std);

  const double corr_bg = Correlation(channels[0], channels[1]);
  const double corr_br = Correlation(channels[0], channels[2]);
  const double corr_gr = Correlation(channels[1], channels[2]);
  const double mean_abs_corr =
      (std::abs(corr_bg) + std::abs(corr_br) + std::abs(corr_gr)) / 3.0;

  const int tile = 32;
  std::vector<double> tile_std;
  for (int y = 0; y + tile <= residual_abs.rows; y += tile) {
    for (int x = 0; x + tile <= residual_abs.cols; x += tile) {
      const cv::Mat patch = residual_abs(cv::Rect(x, y, tile, tile));
      cv::Scalar p_mean;
      cv::Scalar p_std;
      cv::meanStdDev(patch, p_mean, p_std);
      tile_std.push_back(p_std[0]);
    }
  }

  if (tile_std.empty()) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary =
        "Image too small for tile-based sensor-noise stationarity analysis.";
    return result;
  }

  const double tile_mean =
      std::accumulate(tile_std.begin(), tile_std.end(), 0.0) / tile_std.size();
  double tile_var = 0.0;
  for (const double v : tile_std) {
    const double d = v - tile_mean;
    tile_var += d * d;
  }
  tile_var /= tile_std.size();
  const double tile_stddev = std::sqrt(tile_var);
  const double stationarity_cv = tile_stddev / (tile_mean + 1e-9);

  cv::Mat gray;
  cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
  cv::Mat gray_f;
  gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);

  cv::Mat gray_blur;
  cv::GaussianBlur(gray_f, gray_blur, cv::Size(0, 0), 1.2, 1.2);
  cv::Mat hp = gray_f - gray_blur;

  std::array<double, 4> phase_energy{0.0, 0.0, 0.0, 0.0};
  for (int y = 0; y < hp.rows; ++y) {
    const float* row = hp.ptr<float>(y);
    for (int x = 0; x < hp.cols; ++x) {
      const int idx = (y & 1) * 2 + (x & 1);
      phase_energy[idx] += std::abs(static_cast<double>(row[x]));
    }
  }

  const double phase_sum =
      std::accumulate(phase_energy.begin(), phase_energy.end(), 0.0);
  const double phase_max = *std::max_element(phase_energy.begin(), phase_energy.end());
  const double cfa_phase_dominance = phase_max / (phase_sum + 1e-9);

  double phase_entropy = 0.0;
  for (const double e : phase_energy) {
    if (e <= 0.0) {
      continue;
    }
    const double p = e / (phase_sum + 1e-9);
    phase_entropy -= p * std::log2(p);
  }
  const double cfa_entropy_norm = phase_entropy / 2.0;

    const double score = Clamp(
      100.0 * (0.50 * Clamp(stationarity_cv / 1.10, 0.0, 1.0) +
           0.30 * Clamp((0.985 - mean_abs_corr) / 0.085, 0.0, 1.0) +
           0.15 * Clamp((0.030 - res_mean[0]) / 0.020, 0.0, 1.0) +
           0.05 * Clamp((0.254 - cfa_phase_dominance) / 0.012, 0.0, 1.0)),
      0.0, 100.0);

  result.score_percent = score;
  result.confidence = Clamp(std::abs(score - 50.0) / 50.0, 0.0, 1.0);
  result.status = score >= 70.0 ? TestStatus::kFail
                                : (score <= 30.0 ? TestStatus::kPass
                                                 : TestStatus::kInconclusive);

  std::ostringstream summary;
  summary << "Residual correlation=" << mean_abs_corr
          << ", stationarity CV=" << stationarity_cv
      << ", residual mean=" << res_mean[0]
      << ". Weaker camera-like channel coupling and unstable residual noise increase "
             "AI-likelihood score.";
  result.evidence_summary = summary.str();

  result.raw_metrics["residual_mean_abs"] = res_mean[0];
  result.raw_metrics["residual_std_abs"] = res_std[0];
  result.raw_metrics["channel_corr_bg"] = corr_bg;
  result.raw_metrics["channel_corr_br"] = corr_br;
  result.raw_metrics["channel_corr_gr"] = corr_gr;
  result.raw_metrics["mean_abs_channel_corr"] = mean_abs_corr;
  result.raw_metrics["stationarity_cv"] = stationarity_cv;
  result.raw_metrics["cfa_phase_dominance"] = cfa_phase_dominance;
  result.raw_metrics["cfa_entropy_norm"] = cfa_entropy_norm;
  result.raw_metrics["tile_count"] = static_cast<double>(tile_std.size());

  cv::Mat residual_vis;
  cv::normalize(residual_abs, residual_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::Mat residual_color;
  cv::applyColorMap(residual_vis, residual_color, cv::COLORMAP_BONE);

  cv::Mat cfa_plot = BuildCfaPeriodicityPlot(phase_energy);

  std::filesystem::create_directories(context.output_dir);
  const std::string residual_path = context.output_dir + "/sensor_noise_residual.png";
  const std::string cfa_path = context.output_dir + "/sensor_cfa_periodicity.png";
  cv::imwrite(residual_path, residual_color);
  cv::imwrite(cfa_path, cfa_plot);

  result.artifact_paths.push_back(residual_path);
  result.artifact_paths.push_back(cfa_path);

  return result;
}

}  // namespace pixelaudit
