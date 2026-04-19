#include "pixelaudit/tests/optical_photometric_test.hpp"

#include <algorithm>
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

double WeightedOrientationEntropy(const cv::Mat& angle, const cv::Mat& mag,
                                  int bins) {
  std::vector<double> hist(bins, 0.0);
  double total = 0.0;

  for (int y = 0; y < angle.rows; ++y) {
    const float* ap = angle.ptr<float>(y);
    const float* mp = mag.ptr<float>(y);
    for (int x = 0; x < angle.cols; ++x) {
      float a = ap[x];
      if (a < 0.0f) {
        a += static_cast<float>(CV_PI);
      }
      const int idx = std::min(
          bins - 1,
          static_cast<int>((a / static_cast<float>(CV_PI)) * static_cast<float>(bins)));
      const double w = mp[x];
      hist[idx] += w;
      total += w;
    }
  }

  if (total <= 1e-12) {
    return 0.0;
  }

  double ent = 0.0;
  for (double v : hist) {
    if (v <= 0.0) {
      continue;
    }
    const double p = v / total;
    ent -= p * std::log2(p);
  }
  return ent / std::log2(static_cast<double>(bins));
}

cv::Mat BuildIlluminationMap(const cv::Mat& gray_f32) {
  cv::Mat illum;
  cv::GaussianBlur(gray_f32, illum, cv::Size(0, 0), 15.0, 15.0);

  cv::Mat illum_u8;
  cv::normalize(illum, illum_u8, 0, 255, cv::NORM_MINMAX, CV_8U);

  cv::Mat color;
  cv::applyColorMap(illum_u8, color, cv::COLORMAP_CIVIDIS);
  return color;
}

cv::Mat BuildEdgeShiftDiagnostic(const cv::Mat& edge_b, const cv::Mat& edge_g,
                                 const cv::Mat& edge_r) {
  cv::Mat b_u8;
  cv::Mat g_u8;
  cv::Mat r_u8;
  cv::normalize(edge_b, b_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::normalize(edge_g, g_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::normalize(edge_r, r_u8, 0, 255, cv::NORM_MINMAX, CV_8U);

  std::vector<cv::Mat> ch = {b_u8, g_u8, r_u8};
  cv::Mat merged;
  cv::merge(ch, merged);

  cv::putText(merged, "B/G/R edge alignment diagnostic", cv::Point(14, 28),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2,
              cv::LINE_AA);
  cv::putText(merged, "white=aligned edges, colored fringes=channel shift",
              cv::Point(14, 54), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

  return merged;
}

double ShiftMagnitude(const cv::Point2d& s) {
  return std::sqrt(s.x * s.x + s.y * s.y);
}

}  // namespace

std::string OpticalPhotometricPlausibilityTest::Id() const {
  return "optical_photometric_plausibility";
}

std::string OpticalPhotometricPlausibilityTest::Name() const {
  return "Optical/Photometric Plausibility Checks";
}

std::string OpticalPhotometricPlausibilityTest::Description() const {
  return "Evaluates illumination coherence and cross-channel edge alignment to "
         "check whether lens/lighting behavior is physically plausible for a "
         "single-camera capture.";
}

TestResult OpticalPhotometricPlausibilityTest::Run(const cv::Mat& bgr_image,
                                                   const TestContext& context) {
  TestResult result;
  result.test_id = Id();
  result.name = Name();
  result.description = Description();

  if (bgr_image.empty()) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.0;
    result.evidence_summary =
        "Input image is empty; optical/photometric audit skipped.";
    return result;
  }

  if (bgr_image.channels() != 3) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary =
        "Expected 3-channel image for optical/photometric consistency checks.";
    return result;
  }

  cv::Mat gray_u8;
  cv::cvtColor(bgr_image, gray_u8, cv::COLOR_BGR2GRAY);
  cv::Mat gray;
  gray_u8.convertTo(gray, CV_32F, 1.0 / 255.0);

  cv::Mat gx;
  cv::Mat gy;
  cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

  cv::Mat mag;
  cv::Mat angle;
  cv::cartToPolar(gx, gy, mag, angle, false);

  const double illum_entropy = WeightedOrientationEntropy(angle, mag, 36);

  cv::Mat illum_map = BuildIlluminationMap(gray);

  std::vector<cv::Mat> bgr_f;
  cv::Mat bgr_float;
  bgr_image.convertTo(bgr_float, CV_32FC3, 1.0 / 255.0);
  cv::split(bgr_float, bgr_f);

  cv::Mat edge_b;
  cv::Mat edge_g;
  cv::Mat edge_r;
  cv::Laplacian(bgr_f[0], edge_b, CV_32F, 3);
  cv::Laplacian(bgr_f[1], edge_g, CV_32F, 3);
  cv::Laplacian(bgr_f[2], edge_r, CV_32F, 3);
  edge_b = cv::abs(edge_b);
  edge_g = cv::abs(edge_g);
  edge_r = cv::abs(edge_r);

  double resp_rg = 0.0;
  double resp_bg = 0.0;
  const cv::Point2d shift_rg = cv::phaseCorrelate(edge_r, edge_g, cv::noArray(), &resp_rg);
  const cv::Point2d shift_bg = cv::phaseCorrelate(edge_b, edge_g, cv::noArray(), &resp_bg);

  const double global_shift_rg = ShiftMagnitude(shift_rg);
  const double global_shift_bg = ShiftMagnitude(shift_bg);

  std::vector<double> tile_shifts;
  const int tile = 96;
  for (int y = 0; y + tile <= edge_g.rows; y += tile) {
    for (int x = 0; x + tile <= edge_g.cols; x += tile) {
      const cv::Rect roi(x, y, tile, tile);
      const cv::Mat rg_r = edge_r(roi);
      const cv::Mat rg_g = edge_g(roi);
      const cv::Mat bg_b = edge_b(roi);

      double tr1 = 0.0;
      double tr2 = 0.0;
      const cv::Point2d s1 = cv::phaseCorrelate(rg_r, rg_g, cv::noArray(), &tr1);
      const cv::Point2d s2 = cv::phaseCorrelate(bg_b, rg_g, cv::noArray(), &tr2);

      if (std::isfinite(s1.x) && std::isfinite(s1.y) && tr1 > 0.01) {
        tile_shifts.push_back(ShiftMagnitude(s1));
      }
      if (std::isfinite(s2.x) && std::isfinite(s2.y) && tr2 > 0.01) {
        tile_shifts.push_back(ShiftMagnitude(s2));
      }
    }
  }

  double shift_var = 0.0;
  double shift_mean = 0.0;
  if (!tile_shifts.empty()) {
    shift_mean =
        std::accumulate(tile_shifts.begin(), tile_shifts.end(), 0.0) / tile_shifts.size();
    for (double v : tile_shifts) {
      const double d = v - shift_mean;
      shift_var += d * d;
    }
    shift_var /= tile_shifts.size();
  }
  const double shift_std = std::sqrt(shift_var);

    const double chroma_inconsistency =
      0.60 * Clamp(shift_std / 0.60, 0.0, 1.0) +
      0.30 * Clamp(shift_mean / 0.26, 0.0, 1.0) +
      0.10 * Clamp((1.98 - (resp_rg + resp_bg)) / 0.22, 0.0, 1.0);

    const double photometric_inconsistency =
      0.35 * Clamp((illum_entropy - 0.64) / 0.12, 0.0, 1.0) +
      0.65 * Clamp(shift_mean / 0.26, 0.0, 1.0);

    const double score = Clamp(
      100.0 * (0.75 * chroma_inconsistency + 0.25 * photometric_inconsistency),
      0.0, 100.0);

  result.score_percent = score;
  result.confidence = Clamp(std::abs(score - 50.0) / 50.0, 0.0, 1.0);
  result.status = score >= 70.0 ? TestStatus::kFail
                                : (score <= 30.0 ? TestStatus::kPass
                                                 : TestStatus::kInconclusive);

  std::ostringstream summary;
  summary << "Illumination entropy=" << illum_entropy
          << ", edge-shift std=" << shift_std
         << ", edge-shift mean=" << shift_mean << ", phase responses=" << resp_rg
         << "/" << resp_bg
         << ". Higher local chromatic edge-shift instability increases AI-likelihood "
           "score.";
  result.evidence_summary = summary.str();

  result.raw_metrics["illumination_entropy_norm"] = illum_entropy;
  result.raw_metrics["global_shift_rg"] = global_shift_rg;
  result.raw_metrics["global_shift_bg"] = global_shift_bg;
  result.raw_metrics["phase_response_rg"] = resp_rg;
  result.raw_metrics["phase_response_bg"] = resp_bg;
  result.raw_metrics["tile_shift_mean"] = shift_mean;
  result.raw_metrics["tile_shift_std"] = shift_std;
  result.raw_metrics["tile_shift_samples"] = static_cast<double>(tile_shifts.size());
  result.raw_metrics["photometric_inconsistency"] = photometric_inconsistency;
  result.raw_metrics["chromatic_inconsistency"] = chroma_inconsistency;

  cv::Mat edge_shift = BuildEdgeShiftDiagnostic(edge_b, edge_g, edge_r);

  std::filesystem::create_directories(context.output_dir);
  const std::string illum_path = context.output_dir + "/optical_illumination_map.png";
  const std::string edge_path = context.output_dir + "/optical_edge_shift_diagnostic.png";
  cv::imwrite(illum_path, illum_map);
  cv::imwrite(edge_path, edge_shift);

  result.artifact_paths.push_back(illum_path);
  result.artifact_paths.push_back(edge_path);

  return result;
}

}  // namespace pixelaudit
