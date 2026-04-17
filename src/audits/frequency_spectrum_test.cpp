#include "pixelaudit/tests/frequency_spectrum_test.hpp"

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

void ShiftDftQuadrants(cv::Mat* mag) {
  const int cx = mag->cols / 2;
  const int cy = mag->rows / 2;

  cv::Mat q0(*mag, cv::Rect(0, 0, cx, cy));
  cv::Mat q1(*mag, cv::Rect(cx, 0, cx, cy));
  cv::Mat q2(*mag, cv::Rect(0, cy, cx, cy));
  cv::Mat q3(*mag, cv::Rect(cx, cy, cx, cy));

  cv::Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

cv::Mat BuildRadialProfilePlot(const std::vector<double>& profile,
                               const std::vector<double>& smooth_profile,
                               const double low_edge, const double mid_edge,
                               const double high_edge) {
  const int width = 820;
  const int height = 420;
  const int margin = 52;

  cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(250, 250, 250));
  cv::rectangle(canvas, cv::Point(margin, margin),
                cv::Point(width - margin, height - margin),
                cv::Scalar(220, 220, 220), 1);

  if (profile.empty()) {
    return canvas;
  }

  const double max_v =
      std::max(1e-8, *std::max_element(profile.begin(), profile.end()));

  auto to_x = [&](double r) {
    return margin + static_cast<int>(
                        r * static_cast<double>(width - 2 * margin - 1));
  };
  auto to_y = [&](double v) {
    const double nv = Clamp(v / max_v, 0.0, 1.0);
    return height - margin -
           static_cast<int>(nv * static_cast<double>(height - 2 * margin - 1));
  };

  for (std::size_t i = 1; i < profile.size(); ++i) {
    const double r0 = static_cast<double>(i - 1) / static_cast<double>(profile.size() - 1);
    const double r1 = static_cast<double>(i) / static_cast<double>(profile.size() - 1);
    cv::line(canvas, cv::Point(to_x(r0), to_y(profile[i - 1])),
             cv::Point(to_x(r1), to_y(profile[i])), cv::Scalar(190, 190, 190), 1,
             cv::LINE_AA);
  }

  for (std::size_t i = 1; i < smooth_profile.size(); ++i) {
    const double r0 = static_cast<double>(i - 1) /
                      static_cast<double>(smooth_profile.size() - 1);
    const double r1 =
        static_cast<double>(i) / static_cast<double>(smooth_profile.size() - 1);
    cv::line(canvas, cv::Point(to_x(r0), to_y(smooth_profile[i - 1])),
             cv::Point(to_x(r1), to_y(smooth_profile[i])), cv::Scalar(40, 95, 220),
             2, cv::LINE_AA);
  }

  cv::line(canvas, cv::Point(to_x(low_edge), margin),
           cv::Point(to_x(low_edge), height - margin), cv::Scalar(230, 180, 60), 1,
           cv::LINE_AA);
  cv::line(canvas, cv::Point(to_x(mid_edge), margin),
           cv::Point(to_x(mid_edge), height - margin), cv::Scalar(230, 180, 60), 1,
           cv::LINE_AA);
  cv::line(canvas, cv::Point(to_x(high_edge), margin),
           cv::Point(to_x(high_edge), height - margin), cv::Scalar(230, 180, 60), 1,
           cv::LINE_AA);

  cv::putText(canvas, "Radial Frequency Energy Profile", cv::Point(56, 34),
              cv::FONT_HERSHEY_SIMPLEX, 0.68, cv::Scalar(55, 55, 55), 1,
              cv::LINE_AA);
  cv::putText(canvas, "r=0 (low freq) -> r=1 (high freq)",
              cv::Point(56, height - 18), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(110, 110, 110), 1, cv::LINE_AA);

  return canvas;
}

}  // namespace

std::string FrequencySpectrumTest::Id() const { return "frequency_spectrum"; }

std::string FrequencySpectrumTest::Name() const {
  return "Frequency Spectrum and Ringing Analysis";
}

std::string FrequencySpectrumTest::Description() const {
  return "Uses Fourier-domain statistics to measure unusual frequency energy "
         "distribution, emphasizing high-frequency excess, radial imbalance, and "
         "periodic ringing signatures that can appear in synthetic image pipelines.";
}

TestResult FrequencySpectrumTest::Run(const cv::Mat& bgr_image,
                                      const TestContext& context) {
  TestResult result;
  result.test_id = Id();
  result.name = Name();
  result.description = Description();

  if (bgr_image.empty()) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.0;
    result.evidence_summary = "Input image is empty; spectrum analysis skipped.";
    return result;
  }

  cv::Mat gray_u8;
  if (bgr_image.channels() == 3) {
    cv::cvtColor(bgr_image, gray_u8, cv::COLOR_BGR2GRAY);
  } else if (bgr_image.channels() == 1) {
    gray_u8 = bgr_image;
  } else {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary = "Unsupported channel layout for spectrum analysis.";
    return result;
  }

  cv::Mat gray;
  gray_u8.convertTo(gray, CV_32F, 1.0 / 255.0);

  const int optimal_rows = cv::getOptimalDFTSize(gray.rows);
  const int optimal_cols = cv::getOptimalDFTSize(gray.cols);

  cv::Mat padded;
  cv::copyMakeBorder(gray, padded, 0, optimal_rows - gray.rows, 0,
                     optimal_cols - gray.cols, cv::BORDER_CONSTANT,
                     cv::Scalar::all(0));

  cv::Mat planes[] = {padded.clone(), cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complex;
  cv::merge(planes, 2, complex);
  cv::dft(complex, complex);

  cv::split(complex, planes);
  cv::Mat mag;
  cv::magnitude(planes[0], planes[1], mag);
  mag += 1.0f;
  cv::log(mag, mag);

  cv::Mat mag_even = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2)).clone();
  ShiftDftQuadrants(&mag_even);

  cv::Mat mag_vis_u8;
  cv::normalize(mag_even, mag_vis_u8, 0, 255, cv::NORM_MINMAX, CV_8U);

  cv::Mat spectrum_artifact;
  cv::applyColorMap(mag_vis_u8, spectrum_artifact, cv::COLORMAP_INFERNO);

  const cv::Point2f center(static_cast<float>(mag_even.cols) * 0.5f,
                           static_cast<float>(mag_even.rows) * 0.5f);
  const float max_radius =
      std::min(static_cast<float>(mag_even.cols), static_cast<float>(mag_even.rows)) *
      0.5f;
  const int bins = std::max(80, static_cast<int>(max_radius));

  std::vector<double> radial_sum(bins, 0.0);
  std::vector<double> radial_count(bins, 0.0);

  for (int y = 0; y < mag_even.rows; ++y) {
    const float* row = mag_even.ptr<float>(y);
    for (int x = 0; x < mag_even.cols; ++x) {
      const float dx = static_cast<float>(x) - center.x;
      const float dy = static_cast<float>(y) - center.y;
      const float r = std::sqrt(dx * dx + dy * dy);
      const int bin = static_cast<int>(std::floor((r / max_radius) * (bins - 1)));
      if (bin < 0 || bin >= bins) {
        continue;
      }
      radial_sum[bin] += row[x];
      radial_count[bin] += 1.0;
    }
  }

  std::vector<double> radial_profile(bins, 0.0);
  for (int i = 0; i < bins; ++i) {
    radial_profile[i] = radial_count[i] > 0.0 ? radial_sum[i] / radial_count[i] : 0.0;
  }

  std::vector<double> smooth_profile(bins, 0.0);
  constexpr int kWin = 3;
  for (int i = 0; i < bins; ++i) {
    int c = 0;
    double s = 0.0;
    for (int k = -kWin; k <= kWin; ++k) {
      const int j = i + k;
      if (j < 0 || j >= bins) {
        continue;
      }
      s += radial_profile[j];
      ++c;
    }
    smooth_profile[i] = c > 0 ? s / static_cast<double>(c) : 0.0;
  }

  const int low_idx = static_cast<int>(0.20 * bins);
  const int mid_idx = static_cast<int>(0.55 * bins);
  const int high_idx = static_cast<int>(0.85 * bins);

  double low_energy = 0.0;
  double mid_energy = 0.0;
  double high_energy = 0.0;
  double total_energy = 0.0;

  for (int i = 0; i < bins; ++i) {
    const double v = std::max(0.0, smooth_profile[i]);
    total_energy += v;
    if (i <= low_idx) {
      low_energy += v;
    } else if (i <= mid_idx) {
      mid_energy += v;
    } else {
      high_energy += v;
    }
  }

  const double high_ratio = high_energy / (total_energy + 1e-9);
  const double low_ratio = low_energy / (total_energy + 1e-9);

  // Ringing proxy: measure oscillation in upper-mid to high radial bands.
  double ringing_acc = 0.0;
  int ringing_n = 0;
  for (int i = mid_idx + 2; i < bins - 2; ++i) {
    const double d2 = std::abs(smooth_profile[i + 1] - 2.0 * smooth_profile[i] +
                               smooth_profile[i - 1]);
    ringing_acc += d2;
    ++ringing_n;
  }
  const double ringing_strength =
      ringing_n > 0 ? ringing_acc / static_cast<double>(ringing_n) : 0.0;

  // Directional anisotropy in Fourier domain.
  const int wedge_bins = 36;
  std::vector<double> angle_energy(wedge_bins, 0.0);
  for (int y = 0; y < mag_even.rows; ++y) {
    const float* row = mag_even.ptr<float>(y);
    for (int x = 0; x < mag_even.cols; ++x) {
      const float dx = static_cast<float>(x) - center.x;
      const float dy = static_cast<float>(y) - center.y;
      const float r = std::sqrt(dx * dx + dy * dy);
      if (r < 0.15f * max_radius || r > 0.90f * max_radius) {
        continue;
      }
      float theta = std::atan2(dy, dx);
      if (theta < 0.0f) {
        theta += static_cast<float>(2.0 * CV_PI);
      }
      const int ab = std::min(wedge_bins - 1,
                              static_cast<int>((theta / (2.0f * CV_PI)) * wedge_bins));
      angle_energy[ab] += row[x];
    }
  }
  const auto [amin_it, amax_it] = std::minmax_element(angle_energy.begin(), angle_energy.end());
  const double anisotropy =
      (*amax_it - *amin_it) / (std::accumulate(angle_energy.begin(), angle_energy.end(), 0.0) /
                                   static_cast<double>(wedge_bins) +
                               1e-9);

  const double score = Clamp(
      100.0 * (0.50 * Clamp(high_ratio / 0.40, 0.0, 1.0) +
               0.30 * Clamp(ringing_strength / 0.020, 0.0, 1.0) +
               0.20 * Clamp(anisotropy / 1.6, 0.0, 1.0)),
      0.0, 100.0);

  result.score_percent = score;
  result.confidence = Clamp(std::abs(score - 50.0) / 50.0, 0.0, 1.0);
  result.status = score >= 70.0 ? TestStatus::kFail
                                : (score <= 30.0 ? TestStatus::kPass
                                                 : TestStatus::kInconclusive);

  std::ostringstream summary;
  summary << "Frequency evidence from log-spectrum: high-frequency ratio=" << high_ratio
          << ", ringing proxy=" << ringing_strength << ", anisotropy=" << anisotropy
          << ". Elevated high-frequency concentration and oscillation increase "
             "AI-likelihood score.";
  result.evidence_summary = summary.str();

  result.raw_metrics["low_band_ratio"] = low_ratio;
  result.raw_metrics["mid_band_ratio"] = mid_energy / (total_energy + 1e-9);
  result.raw_metrics["high_band_ratio"] = high_ratio;
  result.raw_metrics["ringing_strength"] = ringing_strength;
  result.raw_metrics["directional_anisotropy"] = anisotropy;
  result.raw_metrics["profile_bins"] = static_cast<double>(bins);

  cv::Mat radial_plot = BuildRadialProfilePlot(
      radial_profile, smooth_profile, static_cast<double>(low_idx) / bins,
      static_cast<double>(mid_idx) / bins, static_cast<double>(high_idx) / bins);

  std::filesystem::create_directories(context.output_dir);
  const std::string spectrum_path = context.output_dir + "/frequency_log_spectrum.png";
  const std::string profile_path = context.output_dir + "/frequency_radial_profile.png";

  cv::imwrite(spectrum_path, spectrum_artifact);
  cv::imwrite(profile_path, radial_plot);

  result.artifact_paths.push_back(spectrum_path);
  result.artifact_paths.push_back(profile_path);

  return result;
}

}  // namespace pixelaudit
