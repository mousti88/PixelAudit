#include "pixelaudit/tests/jpeg_forensics_test.hpp"

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

cv::Mat BuildHistogramPlot(const std::vector<double>& hist,
                           const std::string& title) {
  const int width = 900;
  const int height = 440;
  const int margin = 56;

  cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(248, 248, 248));
  cv::rectangle(canvas, cv::Point(margin, margin),
                cv::Point(width - margin, height - margin),
                cv::Scalar(220, 220, 220), 1);

  if (hist.empty()) {
    cv::putText(canvas, title, cv::Point(58, 34), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
    return canvas;
  }

  const double max_v = std::max(1e-9, *std::max_element(hist.begin(), hist.end()));
  const int plot_w = width - 2 * margin;
  const int plot_h = height - 2 * margin;

  const int n = static_cast<int>(hist.size());
  const double bar_w = static_cast<double>(plot_w) / n;

  for (int i = 0; i < n; ++i) {
    const double nv = Clamp(hist[i] / max_v, 0.0, 1.0);
    const int x0 = margin + static_cast<int>(std::floor(i * bar_w));
    const int x1 = margin + static_cast<int>(std::floor((i + 1) * bar_w));
    const int y1 = height - margin;
    const int y0 = y1 - static_cast<int>(nv * plot_h);
    cv::rectangle(canvas, cv::Point(x0, y0), cv::Point(std::max(x0 + 1, x1 - 1), y1),
                  cv::Scalar(45, 130, 220), cv::FILLED);
  }

  cv::putText(canvas, title, cv::Point(58, 34), cv::FONT_HERSHEY_SIMPLEX, 0.65,
              cv::Scalar(50, 50, 50), 1, cv::LINE_AA);
  cv::putText(canvas, "abs(DCT AC coefficient) bins", cv::Point(58, height - 16),
              cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(90, 90, 90), 1,
              cv::LINE_AA);

  return canvas;
}

double HistogramPeriodicity(const std::vector<double>& hist, int min_lag,
                            int max_lag) {
  if (hist.size() < static_cast<std::size_t>(max_lag + 2)) {
    return 0.0;
  }

  const double mean =
      std::accumulate(hist.begin(), hist.end(), 0.0) / std::max<std::size_t>(1, hist.size());

  std::vector<double> centered(hist.size(), 0.0);
  for (std::size_t i = 0; i < hist.size(); ++i) {
    centered[i] = hist[i] - mean;
  }

  double var = 0.0;
  for (double v : centered) {
    var += v * v;
  }
  if (var <= 1e-12) {
    return 0.0;
  }

  double best = 0.0;
  for (int lag = min_lag; lag <= max_lag; ++lag) {
    double corr = 0.0;
    for (std::size_t i = 0; i + lag < centered.size(); ++i) {
      corr += centered[i] * centered[i + lag];
    }
    best = std::max(best, std::abs(corr) / var);
  }
  return best;
}

int FirstDigitFromScaledMagnitude(const double value) {
  int v = static_cast<int>(std::round(std::abs(value) * 1000.0));
  if (v <= 0) {
    return 0;
  }
  while (v >= 10) {
    v /= 10;
  }
  return v;
}

double BenfordChiSquare(const std::vector<double>& observed) {
  if (observed.size() != 9) {
    return 0.0;
  }
  const double s = std::accumulate(observed.begin(), observed.end(), 0.0);
  if (s <= 1e-12) {
    return 0.0;
  }
  double chi2 = 0.0;
  for (int d = 1; d <= 9; ++d) {
    const double p_obs = observed[d - 1] / s;
    const double p_exp = std::log10(1.0 + 1.0 / static_cast<double>(d));
    chi2 += ((p_obs - p_exp) * (p_obs - p_exp)) / (p_exp + 1e-12);
  }
  return chi2;
}

}  // namespace

std::string JpegCompressionForensicsTest::Id() const {
  return "jpeg_compression_forensics";
}

std::string JpegCompressionForensicsTest::Name() const {
  return "JPEG and Compression Forensics";
}

std::string JpegCompressionForensicsTest::Description() const {
  return "Analyzes 8x8 block-boundary discontinuities and DCT coefficient "
         "statistics to detect compression and quantization inconsistencies that are "
         "often atypical in camera-native photographs.";
}

TestResult JpegCompressionForensicsTest::Run(const cv::Mat& bgr_image,
                                             const TestContext& context) {
  TestResult result;
  result.test_id = Id();
  result.name = Name();
  result.description = Description();

  if (bgr_image.empty()) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.0;
    result.evidence_summary = "Input image is empty; JPEG forensic audit skipped.";
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
    result.evidence_summary = "Unsupported channel layout for JPEG forensic audit.";
    return result;
  }

  cv::Mat gray;
  gray_u8.convertTo(gray, CV_32F, 1.0 / 255.0);

  // Block-boundary discontinuity map and metrics.
  cv::Mat block_map(gray.size(), CV_32F, cv::Scalar(0.0f));
  double boundary_sum = 0.0;
  double interior_sum = 0.0;
  int boundary_count = 0;
  int interior_count = 0;

  for (int y = 1; y < gray.rows - 1; ++y) {
    const float* row = gray.ptr<float>(y);
    const float* row_prev = gray.ptr<float>(y - 1);
    for (int x = 1; x < gray.cols - 1; ++x) {
      const float d_left = std::abs(row[x] - row[x - 1]);
      const float d_up = std::abs(row[x] - row_prev[x]);
      const float diff = 0.5f * (d_left + d_up);

      const bool is_block_boundary = ((x % 8) == 0) || ((y % 8) == 0);
      const bool is_interior_ref = ((x % 8) == 4) || ((y % 8) == 4);

      if (is_block_boundary) {
        block_map.at<float>(y, x) = diff;
        boundary_sum += diff;
        ++boundary_count;
      } else if (is_interior_ref) {
        interior_sum += diff;
        ++interior_count;
      }
    }
  }

  if (boundary_count < 50 || interior_count < 50) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary =
        "Insufficient pixels for stable 8x8 boundary forensic analysis.";
    return result;
  }

  const double boundary_mean = boundary_sum / boundary_count;
  const double interior_mean = interior_sum / interior_count;
  const double boundary_ratio = boundary_mean / (interior_mean + 1e-9);

  // DCT-statistics over 8x8 blocks.
  const int rows8 = (gray.rows / 8) * 8;
  const int cols8 = (gray.cols / 8) * 8;
  if (rows8 < 16 || cols8 < 16) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary = "Image too small for DCT block-statistic analysis.";
    return result;
  }

  cv::Mat cropped = gray(cv::Rect(0, 0, cols8, rows8)).clone();

  constexpr int kHistBins = 64;
  constexpr double kMaxCoeff = 0.5;
  std::vector<double> dct_hist(kHistBins, 0.0);
  std::vector<double> benford_digits(9, 0.0);

  double low_energy = 0.0;
  double mid_energy = 0.0;
  double high_energy = 0.0;
  int high_count = 0;
  int high_near_zero = 0;
  int block_count = 0;

  cv::Mat block(8, 8, CV_32F);
  cv::Mat coeff(8, 8, CV_32F);

  for (int y = 0; y < rows8; y += 8) {
    for (int x = 0; x < cols8; x += 8) {
      cropped(cv::Rect(x, y, 8, 8)).copyTo(block);
      block -= 0.5f;
      cv::dct(block, coeff);

      for (int v = 0; v < 8; ++v) {
        const float* crow = coeff.ptr<float>(v);
        for (int u = 0; u < 8; ++u) {
          if (u == 0 && v == 0) {
            continue;
          }
          const double av = std::abs(crow[u]);
          const int freq_idx = u + v;

          if (freq_idx <= 2) {
            low_energy += av;
          } else if (freq_idx <= 6) {
            mid_energy += av;
          } else {
            high_energy += av;
            ++high_count;
            if (av < 0.010) {
              ++high_near_zero;
            }
          }

          const int bin = std::min(
              kHistBins - 1,
              static_cast<int>(std::floor((std::min(av, kMaxCoeff) / kMaxCoeff) *
                                          static_cast<double>(kHistBins - 1))));
          dct_hist[bin] += 1.0;

          const int first_digit = FirstDigitFromScaledMagnitude(av);
          if (first_digit >= 1 && first_digit <= 9) {
            benford_digits[first_digit - 1] += 1.0;
          }
        }
      }
      ++block_count;
    }
  }

  if (block_count < 4 || high_count == 0) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary =
        "Insufficient DCT samples for compression forensic scoring.";
    return result;
  }

  const double total_hist = std::accumulate(dct_hist.begin(), dct_hist.end(), 0.0);
  if (total_hist > 0.0) {
    for (double& v : dct_hist) {
      v /= total_hist;
    }
  }

  const double high_low_ratio = high_energy / (low_energy + 1e-9);
  const double high_zero_ratio =
      static_cast<double>(high_near_zero) / static_cast<double>(high_count);
  const double hist_periodicity = HistogramPeriodicity(dct_hist, 2, 10);
    const double benford_chi2 = BenfordChiSquare(benford_digits);

  const double score = Clamp(
      100.0 * (0.28 * Clamp((boundary_ratio - 1.05) / 0.35, 0.0, 1.0) +
           0.24 * Clamp((high_zero_ratio - 0.90) / 0.10, 0.0, 1.0) +
           0.18 * Clamp((0.06 - hist_periodicity) / 0.06, 0.0, 1.0) +
           0.15 * Clamp((0.65 - high_low_ratio) / 0.45, 0.0, 1.0) +
           0.15 * Clamp(benford_chi2 / 0.30, 0.0, 1.0)),
      0.0, 100.0);

  result.score_percent = score;
  result.confidence = Clamp(std::abs(score - 50.0) / 50.0, 0.0, 1.0);
  result.status = score >= 70.0 ? TestStatus::kFail
                                : (score <= 30.0 ? TestStatus::kPass
                                                 : TestStatus::kInconclusive);

  std::ostringstream summary;
  summary << "Block-boundary ratio=" << boundary_ratio
          << ", high-frequency near-zero ratio=" << high_zero_ratio
      << ", DCT histogram periodicity=" << hist_periodicity
      << ", Benford chi2=" << benford_chi2
          << ". Strong 8x8 boundary contrast and quantization-like DCT signatures "
             "increase AI-likelihood score.";
  result.evidence_summary = summary.str();

  result.raw_metrics["block_boundary_mean"] = boundary_mean;
  result.raw_metrics["block_interior_mean"] = interior_mean;
  result.raw_metrics["block_boundary_ratio"] = boundary_ratio;
  result.raw_metrics["dct_high_zero_ratio"] = high_zero_ratio;
  result.raw_metrics["dct_hist_periodicity"] = hist_periodicity;
  result.raw_metrics["dct_benford_chi2"] = benford_chi2;
  result.raw_metrics["dct_high_low_ratio"] = high_low_ratio;
  result.raw_metrics["dct_low_energy"] = low_energy;
  result.raw_metrics["dct_mid_energy"] = mid_energy;
  result.raw_metrics["dct_high_energy"] = high_energy;
  result.raw_metrics["block_count"] = static_cast<double>(block_count);

  cv::Mat block_u8;
  cv::normalize(block_map, block_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::Mat block_heatmap;
  cv::applyColorMap(block_u8, block_heatmap, cv::COLORMAP_TURBO);

  cv::Mat hist_plot = BuildHistogramPlot(dct_hist, "DCT AC Coefficient Histogram");

  std::filesystem::create_directories(context.output_dir);
  const std::string block_path = context.output_dir + "/jpeg_block_heatmap.png";
  const std::string hist_path = context.output_dir + "/jpeg_dct_histogram.png";
  cv::imwrite(block_path, block_heatmap);
  cv::imwrite(hist_path, hist_plot);

  result.artifact_paths.push_back(block_path);
  result.artifact_paths.push_back(hist_path);

  return result;
}

}  // namespace pixelaudit
