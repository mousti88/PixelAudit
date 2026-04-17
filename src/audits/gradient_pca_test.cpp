#include "pixelaudit/tests/gradient_pca_test.hpp"

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

cv::Mat BuildProjectionPlot(const cv::Mat& projected) {
  const int width = 640;
  const int height = 420;
  cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

  if (projected.rows <= 0 || projected.cols < 2) {
    return canvas;
  }

  double min_x = 0.0;
  double max_x = 0.0;
  double min_y = 0.0;
  double max_y = 0.0;
  cv::minMaxLoc(projected.col(0), &min_x, &max_x);
  cv::minMaxLoc(projected.col(1), &min_y, &max_y);

  const double x_span = std::max(1e-6, max_x - min_x);
  const double y_span = std::max(1e-6, max_y - min_y);

  for (int i = 0; i < projected.rows; ++i) {
    const float px = projected.at<float>(i, 0);
    const float py = projected.at<float>(i, 1);
    const int x = static_cast<int>(40 + ((px - min_x) / x_span) * (width - 80));
    const int y = static_cast<int>(height - 40 - ((py - min_y) / y_span) * (height - 80));
    cv::circle(canvas, cv::Point(x, y), 2, cv::Scalar(36, 99, 193), cv::FILLED);
  }

  cv::rectangle(canvas, cv::Point(40, 20), cv::Point(width - 40, height - 40),
                cv::Scalar(220, 220, 220), 1);
  cv::putText(canvas, "Patch PCA Projection (PC1 vs PC2)", cv::Point(45, 35),
              cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(40, 40, 40), 1,
              cv::LINE_AA);

  return canvas;
}

cv::Mat BuildOrientationVisualization(const cv::Mat& angle_rad,
                                      const cv::Mat& magnitude) {
  cv::Mat angle_deg;
  angle_rad.convertTo(angle_deg, CV_32F, 180.0 / CV_PI);

  cv::Mat hue_u8;
  // OpenCV HSV hue in 8-bit images is [0, 179].
  angle_deg.convertTo(hue_u8, CV_8U, 0.5);

  cv::Mat hsv[3];
  hsv[0] = hue_u8;

  cv::Mat mag_norm;
  cv::normalize(magnitude, mag_norm, 0.0, 255.0, cv::NORM_MINMAX);
  mag_norm.convertTo(hsv[1], CV_8U);
  hsv[2] = cv::Mat(magnitude.size(), CV_8U, cv::Scalar(255));

  cv::Mat hsv_merged;
  cv::merge(hsv, 3, hsv_merged);

  cv::Mat bgr;
  cv::cvtColor(hsv_merged, bgr, cv::COLOR_HSV2BGR);
  return bgr;
}

double ComputeEntropy(const cv::Mat& angle_rad, const cv::Mat& magnitude,
                      const int bins) {
  std::vector<double> hist(bins, 0.0);
  double total_weight = 0.0;

  for (int y = 0; y < angle_rad.rows; ++y) {
    const float* angle_ptr = angle_rad.ptr<float>(y);
    const float* mag_ptr = magnitude.ptr<float>(y);
    for (int x = 0; x < angle_rad.cols; ++x) {
      float angle = angle_ptr[x];
      if (angle < 0.0f) {
        angle += static_cast<float>(CV_PI);
      }
      const int idx = std::min(bins - 1,
                               static_cast<int>((angle / CV_PI) * static_cast<float>(bins)));
      const double weight = static_cast<double>(mag_ptr[x]);
      hist[idx] += weight;
      total_weight += weight;
    }
  }

  if (total_weight <= 1e-9) {
    return 0.0;
  }

  double entropy = 0.0;
  for (const double count : hist) {
    if (count <= 0.0) {
      continue;
    }
    const double p = count / total_weight;
    entropy -= p * std::log2(p);
  }

  return entropy / std::log2(static_cast<double>(bins));
}

}  // namespace

std::string LuminanceGradientPcaTest::Id() const { return "gradient_pca"; }

std::string LuminanceGradientPcaTest::Name() const {
  return "Luminance-Gradient PCA Analysis";
}

std::string LuminanceGradientPcaTest::Description() const {
  return "Analyzes luminance gradient coherence and patch-level PCA structure to "
         "identify denoising-related instability patterns common in synthetic images.";
}

TestResult LuminanceGradientPcaTest::Run(const cv::Mat& bgr_image,
                                         const TestContext& context) {
  TestResult result;
  result.test_id = Id();
  result.name = Name();
  result.description = Description();

  if (bgr_image.empty()) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.0;
    result.evidence_summary = "Input image is empty; gradient analysis skipped.";
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
    result.evidence_summary = "Unsupported channel layout for luminance conversion.";
    return result;
  }

  cv::Mat gray;
  gray_u8.convertTo(gray, CV_32F, 1.0 / 255.0);

  cv::Mat gx;
  cv::Mat gy;
  cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

  cv::Mat magnitude;
  cv::Mat angle;
  cv::cartToPolar(gx, gy, magnitude, angle, false);

  const double orientation_entropy = ComputeEntropy(angle, magnitude, 36);
  cv::Scalar mag_mean;
  cv::Scalar mag_std;
  cv::meanStdDev(magnitude, mag_mean, mag_std);

  cv::Mat lap;
  cv::Laplacian(gray, lap, CV_32F, 3);
  cv::Scalar lap_mean;
  cv::Scalar lap_std;
  cv::meanStdDev(lap, lap_mean, lap_std);

  constexpr int patch_size = 16;
  const int usable_rows = (gray.rows / patch_size) * patch_size;
  const int usable_cols = (gray.cols / patch_size) * patch_size;

  std::vector<cv::Vec4f> features;
  features.reserve((usable_rows / patch_size) * (usable_cols / patch_size));

  for (int y = 0; y < usable_rows; y += patch_size) {
    for (int x = 0; x < usable_cols; x += patch_size) {
      const cv::Rect roi(x, y, patch_size, patch_size);
      const cv::Mat mag_patch = magnitude(roi);
      const cv::Mat gx_patch = gx(roi);
      const cv::Mat gy_patch = gy(roi);

      cv::Scalar p_mean;
      cv::Scalar p_std;
      cv::meanStdDev(mag_patch, p_mean, p_std);

      const double gx_e = cv::norm(gx_patch, cv::NORM_L2);
      const double gy_e = cv::norm(gy_patch, cv::NORM_L2);
      const double anisotropy = std::abs(gx_e - gy_e) / (gx_e + gy_e + 1e-6);
      const double coherence = p_std[0] / (p_mean[0] + 1e-6);

      features.emplace_back(static_cast<float>(p_mean[0]),
                            static_cast<float>(p_std[0]),
                            static_cast<float>(anisotropy),
                            static_cast<float>(coherence));
    }
  }

  if (features.size() < 4) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary =
        "Image is too small for stable patch PCA analysis at 16x16 granularity.";
    return result;
  }

  cv::Mat feature_mat(static_cast<int>(features.size()), 4, CV_32F);
  for (int i = 0; i < feature_mat.rows; ++i) {
    float* row = feature_mat.ptr<float>(i);
    row[0] = features[i][0];
    row[1] = features[i][1];
    row[2] = features[i][2];
    row[3] = features[i][3];
  }

  cv::PCA pca(feature_mat, cv::Mat(), cv::PCA::DATA_AS_ROW, 2);
  cv::Mat projected;
  pca.project(feature_mat, projected);

  cv::Scalar pc1_mean;
  cv::Scalar pc1_std;
  cv::meanStdDev(projected.col(0), pc1_mean, pc1_std);

  cv::Mat covar;
  cv::Mat mean;
  cv::calcCovarMatrix(feature_mat, covar, mean,
                      cv::COVAR_ROWS | cv::COVAR_NORMAL, CV_64F);

  cv::Mat eigenvals;
  cv::eigen(covar, eigenvals);
  const double lambda0 = eigenvals.at<double>(0, 0);
  const double lambda1 = eigenvals.rows > 1 ? eigenvals.at<double>(1, 0) : 1e-6;
  const double pca_ratio = lambda0 / (lambda1 + 1e-6);

  const double instability =
      0.45 * orientation_entropy +
      0.25 * Clamp(lap_std[0] / 0.2, 0.0, 1.0) +
      0.20 * Clamp(pc1_std[0] / 0.4, 0.0, 1.0) +
      0.10 * Clamp((3.0 - pca_ratio) / 3.0, 0.0, 1.0);

  const double score = Clamp(instability * 100.0, 0.0, 100.0);

  result.score_percent = score;
  result.confidence = Clamp(std::abs(score - 50.0) / 50.0, 0.0, 1.0);
  result.status = score >= 70.0 ? TestStatus::kFail
                                : (score <= 30.0 ? TestStatus::kPass
                                                 : TestStatus::kInconclusive);

  std::ostringstream summary;
  summary << "Gradient orientation entropy=" << orientation_entropy
          << ", Laplacian std=" << lap_std[0]
          << ", patch-PC1 std=" << pc1_std[0]
          << ". Higher instability patterns increase AI-likelihood score.";
  result.evidence_summary = summary.str();

  result.raw_metrics["orientation_entropy_norm"] = orientation_entropy;
  result.raw_metrics["gradient_mean"] = mag_mean[0];
  result.raw_metrics["gradient_std"] = mag_std[0];
  result.raw_metrics["laplacian_std"] = lap_std[0];
  result.raw_metrics["pc1_std"] = pc1_std[0];
  result.raw_metrics["pca_lambda_ratio"] = pca_ratio;
  result.raw_metrics["patch_count"] = static_cast<double>(features.size());

  std::filesystem::create_directories(context.output_dir);

  cv::Mat mag_vis;
  cv::normalize(magnitude, mag_vis, 0.0, 255.0, cv::NORM_MINMAX);
  mag_vis.convertTo(mag_vis, CV_8U);

  cv::Mat orientation_vis = BuildOrientationVisualization(angle, magnitude);
  cv::Mat pca_plot = BuildProjectionPlot(projected);

  const std::string mag_path = context.output_dir + "/gradient_magnitude.png";
  const std::string ori_path = context.output_dir + "/gradient_orientation.png";
  const std::string pca_path = context.output_dir + "/gradient_pca_projection.png";

  cv::imwrite(mag_path, mag_vis);
  cv::imwrite(ori_path, orientation_vis);
  cv::imwrite(pca_path, pca_plot);

  result.artifact_paths.push_back(mag_path);
  result.artifact_paths.push_back(ori_path);
  result.artifact_paths.push_back(pca_path);

  return result;
}

}  // namespace pixelaudit
