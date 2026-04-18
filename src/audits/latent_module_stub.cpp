#include "pixelaudit/tests/latent_module_stub.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pixelaudit {
namespace {

double Clamp(const double value, const double lo, const double hi) {
  return std::max(lo, std::min(hi, value));
}

}  // namespace

std::string LatentManifoldReconstructionTest::Id() const {
  return "latent_manifold_optional";
}

std::string LatentManifoldReconstructionTest::Name() const {
  return "Latent-Manifold Reconstruction Error (Optional Non-Classical)";
}

std::string LatentManifoldReconstructionTest::Description() const {
  return "Projects local luminance patches onto a low-dimensional latent manifold "
         "(PCA proxy for autoencoder latent space), reconstructs the image, and "
         "scores manifold fit from reconstruction error. This audit is explicitly "
         "non-classical and optional.";
}

TestResult LatentManifoldReconstructionTest::Run(const cv::Mat& bgr_image,
                                                 const TestContext& context) {

  TestResult result;
  result.test_id = Id();
  result.name = Name();
  result.description = Description();
  result.is_non_classical = true;

  if (bgr_image.empty()) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.0;
    result.evidence_summary =
        "Input image is empty; latent-manifold reconstruction skipped.";
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
    result.evidence_summary =
        "Unsupported channel layout for latent-manifold reconstruction.";
    return result;
  }

  const int max_side = 384;
  cv::Mat work_u8 = gray_u8;
  if (std::max(gray_u8.cols, gray_u8.rows) > max_side) {
    const double scale =
        static_cast<double>(max_side) / static_cast<double>(std::max(gray_u8.cols, gray_u8.rows));
    cv::resize(gray_u8, work_u8, cv::Size(), scale, scale, cv::INTER_AREA);
  }

  cv::Mat work;
  work_u8.convertTo(work, CV_32F, 1.0 / 255.0);

  constexpr int patch = 8;
  constexpr int stride = 4;
  const int rows = work.rows;
  const int cols = work.cols;

  std::vector<cv::Point> patch_origins;
  for (int y = 0; y + patch <= rows; y += stride) {
    for (int x = 0; x + patch <= cols; x += stride) {
      patch_origins.emplace_back(x, y);
    }
  }

  if (patch_origins.size() < 40) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary =
        "Image too small for stable latent patch-manifold reconstruction.";
    return result;
  }

  cv::Mat features(static_cast<int>(patch_origins.size()), patch * patch, CV_32F);
  for (int i = 0; i < features.rows; ++i) {
    const cv::Rect roi(patch_origins[i].x, patch_origins[i].y, patch, patch);
    const cv::Mat p = work(roi);
    cv::Mat row = features.row(i);
    int k = 0;
    for (int yy = 0; yy < patch; ++yy) {
      const float* pr = p.ptr<float>(yy);
      float* rr = row.ptr<float>(0);
      for (int xx = 0; xx < patch; ++xx) {
        rr[k++] = pr[xx];
      }
    }
  }

  const int max_components = 24;
  const int components = std::max(4, std::min(max_components, features.rows - 1));
  cv::PCA pca(features, cv::Mat(), cv::PCA::DATA_AS_ROW, components);

  cv::Mat projected;
  pca.project(features, projected);
  cv::Mat reconstructed_features;
  pca.backProject(projected, reconstructed_features);

  cv::Mat recon_sum(work.size(), CV_32F, cv::Scalar(0.0f));
  cv::Mat recon_count(work.size(), CV_32F, cv::Scalar(0.0f));

  for (int i = 0; i < reconstructed_features.rows; ++i) {
    const float* row = reconstructed_features.ptr<float>(i);
    const cv::Point o = patch_origins[i];
    int k = 0;
    for (int yy = 0; yy < patch; ++yy) {
      float* sum_ptr = recon_sum.ptr<float>(o.y + yy);
      float* cnt_ptr = recon_count.ptr<float>(o.y + yy);
      for (int xx = 0; xx < patch; ++xx) {
        sum_ptr[o.x + xx] += row[k++];
        cnt_ptr[o.x + xx] += 1.0f;
      }
    }
  }

  cv::Mat reconstruction = recon_sum / (recon_count + 1e-6f);
  cv::Mat error_map = cv::abs(work - reconstruction);

  cv::Scalar mse_mean = cv::mean(error_map.mul(error_map));

  std::vector<float> errors;
  errors.reserve(static_cast<std::size_t>(error_map.rows * error_map.cols));
  for (int y = 0; y < error_map.rows; ++y) {
    const float* ep = error_map.ptr<float>(y);
    for (int x = 0; x < error_map.cols; ++x) {
      errors.push_back(ep[x]);
    }
  }
  std::nth_element(errors.begin(), errors.begin() + errors.size() * 95 / 100,
                   errors.end());
  const double p95_error = errors[errors.size() * 95 / 100];

  const cv::Mat evals = pca.eigenvalues;
  const double leading_ev = evals.at<float>(0, 0);
  const double trailing_ev = evals.at<float>(components - 1, 0);
  const double eig_ratio = leading_ev / (trailing_ev + 1e-6);

  const double score = Clamp(
      100.0 * (0.60 * Clamp((0.018 - mse_mean[0]) / 0.018, 0.0, 1.0) +
               0.30 * Clamp((0.050 - p95_error) / 0.050, 0.0, 1.0) +
               0.10 * Clamp((12.0 - eig_ratio) / 12.0, 0.0, 1.0)),
      0.0, 100.0);

  result.score_percent = score;
  result.confidence = Clamp(std::abs(score - 50.0) / 50.0, 0.0, 1.0);
  result.status = score >= 70.0 ? TestStatus::kFail
                                : (score <= 30.0 ? TestStatus::kPass
                                                 : TestStatus::kInconclusive);

  std::ostringstream summary;
  summary << "Latent manifold fit via PCA reconstruction: MSE=" << mse_mean[0]
          << ", p95 error=" << p95_error << ", eigen ratio=" << eig_ratio
          << ". Lower reconstruction error implies stronger manifold adherence and "
             "increases AI-likelihood score in this optional non-classical test.";
  result.evidence_summary = summary.str();

  result.raw_metrics["reconstruction_mse"] = mse_mean[0];
  result.raw_metrics["reconstruction_p95_error"] = p95_error;
  result.raw_metrics["pca_components"] = static_cast<double>(components);
  result.raw_metrics["pca_eigen_ratio"] = eig_ratio;
  result.raw_metrics["patch_count"] = static_cast<double>(patch_origins.size());

  cv::Mat recon_u8;
  cv::Mat recon_clamped;
  cv::min(cv::max(reconstruction, 0.0f), 1.0f, recon_clamped);
  recon_clamped.convertTo(recon_u8, CV_8U, 255.0);

  cv::Mat err_u8;
  cv::normalize(error_map, err_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::Mat err_color;
  cv::applyColorMap(err_u8, err_color, cv::COLORMAP_MAGMA);

  std::filesystem::create_directories(context.output_dir);
  const std::string recon_path = context.output_dir + "/latent_reconstruction.png";
  const std::string err_path = context.output_dir + "/latent_error_heatmap.png";
  cv::imwrite(recon_path, recon_u8);
  cv::imwrite(err_path, err_color);

  result.artifact_paths.push_back(recon_path);
  result.artifact_paths.push_back(err_path);

  result.raw_metrics["implemented"] = 1.0;
  return result;
}

}  // namespace pixelaudit
