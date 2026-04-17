#include "pixelaudit/core/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace pixelaudit {
namespace {

double Clamp01(const double value) {
  return std::max(0.0, std::min(1.0, value));
}

double Sigmoid(const double value) {
  return 1.0 / (1.0 + std::exp(-value));
}

}  // namespace

void DetectionPipeline::AddTest(std::unique_ptr<IDetectionTest> test,
                                const double weight) {
  tests_.push_back({std::move(test), std::max(0.0, weight)});
}

void DetectionPipeline::SetFusionBias(const double bias) { fusion_bias_ = bias; }

FinalReport DetectionPipeline::Run(const cv::Mat& bgr_image,
                                   const std::string& input_image,
                                   const std::string& output_dir) const {
  const auto start = std::chrono::high_resolution_clock::now();

  FinalReport report;
  report.input_image = input_image;

  TestContext context{.output_dir = output_dir};

  double weighted_logit_sum = 0.0;
  double weight_sum = 0.0;

  for (const auto& [test, weight] : tests_) {
    TestResult result = test->Run(bgr_image, context);
    result.weight = weight;

    const double calibrated_score = Clamp01(result.score_percent / 100.0);
    const double centered = (calibrated_score - 0.5) * 4.0;
    weighted_logit_sum += centered * weight;
    weight_sum += weight;

    report.tests.push_back(result);
  }

  if (weight_sum <= 0.0) {
    report.ai_probability_percent = 50.0;
  } else {
    const double final_logit = fusion_bias_ + (weighted_logit_sum / weight_sum);
    report.ai_probability_percent = Sigmoid(final_logit) * 100.0;
  }

  report.confidence = std::abs(report.ai_probability_percent - 50.0) / 50.0;

  for (const auto& test : report.tests) {
    const double contribution =
        weight_sum <= 0.0 ? 0.0 : (test.weight / weight_sum) * test.score_percent;
    report.contribution_percent[test.name] = contribution;
  }

  const auto end = std::chrono::high_resolution_clock::now();
  report.elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  return report;
}

}  // namespace pixelaudit
