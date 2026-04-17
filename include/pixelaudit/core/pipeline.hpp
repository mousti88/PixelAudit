#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "pixelaudit/core/test_interface.hpp"
#include "pixelaudit/core/types.hpp"

namespace pixelaudit {

class DetectionPipeline {
 public:
  void AddTest(std::unique_ptr<IDetectionTest> test, double weight);
  void SetFusionBias(double bias);
  FinalReport Run(const cv::Mat& bgr_image, const std::string& input_image,
                  const std::string& output_dir) const;

 private:
  std::vector<std::pair<std::unique_ptr<IDetectionTest>, double>> tests_;
  double fusion_bias_ = 0.0;
};

}  // namespace pixelaudit
