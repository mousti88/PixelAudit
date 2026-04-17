#pragma once

#include "pixelaudit/core/test_interface.hpp"

namespace pixelaudit {

class LatentManifoldStubTest final : public IDetectionTest {
 public:
  std::string Id() const override;
  std::string Name() const override;
  std::string Description() const override;
  bool IsNonClassical() const override { return true; }
  TestResult Run(const cv::Mat& bgr_image, const TestContext& context) override;
};

}  // namespace pixelaudit
