#pragma once

#include "pixelaudit/core/test_interface.hpp"

namespace pixelaudit {

class VanishingPointGeometryTest final : public IDetectionTest {
 public:
  std::string Id() const override;
  std::string Name() const override;
  std::string Description() const override;
  TestResult Run(const cv::Mat& bgr_image, const TestContext& context) override;
};

}  // namespace pixelaudit
