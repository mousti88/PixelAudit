#pragma once

#include <opencv2/core.hpp>

#include <string>

#include "pixelaudit/core/types.hpp"

namespace pixelaudit {

struct TestContext {
  std::string output_dir;
};

class IDetectionTest {
 public:
  virtual ~IDetectionTest() = default;
  virtual std::string Id() const = 0;
  virtual std::string Name() const = 0;
  virtual std::string Description() const = 0;
  virtual bool IsNonClassical() const { return false; }
  virtual TestResult Run(const cv::Mat& bgr_image, const TestContext& context) = 0;
};

}  // namespace pixelaudit
