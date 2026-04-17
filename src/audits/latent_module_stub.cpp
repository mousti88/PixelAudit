#include "pixelaudit/tests/latent_module_stub.hpp"

namespace pixelaudit {

std::string LatentManifoldStubTest::Id() const { return "latent_manifold_optional"; }

std::string LatentManifoldStubTest::Name() const {
  return "Latent-Manifold Reconstruction Error (Optional Non-Classical)";
}

std::string LatentManifoldStubTest::Description() const {
  return "Optional add-on module using a pretrained autoencoder to compare "
         "reconstruction error. This test is explicitly non-classical and disabled "
         "unless enabled by user flag.";
}

TestResult LatentManifoldStubTest::Run(const cv::Mat& bgr_image,
                                       const TestContext& context) {
  (void)bgr_image;
  (void)context;

  TestResult result;
  result.test_id = Id();
  result.name = Name();
  result.description = Description();
  result.is_non_classical = true;
  result.status = TestStatus::kInconclusive;
  result.score_percent = 50.0;
  result.confidence = 0.0;
  result.evidence_summary =
      "Latent-manifold add-on is enabled but not implemented in v0.1. "
      "No effect should be assumed from this placeholder result.";
  result.raw_metrics["implemented"] = 0.0;
  return result;
}

}  // namespace pixelaudit
