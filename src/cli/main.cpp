#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "pixelaudit/core/pipeline.hpp"
#include "pixelaudit/core/types.hpp"
#include "pixelaudit/tests/frequency_spectrum_test.hpp"
#include "pixelaudit/tests/gradient_pca_test.hpp"
#include "pixelaudit/tests/jpeg_forensics_test.hpp"
#include "pixelaudit/tests/latent_module_stub.hpp"
#include "pixelaudit/tests/optical_photometric_test.hpp"
#include "pixelaudit/tests/sensor_noise_prnu_test.hpp"
#include "pixelaudit/tests/vanishing_point_test.hpp"

namespace {

void PrintUsage() {
  std::cout << "Usage:\n";
  std::cout << "  pixelaudit_cli --input <image_path> --output-dir <dir> [--enable-latent] [--calibration-file <path>]\n";
}

struct CalibrationConfig {
  double fusion_bias = 0.0;
  std::unordered_map<std::string, double> weights;
};

CalibrationConfig LoadCalibrationConfig(const std::string& calibration_path) {
  CalibrationConfig out;
  cv::FileStorage fs(calibration_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open calibration file: " + calibration_path);
  }

  if (!fs["fusion_bias"].empty()) {
    fs["fusion_bias"] >> out.fusion_bias;
  }

  const cv::FileNode weights = fs["weights"];
  if (!weights.empty() && weights.type() == cv::FileNode::MAP) {
    for (auto it = weights.begin(); it != weights.end(); ++it) {
      const cv::FileNode node = *it;
      double v = 1.0;
      node >> v;
      out.weights[node.name()] = v;
    }
  }

  return out;
}

double GetWeightOrDefault(const CalibrationConfig* cfg, const std::string& test_id,
                          const double fallback) {
  if (cfg == nullptr) {
    return fallback;
  }
  const auto it = cfg->weights.find(test_id);
  if (it == cfg->weights.end()) {
    return fallback;
  }
  return it->second;
}

}  // namespace

int main(int argc, char** argv) {
  try {
  std::string input_image_path;
  std::string output_dir = "artifacts";
  std::string calibration_file_path;
  bool enable_latent = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_image_path = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--calibration-file" && i + 1 < argc) {
      calibration_file_path = argv[++i];
    } else if (arg == "--enable-latent") {
      enable_latent = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage();
      return 2;
    }
  }

  if (input_image_path.empty()) {
    std::cerr << "Missing required --input argument.\n";
    PrintUsage();
    return 2;
  }

  std::filesystem::create_directories(output_dir);
  const std::string artifact_dir = output_dir + "/artifacts";
  std::filesystem::create_directories(artifact_dir);

  const cv::Mat image = cv::imread(input_image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Failed to load image: " << input_image_path << "\n";
    return 1;
  }

  std::optional<CalibrationConfig> calibration;
  if (!calibration_file_path.empty()) {
    calibration = LoadCalibrationConfig(calibration_file_path);
  }

  pixelaudit::DetectionPipeline pipeline;
  if (calibration.has_value()) {
    pipeline.SetFusionBias(calibration->fusion_bias);
  }

  pipeline.AddTest(std::make_unique<pixelaudit::LuminanceGradientPcaTest>(),
                   GetWeightOrDefault(calibration ? &*calibration : nullptr,
                                      "gradient_pca", 1.0));
  pipeline.AddTest(std::make_unique<pixelaudit::FrequencySpectrumTest>(),
                   GetWeightOrDefault(calibration ? &*calibration : nullptr,
                                      "frequency_spectrum", 1.0));
  pipeline.AddTest(std::make_unique<pixelaudit::JpegCompressionForensicsTest>(),
                   GetWeightOrDefault(calibration ? &*calibration : nullptr,
                                      "jpeg_compression_forensics", 1.0));
  pipeline.AddTest(std::make_unique<pixelaudit::OpticalPhotometricPlausibilityTest>(),
                   GetWeightOrDefault(calibration ? &*calibration : nullptr,
                                      "optical_photometric_plausibility", 1.0));
  pipeline.AddTest(std::make_unique<pixelaudit::SensorNoisePrnuCfaTest>(),
                   GetWeightOrDefault(calibration ? &*calibration : nullptr,
                                      "sensor_noise_prnu_cfa", 1.0));
  pipeline.AddTest(std::make_unique<pixelaudit::VanishingPointGeometryTest>(),
                   GetWeightOrDefault(calibration ? &*calibration : nullptr,
                                      "vanishing_point_geometry", 1.0));

  if (enable_latent) {
    pipeline.AddTest(std::make_unique<pixelaudit::LatentManifoldStubTest>(),
                     GetWeightOrDefault(calibration ? &*calibration : nullptr,
                                        "latent_manifold_optional", 0.0));
  }

  const pixelaudit::FinalReport report =
      pipeline.Run(image, input_image_path, artifact_dir);

  const std::string report_path = output_dir + "/report.json";
  std::ofstream out(report_path);
  out << pixelaudit::ToJson(report);
  out.close();

  std::cout << "PixelAudit report written: " << report_path << "\n";
  std::cout << "AI probability (%): " << report.ai_probability_percent << "\n";
  std::cout << "Elapsed (ms): " << report.elapsed_ms << "\n";

  return 0;
  } catch (const cv::Exception& ex) {
    std::cerr << "OpenCV error: " << ex.what() << "\n";
    return 3;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 4;
  }
}
