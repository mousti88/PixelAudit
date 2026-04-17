#pragma once

#include <map>
#include <string>
#include <vector>

namespace pixelaudit {

enum class TestStatus {
  kPass,
  kFail,
  kInconclusive,
};

struct TestResult {
  std::string test_id;
  std::string name;
  std::string description;
  double score_percent = 50.0;
  double confidence = 0.5;
  TestStatus status = TestStatus::kInconclusive;
  std::string evidence_summary;
  std::vector<std::string> artifact_paths;
  std::map<std::string, double> raw_metrics;
  bool is_non_classical = false;
  double weight = 1.0;
};

struct FinalReport {
  std::string input_image;
  std::vector<TestResult> tests;
  double ai_probability_percent = 50.0;
  double confidence = 0.5;
  double elapsed_ms = 0.0;
  std::map<std::string, double> contribution_percent;
};

std::string ToString(TestStatus status);
std::string EscapeJson(const std::string& input);
std::string ToJson(const FinalReport& report);

}  // namespace pixelaudit
