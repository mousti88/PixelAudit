#include "pixelaudit/core/types.hpp"

#include <iomanip>
#include <sstream>

namespace pixelaudit {

std::string ToString(TestStatus status) {
  switch (status) {
    case TestStatus::kPass:
      return "pass";
    case TestStatus::kFail:
      return "fail";
    case TestStatus::kInconclusive:
      return "inconclusive";
  }
  return "inconclusive";
}

std::string EscapeJson(const std::string& input) {
  std::ostringstream out;
  for (const char c : input) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

std::string ToJson(const FinalReport& report) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"input_image\": \"" << EscapeJson(report.input_image) << "\",\n";
  out << "  \"ai_probability_percent\": " << report.ai_probability_percent << ",\n";
  out << "  \"confidence\": " << report.confidence << ",\n";
  out << "  \"elapsed_ms\": " << report.elapsed_ms << ",\n";

  out << "  \"contribution_percent\": {";
  bool first_contrib = true;
  for (const auto& [key, value] : report.contribution_percent) {
    if (!first_contrib) {
      out << ", ";
    }
    first_contrib = false;
    out << "\"" << EscapeJson(key) << "\": " << value;
  }
  out << "},\n";

  out << "  \"tests\": [\n";
  for (std::size_t i = 0; i < report.tests.size(); ++i) {
    const auto& test = report.tests[i];
    out << "    {\n";
    out << "      \"test_id\": \"" << EscapeJson(test.test_id) << "\",\n";
    out << "      \"name\": \"" << EscapeJson(test.name) << "\",\n";
    out << "      \"description\": \"" << EscapeJson(test.description) << "\",\n";
    out << "      \"score_percent\": " << test.score_percent << ",\n";
    out << "      \"confidence\": " << test.confidence << ",\n";
    out << "      \"status\": \"" << ToString(test.status) << "\",\n";
    out << "      \"evidence_summary\": \"" << EscapeJson(test.evidence_summary) << "\",\n";
    out << "      \"is_non_classical\": " << (test.is_non_classical ? "true" : "false") << ",\n";
    out << "      \"weight\": " << test.weight << ",\n";

    out << "      \"artifact_paths\": [";
    for (std::size_t j = 0; j < test.artifact_paths.size(); ++j) {
      if (j != 0) {
        out << ", ";
      }
      out << "\"" << EscapeJson(test.artifact_paths[j]) << "\"";
    }
    out << "],\n";

    out << "      \"raw_metrics\": {";
    bool first_metric = true;
    for (const auto& [key, value] : test.raw_metrics) {
      if (!first_metric) {
        out << ", ";
      }
      first_metric = false;
      out << "\"" << EscapeJson(key) << "\": " << value;
    }
    out << "}\n";
    out << "    }";
    if (i + 1 != report.tests.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";

  return out.str();
}

}  // namespace pixelaudit
