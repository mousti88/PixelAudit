#include "pixelaudit/tests/vanishing_point_test.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pixelaudit {
namespace {

struct LineEq {
  cv::Point2f p0;
  cv::Point2f p1;
  float a = 0.0f;
  float b = 0.0f;
  float c = 0.0f;
  float angle_rad = 0.0f;
  float length = 0.0f;
};

double Clamp(const double value, const double lo, const double hi) {
  return std::max(lo, std::min(hi, value));
}

LineEq ToLineEq(const cv::Vec4i& line) {
  LineEq out;
  out.p0 = cv::Point2f(static_cast<float>(line[0]), static_cast<float>(line[1]));
  out.p1 = cv::Point2f(static_cast<float>(line[2]), static_cast<float>(line[3]));
  const float dx = out.p1.x - out.p0.x;
  const float dy = out.p1.y - out.p0.y;
  out.length = std::sqrt(dx * dx + dy * dy);
  out.angle_rad = std::atan2(dy, dx);

  out.a = out.p0.y - out.p1.y;
  out.b = out.p1.x - out.p0.x;
  out.c = out.p0.x * out.p1.y - out.p1.x * out.p0.y;
  return out;
}

bool Intersect(const LineEq& l1, const LineEq& l2, cv::Point2f* out_pt) {
  const float det = l1.a * l2.b - l2.a * l1.b;
  if (std::abs(det) < 1e-6f) {
    return false;
  }

  out_pt->x = (l1.b * l2.c - l2.b * l1.c) / det;
  out_pt->y = (l1.c * l2.a - l2.c * l1.a) / det;
  return std::isfinite(out_pt->x) && std::isfinite(out_pt->y);
}

cv::Mat BuildVpDensityImage(const cv::Size size,
                            const std::vector<cv::Point2f>& points,
                            const std::vector<double>& weights,
                            const cv::Point2f& centroid) {
  cv::Mat heat(size, CV_32F, cv::Scalar(0.0f));

  for (std::size_t i = 0; i < points.size(); ++i) {
    const int x = static_cast<int>(std::round(points[i].x));
    const int y = static_cast<int>(std::round(points[i].y));
    if (x < 0 || x >= size.width || y < 0 || y >= size.height) {
      continue;
    }
    heat.at<float>(y, x) += static_cast<float>(weights[i]);
  }

  cv::GaussianBlur(heat, heat, cv::Size(0, 0), 8.0, 8.0);

  cv::Mat heat_u8;
  cv::normalize(heat, heat_u8, 0, 255, cv::NORM_MINMAX, CV_8U);

  cv::Mat colored;
  cv::applyColorMap(heat_u8, colored, cv::COLORMAP_TURBO);

  if (centroid.x >= 0.0f && centroid.x < size.width && centroid.y >= 0.0f &&
      centroid.y < size.height) {
    cv::drawMarker(colored,
                   cv::Point(static_cast<int>(std::round(centroid.x)),
                             static_cast<int>(std::round(centroid.y))),
                   cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 28, 2,
                   cv::LINE_AA);
  }

  return colored;
}

}  // namespace

std::string VanishingPointGeometryTest::Id() const { return "vanishing_point_geometry"; }

std::string VanishingPointGeometryTest::Name() const {
  return "Vanishing Point Geometry Consistency";
}

std::string VanishingPointGeometryTest::Description() const {
  return "Detects structural perspective consistency by extracting strong line "
         "segments and measuring whether intersections converge to a compact set of "
         "vanishing points or spread across many inconsistent locations.";
}

TestResult VanishingPointGeometryTest::Run(const cv::Mat& bgr_image,
                                           const TestContext& context) {
  TestResult result;
  result.test_id = Id();
  result.name = Name();
  result.description = Description();

  if (bgr_image.empty()) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.0;
    result.evidence_summary = "Input image is empty; vanishing-point test skipped.";
    return result;
  }

  cv::Mat gray;
  if (bgr_image.channels() == 3) {
    cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
  } else if (bgr_image.channels() == 1) {
    gray = bgr_image;
  } else {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary = "Unsupported channel layout for line extraction.";
    return result;
  }

  cv::Mat blur;
  cv::GaussianBlur(gray, blur, cv::Size(5, 5), 1.2, 1.2);

  cv::Mat edges;
  cv::Canny(blur, edges, 50, 150, 3);

  std::vector<cv::Vec4i> raw_lines;
  cv::HoughLinesP(edges, raw_lines, 1.0, CV_PI / 180.0, 60,
                  std::max(35.0, 0.04 * std::min(gray.cols, gray.rows)), 12.0);

  std::vector<LineEq> lines;
  lines.reserve(raw_lines.size());
  for (const auto& line : raw_lines) {
    LineEq l = ToLineEq(line);
    if (l.length < 30.0f) {
      continue;
    }
    lines.push_back(l);
  }

  if (lines.size() < 6) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.1;
    result.evidence_summary =
        "Insufficient strong line segments to estimate stable vanishing geometry.";
    result.raw_metrics["line_count"] = static_cast<double>(lines.size());
    return result;
  }

  const float w = static_cast<float>(gray.cols);
  const float h = static_cast<float>(gray.rows);
  const cv::Rect2f expanded(-0.5f * w, -0.5f * h, 2.0f * w, 2.0f * h);

  std::vector<cv::Point2f> intersections;
  std::vector<double> inter_weights;
  intersections.reserve(lines.size() * 2);
  inter_weights.reserve(lines.size() * 2);

  for (std::size_t i = 0; i < lines.size(); ++i) {
    for (std::size_t j = i + 1; j < lines.size(); ++j) {
      const float angle_diff =
          std::abs(std::atan2(std::sin(lines[i].angle_rad - lines[j].angle_rad),
                              std::cos(lines[i].angle_rad - lines[j].angle_rad)));

      if (angle_diff < 0.2f || angle_diff > (CV_PI - 0.2f)) {
        continue;
      }

      cv::Point2f p;
      if (!Intersect(lines[i], lines[j], &p)) {
        continue;
      }

      if (!expanded.contains(p)) {
        continue;
      }

      intersections.push_back(p);
      inter_weights.push_back(static_cast<double>(lines[i].length * lines[j].length));

      if (intersections.size() > 12000) {
        break;
      }
    }
    if (intersections.size() > 12000) {
      break;
    }
  }

  if (intersections.size() < 20) {
    result.status = TestStatus::kInconclusive;
    result.score_percent = 50.0;
    result.confidence = 0.15;
    result.evidence_summary =
        "Not enough robust line intersections to infer vanishing-point consistency.";
    result.raw_metrics["line_count"] = static_cast<double>(lines.size());
    result.raw_metrics["intersection_count"] =
        static_cast<double>(intersections.size());
    return result;
  }

  const int cell_size = std::max(20, std::min(gray.cols, gray.rows) / 24);
  std::unordered_map<long long, double> grid;
  grid.reserve(intersections.size());

  auto make_key = [](int gx, int gy) {
    return (static_cast<long long>(gx) << 32) ^ static_cast<unsigned int>(gy);
  };

  for (std::size_t i = 0; i < intersections.size(); ++i) {
    const int gx = static_cast<int>(std::floor(intersections[i].x / cell_size));
    const int gy = static_cast<int>(std::floor(intersections[i].y / cell_size));
    grid[make_key(gx, gy)] += inter_weights[i];
  }

  std::vector<double> cluster_weights;
  cluster_weights.reserve(grid.size());
  for (const auto& kv : grid) {
    cluster_weights.push_back(kv.second);
  }

  std::sort(cluster_weights.begin(), cluster_weights.end(), std::greater<double>());
  const int dominant_count =
      static_cast<int>(std::count_if(cluster_weights.begin(), cluster_weights.end(),
                                     [](const double wv) { return wv > 0.0; }));

  const int top_k = std::min<int>(3, static_cast<int>(cluster_weights.size()));
  double top_sum = 0.0;
  double total_sum = 0.0;
  for (std::size_t i = 0; i < cluster_weights.size(); ++i) {
    total_sum += cluster_weights[i];
    if (static_cast<int>(i) < top_k) {
      top_sum += cluster_weights[i];
    }
  }
  const double concentration = top_sum / (total_sum + 1e-9);

  cv::Point2f centroid(0.0f, 0.0f);
  double weight_total = 0.0;
  for (std::size_t i = 0; i < intersections.size(); ++i) {
    centroid += intersections[i] * static_cast<float>(inter_weights[i]);
    weight_total += inter_weights[i];
  }
  centroid *= static_cast<float>(1.0 / (weight_total + 1e-9));

  const double img_diag =
      std::sqrt(static_cast<double>(gray.cols * gray.cols + gray.rows * gray.rows));
  double weighted_dispersion = 0.0;
  for (std::size_t i = 0; i < intersections.size(); ++i) {
    const double d = cv::norm(intersections[i] - centroid);
    weighted_dispersion += d * inter_weights[i];
  }
  weighted_dispersion =
      (weighted_dispersion / (weight_total + 1e-9)) / (img_diag + 1e-9);

  double entropy = 0.0;
  for (const double cw : cluster_weights) {
    if (cw <= 0.0) {
      continue;
    }
    const double p = cw / (total_sum + 1e-9);
    entropy -= p * std::log2(p);
  }
  const double entropy_norm =
      cluster_weights.size() > 1
          ? entropy / std::log2(static_cast<double>(cluster_weights.size()))
          : 0.0;

  const double score = Clamp(
      100.0 * (0.45 * Clamp(weighted_dispersion / 0.35, 0.0, 1.0) +
               0.35 * (1.0 - Clamp(concentration, 0.0, 1.0)) +
               0.20 * Clamp(entropy_norm, 0.0, 1.0)),
      0.0, 100.0);

  result.score_percent = score;
  result.confidence = Clamp(std::abs(score - 50.0) / 50.0, 0.0, 1.0);
  result.status = score >= 70.0 ? TestStatus::kFail
                                : (score <= 30.0 ? TestStatus::kPass
                                                 : TestStatus::kInconclusive);

  std::ostringstream summary;
  summary << "Perspective consistency derived from " << lines.size()
          << " lines and " << intersections.size() << " intersections. "
          << "Higher dispersion and lower cluster concentration increase AI-likelihood.";
  result.evidence_summary = summary.str();

  result.raw_metrics["line_count"] = static_cast<double>(lines.size());
  result.raw_metrics["intersection_count"] =
      static_cast<double>(intersections.size());
  result.raw_metrics["vp_cluster_count"] = static_cast<double>(dominant_count);
  result.raw_metrics["cluster_concentration_top3"] = concentration;
  result.raw_metrics["intersection_dispersion_norm"] = weighted_dispersion;
  result.raw_metrics["cluster_entropy_norm"] = entropy_norm;

  std::filesystem::create_directories(context.output_dir);

  cv::Mat line_overlay = bgr_image.clone();
  for (const auto& line : lines) {
    cv::line(line_overlay, line.p0, line.p1, cv::Scalar(40, 220, 80), 1, cv::LINE_AA);
  }

  if (centroid.x >= 0.0f && centroid.x < bgr_image.cols && centroid.y >= 0.0f &&
      centroid.y < bgr_image.rows) {
    cv::drawMarker(line_overlay,
                   cv::Point(static_cast<int>(std::round(centroid.x)),
                             static_cast<int>(std::round(centroid.y))),
                   cv::Scalar(30, 50, 240), cv::MARKER_CROSS, 24, 2,
                   cv::LINE_AA);
  }

  cv::Mat density = BuildVpDensityImage(bgr_image.size(), intersections, inter_weights,
                                        centroid);

  const std::string line_path = context.output_dir + "/vanishing_lines_overlay.png";
  const std::string density_path = context.output_dir + "/vanishing_density_map.png";
  cv::imwrite(line_path, line_overlay);
  cv::imwrite(density_path, density);

  result.artifact_paths.push_back(line_path);
  result.artifact_paths.push_back(density_path);

  return result;
}

}  // namespace pixelaudit
