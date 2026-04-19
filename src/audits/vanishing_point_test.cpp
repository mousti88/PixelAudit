#include "pixelaudit/tests/vanishing_point_test.hpp"

#include <algorithm>
#include <array>
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

struct VpCandidate {
  cv::Point2f vp;
  int family_bin_a = -1;
  int family_bin_b = -1;
  int support_count = 0;
  int family_support_count = 0;
  double support_weight = 0.0;
  double median_err_deg = 90.0;
  std::vector<int> inlier_indices;
  std::vector<int> family_a_inlier_indices;
  std::vector<int> family_b_inlier_indices;
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

double AngleResidualToVp(const LineEq& line, const cv::Point2f& vp) {
  cv::Point2f line_dir = line.p1 - line.p0;
  const cv::Point2f mid = 0.5f * (line.p0 + line.p1);
  cv::Point2f vp_dir = vp - mid;

  const double ln = std::sqrt(line_dir.x * line_dir.x + line_dir.y * line_dir.y);
  const double vn = std::sqrt(vp_dir.x * vp_dir.x + vp_dir.y * vp_dir.y);
  if (ln <= 1e-6 || vn <= 1e-6) {
    return 90.0;
  }

  const double cosv =
      std::abs((line_dir.x * vp_dir.x + line_dir.y * vp_dir.y) / (ln * vn));
  const double c = Clamp(cosv, 0.0, 1.0);
  return std::acos(c) * 180.0 / CV_PI;
}

double ComputeMedian(std::vector<double>* values) {
  if (values->empty()) {
    return 0.0;
  }
  std::nth_element(values->begin(), values->begin() + values->size() / 2,
                   values->end());
  return (*values)[values->size() / 2];
}

cv::Mat BuildDominantVpInlierOverlay(const cv::Mat& image,
                                     const std::vector<LineEq>& lines,
                                     const VpCandidate& best) {
  cv::Mat overlay = image.clone();

  std::vector<unsigned char> line_state(lines.size(), 0);
  for (int idx : best.inlier_indices) {
    if (idx >= 0 && idx < static_cast<int>(line_state.size())) {
      line_state[idx] = 1;
    }
  }
  for (int idx : best.family_a_inlier_indices) {
    if (idx >= 0 && idx < static_cast<int>(line_state.size())) {
      line_state[idx] = 2;
    }
  }
  for (int idx : best.family_b_inlier_indices) {
    if (idx >= 0 && idx < static_cast<int>(line_state.size())) {
      line_state[idx] = 3;
    }
  }

  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    const auto& l = lines[i];
    cv::Scalar color(90, 90, 90);
    int thickness = 1;
    if (line_state[i] == 2) {
      color = cv::Scalar(255, 190, 60);  // family A inliers
      thickness = 2;
    } else if (line_state[i] == 3) {
      color = cv::Scalar(70, 220, 240);  // family B inliers
      thickness = 2;
    } else if (line_state[i] == 1) {
      color = cv::Scalar(80, 200, 120);  // general inlier
      thickness = 2;
    }
    cv::line(overlay, l.p0, l.p1, color, thickness, cv::LINE_AA);
  }

  if (best.vp.x >= 0.0f && best.vp.x < image.cols && best.vp.y >= 0.0f &&
      best.vp.y < image.rows) {
    cv::drawMarker(overlay,
                   cv::Point(static_cast<int>(std::round(best.vp.x)),
                             static_cast<int>(std::round(best.vp.y))),
                   cv::Scalar(40, 50, 240), cv::MARKER_CROSS, 26, 2,
                   cv::LINE_AA);
  }

  cv::putText(overlay, "Dominant VP inlier overlay", cv::Point(14, 26),
              cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(240, 240, 240), 2,
              cv::LINE_AA);
  cv::putText(overlay, "orange/cyan: family inliers, green: other inliers, gray: outliers",
              cv::Point(14, 50), cv::FONT_HERSHEY_SIMPLEX, 0.45,
              cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

  return overlay;
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
  const cv::Rect2f expanded(-0.9f * w, -0.9f * h, 2.8f * w, 2.8f * h);

  constexpr int kAngleBins = 18;
  std::array<std::vector<int>, kAngleBins> line_bins;
  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    float a = lines[i].angle_rad;
    if (a < 0.0f) {
      a += static_cast<float>(CV_PI);
    }
    const int b = std::min(
        kAngleBins - 1,
        static_cast<int>((a / static_cast<float>(CV_PI)) * static_cast<float>(kAngleBins)));
    line_bins[b].push_back(i);
  }

  std::vector<int> dominant_bins;
  for (int b = 0; b < kAngleBins; ++b) {
    if (line_bins[b].size() >= 6) {
      dominant_bins.push_back(b);
    }
  }

  std::vector<cv::Point2f> intersections;
  std::vector<double> inter_weights;
  intersections.reserve(lines.size() * 2);
  inter_weights.reserve(lines.size() * 2);

  std::vector<VpCandidate> candidates;
  for (std::size_t bi = 0; bi < dominant_bins.size(); ++bi) {
    for (std::size_t bj = bi + 1; bj < dominant_bins.size(); ++bj) {
      const int b1 = dominant_bins[bi];
      const int b2 = dominant_bins[bj];
      const auto& g1 = line_bins[b1];
      const auto& g2 = line_bins[b2];

      std::vector<cv::Point2f> local_vps;
      local_vps.reserve(g1.size() * g2.size());

      for (int i1 : g1) {
        for (int i2 : g2) {
          cv::Point2f p;
          if (!Intersect(lines[i1], lines[i2], &p)) {
            continue;
          }
          if (!expanded.contains(p)) {
            continue;
          }
          local_vps.push_back(p);
          intersections.push_back(p);
          inter_weights.push_back(static_cast<double>(lines[i1].length * lines[i2].length));
          if (intersections.size() > 16000) {
            break;
          }
        }
        if (intersections.size() > 16000) {
          break;
        }
      }

      if (local_vps.size() < 8) {
        continue;
      }

      cv::Point2f vp(0.0f, 0.0f);
      for (const auto& p : local_vps) {
        vp += p;
      }
      vp *= static_cast<float>(1.0 / local_vps.size());

      std::vector<double> residuals;
      residuals.reserve(lines.size());
      int support = 0;
      double support_weight = 0.0;
      for (const auto& line : lines) {
        const double err = AngleResidualToVp(line, vp);
        residuals.push_back(err);
        if (err < 12.0) {
          ++support;
          support_weight += line.length;
        }
      }

      VpCandidate c;
      c.vp = vp;
      c.family_bin_a = b1;
      c.family_bin_b = b2;
      c.support_count = support;
      c.support_weight = support_weight;
      c.median_err_deg = ComputeMedian(&residuals);

      c.inlier_indices.reserve(lines.size());
      c.family_a_inlier_indices.reserve(g1.size());
      c.family_b_inlier_indices.reserve(g2.size());
      for (int li = 0; li < static_cast<int>(lines.size()); ++li) {
        const double err = AngleResidualToVp(lines[li], vp);
        if (err < 12.0) {
          c.inlier_indices.push_back(li);
        }
      }

      for (int li : g1) {
        if (li >= 0 && li < static_cast<int>(lines.size())) {
          const double err = AngleResidualToVp(lines[li], vp);
          if (err < 12.0) {
            c.family_a_inlier_indices.push_back(li);
          }
        }
      }
      for (int li : g2) {
        if (li >= 0 && li < static_cast<int>(lines.size())) {
          const double err = AngleResidualToVp(lines[li], vp);
          if (err < 12.0) {
            c.family_b_inlier_indices.push_back(li);
          }
        }
      }
      c.family_support_count =
          static_cast<int>(c.family_a_inlier_indices.size() +
                           c.family_b_inlier_indices.size());
      candidates.push_back(c);

      if (intersections.size() > 16000) {
        break;
      }
    }
    if (intersections.size() > 16000) {
      break;
    }
  }

  if (intersections.size() < 20 || candidates.empty()) {
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

  const auto best_it = std::max_element(
      candidates.begin(), candidates.end(),
      [](const VpCandidate& a, const VpCandidate& b) {
        return a.support_weight < b.support_weight;
      });
  const VpCandidate* best_ptr = &(*best_it);
  const VpCandidate best = *best_it;
  const double support_ratio = static_cast<double>(best.support_count) /
                               static_cast<double>(std::max<std::size_t>(1, lines.size()));
  const double median_err = best.median_err_deg;

  double second_support_weight = 0.0;
  for (const auto& c : candidates) {
    if (&c == best_ptr) {
      continue;
    }
    second_support_weight = std::max(second_support_weight, c.support_weight);
  }
  const double support_margin =
      best.support_weight / (second_support_weight + 1e-9);

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

  const double structure_confidence =
      Clamp(static_cast<double>(lines.size()) / 260.0, 0.0, 1.0) *
      Clamp(static_cast<double>(candidates.size()) / 6.0, 0.0, 1.0);

  const double family_support_ratio =
      static_cast<double>(best.family_support_count) /
      static_cast<double>(std::max<std::size_t>(1, lines.size()));

  const double inconsistency =
      0.42 * Clamp((0.42 - support_ratio) / 0.42, 0.0, 1.0) +
      0.28 * Clamp((median_err - 9.0) / 18.0, 0.0, 1.0) +
      0.20 * Clamp((1.8 - support_margin) / 1.8, 0.0, 1.0) +
      0.10 * Clamp((0.35 - family_support_ratio) / 0.35, 0.0, 1.0);

  const double score = Clamp(
      100.0 * (0.5 + structure_confidence * (inconsistency - 0.5)), 0.0, 100.0);

  result.score_percent = score;
  result.confidence = Clamp(std::abs(score - 50.0) / 50.0, 0.0, 1.0);
  result.status = score >= 70.0 ? TestStatus::kFail
                                : (score <= 30.0 ? TestStatus::kPass
                                                 : TestStatus::kInconclusive);

  std::ostringstream summary;
  summary << "Perspective consistency from " << lines.size() << " lines, "
          << candidates.size() << " VP hypotheses, best support ratio="
          << support_ratio << ", family-support ratio=" << family_support_ratio
          << ", median residual=" << median_err
          << " deg. Weak dominant-VP inlier support and high angular residual "
             "increase AI-likelihood.";
  result.evidence_summary = summary.str();

  result.raw_metrics["line_count"] = static_cast<double>(lines.size());
  result.raw_metrics["intersection_count"] =
      static_cast<double>(intersections.size());
  result.raw_metrics["vp_cluster_count"] = static_cast<double>(dominant_count);
  result.raw_metrics["vp_candidate_count"] = static_cast<double>(candidates.size());
  result.raw_metrics["dominant_vp_support_ratio"] = support_ratio;
    result.raw_metrics["dominant_vp_family_support_ratio"] = family_support_ratio;
  result.raw_metrics["dominant_vp_median_residual_deg"] = median_err;
  result.raw_metrics["dominant_vp_support_margin"] = support_margin;
    result.raw_metrics["dominant_vp_family_a_inliers"] =
      static_cast<double>(best.family_a_inlier_indices.size());
    result.raw_metrics["dominant_vp_family_b_inliers"] =
      static_cast<double>(best.family_b_inlier_indices.size());
    result.raw_metrics["dominant_vp_all_inliers"] =
      static_cast<double>(best.inlier_indices.size());
  result.raw_metrics["scene_structure_confidence"] = structure_confidence;
  result.raw_metrics["cluster_concentration_top3"] = concentration;
  result.raw_metrics["intersection_dispersion_norm"] = weighted_dispersion;
  result.raw_metrics["cluster_entropy_norm"] = entropy_norm;

  std::filesystem::create_directories(context.output_dir);

  cv::Mat line_overlay = bgr_image.clone();
  for (const auto& line : lines) {
    cv::line(line_overlay, line.p0, line.p1, cv::Scalar(40, 220, 80), 1, cv::LINE_AA);
  }

  if (best.vp.x >= 0.0f && best.vp.x < bgr_image.cols && best.vp.y >= 0.0f &&
      best.vp.y < bgr_image.rows) {
    cv::drawMarker(line_overlay,
                   cv::Point(static_cast<int>(std::round(best.vp.x)),
                             static_cast<int>(std::round(best.vp.y))),
                   cv::Scalar(30, 50, 240), cv::MARKER_CROSS, 24, 2,
                   cv::LINE_AA);
  }

  cv::Mat density = BuildVpDensityImage(bgr_image.size(), intersections, inter_weights,
                                        centroid);
  cv::Mat inlier_overlay = BuildDominantVpInlierOverlay(bgr_image, lines, best);

  const std::string line_path = context.output_dir + "/vanishing_lines_overlay.png";
  const std::string density_path = context.output_dir + "/vanishing_density_map.png";
  const std::string inlier_path =
      context.output_dir + "/vanishing_dominant_vp_inliers.png";
  cv::imwrite(line_path, line_overlay);
  cv::imwrite(density_path, density);
  cv::imwrite(inlier_path, inlier_overlay);

  result.artifact_paths.push_back(line_path);
  result.artifact_paths.push_back(density_path);
  result.artifact_paths.push_back(inlier_path);

  return result;
}

}  // namespace pixelaudit
