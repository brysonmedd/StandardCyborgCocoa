//
//  ScanPreparation.hpp
//  StandardCyborgFusion
//
//  Shared steps for working with head scans, used by both scan merging
//  (SCPointCloudMerging) and head-model fitting (SCHeadModelFitting):
//  standing a scan upright with its gravity, sampling it, cutting off the body
//  below the neck, and picking points evenly by surface direction.
//

#pragma once

#import <StandardCyborgFusion/SCPointCloud.h>
#import <StandardCyborgFusion/Surfel.hpp>
#import <standard_cyborg/math/Vec3.hpp>
#import <standard_cyborg/util/IncludeEigen.hpp>
#import <simd/simd.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace scanprep {

using namespace standard_cyborg;

static const int   kNormalBinsPerAxis = 8;       // for picking points evenly by surface direction
// Ignoring the body: the neck is looked for this far below the top of the head (meters)
static const float kNeckSliceHeight = 0.01f;
static const float kNeckSearchTop = 0.16f;
static const float kNeckSearchBottom = 0.32f;
static const float kShoulderToNeckRatio = 1.4f;  // shoulders must be this much wider than the neck

struct Cloud {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> normals;  // unit length

    Eigen::Vector3f centroid() const {
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        for (const auto& p : positions) { sum += p; }
        return positions.empty() ? sum : Eigen::Vector3f(sum / (float)positions.size());
    }
};

/** Rotation that turns `gravity` into straight down (0, -1, 0). */
inline Eigen::Matrix4f uprightTransform(simd_float3 gravity)
{
    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
    Eigen::Vector3f from(gravity.x, gravity.y, gravity.z);
    if (from.norm() < 1e-3f) { return result; }
    from.normalize();
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(from, Eigen::Vector3f(0, -1, 0));
    result.block<3, 3>(0, 0) = q.toRotationMatrix();
    return result;
}

/** Up to `maxCount` evenly spaced points from `pointCloud`, moved by `transform`. */
inline Cloud sampleCloud(SCPointCloud *pointCloud, const Eigen::Matrix4f& transform, size_t maxCount)
{
    Cloud cloud;
    const Surfel *surfels = (const Surfel *)[[pointCloud pointsData] bytes];
    size_t count = (size_t)[pointCloud pointCount];
    size_t step = std::max<size_t>(1, count / std::max<size_t>(1, maxCount));
    Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = transform.block<3, 1>(0, 3);

    cloud.positions.reserve(count / step + 1);
    cloud.normals.reserve(count / step + 1);
    for (size_t i = 0; i < count; i += step) {
        Eigen::Vector3f n = surfels[i].normal;
        float length = n.norm();
        if (length < 1e-9f) { continue; }
        cloud.positions.push_back(R * surfels[i].position + t);
        cloud.normals.push_back(R * (n / length));
    }
    return cloud;
}

/** The `fraction` (0–1) percentile of `values` (reorders them). */
inline float percentile(std::vector<float>& values, float fraction)
{
    if (values.empty()) { return 0; }
    size_t k = std::min(values.size() - 1, (size_t)(fraction * (float)(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + k, values.end());
    return values[k];
}

/** For an upright cloud (gravity = -y) of a head and shoulders, finds the height of the neck:
    the narrowest 1 cm horizontal slice between 16 and 32 cm below the top of the head.
    Returns false (no cut) if the cloud doesn't reach that far down, or if that slice isn't
    clearly narrower than the shoulders below it (e.g. a head-only scan). */
inline bool findNeckHeight(const Cloud& cloud, float& neckY, float& depthBelowTop)
{
    if (cloud.positions.size() < 500) { return false; }

    std::vector<float> heights;
    heights.reserve(cloud.positions.size());
    for (const auto& p : cloud.positions) { heights.push_back(p.y()); }
    float top = percentile(heights, 0.998f);  // ignore a few stray points above the head

    // Sort point indices into 1 cm slices going down from the top
    const float sliceHeight = kNeckSliceHeight;
    const int sliceCount = (int)std::ceil(kNeckSearchBottom / sliceHeight) + 15;  // extra slices for the shoulders
    std::vector<std::vector<size_t>> slices(sliceCount);
    for (size_t i = 0; i < cloud.positions.size(); ++i) {
        int slice = (int)((top - cloud.positions[i].y()) / sliceHeight);
        if (slice >= 0 && slice < sliceCount) { slices[slice].push_back(i); }
    }

    // Width of each slice: how far its points spread out horizontally from the slice's middle
    std::vector<float> widths(sliceCount, -1.0f);
    for (int s = 0; s < sliceCount; ++s) {
        if (slices[s].size() < 20) { continue; }
        std::vector<float> xs, zs;
        for (size_t i : slices[s]) { xs.push_back(cloud.positions[i].x()); zs.push_back(cloud.positions[i].z()); }
        float cx = percentile(xs, 0.5f), cz = percentile(zs, 0.5f);
        std::vector<float> spread;
        for (size_t i : slices[s]) {
            float dx = cloud.positions[i].x() - cx, dz = cloud.positions[i].z() - cz;
            spread.push_back(std::sqrt(dx * dx + dz * dz));
        }
        widths[s] = percentile(spread, 0.9f);
    }

    // Narrowest slice in the neck window
    int first = (int)(kNeckSearchTop / sliceHeight), last = (int)(kNeckSearchBottom / sliceHeight);
    int neck = -1;
    for (int s = first; s <= last && s < sliceCount; ++s) {
        if (widths[s] < 0) { continue; }
        if (neck < 0 || widths[s] < widths[neck]) { neck = s; }
    }
    if (neck < 0) { return false; }

    // Only cut if there are clearly wider shoulders below it
    float widestBelow = 0;
    for (int s = neck + 1; s < sliceCount; ++s) { widestBelow = std::max(widestBelow, widths[s]); }
    if (widestBelow < kShoulderToNeckRatio * widths[neck]) { return false; }

    depthBelowTop = ((float)neck + 0.5f) * sliceHeight;
    neckY = top - depthBelowTop;
    return true;
}

/** Only the points at or above `minY`. */
inline Cloud keepAbove(const Cloud& cloud, float minY)
{
    Cloud result;
    for (size_t i = 0; i < cloud.positions.size(); ++i) {
        if (cloud.positions[i].y() < minY) { continue; }
        result.positions.push_back(cloud.positions[i]);
        result.normals.push_back(cloud.normals[i]);
    }
    return result;
}

/** Removes everything below the neck, if a neck is found. Reports the cut depth
    (meters below the top of the head), or 0 if nothing was cut. */
inline Cloud withoutBody(const Cloud& cloud, float& cutDepth)
{
    float neckY = 0;
    cutDepth = 0;
    if (!findNeckHeight(cloud, neckY, cutDepth)) { return cloud; }
    return keepAbove(cloud, neckY);
}

/** Up to `maxCount` points from `all`, picked so that every surface direction is equally
    represented ("normal-space sampling").

    On a head, most points face roughly the same few ways (big rounded areas), so a plain
    sample lets those areas outvote small sharp features like the nose and ears, and the
    scans can slide over each other. Grouping points by which way they face and taking
    the same number from each group gives the features a fair say. */
inline Cloud pickByNormals(const Cloud& all, size_t maxCount)
{
    if (all.positions.size() <= maxCount) { return all; }

    // Group point indices by normal direction (a grid over the unit cube of directions)
    const int binsPerAxis = kNormalBinsPerAxis;
    std::vector<std::vector<size_t>> bins(binsPerAxis * binsPerAxis * binsPerAxis);
    auto binOf = [binsPerAxis](float v) {
        int b = (int)((v + 1.0f) * 0.5f * binsPerAxis);
        return std::min(std::max(b, 0), binsPerAxis - 1);
    };
    for (size_t i = 0; i < all.normals.size(); ++i) {
        const Eigen::Vector3f& n = all.normals[i];
        bins[(binOf(n.x()) * binsPerAxis + binOf(n.y())) * binsPerAxis + binOf(n.z())].push_back(i);
    }

    // Shuffle each group (fixed seed, so results repeat), then take one from each in turn
    std::mt19937 random(12345);
    std::vector<std::vector<size_t>*> groups;
    for (auto& bin : bins) {
        if (bin.empty()) { continue; }
        std::shuffle(bin.begin(), bin.end(), random);
        groups.push_back(&bin);
    }

    Cloud picked;
    picked.positions.reserve(maxCount);
    picked.normals.reserve(maxCount);
    for (size_t round = 0; picked.positions.size() < maxCount; ++round) {
        bool tookAny = false;
        for (auto *group : groups) {
            if (round >= group->size()) { continue; }
            size_t i = (*group)[round];
            picked.positions.push_back(all.positions[i]);
            picked.normals.push_back(all.normals[i]);
            tookAny = true;
            if (picked.positions.size() >= maxCount) { break; }
        }
        if (!tookAny) { break; }
    }
    return picked;
}

inline std::vector<math::Vec3> toVec3(const std::vector<Eigen::Vector3f>& points)
{
    std::vector<math::Vec3> result;
    result.reserve(points.size());
    for (const auto& p : points) { result.emplace_back(p.x(), p.y(), p.z()); }
    return result;
}

/** Turn about the vertical (y) axis of a rotation, in degrees 0–360. */
inline float yawDegrees(const Eigen::Matrix4f& transform)
{
    float yaw = std::atan2(transform(0, 2), transform(2, 2)) * 180.0f / (float)M_PI;
    return yaw < 0 ? yaw + 360.0f : yaw;
}

inline float yawDifference(float a, float b)
{
    float d = std::fabs(a - b);
    return std::min(d, 360.0f - d);
}

inline simd_float4x4 toSimd(const Eigen::Matrix4f& m)
{
    simd_float4x4 result;
    for (int column = 0; column < 4; ++column) {
        result.columns[column] = simd_make_float4(m(0, column), m(1, column), m(2, column), m(3, column));
    }
    return result;
}

inline Eigen::Matrix4f fromSimd(simd_float4x4 m)
{
    Eigen::Matrix4f result;
    for (int column = 0; column < 4; ++column) {
        result(0, column) = m.columns[column].x;
        result(1, column) = m.columns[column].y;
        result(2, column) = m.columns[column].z;
        result(3, column) = m.columns[column].w;
    }
    return result;
}

} // namespace scanprep
