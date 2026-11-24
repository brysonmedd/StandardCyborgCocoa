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
// Head-only crop (merging and showing scans), found from the face's side profile:
static const float kNoseSearchTop = 0.06f;       // the nose tip is looked for this far below the top...
static const float kNoseSearchBottom = 0.22f;    // ...down to this
static const int   kNoseTipPointCount = 20;      // most-forward points averaged for the nose tip
static const float kMidlineHalfWidth = 0.015f;   // the profile uses points this close to the nose's midline
static const float kProfileSlice = 0.005f;
static const float kProfileStart = 0.02f;        // profile from 2 cm below the nose tip...
static const float kProfileEnd = 0.16f;          // ...to 16 cm below it
static const float kChinSearchStart = 0.04f;     // the chin is at least this far below the nose tip
static const float kChinDrop = 0.035f;           // under the chin, the profile steps back at least this much
static const float kCutBelowChin = 0.005f;       // cut this far below the chin (keeps the nape hairline, drops the collar)
static const float kChinMinDepth = 0.15f;        // the chin must be this far below the top of the head...
static const float kChinMaxDepth = 0.32f;        // ...and no farther
// Near the cut (from kLowerHeadBandAboveChin above the chin down), only points this close
// (horizontally) to the head's vertical axis are kept, so shoulders and collar raised by a
// tilted head go too. The axis is kHeadAxisBehindNose behind the nose tip. At that height
// chin, jaw and nape are within ~9 cm of it, shoulders farther out. (Higher up, the back of
// the skull reaches ~11 cm behind the axis, so the band must not start higher.)
static const float kHeadAxisBehindNose = 0.10f;
static const float kLowerHeadRadius = 0.12f;
static const float kLowerHeadBandAboveChin = 0.03f;
// Last resort (no face and no neck found): keep this much below the top of the head.
static const float kHeadOnlyMaxDepth = 0.26f;

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

/** Height of the top of the head in an upright cloud (ignores a few stray points above it). */
inline float topHeight(const Cloud& cloud)
{
    std::vector<float> heights;
    heights.reserve(cloud.positions.size());
    for (const auto& p : cloud.positions) { heights.push_back(p.y()); }
    return percentile(heights, 0.998f);
}

/** Finds the bottom of the chin in an upright cloud of a head scan, from the face's side
    profile. Every scan starts facing the camera, and the camera is at the origin (scans are
    stored in the camera's space at their first frame; standing upright only rotates), so
    "forward" is the horizontal direction from the head toward the origin.

    1. Nose tip: the most forward points 6–22 cm below the top of the head.
    2. Profile: in 0.5 cm horizontal slices going down from the nose, how far forward the
       face reaches along its midline (points within 1.5 cm of the nose, sideways).
    3. Chin: going down from 4 cm below the nose (past the lips), the profile stays near
       its furthest-forward value until, under the chin, it steps back ≥ 3.5 cm to the neck.
    Returns false if there's no clear chin (e.g. the scan doesn't show the face), or if it
    is implausibly close to or far from the top of the head. */
struct FaceProfile {
    float chinY = 0;
    float noseY = 0;
    Eigen::Vector3f noseTip = Eigen::Vector3f::Zero();
    Eigen::Vector3f forward = Eigen::Vector3f::UnitZ();  // horizontal, toward the camera
};

inline bool findChinHeight(const Cloud& cloud, FaceProfile& face, float& depthBelowTop)
{
    if (cloud.positions.size() < 500) { return false; }
    float top = topHeight(cloud);

    // Forward: from the head's center toward the camera, horizontal
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    size_t headCount = 0;
    for (const auto& p : cloud.positions) {
        if (p.y() > top - 0.25f) { center += p; ++headCount; }
    }
    if (headCount == 0) { return false; }
    center /= (float)headCount;
    Eigen::Vector3f forward(-center.x(), 0, -center.z());
    if (forward.norm() < 1e-3f) { return false; }
    forward.normalize();
    Eigen::Vector3f side = Eigen::Vector3f::UnitY().cross(forward);

    // 1. Nose tip
    std::vector<std::pair<float, size_t>> band;
    for (size_t i = 0; i < cloud.positions.size(); ++i) {
        float y = cloud.positions[i].y();
        if (y < top - kNoseSearchTop && y > top - kNoseSearchBottom) {
            band.push_back({cloud.positions[i].dot(forward), i});
        }
    }
    if ((int)band.size() < kNoseTipPointCount) { return false; }
    std::partial_sort(band.begin(), band.begin() + kNoseTipPointCount, band.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<float> tipSides, tipHeights;
    for (int k = 0; k < kNoseTipPointCount; ++k) {
        tipSides.push_back(cloud.positions[band[k].second].dot(side));
        tipHeights.push_back(cloud.positions[band[k].second].y());
    }
    std::vector<float> tipReach;
    for (int k = 0; k < kNoseTipPointCount; ++k) { tipReach.push_back(band[k].first); }
    float noseSide = percentile(tipSides, 0.5f);
    float noseY = percentile(tipHeights, 0.5f);
    float noseReach = percentile(tipReach, 0.5f);

    // 2. Profile along the midline
    const float profileTop = noseY - kProfileStart;
    const int sliceCount = (int)std::ceil((kProfileEnd - kProfileStart) / kProfileSlice);
    std::vector<std::vector<float>> slices(sliceCount);
    for (const auto& p : cloud.positions) {
        if (std::fabs(p.dot(side) - noseSide) > kMidlineHalfWidth) { continue; }
        int slice = (int)((profileTop - p.y()) / kProfileSlice);
        if (slice >= 0 && slice < sliceCount) { slices[slice].push_back(p.dot(forward)); }
    }

    // 3. Chin: the first big step back below the lips
    int first = (int)((kChinSearchStart - kProfileStart) / kProfileSlice);
    float furthest = -INFINITY;
    for (int slice = first; slice < sliceCount; ++slice) {
        bool empty = slices[slice].size() < 5;
        float reach = empty ? -INFINITY : percentile(slices[slice], 0.95f);
        bool steppedBack = furthest > -INFINITY && (empty || furthest - reach > kChinDrop);
        if (steppedBack) {
            face.chinY = profileTop - (float)slice * kProfileSlice;
            face.noseY = noseY;
            face.noseTip = noseReach * forward + noseSide * side + noseY * Eigen::Vector3f::UnitY();
            face.forward = forward;
            depthBelowTop = top - face.chinY;
            return depthBelowTop >= kChinMinDepth && depthBelowTop <= kChinMaxDepth;
        }
        if (!empty) { furthest = std::max(furthest, reach); }
    }
    return false;
}

/** Which points of an upright cloud belong to the head. */
struct HeadCut {
    float cutY = -INFINITY;      // nothing below this
    bool usesAxis = false;       // below `axisFromY`, also nothing farther than `radius` from the axis
    float axisFromY = 0;
    Eigen::Vector2f axis = Eigen::Vector2f::Zero();  // (x, z) of the head's vertical axis
    float radius = INFINITY;

    bool keeps(const Eigen::Vector3f& p) const {
        if (p.y() < cutY) { return false; }
        if (!usesAxis || p.y() >= axisFromY) { return true; }
        return (Eigen::Vector2f(p.x(), p.z()) - axis).norm() <= radius;
    }
};

/** Where to cut an upright cloud (camera at the origin) to keep only the head:
    - If the face's profile shows a chin: 0.5 cm below the chin, and from 3 cm above the
      chin down only what's within 12 cm of the head's vertical axis (drops shoulders and
      collar, even ones raised by a tilted head).
    - Else at the neck (narrowest slice above wider shoulders).
    - Else, as a last resort, `kHeadOnlyMaxDepth` below the top of the head.
    Returns false (keep everything) if there's nothing to cut. `cutDepth` is meters below
    the top of the head. */
inline bool findHeadCut(const Cloud& cloud, HeadCut& cut, float& cutDepth)
{
    cut = HeadCut();
    cutDepth = 0;
    if (cloud.positions.empty()) { return false; }
    float top = topHeight(cloud);
    FaceProfile face;
    float chinDepth = 0;
    if (findChinHeight(cloud, face, chinDepth)) {
        cut.cutY = face.chinY - kCutBelowChin;
        cut.usesAxis = true;
        cut.axisFromY = face.chinY + kLowerHeadBandAboveChin;
        Eigen::Vector3f axis = face.noseTip - kHeadAxisBehindNose * face.forward;
        cut.axis = Eigen::Vector2f(axis.x(), axis.z());
        cut.radius = kLowerHeadRadius;
        cutDepth = top - cut.cutY;
        return true;
    }
    float neckY = 0;
    if (findNeckHeight(cloud, neckY, cutDepth)) {
        cut.cutY = neckY;
        return true;
    }
    float lowest = INFINITY;
    for (const auto& p : cloud.positions) { lowest = std::min(lowest, p.y()); }
    if (top - lowest <= kHeadOnlyMaxDepth) { return false; }
    cutDepth = kHeadOnlyMaxDepth;
    cut.cutY = top - kHeadOnlyMaxDepth;
    return true;
}

/** Only the head (camera at the origin): see findHeadCut.
    (Shirts and shoulders otherwise pull scan alignment around.) */
inline Cloud headOnly(const Cloud& cloud, float& cutDepth)
{
    HeadCut cut;
    if (!findHeadCut(cloud, cut, cutDepth)) { return cloud; }
    Cloud result;
    for (size_t i = 0; i < cloud.positions.size(); ++i) {
        if (!cut.keeps(cloud.positions[i])) { continue; }
        result.positions.push_back(cloud.positions[i]);
        result.normals.push_back(cloud.normals[i]);
    }
    return result;
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
