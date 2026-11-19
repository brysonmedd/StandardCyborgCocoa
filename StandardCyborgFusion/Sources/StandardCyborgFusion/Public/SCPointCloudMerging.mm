//
//  SCPointCloudMerging.mm
//  StandardCyborgFusion
//
//  How alignment works:
//   1. Rotate each scan so its gravity points straight down (0, -1, 0).
//      Now the only unknowns are a turn about the vertical (y) axis and a shift.
//   2. For 36 turns (every 10°): turn the source about its center, move its center
//      onto the target's center, and run a short ICP. Score each by overlap
//      (how many source points land on the target's surface).
//   3. Refine the best with a longer ICP on more points.
//
//  ICP here is point-to-plane: each source point is matched to its nearest target
//  point, and we solve for the small rotation + shift that best moves every source
//  point onto its match's surface (the plane through the match, along its normal).
//

#import "SCPointCloudMerging.h"
#import "SCPointCloud_Private.h"

#import <StandardCyborgFusion/Surfel.hpp>
#import <standard_cyborg/math/Vec3.hpp>
#import <standard_cyborg/sc3d/Geometry.hpp>
#import <standard_cyborg/util/IncludeEigen.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace standard_cyborg;

// MARK: - Tuning

static const size_t kSearchSampleCount = 1500;   // source points used while searching turns
static const size_t kRefineSampleCount = 12000;  // source points used for the final refinement
static const size_t kTargetSampleCount = 60000;  // target points kept for nearest-point lookups
static const int    kNormalBinsPerAxis = 8;      // for picking points evenly by surface direction
static const int    kYawSteps = 36;              // 10° apart
static const int    kSearchIterations = 15;
static const int    kRefineIterations = 50;
static const float  kSearchStartDistance = 0.05f;  // meters: matches farther apart are ignored...
static const float  kSearchEndDistance = 0.012f;   // ...shrinking to this by the last iteration
// The refinement starts wide enough that a feature ~1 cm off (e.g. a doubled nose)
// still pulls on the result, then narrows for precision.
static const float  kRefineStartDistance = 0.03f;
static const float  kRefineEndDistance = 0.003f;
static const float  kInlierDistance = 0.004f;      // "on the surface" for scoring
static const float  kMinNormalAgreement = 0.3f;    // cos(~72°): skip matches facing opposite ways
static const float  kRunnerUpMinYawDifference = 20.0f;
// Ignoring the body: the neck is looked for this far below the top of the head (meters)
static const float  kNeckSliceHeight = 0.01f;
static const float  kNeckSearchTop = 0.16f;
static const float  kNeckSearchBottom = 0.32f;
static const float  kShoulderToNeckRatio = 1.4f;   // shoulders must be this much wider than the neck

// MARK: - Helpers

namespace {

struct Cloud {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> normals;  // unit length

    Eigen::Vector3f centroid() const {
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        for (const auto& p : positions) { sum += p; }
        return positions.empty() ? sum : Eigen::Vector3f(sum / (float)positions.size());
    }
};

struct ICPResult {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    float rms = INFINITY;
    float overlap = 0;
};

/** Rotation that turns `gravity` into straight down (0, -1, 0). */
Eigen::Matrix4f uprightTransform(simd_float3 gravity)
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
Cloud sampleCloud(SCPointCloud *pointCloud, const Eigen::Matrix4f& transform, size_t maxCount)
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
float percentile(std::vector<float>& values, float fraction)
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
bool findNeckHeight(const Cloud& cloud, float& neckY, float& depthBelowTop)
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
Cloud keepAbove(const Cloud& cloud, float minY)
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
Cloud withoutBody(const Cloud& cloud, float& cutDepth)
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
Cloud pickByNormals(const Cloud& all, size_t maxCount)
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

std::vector<math::Vec3> toVec3(const std::vector<Eigen::Vector3f>& points)
{
    std::vector<math::Vec3> result;
    result.reserve(points.size());
    for (const auto& p : points) { result.emplace_back(p.x(), p.y(), p.z()); }
    return result;
}

/** Scores `transform`: overlap fraction and RMS distance of points on the target's surface. */
void score(const Cloud& source, const Cloud& target, const sc3d::Geometry& targetIndex,
           const Eigen::Matrix4f& transform, ICPResult& result)
{
    Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = transform.block<3, 1>(0, 3);
    double sumSquares = 0;
    size_t inliers = 0;

    for (size_t i = 0; i < source.positions.size(); ++i) {
        Eigen::Vector3f p = R * source.positions[i] + t;
        int index = targetIndex.getClosestVertexIndex(math::Vec3(p.x(), p.y(), p.z()));
        const Eigen::Vector3f& q = target.positions[index];
        const Eigen::Vector3f& n = target.normals[index];
        if ((p - q).squaredNorm() > kInlierDistance * kInlierDistance) { continue; }
        if ((R * source.normals[i]).dot(n) < kMinNormalAgreement) { continue; }
        float r = (p - q).dot(n);
        sumSquares += r * r;
        ++inliers;
    }

    result.transform = transform;
    result.overlap = source.positions.empty() ? 0 : (float)inliers / (float)source.positions.size();
    result.rms = inliers > 0 ? (float)std::sqrt(sumSquares / inliers) : INFINITY;
}

/** Point-to-plane ICP starting from `transform`. */
ICPResult runICP(const Cloud& source, const Cloud& target, const sc3d::Geometry& targetIndex,
                 Eigen::Matrix4f transform, int iterations, float startDistance, float endDistance)
{
    for (int iteration = 0; iteration < iterations; ++iteration) {
        float fraction = iterations > 1 ? (float)iteration / (float)(iterations - 1) : 1.0f;
        float maxDistance = startDistance + (endDistance - startDistance) * fraction;
        float maxDistanceSquared = maxDistance * maxDistance;

        Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = transform.block<3, 1>(0, 3);

        // Build the 6x6 least-squares system for (small rotation, shift)
        Eigen::Matrix<double, 6, 6> AtA = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> Atb = Eigen::Matrix<double, 6, 1>::Zero();
        int used = 0;

        for (size_t i = 0; i < source.positions.size(); ++i) {
            Eigen::Vector3f p = R * source.positions[i] + t;
            int index = targetIndex.getClosestVertexIndex(math::Vec3(p.x(), p.y(), p.z()));
            const Eigen::Vector3f& q = target.positions[index];
            const Eigen::Vector3f& n = target.normals[index];

            float distanceSquared = (p - q).squaredNorm();
            if (distanceSquared > maxDistanceSquared) { continue; }
            if ((R * source.normals[i]).dot(n) < kMinNormalAgreement) { continue; }

            // Soft weight (Tukey): close matches count fully, farther ones fade out
            // smoothly toward the cutoff instead of dropping from full to zero.
            double u = 1.0 - (double)distanceSquared / (double)maxDistanceSquared;
            double weight = u * u;

            Eigen::Matrix<double, 6, 1> J;
            J << p.cross(n).cast<double>(), n.cast<double>();
            double r = (double)(p - q).dot(n);
            AtA += weight * (J * J.transpose());
            Atb -= weight * (J * r);
            ++used;
        }

        if (used < 30) { break; }  // not enough overlap to say anything

        Eigen::Matrix<double, 6, 1> x = AtA.ldlt().solve(Atb);
        Eigen::Vector3d angles = x.head<3>();
        Eigen::Vector3d shift = x.tail<3>();

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        double angle = angles.norm();
        if (angle > 1e-12) {
            delta.block<3, 3>(0, 0) = Eigen::AngleAxisd(angle, angles / angle).toRotationMatrix().cast<float>();
        }
        delta.block<3, 1>(0, 3) = shift.cast<float>();
        transform = delta * transform;

        if (angle < 1e-5 && shift.norm() < 1e-5) { break; }  // converged
    }

    ICPResult result;
    score(source, target, targetIndex, transform, result);
    return result;
}

/** Turn about the vertical (y) axis of a rotation, in degrees 0–360. */
float yawDegrees(const Eigen::Matrix4f& transform)
{
    float yaw = std::atan2(transform(0, 2), transform(2, 2)) * 180.0f / (float)M_PI;
    return yaw < 0 ? yaw + 360.0f : yaw;
}

float yawDifference(float a, float b)
{
    float d = std::fabs(a - b);
    return std::min(d, 360.0f - d);
}

simd_float4x4 toSimd(const Eigen::Matrix4f& m)
{
    simd_float4x4 result;
    for (int column = 0; column < 4; ++column) {
        result.columns[column] = simd_make_float4(m(0, column), m(1, column), m(2, column), m(3, column));
    }
    return result;
}

Eigen::Matrix4f fromSimd(simd_float4x4 m)
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

} // namespace

// MARK: - SCAlignmentResult

@interface SCAlignmentResult ()
@property (nonatomic, readwrite) simd_float4x4 transform;
@property (nonatomic, readwrite) float rmsError;
@property (nonatomic, readwrite) float overlap;
@property (nonatomic, readwrite) float yawDegrees;
@property (nonatomic, readwrite) float runnerUpOverlap;
@property (nonatomic, readwrite) float runnerUpYawDegrees;
@property (nonatomic, readwrite) float sourceBodyCutDepth;
@property (nonatomic, readwrite) float targetBodyCutDepth;
@end

@implementation SCAlignmentResult
@end

// MARK: - SCPointCloudMerging

@implementation SCPointCloudMerging

+ (SCAlignmentResult *)alignPointCloud:(SCPointCloud *)source toPointCloud:(SCPointCloud *)target
{
    // 1. Stand both scans upright
    Eigen::Matrix4f sourceUpright = uprightTransform([source gravity]);
    Eigen::Matrix4f targetUpright = uprightTransform([target gravity]);

    // Ignore everything below the neck (shirt, shoulders): the head turns on the neck
    // between scans, so the body isn't rigidly attached to the head.
    float targetCutDepth = 0, sourceCutDepth = 0;
    Cloud targetCloud = withoutBody(sampleCloud(target, targetUpright, kTargetSampleCount), targetCutDepth);
    Cloud sourceHead = withoutBody(sampleCloud(source, sourceUpright, 20 * kRefineSampleCount), sourceCutDepth);
    Cloud searchCloud = pickByNormals(sourceHead, kSearchSampleCount);
    Cloud refineCloud = pickByNormals(sourceHead, kRefineSampleCount);

    SCAlignmentResult *result = [[SCAlignmentResult alloc] init];
    result.sourceBodyCutDepth = sourceCutDepth;
    result.targetBodyCutDepth = targetCutDepth;
    result.transform = matrix_identity_float4x4;
    result.rmsError = INFINITY;
    if (targetCloud.positions.empty() || searchCloud.positions.empty()) { return result; }

    sc3d::Geometry targetIndex(toVec3(targetCloud.positions));  // builds a KD-tree for nearest-point lookups

    Eigen::Vector3f sourceCenter = searchCloud.centroid();
    Eigen::Vector3f targetCenter = targetCloud.centroid();

    // 2. Try every turn about the vertical axis
    std::vector<ICPResult> candidates;
    candidates.reserve(kYawSteps);
    for (int step = 0; step < kYawSteps; ++step) {
        float yaw = (float)step * 2.0f * (float)M_PI / (float)kYawSteps;

        Eigen::Affine3f start = Eigen::Translation3f(targetCenter)
                              * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitY())
                              * Eigen::Translation3f(-sourceCenter);

        candidates.push_back(runICP(searchCloud, targetCloud, targetIndex, start.matrix(),
                                    kSearchIterations, kSearchStartDistance, kSearchEndDistance));
    }

    auto better = [](const ICPResult& a, const ICPResult& b) {
        if (std::fabs(a.overlap - b.overlap) > 0.01f) { return a.overlap > b.overlap; }
        return a.rms < b.rms;
    };
    const ICPResult *best = &candidates[0];
    for (const auto& candidate : candidates) {
        if (better(candidate, *best)) { best = &candidate; }
    }

    // Best result at a clearly different final turn
    float bestYaw = yawDegrees(best->transform);
    const ICPResult *runnerUp = nullptr;
    for (const auto& candidate : candidates) {
        if (yawDifference(yawDegrees(candidate.transform), bestYaw) <= kRunnerUpMinYawDifference) { continue; }
        if (runnerUp == nullptr || better(candidate, *runnerUp)) { runnerUp = &candidate; }
    }

    // 3. Refine with more points
    ICPResult refined = runICP(refineCloud, targetCloud, targetIndex, best->transform,
                               kRefineIterations, kRefineStartDistance, kRefineEndDistance);

    // Source (as recorded) → upright source → aligned → target (as recorded)
    Eigen::Matrix4f total = targetUpright.inverse() * refined.transform * sourceUpright;

    result.transform = toSimd(total);
    result.rmsError = refined.rms;
    result.overlap = refined.overlap;
    result.yawDegrees = yawDegrees(refined.transform);
    result.runnerUpOverlap = runnerUp ? runnerUp->overlap : 0;
    result.runnerUpYawDegrees = runnerUp ? yawDegrees(runnerUp->transform) : 0;
    return result;
}

+ (SCPointCloud *)mergePointCloud:(SCPointCloud *)source
                    transformedBy:(simd_float4x4)transform
                   intoPointCloud:(SCPointCloud *)target
{
    Eigen::Matrix4f m = fromSimd(transform);
    Eigen::Matrix3f R = m.block<3, 3>(0, 0);
    Eigen::Vector3f t = m.block<3, 1>(0, 3);

    const Surfel *targetSurfels = (const Surfel *)[[target pointsData] bytes];
    const Surfel *sourceSurfels = (const Surfel *)[[source pointsData] bytes];
    size_t targetCount = (size_t)[target pointCount];
    size_t sourceCount = (size_t)[source pointCount];

    std::vector<Surfel> merged(targetSurfels, targetSurfels + targetCount);
    merged.reserve(targetCount + sourceCount);
    for (size_t i = 0; i < sourceCount; ++i) {
        Surfel surfel = sourceSurfels[i];
        surfel.position = R * surfel.position + t;
        surfel.normal = R * surfel.normal;  // rotation keeps the length (which encodes size)
        merged.push_back(surfel);
    }

    NSData *data = [NSData dataWithBytes:merged.data() length:merged.size() * sizeof(Surfel)];
    return [[SCPointCloud alloc] initWithSurfelData:data gravity:[target gravity]];
}

@end
