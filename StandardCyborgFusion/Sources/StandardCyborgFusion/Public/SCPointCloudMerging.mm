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
#import "ScanPreparation.hpp"

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

// MARK: - Helpers

namespace {

using namespace scanprep;

struct ICPResult {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    float rms = INFINITY;
    float overlap = 0;
};

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
