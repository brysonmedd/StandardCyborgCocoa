//
//  SCPointCloudMerging.mm
//  StandardCyborgFusion
//
//  How alignment works:
//   1. Rotate each scan so its gravity points straight down (0, -1, 0).
//      Now the only unknowns are a turn about the vertical (y) axis and a shift.
//      Keep only the head: cut just below the chin, found from the face's side profile
//      (ScanPreparation.hpp: findHeadCut), with fallbacks if no face is seen.
//   2. Starting positions, each refined with a short ICP:
//      - near the recorded pose (if asked): the scans as recorded, turned -25°…+25°
//        in 5° steps about the head's vertical axis. Scans that start from the same
//        view (phone on a stand) are already nearly lined up as recorded.
//      - full search (always): 36 turns, every 10°, with the source's center moved onto
//        the target's center. So if the recorded pose turns out wrong (the stand moved),
//        a clearly better answer still wins.
//   3. Score each: overlap (share of source points on the target's surface) minus
//      2 × see-through (share of points sitting clearly in front of the other scan's
//      surface, where that scan saw empty space; checked both ways), plus a small head
//      start for candidates near the recorded pose. See-through is what rules out
//      doubled faces: two heads slid partly into each other still have lots of touching
//      surface, but one always pokes out in front of the other. A candidate where almost
//      nothing touches (< 5% overlap) never wins, however little see-through it has.
//   4. Refine the best with a longer ICP on more points.
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
static const float  kAmbiguousScoreRatio = 0.8f;   // runner-up this close to the best → ambiguous
// Near the recorded pose
static const float  kNearStartMaxYawDegrees = 25.0f;
static const float  kNearStartYawStepDegrees = 5.0f;
static const float  kRecordedPoseBonus = 0.05f;     // head start for candidates near the recorded pose
static const float  kMinUsableOverlap = 0.05f;      // less than this never wins
// See-through: a point counts if it sits in front of the other surface (along its normal)
// by more than kSeeThroughMinDepth (above noise) and less than kSeeThroughMaxDepth (beyond
// that the space may not have been seen), and roughly straight above it (not off to the side).
static const float  kSeeThroughMinDepth = 0.006f;
static const float  kSeeThroughMaxDepth = 0.04f;
static const float  kSeeThroughMaxSideways = 0.006f;
static const float  kSeeThroughWeight = 2.0f;
static const size_t kSeeThroughSampleCount = 1500;  // points checked each way
static const size_t kHeadPointCount = 240000;       // source points kept before picking samples

// MARK: - Helpers

namespace {

using namespace scanprep;

struct ICPResult {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    float rms = INFINITY;
    float overlap = 0;
    float seeThrough = 0;
    bool nearRecordedPose = false;

    float rank() const {
        if (overlap < kMinUsableOverlap) { return overlap - 1.0f; }
        return overlap - kSeeThroughWeight * seeThrough + (nearRecordedPose ? kRecordedPoseBonus : 0.0f);
    }
};

/** Share of `points` (moved by `transform`) that sit clearly in front of `surface`,
    i.e. in space the scanner of `surface` saw as empty. */
float seeThroughFraction(const Cloud& points, const Eigen::Matrix4f& transform,
                         const Cloud& surface, const sc3d::Geometry& surfaceIndex)
{
    if (points.positions.empty()) { return 0; }
    Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = transform.block<3, 1>(0, 3);
    size_t count = 0;
    for (const auto& position : points.positions) {
        Eigen::Vector3f p = R * position + t;
        int index = surfaceIndex.getClosestVertexIndex(math::Vec3(p.x(), p.y(), p.z()));
        Eigen::Vector3f d = p - surface.positions[index];
        const Eigen::Vector3f& n = surface.normals[index];
        float inFront = d.dot(n);
        if (inFront < kSeeThroughMinDepth || inFront > kSeeThroughMaxDepth) { continue; }
        if ((d - inFront * n).norm() > kSeeThroughMaxSideways) { continue; }
        ++count;
    }
    return (float)count / (float)points.positions.size();
}

/** Everything needed to score a candidate, both ways. */
struct Scorer {
    const Cloud& target;
    const sc3d::Geometry& targetIndex;
    const Cloud& sourceHead;
    const sc3d::Geometry& sourceIndex;
    const Cloud& sourceSample;   // source points checked against the target
    const Cloud& targetSample;   // target points checked against the source

    float seeThrough(const Eigen::Matrix4f& sourceToTarget) const {
        float forward = seeThroughFraction(sourceSample, sourceToTarget, target, targetIndex);
        float backward = seeThroughFraction(targetSample, sourceToTarget.inverse(), sourceHead, sourceIndex);
        return 0.5f * (forward + backward);
    }
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
@property (nonatomic, readwrite) float seeThrough;
@property (nonatomic, readwrite) float score;
@property (nonatomic, readwrite) BOOL usedRecordedPose;
@property (nonatomic, readwrite) float runnerUpOverlap;
@property (nonatomic, readwrite) float runnerUpScore;
@property (nonatomic, readwrite) float runnerUpYawDegrees;
@property (nonatomic, readwrite) BOOL isAmbiguous;
@property (nonatomic, readwrite) float sourceBodyCutDepth;
@property (nonatomic, readwrite) float targetBodyCutDepth;
@end

@implementation SCAlignmentResult
@end

// MARK: - SCPointCloudMerging

@implementation SCPointCloudMerging

+ (SCAlignmentResult *)alignPointCloud:(SCPointCloud *)source toPointCloud:(SCPointCloud *)target
{
    return [self alignPointCloud:source toPointCloud:target startingFromRecordedPose:NO];
}

+ (SCAlignmentResult *)alignPointCloud:(SCPointCloud *)source
                          toPointCloud:(SCPointCloud *)target
              startingFromRecordedPose:(BOOL)startingFromRecordedPose
{
    // 1. Stand both scans upright
    Eigen::Matrix4f sourceUpright = uprightTransform([source gravity]);
    Eigen::Matrix4f targetUpright = uprightTransform([target gravity]);

    // Keep only the head (shirt, shoulders): the head turns on the neck between scans,
    // so the body isn't rigidly attached to the head, and cloth wrinkles differently.
    float targetCutDepth = 0, sourceCutDepth = 0;
    Cloud targetCloud = headOnly(sampleCloud(target, targetUpright, kTargetSampleCount), targetCutDepth);
    Cloud sourceHead = headOnly(sampleCloud(source, sourceUpright, kHeadPointCount), sourceCutDepth);
    Cloud searchCloud = pickByNormals(sourceHead, kSearchSampleCount);
    Cloud refineCloud = pickByNormals(sourceHead, kRefineSampleCount);

    SCAlignmentResult *result = [[SCAlignmentResult alloc] init];
    result.sourceBodyCutDepth = sourceCutDepth;
    result.targetBodyCutDepth = targetCutDepth;
    result.transform = matrix_identity_float4x4;
    result.rmsError = INFINITY;
    if (targetCloud.positions.empty() || searchCloud.positions.empty()) { return result; }

    // KD-trees for nearest-point lookups
    sc3d::Geometry targetIndex(toVec3(targetCloud.positions));
    sc3d::Geometry sourceIndex(toVec3(sourceHead.positions));
    Cloud seeThroughSource = pickByNormals(sourceHead, kSeeThroughSampleCount);
    Cloud seeThroughTarget = pickByNormals(targetCloud, kSeeThroughSampleCount);
    Scorer scorer{targetCloud, targetIndex, sourceHead, sourceIndex, seeThroughSource, seeThroughTarget};

    Eigen::Vector3f sourceCenter = searchCloud.centroid();
    Eigen::Vector3f targetCenter = targetCloud.centroid();

    std::vector<ICPResult> candidates;
    auto tryStart = [&](const Eigen::Matrix4f& start, bool nearRecordedPose) {
        ICPResult candidate = runICP(searchCloud, targetCloud, targetIndex, start,
                                     kSearchIterations, kSearchStartDistance, kSearchEndDistance);
        candidate.seeThrough = scorer.seeThrough(candidate.transform);
        candidate.nearRecordedPose = nearRecordedPose;
        candidates.push_back(candidate);
    };
    auto better = [](const ICPResult& a, const ICPResult& b) {
        if (std::fabs(a.rank() - b.rank()) > 0.01f) { return a.rank() > b.rank(); }
        return a.rms < b.rms;
    };
    auto bestOf = [&]() {
        const ICPResult *best = &candidates[0];
        for (const auto& candidate : candidates) {
            if (better(candidate, *best)) { best = &candidate; }
        }
        return *best;
    };

    // 2a. Near the recorded pose: as recorded, the source is already in the target's space
    // (source as recorded → target as recorded is "no move"), so in upright spaces the start
    // is targetUpright · sourceUpright⁻¹, turned a little about the head's vertical axis.
    if (startingFromRecordedPose) {
        Eigen::Matrix4f recorded = targetUpright * sourceUpright.inverse();
        Eigen::Vector3f center = targetCenter;
        for (float yaw = -kNearStartMaxYawDegrees; yaw <= kNearStartMaxYawDegrees + 0.01f;
             yaw += kNearStartYawStepDegrees) {
            Eigen::Affine3f turn = Eigen::Translation3f(center)
                                 * Eigen::AngleAxisf(yaw * (float)M_PI / 180.0f, Eigen::Vector3f::UnitY())
                                 * Eigen::Translation3f(-center);
            tryStart(turn.matrix() * recorded, true);
        }
    }

    // 2b. Full search: every turn, centers on top of each other
    for (int step = 0; step < kYawSteps; ++step) {
        float yaw = (float)step * 2.0f * (float)M_PI / (float)kYawSteps;
        Eigen::Affine3f start = Eigen::Translation3f(targetCenter)
                              * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitY())
                              * Eigen::Translation3f(-sourceCenter);
        tryStart(start.matrix(), false);
    }

    // 3. Best, and the best at a clearly different final turn
    ICPResult best = bestOf();
    float bestYaw = yawDegrees(best.transform);
    const ICPResult *runnerUp = nullptr;
    for (const auto& candidate : candidates) {
        if (yawDifference(yawDegrees(candidate.transform), bestYaw) <= kRunnerUpMinYawDifference) { continue; }
        if (runnerUp == nullptr || better(candidate, *runnerUp)) { runnerUp = &candidate; }
    }

    // 4. Refine with more points
    ICPResult refined = runICP(refineCloud, targetCloud, targetIndex, best.transform,
                               kRefineIterations, kRefineStartDistance, kRefineEndDistance);
    refined.seeThrough = scorer.seeThrough(refined.transform);
    refined.nearRecordedPose = best.nearRecordedPose;

    // Source (as recorded) → upright source → aligned → target (as recorded)
    Eigen::Matrix4f total = targetUpright.inverse() * refined.transform * sourceUpright;

    result.transform = toSimd(total);
    result.rmsError = refined.rms;
    result.overlap = refined.overlap;
    result.seeThrough = refined.seeThrough;
    result.score = refined.rank();
    result.usedRecordedPose = best.nearRecordedPose;
    result.yawDegrees = yawDegrees(refined.transform);
    result.runnerUpOverlap = runnerUp ? runnerUp->overlap : 0;
    result.runnerUpScore = runnerUp ? runnerUp->rank() : 0;
    result.runnerUpYawDegrees = runnerUp ? yawDegrees(runnerUp->transform) : 0;
    result.isAmbiguous = runnerUp != nullptr && runnerUp->rank() > 0
                      && runnerUp->rank() >= kAmbiguousScoreRatio * best.rank();
    return result;
}

+ (SCPointCloud *)headOnlyPointCloud:(SCPointCloud *)pointCloud
{
    Eigen::Matrix4f upright = uprightTransform([pointCloud gravity]);
    HeadCut cut;
    float cutDepth = 0;
    if (!findHeadCut(sampleCloud(pointCloud, upright, kTargetSampleCount), cut, cutDepth)) {
        return pointCloud;
    }
    Eigen::Matrix3f R = upright.block<3, 3>(0, 0);
    Eigen::Vector3f t = upright.block<3, 1>(0, 3);
    const Surfel *surfels = (const Surfel *)[[pointCloud pointsData] bytes];
    size_t count = (size_t)[pointCloud pointCount];
    std::vector<Surfel> kept;
    kept.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (cut.keeps(R * surfels[i].position + t)) { kept.push_back(surfels[i]); }
    }
    NSData *data = [NSData dataWithBytes:kept.data() length:kept.size() * sizeof(Surfel)];
    return [[SCPointCloud alloc] initWithSurfelData:data gravity:[pointCloud gravity]];
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
