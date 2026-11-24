//
//  SCPointCloudMerging.h
//  StandardCyborgFusion
//
//  Lines up two partial scans of the same object and merges them.
//
//  Alignment uses each scan's gravity to stand both upright, searches turns
//  about the vertical axis, and refines the best with point-to-plane ICP.
//  Positions where one scan would sit in front of the other's surface ("see-through")
//  are penalized, and only the head is used (the body is cut off).
//

#import <Foundation/Foundation.h>
#import <simd/simd.h>
#import <StandardCyborgFusion/SCPointCloud.h>

NS_ASSUME_NONNULL_BEGIN

/** The result of lining up a source scan onto a target scan. */
@interface SCAlignmentResult : NSObject

/** Maps points from the source scan's space into the target scan's space (meters). */
@property (nonatomic, readonly) simd_float4x4 transform;

/** Root-mean-square point-to-plane distance (meters) of the matched overlapping points. */
@property (nonatomic, readonly) float rmsError;

/** Fraction (0–1) of the source's points that ended up on the target's surface. */
@property (nonatomic, readonly) float overlap;

/** Turn about the vertical axis (degrees, 0–360) of the chosen alignment. */
@property (nonatomic, readonly) float yawDegrees;

/** Fraction (0–1) of points that sit clearly in front of the other scan's surface, where
    that scan saw empty space (checked both ways, averaged). A correct alignment is near 0;
    a doubled face or nose gives a large value. */
@property (nonatomic, readonly) float seeThrough;

/** How the alignment was ranked: overlap − 2 × seeThrough, plus 0.05 if it came from near the
    recorded pose. Higher is better. */
@property (nonatomic, readonly) float score;

/** True if the result came from the candidates near the recorded pose (see
    align(_:onto:startingFromRecordedPose:)); false if from the full search. */
@property (nonatomic, readonly) BOOL usedRecordedPose;

/** The best alignment found at a clearly different turn (> 20° away), to judge
    whether the choice was ambiguous. Overlap and score are 0 if there was none. */
@property (nonatomic, readonly) float runnerUpOverlap;
@property (nonatomic, readonly) float runnerUpScore;
@property (nonatomic, readonly) float runnerUpYawDegrees;

/** True if the runner-up scored almost as well (≥ 80% of the best score). */
@property (nonatomic, readonly) BOOL isAmbiguous;

/** Where the body was cut off before lining up: meters below the top of the head
    (just below the chin). 0 means nothing was cut. Points below the cut
    are ignored for lining up but still kept in the merged result. */
@property (nonatomic, readonly) float sourceBodyCutDepth;
@property (nonatomic, readonly) float targetBodyCutDepth;

@end

@interface SCPointCloudMerging : NSObject

/** Finds the transform that puts `source` onto `target`. Both need gravity.
    Searches every turn about the vertical axis. Slow-ish (a second or so): call off the
    main thread. */
+ (SCAlignmentResult *)alignPointCloud:(SCPointCloud *)source
                          toPointCloud:(SCPointCloud *)target
    NS_SWIFT_NAME(align(_:onto:));

/** Like align(_:onto:). With `startingFromRecordedPose`, the scans are assumed to already
    be roughly lined up as recorded: each scan is stored in the camera's space at its first
    frame, so two scans that start from the same view (phone on a stand, person facing it)
    start out nearly on top of each other. Turns within ±25° of that are tried first and get
    a small head start in the ranking; the full search still runs, so a clearly better
    answer wins if the recorded pose was wrong (e.g. the stand moved). */
+ (SCAlignmentResult *)alignPointCloud:(SCPointCloud *)source
                          toPointCloud:(SCPointCloud *)target
              startingFromRecordedPose:(BOOL)startingFromRecordedPose
    NS_SWIFT_NAME(align(_:onto:startingFromRecordedPose:));

/** Returns a new point cloud with `target`'s points plus `source`'s points moved by
    `transform`. Overlapping areas keep both sets of points (no averaging), so any
    misalignment stays visible. The result uses `target`'s gravity. */
+ (SCPointCloud *)mergePointCloud:(SCPointCloud *)source
                    transformedBy:(simd_float4x4)transform
                   intoPointCloud:(SCPointCloud *)target
    NS_SWIFT_NAME(merge(_:transformedBy:into:));

/** Only the head of a scan (needs gravity, and must be in the camera's space at the scan's
    first frame, with the face toward the camera then, as scans are recorded). Cuts just
    below the chin, found from the face's side profile, and drops shoulders beside the neck.
    Without a visible face it cuts at the neck, or 26 cm below the top of the head.
    Returns `pointCloud` itself if there's nothing to cut. For showing scans without the
    shirt and shoulders. */
+ (SCPointCloud *)headOnlyPointCloud:(SCPointCloud *)pointCloud
    NS_SWIFT_NAME(headOnly(_:));

@end

NS_ASSUME_NONNULL_END
