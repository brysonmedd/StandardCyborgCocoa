//
//  SCPointCloudMerging.h
//  StandardCyborgFusion
//
//  Lines up two partial scans of the same object and merges them.
//
//  Alignment uses each scan's gravity to stand both upright, searches turns
//  about the vertical axis, and refines the best with point-to-plane ICP.
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

/** The best alignment found at a clearly different turn (> 20° away), to judge
    whether the choice was ambiguous. Overlap is 0 if there was none. */
@property (nonatomic, readonly) float runnerUpOverlap;
@property (nonatomic, readonly) float runnerUpYawDegrees;

/** Where the body was cut off before lining up: meters below the top of the head
    (the neck). 0 means no neck was found and nothing was cut. Points below the cut
    are ignored for lining up but still kept in the merged result. */
@property (nonatomic, readonly) float sourceBodyCutDepth;
@property (nonatomic, readonly) float targetBodyCutDepth;

@end

@interface SCPointCloudMerging : NSObject

/** Finds the transform that puts `source` onto `target`. Both need gravity.
    Slow-ish (a second or so): call off the main thread. */
+ (SCAlignmentResult *)alignPointCloud:(SCPointCloud *)source
                          toPointCloud:(SCPointCloud *)target
    NS_SWIFT_NAME(align(_:onto:));

/** Returns a new point cloud with `target`'s points plus `source`'s points moved by
    `transform`. Overlapping areas keep both sets of points (no averaging), so any
    misalignment stays visible. The result uses `target`'s gravity. */
+ (SCPointCloud *)mergePointCloud:(SCPointCloud *)source
                    transformedBy:(simd_float4x4)transform
                   intoPointCloud:(SCPointCloud *)target
    NS_SWIFT_NAME(merge(_:transformedBy:into:));

@end

NS_ASSUME_NONNULL_END
