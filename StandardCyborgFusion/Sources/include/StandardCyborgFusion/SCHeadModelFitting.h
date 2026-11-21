//
//  SCHeadModelFitting.h
//  StandardCyborgFusion
//
//  Fits a statistical head model (FLAME) to a scan: finds the head's position,
//  turn, and shape coefficients so the model's surface lies on the scan's points.
//  Where the scan has no data (e.g. the back of a front-only scan), the model
//  fills in a typical head for that shape.
//

#import <Foundation/Foundation.h>
#import <simd/simd.h>
#import <StandardCyborgFusion/SCPointCloud.h>

NS_ASSUME_NONNULL_BEGIN

/** A linear head shape model: an average head plus shape components.
    Loaded from the files written by tools/flame/convert_flame.py. */
@interface SCHeadModel : NSObject

/** `modelPath`: FLAMEHead.bin. `masksPath`: FLAMEMasks.bin (vertex groups such as
    scalp, face, eyeballs, neck). */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                 masksPath:(NSString *)masksPath
                                     error:(NSError **)error;

@property (nonatomic, readonly) NSInteger vertexCount;
@property (nonatomic, readonly) NSInteger triangleCount;
@property (nonatomic, readonly) NSInteger shapeComponentCount;

/** The turn (degrees about the vertical axis, model → scan) expected for a scan of the
    given side: "front", "left", "right", "back". nil for anything else. */
- (nullable NSNumber *)expectedYawDegreesForLabel:(NSString *)label NS_SWIFT_NAME(expectedYawDegrees(forLabel:));

@end

/** The result of fitting the head model to a scan. Positions are in the scan's own space. */
@interface SCHeadFitResult : NSObject

/** Fitted head: vertexCount × 3 float32 positions (meters), and triangleCount × 3 uint32 indices. */
@property (nonatomic, readonly) NSData *vertices;
@property (nonatomic, readonly) NSData *triangles;

/** Shape coefficients (float32, one per component); ~0 is average, ±2 is unusual. */
@property (nonatomic, readonly) NSData *shapeCoefficients;

/** Distances between scan points and the fitted surface (meters), over the scan points
    that matched the head (within 2 cm): root-mean-square, median, and 90th percentile. */
@property (nonatomic, readonly) float rmsError;
@property (nonatomic, readonly) float medianError;
@property (nonatomic, readonly) float p90Error;

/** RMS distance split by head region (meters; NaN if no scan points there). */
@property (nonatomic, readonly) float scalpRmsError;
@property (nonatomic, readonly) float faceRmsError;

/** Fraction (0–1) of the scan's head points within 1 cm of the fitted surface. */
@property (nonatomic, readonly) float coverage;

/** Chosen turn about the vertical axis (model → scan, degrees 0–360). */
@property (nonatomic, readonly) float yawDegrees;

/** Where the scan's body was cut off (meters below the top of the head; 0 = not cut). */
@property (nonatomic, readonly) float bodyCutDepth;

/** A sample of the scan's head points (N × 3 float32, scan space) and each one's signed
    distance to the fitted surface (N float32, meters; positive = outside the head). */
@property (nonatomic, readonly) NSData *scanPoints;
@property (nonatomic, readonly) NSData *scanDistances;

@end

@interface SCHeadModelFitting : NSObject

/** Fits `model` to `scan` (which needs gravity). `expectedYawDegrees` (from the scan's
    label) narrows the turn search; pass nil to search all turns.
    If `fixedShape` is given, only position and turn are fitted (shape stays as given):
    use this to check how well a fitted head predicts a different scan.
    Takes a second or two: call off the main thread. */
+ (SCHeadFitResult *)fitModel:(SCHeadModel *)model
                 toPointCloud:(SCPointCloud *)scan
           expectedYawDegrees:(nullable NSNumber *)expectedYawDegrees
                   fixedShape:(nullable NSData *)fixedShape
    NS_SWIFT_NAME(fit(_:to:expectedYawDegrees:fixedShape:));

@end

NS_ASSUME_NONNULL_END
