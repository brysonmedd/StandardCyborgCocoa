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

/** The model's triangles: triangleCount × 3 uint32 vertex indices. */
@property (nonatomic, readonly) NSData *triangles;

/** The head for the given shape coefficients (float32, as in SCHeadFitResult.shapeCoefficients),
    in the model's own head space: meters, y = up, z = out of the face. Returns
    vertexCount × 3 float32 positions. Missing coefficients count as 0 (average). */
- (NSData *)verticesForShapeCoefficients:(NSData *)shapeCoefficients NS_SWIFT_NAME(vertices(forShape:));

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

/** Fraction (0–1) of the scan's head points within 1 cm of the fitted surface. Points
    nearest the model's neck aren't head points: they're left out of the fit and of all
    these numbers (their scanDistances are NaN). */
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

/** The result of lining up a face mask (e.g. ARKit's face geometry) with a fitted head. */
@interface SCFaceLineUpResult : NSObject

/** Transform from the mask's space (face space) to the head's space. */
@property (nonatomic, readonly) simd_float4x4 faceToHead;

/** For each mask point: where it landed on the head's surface (N × 3 float32, head space),
    and whether it was used (N uint8: 1 = matched the head's face, 0 = ignored). */
@property (nonatomic, readonly) NSData *matchedHeadPoints;
@property (nonatomic, readonly) NSData *matchedFlags;

/** Root-mean-square distance between the used mask points and the head's surface (meters). */
@property (nonatomic, readonly) float rmsError;
/** How many mask points were used, and how many rounds the snapping took. */
@property (nonatomic, readonly) NSInteger usedPointCount;
@property (nonatomic, readonly) NSInteger iterations;

@end

@interface SCFaceLineUp : NSObject

/** Snaps `facePoints` (N × 3 float32, meters, y = up, z = out of the face, e.g. ARKit's
    face mesh in face-anchor space) onto the face of the head with shape `shapeCoefficients`.
    Starts with no turn and the nose tips together, then repeats "find nearest, nudge" (rigid
    ICP) on the face region only (no lips, eyes, ears, scalp or neck). Fast: call it anywhere. */
+ (SCFaceLineUpResult *)lineUpFacePoints:(NSData *)facePoints
                                withHead:(SCHeadModel *)model
                                   shape:(NSData *)shapeCoefficients
    NS_SWIFT_NAME(lineUp(facePoints:head:shape:));

@end

/** Head regions reported by SCHeadSurfaceCheck. */
typedef NS_ENUM(uint8_t, SCHeadRegion) {
    SCHeadRegionNone = 0,      // farther than the gate from the head (hair, background, hands)
    SCHeadRegionFace = 1,
    SCHeadRegionForehead = 2,
    SCHeadRegionScalp = 3,
    SCHeadRegionLeftEar = 4,   // the subject's own left
    SCHeadRegionRightEar = 5,
    SCHeadRegionOther = 6,     // neck, boundary
};

/** The result of snapping measured points onto a head (SCHeadSurfaceCheck.snap). */
@interface SCHeadSnapResult : NSObject
/** Moves the points (head space) onto the head: corrected = correction × point. */
@property (nonatomic, readonly) simd_float4x4 correction;
/** Distance left between the used points and the head (meters, root-mean-square). */
@property (nonatomic, readonly) float rmsError;
@property (nonatomic, readonly) NSInteger usedPointCount;
@property (nonatomic, readonly) NSInteger iterations;
@end

/** Distances from measured points to a fixed head's surface. */
@interface SCHeadSurfaceCheck : NSObject

/** Builds a fast nearest-point search over the head with shape `shapeCoefficients`. */
- (instancetype)initWithModel:(SCHeadModel *)model shape:(NSData *)shapeCoefficients;

/** For points in head space (N × 3 float32): fills `distances` (N float32, meters, signed:
    positive = outside the head) and `regions` (N uint8 SCHeadRegion; None if the point is
    farther than `gate` meters from the head). */
- (void)measurePoints:(NSData *)points
                 gate:(float)gate
            distances:(NSMutableData *)distances
              regions:(NSMutableData *)regions
    NS_SWIFT_NAME(measure(points:gate:distances:regions:));

/** Snaps points (head space, N × 3 float32) onto the head's firm, hair-free parts (face
    without lips and eyes, nose, ears) with a rigid "find nearest, nudge, repeat".
    Used to correct a tracked head pose with measured depth. */
- (SCHeadSnapResult *)snapPoints:(NSData *)points
                       startGate:(float)startGate
                         endGate:(float)endGate
                      iterations:(NSInteger)iterations
    NS_SWIFT_NAME(snap(points:startGate:endGate:iterations:));

@end

NS_ASSUME_NONNULL_END
