//
//  SCHeadModelFitting.mm
//  StandardCyborgFusion
//
//  The head model is linear: vertices = average + Σ coefficient_k × component_k.
//  Fitting finds a rigid placement (rotation R, shift t) and coefficients β that put
//  the model's surface on the scan, while a "realism" penalty keeps β near 0
//  (the components are scaled so that β ~ N(0, 1) across real heads).
//
//  Steps:
//   1. Stand the scan upright with its gravity and cut off the body just below the chin
//      (same as merging). The model is upright too (y = up, z = toward the face).
//   2. Placement search: for several turns about the vertical axis, place the average
//      head on the scan and run a short rigid ICP. Keep the best.
//   3. Joint fit in stages: rigid + the first 10 components with a strong penalty,
//      then 40, then all, each time relaxing the penalty. Each iteration matches scan
//      points to the nearest model vertex and solves one Gauss-Newton step on the
//      point-to-plane distances. Points nearest the model's neck are skipped: the neck
//      under the chin is in the scan, but the model's neck isn't fitted.
//

#import "SCHeadModelFitting.h"
#import "ScanPreparation.hpp"

#import <standard_cyborg/sc3d/Geometry.hpp>

#include <map>
#include <string>

using namespace scanprep;

// MARK: - Tuning

static const size_t kFitSampleCount = 6000;     // scan points used for fitting
static const size_t kEvalSampleCount = 20000;   // scan points used for the reported numbers
static const float  kNoiseSigma = 0.002f;       // assumed scan noise (m): weighs data vs. penalty
static const float  kMinNormalAgreement = 0.3f;
static const float  kMatchDistance = 0.02f;     // scan points farther than this count as "not head"
static const float  kCoverageDistance = 0.01f;
static const float  kLargeDistance = 1e9f;      // stands in for "infinitely far"

// This library is compiled with -ffast-math, which assumes NaN and infinity never occur
// (std::isnan etc. become constants). So this file never relies on them: "no value" is
// passed as nil, and "very far" is kLargeDistance.

struct Stage { int components; float penalty; int iterations; float startGate; float endGate; };
static const Stage kRigidSearch   = {  0,  0.0f, 12, 0.04f,  0.015f };
static const Stage kStages[] = {
    { 10, 20.0f, 10, 0.020f, 0.010f },
    { 40,  5.0f, 10, 0.015f, 0.008f },
    {  0,  1.0f, 15, 0.010f, 0.006f },  // 0 = all components
};

// MARK: - SCHeadModel

@implementation SCHeadModel {
    @public
    int _vertexCount, _triangleCount, _componentCount;
    std::vector<Eigen::Vector3f> _template;
    std::vector<uint32_t> _triangles;
    std::vector<float> _shape;                 // [component][vertex][xyz]
    std::map<std::string, std::vector<int>> _masks;
    std::vector<bool> _usableForFitting;       // false for eyeballs and neck
    std::vector<bool> _searchedForFitting;     // usable, plus the neck: scan points nearest the
                                               // neck are found there and then skipped, instead
                                               // of being forced onto the jaw
    std::vector<uint8_t> _region;              // 0 = other, 1 = scalp/forehead, 2 = face
    float _subjectLeftSign;                    // +1 if the subject's left is +x
}

static NSError *ModelError(NSString *message)
{
    return [NSError errorWithDomain:@"SCHeadModel" code:1 userInfo:@{ NSLocalizedDescriptionKey: message }];
}

- (instancetype)initWithModelPath:(NSString *)modelPath masksPath:(NSString *)masksPath error:(NSError **)error
{
    self = [super init];
    if (!self) { return nil; }

    // --- Model file ---
    NSData *model = [NSData dataWithContentsOfFile:modelPath];
    if (model.length < 20 || memcmp(model.bytes, "FLMH", 4) != 0) {
        if (error) { *error = ModelError(@"The head model file is missing or not a FLAMEHead.bin."); }
        return nil;
    }
    const uint8_t *bytes = (const uint8_t *)model.bytes;
    uint32_t header[4];
    memcpy(header, bytes + 4, sizeof(header));
    _vertexCount = (int)header[1];
    _triangleCount = (int)header[2];
    _componentCount = (int)header[3];
    size_t expected = 20 + (size_t)_vertexCount * 12 + (size_t)_triangleCount * 12
                    + (size_t)_componentCount * _vertexCount * 12;
    if (header[0] != 1 || model.length != expected) {
        if (error) { *error = ModelError(@"The head model file has an unexpected size or version."); }
        return nil;
    }

    const float *floats = (const float *)(bytes + 20);
    _template.resize(_vertexCount);
    for (int v = 0; v < _vertexCount; ++v) {
        _template[v] = Eigen::Vector3f(floats[3 * v], floats[3 * v + 1], floats[3 * v + 2]);
    }
    const uint32_t *tris = (const uint32_t *)(bytes + 20 + (size_t)_vertexCount * 12);
    _triangles.assign(tris, tris + (size_t)_triangleCount * 3);
    const float *shape = (const float *)(bytes + 20 + (size_t)_vertexCount * 12 + (size_t)_triangleCount * 12);
    _shape.assign(shape, shape + (size_t)_componentCount * _vertexCount * 3);

    // --- Masks file ---
    NSData *masks = [NSData dataWithContentsOfFile:masksPath];
    if (masks.length < 12 || memcmp(masks.bytes, "FLMK", 4) != 0) {
        if (error) { *error = ModelError(@"The head model masks file (FLAMEMasks.bin) is missing. Run tools/flame/convert_flame.py."); }
        return nil;
    }
    const uint8_t *m = (const uint8_t *)masks.bytes;
    const uint8_t *end = m + masks.length;
    uint32_t maskHeader[2];
    memcpy(maskHeader, m + 4, sizeof(maskHeader));
    m += 12;
    for (uint32_t i = 0; i < maskHeader[1] && m < end; ++i) {
        uint8_t nameLength = *m++;
        std::string name((const char *)m, nameLength);
        m += nameLength;
        uint32_t count;
        memcpy(&count, m, 4);
        m += 4;
        if (m + (size_t)count * 4 > end) { break; }
        std::vector<int> indices(count);
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t index;
            memcpy(&index, m + 4 * k, 4);
            indices[k] = (int)index;
        }
        m += (size_t)count * 4;
        _masks[name] = indices;
    }

    // Which vertices take part in fitting, and which region each is in
    _usableForFitting.assign(_vertexCount, true);
    for (const char *name : { "left_eyeball", "right_eyeball", "neck", "boundary" }) {
        for (int v : _masks[name]) { if (v >= 0 && v < _vertexCount) { _usableForFitting[v] = false; } }
    }
    _searchedForFitting = _usableForFitting;
    for (const char *name : { "neck", "boundary" }) {
        for (int v : _masks[name]) { if (v >= 0 && v < _vertexCount) { _searchedForFitting[v] = true; } }
    }
    _region.assign(_vertexCount, 0);
    for (const char *name : { "scalp", "forehead" }) {
        for (int v : _masks[name]) { if (v >= 0 && v < _vertexCount) { _region[v] = 1; } }
    }
    for (int v : _masks["face"]) { if (v >= 0 && v < _vertexCount && _region[v] == 0) { _region[v] = 2; } }

    // Which way is the subject's left? (Decides the turn for "left"/"right" scans.)
    float leftEarX = 0;
    for (int v : _masks["left_ear"]) { leftEarX += _template[v].x(); }
    _subjectLeftSign = leftEarX >= 0 ? 1.0f : -1.0f;

    return self;
}

- (NSInteger)vertexCount { return _vertexCount; }
- (NSInteger)triangleCount { return _triangleCount; }
- (NSInteger)shapeComponentCount { return _componentCount; }

- (NSData *)triangles
{
    return [NSData dataWithBytes:_triangles.data() length:_triangles.size() * sizeof(uint32_t)];
}

- (NSData *)verticesForShapeCoefficients:(NSData *)shapeCoefficients
{
    size_t n = std::min<size_t>(_componentCount, shapeCoefficients.length / sizeof(float));
    std::vector<float> beta(n);
    if (n > 0) { memcpy(beta.data(), shapeCoefficients.bytes, n * sizeof(float)); }
    std::vector<Eigen::Vector3f> vertices = [self verticesForShape:beta];
    NSMutableData *data = [NSMutableData dataWithLength:vertices.size() * 3 * sizeof(float)];
    float *out = (float *)data.mutableBytes;
    for (size_t v = 0; v < vertices.size(); ++v) {
        out[3 * v] = vertices[v].x(); out[3 * v + 1] = vertices[v].y(); out[3 * v + 2] = vertices[v].z();
    }
    return data;
}

- (NSNumber *)expectedYawDegreesForLabel:(NSString *)label
{
    // Model → scan turn about y. In a scan, the scanned side faces the camera (+z).
    // A rotation by θ about y sends +x to (cos θ, 0, -sin θ), so the subject's left
    // (±x) ends up facing +z when θ = ∓90°.
    NSString *l = label.lowercaseString;
    if ([l isEqualToString:@"front"]) { return @0; }
    if ([l isEqualToString:@"back"]) { return @180; }
    if ([l isEqualToString:@"left"]) { return _subjectLeftSign > 0 ? @270 : @90; }
    if ([l isEqualToString:@"right"]) { return _subjectLeftSign > 0 ? @90 : @270; }
    return nil;
}

/** Model vertices for coefficients β (only the first β.size() components). */
- (std::vector<Eigen::Vector3f>)verticesForShape:(const std::vector<float>&)beta
{
    std::vector<Eigen::Vector3f> result = _template;
    int k = std::min((int)beta.size(), _componentCount);
    for (int c = 0; c < k; ++c) {
        if (beta[c] == 0) { continue; }
        const float *component = _shape.data() + (size_t)c * _vertexCount * 3;
        for (int v = 0; v < _vertexCount; ++v) {
            result[v] += beta[c] * Eigen::Vector3f(component[3 * v], component[3 * v + 1], component[3 * v + 2]);
        }
    }
    return result;
}

/** Area-weighted vertex normals, pointing out of the head. */
- (std::vector<Eigen::Vector3f>)normalsForVertices:(const std::vector<Eigen::Vector3f>&)vertices
{
    std::vector<Eigen::Vector3f> normals(_vertexCount, Eigen::Vector3f::Zero());
    for (int f = 0; f < _triangleCount; ++f) {
        uint32_t a = _triangles[3 * f], b = _triangles[3 * f + 1], c = _triangles[3 * f + 2];
        Eigen::Vector3f n = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]);
        normals[a] += n; normals[b] += n; normals[c] += n;
    }
    // Make sure they point outward (FLAME's triangle order decides the sign)
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& v : vertices) { center += v; }
    center /= (float)vertices.size();
    double outward = 0;
    for (int v = 0; v < _vertexCount; ++v) {
        if (!_usableForFitting[v]) { continue; }
        outward += normals[v].dot(vertices[v] - center);
    }
    float sign = outward >= 0 ? 1.0f : -1.0f;
    for (auto& n : normals) {
        float length = n.norm();
        n = length > 1e-12f ? Eigen::Vector3f(sign * n / length) : Eigen::Vector3f(0, 0, 1);
    }
    return normals;
}

@end

// MARK: - SCHeadFitResult

@interface SCHeadFitResult ()
@property (nonatomic, readwrite) NSData *vertices;
@property (nonatomic, readwrite) NSData *triangles;
@property (nonatomic, readwrite) NSData *shapeCoefficients;
@property (nonatomic, readwrite) float rmsError;
@property (nonatomic, readwrite) float medianError;
@property (nonatomic, readwrite) float p90Error;
@property (nonatomic, readwrite) float scalpRmsError;
@property (nonatomic, readwrite) float faceRmsError;
@property (nonatomic, readwrite) float coverage;
@property (nonatomic, readwrite) float yawDegrees;
@property (nonatomic, readwrite) float bodyCutDepth;
@property (nonatomic, readwrite) NSData *scanPoints;
@property (nonatomic, readwrite) NSData *scanDistances;
@end

@implementation SCHeadFitResult
@end

// MARK: - Fitting

namespace {

struct Placement {
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
};

struct Matches {
    std::vector<int> scanIndex;
    std::vector<int> vertex;
};

/** Model vertices placed in the (upright) scan space. */
std::vector<Eigen::Vector3f> place(const std::vector<Eigen::Vector3f>& vertices, const Placement& p)
{
    std::vector<Eigen::Vector3f> result(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) { result[i] = p.R * vertices[i] + p.t; }
    return result;
}

/** A nearest-vertex index over the vertices usable for fitting. */
struct VertexIndex {
    std::vector<int> ids;
    std::unique_ptr<sc3d::Geometry> geometry;

    VertexIndex(const std::vector<Eigen::Vector3f>& placed, const std::vector<bool>& usable) {
        std::vector<math::Vec3> positions;
        for (size_t v = 0; v < placed.size(); ++v) {
            if (!usable[v]) { continue; }
            ids.push_back((int)v);
            positions.emplace_back(placed[v].x(), placed[v].y(), placed[v].z());
        }
        geometry.reset(new sc3d::Geometry(positions));
    }

    int nearest(const Eigen::Vector3f& p) const {
        return ids[geometry->getClosestVertexIndex(math::Vec3(p.x(), p.y(), p.z()))];
    }
};

/** One fitting step: match, then solve for a small rigid change (+ shape change if
    `components` > 0). Returns the number of matches used. */
int step(SCHeadModel *model, const Cloud& scan, std::vector<float>& beta, int components,
         float penalty, float gate, Placement& placement)
{
    std::vector<Eigen::Vector3f> vertices = [model verticesForShape:beta];
    std::vector<Eigen::Vector3f> placed = place(vertices, placement);
    std::vector<Eigen::Vector3f> normals = [model normalsForVertices:vertices];
    VertexIndex index(placed, model->_searchedForFitting);

    const int unknowns = 6 + components;
    std::vector<int> rowsScan, rowsVertex;
    std::vector<float> weights;
    rowsScan.reserve(scan.positions.size());

    for (size_t i = 0; i < scan.positions.size(); ++i) {
        const Eigen::Vector3f& p = scan.positions[i];
        int v = index.nearest(p);
        if (!model->_usableForFitting[v]) { continue; }  // on the neck: not fitted
        Eigen::Vector3f n = placement.R * normals[v];
        float distanceSquared = (placed[v] - p).squaredNorm();
        if (distanceSquared > gate * gate) { continue; }
        if (scan.normals[i].dot(n) < kMinNormalAgreement) { continue; }
        float u = 1.0f - distanceSquared / (gate * gate);  // Tukey weight
        rowsScan.push_back((int)i);
        rowsVertex.push_back(v);
        weights.push_back(u * u / (kNoiseSigma * kNoiseSigma));
    }
    int rows = (int)rowsScan.size();
    if (rows < 50) { return rows; }

    // Jacobian of r = n · (q - p), with q = R m + t, for a small rotation ω and shift τ
    // applied after the current placement, and for each shape coefficient.
    Eigen::MatrixXf J(rows, unknowns);
    Eigen::VectorXf r(rows);
    for (int row = 0; row < rows; ++row) {
        int v = rowsVertex[row];
        const Eigen::Vector3f& q = placed[v];
        Eigen::Vector3f n = placement.R * normals[v];
        r(row) = n.dot(q - scan.positions[rowsScan[row]]);
        J.block<1, 3>(row, 0) = q.cross(n).transpose();
        J.block<1, 3>(row, 3) = n.transpose();
        if (components > 0) {
            Eigen::Vector3f modelNormal = placement.R.transpose() * n;  // shape moves in model space
            for (int c = 0; c < components; ++c) {
                const float *s = model->_shape.data() + ((size_t)c * model->_vertexCount + v) * 3;
                J(row, 6 + c) = modelNormal.x() * s[0] + modelNormal.y() * s[1] + modelNormal.z() * s[2];
            }
        }
    }

    Eigen::VectorXf sqrtW = Eigen::Map<Eigen::VectorXf>(weights.data(), rows).cwiseSqrt();
    Eigen::MatrixXf Jw = sqrtW.asDiagonal() * J;
    Eigen::MatrixXd A = (Jw.transpose() * Jw).cast<double>();
    Eigen::VectorXd b = -(Jw.transpose() * (sqrtW.asDiagonal() * r)).cast<double>();

    // Realism penalty: penalty × ||β||², applied to the coefficients after this step
    for (int c = 0; c < components; ++c) {
        A(6 + c, 6 + c) += penalty;
        b(6 + c) -= penalty * beta[c];
    }
    // A tiny damping keeps the solve well-behaved
    for (int k = 0; k < unknowns; ++k) { A(k, k) += 1e-6 * (A(k, k) + 1.0); }

    Eigen::VectorXd x = A.ldlt().solve(b);
    Eigen::Vector3d omega = x.segment<3>(0);
    Eigen::Vector3f tau = x.segment<3>(3).cast<float>();
    double angle = omega.norm();
    Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
    if (angle > 1e-12) {
        dR = Eigen::AngleAxisd(angle, omega / angle).toRotationMatrix().cast<float>();
    }
    placement.R = dR * placement.R;
    placement.t = dR * placement.t + tau;
    for (int c = 0; c < components; ++c) { beta[c] += (float)x(6 + c); }
    return rows;
}

/** How well the model fits: fraction of scan points within `coverageDistance`, and RMS. */
void evaluate(SCHeadModel *model, const Cloud& scan, const std::vector<float>& beta,
              const Placement& placement, float& coverage, float& rms)
{
    std::vector<Eigen::Vector3f> placed = place([model verticesForShape:beta], placement);
    VertexIndex index(placed, model->_searchedForFitting);
    size_t close = 0;
    double sum = 0;
    for (const auto& p : scan.positions) {
        int v = index.nearest(p);
        // Points on the model's neck count as not covered, so a placement can't score well
        // by putting scan points there.
        if (!model->_usableForFitting[v]) { continue; }
        float d = (placed[v] - p).norm();
        if (d < kCoverageDistance) { ++close; sum += d * d; }
    }
    coverage = scan.positions.empty() ? 0 : (float)close / (float)scan.positions.size();
    rms = close > 0 ? (float)std::sqrt(sum / close) : kLargeDistance;
}

/** Places the model on the scan at turn `yawDegrees`: matches the tops of the heads,
    and the middle of the model's camera-facing side to the middle of the scan. */
Placement initialPlacement(SCHeadModel *model, const std::vector<Eigen::Vector3f>& vertices,
                           const Cloud& scan, float yawDegrees)
{
    Placement p;
    p.R = Eigen::AngleAxisf(yawDegrees * (float)M_PI / 180.0f, Eigen::Vector3f::UnitY()).toRotationMatrix();
    std::vector<Eigen::Vector3f> normals = [model normalsForVertices:vertices];

    Eigen::Vector3f visibleSum = Eigen::Vector3f::Zero();
    int visibleCount = 0;
    float modelTop = -kLargeDistance;
    for (size_t v = 0; v < vertices.size(); ++v) {
        if (!model->_usableForFitting[v]) { continue; }
        Eigen::Vector3f q = p.R * vertices[v];
        modelTop = std::max(modelTop, q.y());
        if ((p.R * normals[v]).z() > 0.3f) { visibleSum += q; ++visibleCount; }
    }
    Eigen::Vector3f visibleCenter = visibleCount > 0 ? Eigen::Vector3f(visibleSum / visibleCount) : Eigen::Vector3f::Zero();

    std::vector<float> heights;
    for (const auto& q : scan.positions) { heights.push_back(q.y()); }
    float scanTop = percentile(heights, 0.998f);
    Eigen::Vector3f scanCenter = scan.centroid();

    p.t = Eigen::Vector3f(scanCenter.x() - visibleCenter.x(),
                          scanTop - modelTop,
                          scanCenter.z() - visibleCenter.z());
    return p;
}

NSData *floatData(const std::vector<float>& values)
{
    return [NSData dataWithBytes:values.data() length:values.size() * sizeof(float)];
}

} // namespace

@implementation SCHeadModelFitting

+ (SCHeadFitResult *)fitModel:(SCHeadModel *)model
                 toPointCloud:(SCPointCloud *)scan
           expectedYawDegrees:(NSNumber *)expectedYawDegrees
                   fixedShape:(NSData *)fixedShape
{
    SCHeadFitResult *result = [[SCHeadFitResult alloc] init];

    // 1. Upright scan, body cut off just below the chin (found from the face's side profile)
    Eigen::Matrix4f upright = uprightTransform([scan gravity]);
    float cutDepth = 0;
    Cloud head = headOnly(sampleCloud(scan, upright, 200000), cutDepth);
    Cloud fitCloud = pickByNormals(head, kFitSampleCount);
    Cloud evalCloud;  // evenly spaced sample of the head points, for reporting
    {
        size_t step = std::max<size_t>(1, head.positions.size() / kEvalSampleCount);
        for (size_t i = 0; i < head.positions.size(); i += step) {
            evalCloud.positions.push_back(head.positions[i]);
            evalCloud.normals.push_back(head.normals[i]);
        }
    }
    result.bodyCutDepth = cutDepth;

    const int K = model->_componentCount;
    std::vector<float> beta(K, 0.0f);
    bool shapeIsFixed = fixedShape != nil && fixedShape.length >= sizeof(float);
    if (shapeIsFixed) {
        size_t n = std::min<size_t>(K, fixedShape.length / sizeof(float));
        memcpy(beta.data(), fixedShape.bytes, n * sizeof(float));
    }

    // 2. Placement search over turns
    std::vector<float> yaws;
    if (expectedYawDegrees != nil) {
        float expected = expectedYawDegrees.floatValue;
        for (int d = -60; d <= 60; d += 20) { yaws.push_back(expected + d); }
    } else {
        for (int d = 0; d < 360; d += 30) { yaws.push_back((float)d); }
    }

    std::vector<Eigen::Vector3f> startVertices = [model verticesForShape:beta];
    Placement best;
    float bestCoverage = -1, bestRms = kLargeDistance;
    for (float yaw : yaws) {
        Placement p = initialPlacement(model, startVertices, fitCloud, yaw);
        for (int i = 0; i < kRigidSearch.iterations; ++i) {
            float fraction = (float)i / (float)(kRigidSearch.iterations - 1);
            float gate = kRigidSearch.startGate + (kRigidSearch.endGate - kRigidSearch.startGate) * fraction;
            if (step(model, fitCloud, beta, 0, 0, gate, p) < 50) { break; }
        }
        float coverage, rms;
        evaluate(model, fitCloud, beta, p, coverage, rms);
        if (coverage > bestCoverage + 0.01f || (std::fabs(coverage - bestCoverage) <= 0.01f && rms < bestRms)) {
            best = p; bestCoverage = coverage; bestRms = rms;
        }
    }

    // 3. Joint fit in stages (or rigid only, if the shape is fixed)
    Placement placement = best;
    for (const Stage& stage : kStages) {
        int components = shapeIsFixed ? 0 : (stage.components == 0 ? K : std::min(stage.components, K));
        for (int i = 0; i < stage.iterations; ++i) {
            float fraction = (float)i / (float)std::max(1, stage.iterations - 1);
            float gate = stage.startGate + (stage.endGate - stage.startGate) * fraction;
            step(model, fitCloud, beta, components, stage.penalty, gate, placement);
        }
    }

    // 4. Report: distances from scan points to the fitted surface
    std::vector<Eigen::Vector3f> vertices = [model verticesForShape:beta];
    std::vector<Eigen::Vector3f> placed = place(vertices, placement);
    std::vector<Eigen::Vector3f> normals = [model normalsForVertices:vertices];
    VertexIndex index(placed, model->_searchedForFitting);

    std::vector<float> matched, scalp, face;
    std::vector<float> distances(evalCloud.positions.size(), NAN);
    size_t covered = 0, headPoints = 0;
    for (size_t i = 0; i < evalCloud.positions.size(); ++i) {
        const Eigen::Vector3f& p = evalCloud.positions[i];
        int v = index.nearest(p);
        if (!model->_usableForFitting[v]) { continue; }  // on the neck: not part of the fit
        ++headPoints;
        Eigen::Vector3f n = placement.R * normals[v];
        float signedDistance = n.dot(p - placed[v]);
        float euclidean = (p - placed[v]).norm();
        distances[i] = signedDistance;
        if (euclidean < kCoverageDistance) { ++covered; }
        if (euclidean > kMatchDistance) { continue; }
        float d = std::fabs(signedDistance);
        matched.push_back(d);
        if (model->_region[v] == 1) { scalp.push_back(d); }
        if (model->_region[v] == 2) { face.push_back(d); }
    }
    auto rmsOf = [](const std::vector<float>& values) {
        if (values.empty()) { return (float)NAN; }
        double sum = 0;
        for (float v : values) { sum += v * v; }
        return (float)std::sqrt(sum / values.size());
    };
    result.rmsError = rmsOf(matched);
    result.scalpRmsError = rmsOf(scalp);
    result.faceRmsError = rmsOf(face);
    std::vector<float> sorted = matched;
    result.medianError = sorted.empty() ? NAN : percentile(sorted, 0.5f);
    result.p90Error = sorted.empty() ? NAN : percentile(sorted, 0.9f);
    result.coverage = headPoints == 0 ? 0 : (float)covered / (float)headPoints;

    Eigen::Matrix4f placementMatrix = Eigen::Matrix4f::Identity();
    placementMatrix.block<3, 3>(0, 0) = placement.R;
    placementMatrix.block<3, 1>(0, 3) = placement.t;
    result.yawDegrees = yawDegrees(placementMatrix);

    // Everything back into the scan's own (as recorded) space
    Eigen::Matrix4f toScan = upright.inverse();
    Eigen::Matrix3f backR = toScan.block<3, 3>(0, 0);
    Eigen::Vector3f backT = toScan.block<3, 1>(0, 3);

    std::vector<float> vertexFloats;
    vertexFloats.reserve(placed.size() * 3);
    for (const auto& q : placed) {
        Eigen::Vector3f s = backR * q + backT;
        vertexFloats.insert(vertexFloats.end(), { s.x(), s.y(), s.z() });
    }
    std::vector<float> pointFloats;
    pointFloats.reserve(evalCloud.positions.size() * 3);
    for (const auto& q : evalCloud.positions) {
        Eigen::Vector3f s = backR * q + backT;
        pointFloats.insert(pointFloats.end(), { s.x(), s.y(), s.z() });
    }

    result.vertices = floatData(vertexFloats);
    result.triangles = [NSData dataWithBytes:model->_triangles.data()
                                      length:model->_triangles.size() * sizeof(uint32_t)];
    result.shapeCoefficients = floatData(beta);
    result.scanPoints = floatData(pointFloats);
    result.scanDistances = floatData(distances);
    return result;
}

@end

// MARK: - Rigid snap (shared by the face line-up and the live depth correction)

namespace {

static const float kSnapStopShift = 0.0001f;  // stop when a round moves less than 0.1 mm…
static const float kSnapStopAngle = 0.0002f;  // …and turns less than ~0.01°

/** Moves `points` (by placement p) onto the indexed head vertices: "find nearest, nudge,
    repeat" (point-to-plane rigid ICP with Tukey weights). The gate (how far apart a pair
    may be) narrows from startGate to endGate over the first half of the rounds; after that
    it stops once a round barely moves. Returns the number of rounds. */
int rigidSnap(const std::vector<Eigen::Vector3f>& points,
              const std::vector<Eigen::Vector3f>& vertices,
              const std::vector<Eigen::Vector3f>& normals,
              const VertexIndex& index,
              float startGate, float endGate, int iterations, Placement& p)
{
    int iteration = 0;
    for (; iteration < iterations; ++iteration) {
        float fraction = std::min(1.0f, (float)iteration / (float)std::max(1, iterations / 2));
        float gate = startGate + (endGate - startGate) * fraction;

        Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        int used = 0;
        for (const auto& point : points) {
            Eigen::Vector3f q = p.R * point + p.t;
            int v = index.nearest(q);
            float distanceSquared = (q - vertices[v]).squaredNorm();
            if (distanceSquared > gate * gate) { continue; }
            const Eigen::Vector3f& n = normals[v];
            float u = 1.0f - distanceSquared / (gate * gate);
            double w = (double)(u * u);
            // r = n · (q + ω × q + τ - vertex): derivative is [q × n, n]
            Eigen::Matrix<double, 6, 1> J;
            J << q.cross(n).cast<double>(), n.cast<double>();
            double r = n.dot(q - vertices[v]);
            A += w * J * J.transpose();
            b -= w * J * r;
            ++used;
        }
        if (used < 30) { break; }
        for (int d = 0; d < 6; ++d) { A(d, d) += 1e-9; }
        Eigen::Matrix<double, 6, 1> x = A.ldlt().solve(b);
        Eigen::Vector3d omega = x.segment<3>(0);
        Eigen::Vector3f tau = x.segment<3>(3).cast<float>();
        double angle = omega.norm();
        Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
        if (angle > 1e-12) { dR = Eigen::AngleAxisd(angle, omega / angle).toRotationMatrix().cast<float>(); }
        p.R = dR * p.R;
        p.t = dR * p.t + tau;
        if (fraction >= 1.0f && tau.norm() < kSnapStopShift && angle < kSnapStopAngle) { ++iteration; break; }
    }
    return iteration;
}

simd_float4x4 toSimdMatrix(const Placement& p)
{
    simd_float4x4 m = matrix_identity_float4x4;
    for (int c = 0; c < 3; ++c) {
        m.columns[c] = simd_make_float4(p.R(0, c), p.R(1, c), p.R(2, c), 0);
    }
    m.columns[3] = simd_make_float4(p.t.x(), p.t.y(), p.t.z(), 1);
    return m;
}

} // namespace

// MARK: - Face line-up

@interface SCFaceLineUpResult ()
@property (nonatomic, readwrite) simd_float4x4 faceToHead;
@property (nonatomic, readwrite) NSData *matchedHeadPoints;
@property (nonatomic, readwrite) NSData *matchedFlags;
@property (nonatomic, readwrite) float rmsError;
@property (nonatomic, readwrite) NSInteger usedPointCount;
@property (nonatomic, readwrite) NSInteger iterations;
@end

@implementation SCFaceLineUpResult
@end

static const int   kLineUpIterations = 40;
static const float kLineUpStartGate = 0.02f;   // first rounds: pairs up to 2 cm apart count
static const float kLineUpEndGate = 0.006f;    // last rounds: only pairs within 6 mm

@implementation SCFaceLineUp

+ (SCFaceLineUpResult *)lineUpFacePoints:(NSData *)facePoints
                                withHead:(SCHeadModel *)model
                                   shape:(NSData *)shapeCoefficients
{
    SCFaceLineUpResult *result = [[SCFaceLineUpResult alloc] init];
    const size_t count = facePoints.length / (3 * sizeof(float));
    const float *raw = (const float *)facePoints.bytes;
    std::vector<Eigen::Vector3f> mask(count);
    for (size_t i = 0; i < count; ++i) { mask[i] = Eigen::Vector3f(raw[3 * i], raw[3 * i + 1], raw[3 * i + 2]); }

    // The head, and which of its vertices are "face" (targets for the mask)
    size_t k = std::min<size_t>(model->_componentCount, shapeCoefficients.length / sizeof(float));
    std::vector<float> beta(k);
    if (k > 0) { memcpy(beta.data(), shapeCoefficients.bytes, k * sizeof(float)); }
    std::vector<Eigen::Vector3f> vertices = [model verticesForShape:beta];
    std::vector<Eigen::Vector3f> normals = [model normalsForVertices:vertices];

    std::vector<bool> target(model->_vertexCount, false);
    for (const char *name : { "face", "forehead", "nose" }) {
        for (int v : model->_masks[name]) { if (v >= 0 && v < model->_vertexCount) { target[v] = true; } }
    }
    for (const char *name : { "lips", "left_eyeball", "right_eyeball", "left_ear", "right_ear",
                              "scalp", "neck", "boundary" }) {
        for (int v : model->_masks[name]) { if (v >= 0 && v < model->_vertexCount) { target[v] = false; } }
    }
    VertexIndex index(vertices, target);

    // First guess: no turn (both spaces have y up and z out of the face), nose tips together.
    // The nose tip is the frontmost point (largest z).
    Placement p;
    if (count > 0 && !index.ids.empty()) {
        Eigen::Vector3f maskNose = mask[0];
        for (const auto& m : mask) { if (m.z() > maskNose.z()) { maskNose = m; } }
        Eigen::Vector3f headNose = vertices[index.ids[0]];
        for (int v : index.ids) { if (vertices[v].z() > headNose.z()) { headNose = vertices[v]; } }
        p.t = headNose - maskNose;
    }

    // Snap: find nearest, nudge, repeat
    int iteration = (count > 0 && !index.ids.empty())
        ? rigidSnap(mask, vertices, normals, index, kLineUpStartGate, kLineUpEndGate, kLineUpIterations, p)
        : 0;

    // Final matches: each mask point's closest spot on the head's face surface
    std::vector<float> matched(count * 3, 0.0f);
    std::vector<uint8_t> flags(count, 0);
    double sum = 0;
    int used = 0;
    for (size_t i = 0; i < count && !index.ids.empty(); ++i) {
        Eigen::Vector3f q = p.R * mask[i] + p.t;
        int v = index.nearest(q);
        if ((q - vertices[v]).norm() > kLineUpEndGate) { continue; }
        const Eigen::Vector3f& n = normals[v];
        float d = n.dot(q - vertices[v]);
        Eigen::Vector3f onSurface = q - d * n;  // foot of q on the vertex's tangent plane
        matched[3 * i] = onSurface.x(); matched[3 * i + 1] = onSurface.y(); matched[3 * i + 2] = onSurface.z();
        flags[i] = 1;
        sum += d * d;
        ++used;
    }

    result.faceToHead = toSimdMatrix(p);
    result.matchedHeadPoints = floatData(matched);
    result.matchedFlags = [NSData dataWithBytes:flags.data() length:flags.size()];
    result.rmsError = used > 0 ? (float)std::sqrt(sum / used) : kLargeDistance;
    result.usedPointCount = used;
    result.iterations = iteration;
    return result;
}

@end

// MARK: - Surface check

@interface SCHeadSnapResult ()
@property (nonatomic, readwrite) simd_float4x4 correction;
@property (nonatomic, readwrite) float rmsError;
@property (nonatomic, readwrite) NSInteger usedPointCount;
@property (nonatomic, readwrite) NSInteger iterations;
@end

@implementation SCHeadSnapResult
@end

@implementation SCHeadSurfaceCheck {
    std::vector<Eigen::Vector3f> _vertices;
    std::vector<Eigen::Vector3f> _normals;
    std::vector<uint8_t> _regions;
    std::unique_ptr<VertexIndex> _index;
    std::unique_ptr<VertexIndex> _snapIndex;   // firm, hair-free parts only
}

- (instancetype)initWithModel:(SCHeadModel *)model shape:(NSData *)shapeCoefficients
{
    self = [super init];
    if (!self) { return nil; }
    size_t k = std::min<size_t>(model->_componentCount, shapeCoefficients.length / sizeof(float));
    std::vector<float> beta(k);
    if (k > 0) { memcpy(beta.data(), shapeCoefficients.bytes, k * sizeof(float)); }
    _vertices = [model verticesForShape:beta];
    _normals = [model normalsForVertices:_vertices];

    _regions.assign(model->_vertexCount, SCHeadRegionOther);
    auto label = [&](const char *name, SCHeadRegion region) {
        for (int v : model->_masks[name]) { if (v >= 0 && v < model->_vertexCount) { _regions[v] = region; } }
    };
    label("scalp", SCHeadRegionScalp);
    label("face", SCHeadRegionFace);
    label("forehead", SCHeadRegionForehead);
    label("left_ear", SCHeadRegionLeftEar);
    label("right_ear", SCHeadRegionRightEar);

    // Eyeballs sit inside the head; leave them out of the search
    std::vector<bool> searchable(model->_vertexCount, true);
    for (const char *name : { "left_eyeball", "right_eyeball" }) {
        for (int v : model->_masks[name]) { if (v >= 0 && v < model->_vertexCount) { searchable[v] = false; } }
    }
    _index.reset(new VertexIndex(_vertices, searchable));

    // Snap targets: face, nose and ears; not lips or eyes (they move), not the forehead,
    // scalp or neck (hair and clothes cover them, and the forehead fit is pushed out by bangs)
    std::vector<bool> firm(model->_vertexCount, false);
    for (const char *name : { "face", "nose", "left_ear", "right_ear" }) {
        for (int v : model->_masks[name]) { if (v >= 0 && v < model->_vertexCount) { firm[v] = true; } }
    }
    for (const char *name : { "lips", "eye_region", "left_eyeball", "right_eyeball", "forehead",
                              "scalp", "neck", "boundary" }) {
        for (int v : model->_masks[name]) { if (v >= 0 && v < model->_vertexCount) { firm[v] = false; } }
    }
    _snapIndex.reset(new VertexIndex(_vertices, firm));
    return self;
}

- (SCHeadSnapResult *)snapPoints:(NSData *)points
                       startGate:(float)startGate
                         endGate:(float)endGate
                      iterations:(NSInteger)iterations
{
    size_t count = points.length / (3 * sizeof(float));
    const float *raw = (const float *)points.bytes;
    std::vector<Eigen::Vector3f> cloud(count);
    for (size_t i = 0; i < count; ++i) { cloud[i] = Eigen::Vector3f(raw[3 * i], raw[3 * i + 1], raw[3 * i + 2]); }

    Placement p;
    int rounds = (count > 0 && !_snapIndex->ids.empty())
        ? rigidSnap(cloud, _vertices, _normals, *_snapIndex, startGate, endGate, (int)iterations, p)
        : 0;

    double sum = 0;
    int used = 0;
    for (const auto& point : cloud) {
        if (_snapIndex->ids.empty()) { break; }
        Eigen::Vector3f q = p.R * point + p.t;
        int v = _snapIndex->nearest(q);
        if ((q - _vertices[v]).norm() > endGate) { continue; }
        float d = _normals[v].dot(q - _vertices[v]);
        sum += d * d;
        ++used;
    }
    SCHeadSnapResult *result = [[SCHeadSnapResult alloc] init];
    result.correction = toSimdMatrix(p);
    result.rmsError = used > 0 ? (float)std::sqrt(sum / used) : kLargeDistance;
    result.usedPointCount = used;
    result.iterations = rounds;
    return result;
}

- (void)measurePoints:(NSData *)points
                 gate:(float)gate
            distances:(NSMutableData *)distances
              regions:(NSMutableData *)regions
{
    size_t count = points.length / (3 * sizeof(float));
    distances.length = count * sizeof(float);
    regions.length = count;
    const float *p = (const float *)points.bytes;
    float *outDistance = (float *)distances.mutableBytes;
    uint8_t *outRegion = (uint8_t *)regions.mutableBytes;
    for (size_t i = 0; i < count; ++i) {
        Eigen::Vector3f q(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
        int v = _index->nearest(q);
        Eigen::Vector3f offset = q - _vertices[v];
        outDistance[i] = _normals[v].dot(offset);
        outRegion[i] = offset.norm() > gate ? (uint8_t)SCHeadRegionNone : _regions[v];
    }
}

@end
