import Foundation
import CoreGraphics

/// Service for fitting bezier curves to point sequences
final class BezierFitter {

    struct FittingSettings {
        var errorThreshold: CGFloat = 4.0    // Max allowed error for curve fitting
        var maxIterations: Int = 4            // Max iterations for curve optimization
        var cornerThreshold: CGFloat = 60     // Angle threshold for detecting corners (degrees)

        static let `default` = FittingSettings()
    }

    /// A fitted bezier segment
    struct BezierSegment {
        let start: CGPoint
        let control1: CGPoint
        let control2: CGPoint
        let end: CGPoint

        /// Evaluate the bezier curve at parameter t (0-1)
        func evaluate(at t: CGFloat) -> CGPoint {
            let t2 = t * t
            let t3 = t2 * t
            let mt = 1 - t
            let mt2 = mt * mt
            let mt3 = mt2 * mt

            return CGPoint(
                x: mt3 * start.x + 3 * mt2 * t * control1.x + 3 * mt * t2 * control2.x + t3 * end.x,
                y: mt3 * start.y + 3 * mt2 * t * control1.y + 3 * mt * t2 * control2.y + t3 * end.y
            )
        }

        /// Calculate the derivative at parameter t
        func derivative(at t: CGFloat) -> CGPoint {
            let mt = 1 - t

            return CGPoint(
                x: 3 * mt * mt * (control1.x - start.x) + 6 * mt * t * (control2.x - control1.x) + 3 * t * t * (end.x - control2.x),
                y: 3 * mt * mt * (control1.y - start.y) + 6 * mt * t * (control2.y - control1.y) + 3 * t * t * (end.y - control2.y)
            )
        }
    }

    // MARK: - Main Fitting API

    /// Fit bezier curves to a sequence of points
    static func fitCurves(
        to points: [CGPoint],
        isClosed: Bool,
        settings: FittingSettings = .default
    ) -> [BezierSegment] {
        guard points.count >= 2 else { return [] }

        // Detect corners to split the curve at
        let corners = detectCorners(points: points, threshold: settings.cornerThreshold, isClosed: isClosed)

        if corners.isEmpty {
            // No corners - fit entire curve as one or more bezier segments
            let tangent1 = computeLeftTangent(points: points, index: 0)
            let tangent2 = computeRightTangent(points: points, index: points.count - 1)
            return fitCubic(points: points, first: 0, last: points.count - 1, tangent1: tangent1, tangent2: tangent2, error: settings.errorThreshold)
        }

        // Split at corners and fit each segment
        var segments: [BezierSegment] = []
        var cornerIndices = corners

        // Add start and end points if not closed
        if !isClosed {
            if !cornerIndices.contains(0) {
                cornerIndices.insert(0, at: 0)
            }
            if !cornerIndices.contains(points.count - 1) {
                cornerIndices.append(points.count - 1)
            }
        }

        cornerIndices.sort()

        for i in 0..<(cornerIndices.count - (isClosed ? 0 : 1)) {
            let startIdx = cornerIndices[i]
            let endIdx = cornerIndices[(i + 1) % cornerIndices.count]

            // Extract segment points
            var segmentPoints: [CGPoint]
            if endIdx > startIdx {
                segmentPoints = Array(points[startIdx...endIdx])
            } else if isClosed {
                // Wrap around
                segmentPoints = Array(points[startIdx...]) + Array(points[...endIdx])
            } else {
                continue
            }

            guard segmentPoints.count >= 2 else { continue }

            // Compute tangents at segment ends
            let tangent1 = computeLeftTangent(points: segmentPoints, index: 0)
            let tangent2 = computeRightTangent(points: segmentPoints, index: segmentPoints.count - 1)

            // Fit bezier to segment
            let fitted = fitCubic(
                points: segmentPoints,
                first: 0,
                last: segmentPoints.count - 1,
                tangent1: tangent1,
                tangent2: tangent2,
                error: settings.errorThreshold
            )

            segments.append(contentsOf: fitted)
        }

        return segments
    }

    // MARK: - Core Fitting Algorithm

    /// Fit a cubic bezier to points[first...last] using Schneider's algorithm
    private static func fitCubic(
        points: [CGPoint],
        first: Int,
        last: Int,
        tangent1: CGPoint,
        tangent2: CGPoint,
        error: CGFloat
    ) -> [BezierSegment] {
        let nPoints = last - first + 1

        // Use heuristic for two-point case
        if nPoints == 2 {
            let dist = distance(points[first], points[last]) / 3.0
            let segment = BezierSegment(
                start: points[first],
                control1: CGPoint(x: points[first].x + tangent1.x * dist, y: points[first].y + tangent1.y * dist),
                control2: CGPoint(x: points[last].x + tangent2.x * dist, y: points[last].y + tangent2.y * dist),
                end: points[last]
            )
            return [segment]
        }

        // Parameterize points
        var u = chordLengthParameterize(points: points, first: first, last: last)

        // Generate bezier curve
        var bezier = generateBezier(points: points, first: first, last: last, uPrime: u, tangent1: tangent1, tangent2: tangent2)

        // Find max error
        var (maxError, splitPoint) = computeMaxError(points: points, first: first, last: last, bezier: bezier, u: u)

        if maxError < error {
            return [bezier]
        }

        // If error is small, try reparameterization
        let iterationError = error * 4
        if maxError < iterationError {
            for _ in 0..<4 {
                u = reparameterize(points: points, first: first, last: last, u: u, bezier: bezier)
                bezier = generateBezier(points: points, first: first, last: last, uPrime: u, tangent1: tangent1, tangent2: tangent2)
                (maxError, splitPoint) = computeMaxError(points: points, first: first, last: last, bezier: bezier, u: u)

                if maxError < error {
                    return [bezier]
                }
            }
        }

        // Split and fit recursively
        let centerTangent = computeCenterTangent(points: points, index: splitPoint)
        let left = fitCubic(points: points, first: first, last: splitPoint, tangent1: tangent1, tangent2: centerTangent, error: error)
        let right = fitCubic(points: points, first: splitPoint, last: last, tangent1: negate(centerTangent), tangent2: tangent2, error: error)

        return left + right
    }

    /// Generate bezier curve for points[first...last]
    private static func generateBezier(
        points: [CGPoint],
        first: Int,
        last: Int,
        uPrime: [CGFloat],
        tangent1: CGPoint,
        tangent2: CGPoint
    ) -> BezierSegment {
        let nPoints = last - first + 1

        // Compute A matrix
        var a = [[CGPoint]](repeating: [CGPoint](repeating: .zero, count: 2), count: nPoints)

        for i in 0..<nPoints {
            let t = uPrime[i]
            let b1 = b1Basis(t)
            let b2 = b2Basis(t)

            a[i][0] = CGPoint(x: tangent1.x * b1, y: tangent1.y * b1)
            a[i][1] = CGPoint(x: tangent2.x * b2, y: tangent2.y * b2)
        }

        // Compute C and X matrices
        var c = [[CGFloat]](repeating: [CGFloat](repeating: 0, count: 2), count: 2)
        var x = [CGFloat](repeating: 0, count: 2)

        let firstPoint = points[first]
        let lastPoint = points[last]

        for i in 0..<nPoints {
            c[0][0] += dot(a[i][0], a[i][0])
            c[0][1] += dot(a[i][0], a[i][1])
            c[1][0] = c[0][1]
            c[1][1] += dot(a[i][1], a[i][1])

            let t = uPrime[i]
            let tmp = subtract(
                points[first + i],
                multiply(firstPoint, b0Basis(t) + b1Basis(t)),
                multiply(lastPoint, b2Basis(t) + b3Basis(t))
            )

            x[0] += dot(a[i][0], tmp)
            x[1] += dot(a[i][1], tmp)
        }

        // Solve for alpha values
        let det = c[0][0] * c[1][1] - c[1][0] * c[0][1]
        var alphaL: CGFloat = 0
        var alphaR: CGFloat = 0

        if abs(det) > 1e-10 {
            alphaL = (c[1][1] * x[0] - c[0][1] * x[1]) / det
            alphaR = (c[0][0] * x[1] - c[1][0] * x[0]) / det
        }

        // If alpha values are invalid, use heuristic
        let segLength = distance(firstPoint, lastPoint)
        let epsilon = 1e-6 * segLength

        if alphaL < epsilon || alphaR < epsilon {
            let dist = segLength / 3.0
            return BezierSegment(
                start: firstPoint,
                control1: CGPoint(x: firstPoint.x + tangent1.x * dist, y: firstPoint.y + tangent1.y * dist),
                control2: CGPoint(x: lastPoint.x + tangent2.x * dist, y: lastPoint.y + tangent2.y * dist),
                end: lastPoint
            )
        }

        return BezierSegment(
            start: firstPoint,
            control1: CGPoint(x: firstPoint.x + tangent1.x * alphaL, y: firstPoint.y + tangent1.y * alphaL),
            control2: CGPoint(x: lastPoint.x + tangent2.x * alphaR, y: lastPoint.y + tangent2.y * alphaR),
            end: lastPoint
        )
    }

    // MARK: - Parameterization

    /// Assign parameter values based on chord length
    private static func chordLengthParameterize(points: [CGPoint], first: Int, last: Int) -> [CGFloat] {
        var u = [CGFloat](repeating: 0, count: last - first + 1)
        u[0] = 0

        for i in (first + 1)...last {
            u[i - first] = u[i - first - 1] + distance(points[i], points[i - 1])
        }

        let total = u[last - first]
        if total > 0 {
            for i in 1...(last - first) {
                u[i] /= total
            }
        }

        return u
    }

    /// Newton-Raphson iteration to refine parameters
    private static func reparameterize(
        points: [CGPoint],
        first: Int,
        last: Int,
        u: [CGFloat],
        bezier: BezierSegment
    ) -> [CGFloat] {
        var uPrime = u

        for i in first...last {
            uPrime[i - first] = newtonRaphsonRoot(bezier: bezier, point: points[i], u: u[i - first])
        }

        return uPrime
    }

    /// Find root using Newton-Raphson method
    private static func newtonRaphsonRoot(bezier: BezierSegment, point: CGPoint, u: CGFloat) -> CGFloat {
        let q = bezier.evaluate(at: u)
        let qPrime = bezier.derivative(at: u)

        let numerator = (q.x - point.x) * qPrime.x + (q.y - point.y) * qPrime.y
        let denominator = qPrime.x * qPrime.x + qPrime.y * qPrime.y

        if abs(denominator) < 1e-10 {
            return u
        }

        return u - numerator / denominator
    }

    // MARK: - Error Computation

    /// Compute max error between curve and points
    private static func computeMaxError(
        points: [CGPoint],
        first: Int,
        last: Int,
        bezier: BezierSegment,
        u: [CGFloat]
    ) -> (CGFloat, Int) {
        var maxError: CGFloat = 0
        var splitPoint = (last - first + 1) / 2

        for i in first...last {
            let p = bezier.evaluate(at: u[i - first])
            let dist = distance(p, points[i])

            if dist >= maxError {
                maxError = dist
                splitPoint = i
            }
        }

        return (maxError * maxError, splitPoint)
    }

    // MARK: - Tangent Computation

    private static func computeLeftTangent(points: [CGPoint], index: Int) -> CGPoint {
        let d = subtract(points[index + 1], points[index])
        return normalize(d)
    }

    private static func computeRightTangent(points: [CGPoint], index: Int) -> CGPoint {
        let d = subtract(points[index - 1], points[index])
        return normalize(d)
    }

    private static func computeCenterTangent(points: [CGPoint], index: Int) -> CGPoint {
        let v1 = subtract(points[index - 1], points[index])
        let v2 = subtract(points[index], points[index + 1])
        let sum = CGPoint(x: (v1.x + v2.x) / 2, y: (v1.y + v2.y) / 2)
        return normalize(sum)
    }

    // MARK: - Corner Detection

    private static func detectCorners(points: [CGPoint], threshold: CGFloat, isClosed: Bool) -> [Int] {
        guard points.count >= 3 else { return [] }

        var corners: [Int] = []
        let thresholdRad = threshold * .pi / 180

        let count = points.count
        let iterations = isClosed ? count : count - 2

        for i in 0..<iterations {
            let prevIdx = (i - 1 + count) % count
            let nextIdx = (i + 1) % count
            let idx = isClosed ? i : i + 1

            let v1 = subtract(points[idx], points[prevIdx])
            let v2 = subtract(points[nextIdx], points[idx])

            let angle1 = atan2(v1.y, v1.x)
            let angle2 = atan2(v2.y, v2.x)

            var diff = abs(angle2 - angle1)
            if diff > .pi {
                diff = 2 * .pi - diff
            }

            if diff > thresholdRad {
                corners.append(idx)
            }
        }

        return corners
    }

    // MARK: - Bernstein Basis Functions

    private static func b0Basis(_ t: CGFloat) -> CGFloat {
        let tmp = 1 - t
        return tmp * tmp * tmp
    }

    private static func b1Basis(_ t: CGFloat) -> CGFloat {
        let tmp = 1 - t
        return 3 * t * tmp * tmp
    }

    private static func b2Basis(_ t: CGFloat) -> CGFloat {
        let tmp = 1 - t
        return 3 * t * t * tmp
    }

    private static func b3Basis(_ t: CGFloat) -> CGFloat {
        return t * t * t
    }

    // MARK: - Vector Helpers

    private static func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y))
    }

    private static func dot(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        a.x * b.x + a.y * b.y
    }

    private static func subtract(_ a: CGPoint, _ b: CGPoint) -> CGPoint {
        CGPoint(x: a.x - b.x, y: a.y - b.y)
    }

    private static func subtract(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> CGPoint {
        CGPoint(x: a.x - b.x - c.x, y: a.y - b.y - c.y)
    }

    private static func multiply(_ p: CGPoint, _ s: CGFloat) -> CGPoint {
        CGPoint(x: p.x * s, y: p.y * s)
    }

    private static func normalize(_ p: CGPoint) -> CGPoint {
        let len = sqrt(p.x * p.x + p.y * p.y)
        guard len > 0 else { return CGPoint(x: 1, y: 0) }
        return CGPoint(x: p.x / len, y: p.y / len)
    }

    private static func negate(_ p: CGPoint) -> CGPoint {
        CGPoint(x: -p.x, y: -p.y)
    }

    // MARK: - Convert to PathPoints

    /// Convert bezier segments to PathPoints for GlyphOutline
    static func toPathPoints(segments: [BezierSegment]) -> [PathPoint] {
        guard !segments.isEmpty else { return [] }

        var points: [PathPoint] = []

        // Add first point
        let firstSegment = segments[0]
        points.append(PathPoint(
            position: firstSegment.start,
            type: .smooth,
            controlOut: firstSegment.control1
        ))

        // Add remaining points
        for (index, segment) in segments.enumerated() {
            let isLast = index == segments.count - 1

            points.append(PathPoint(
                position: segment.end,
                type: .smooth,
                controlIn: segment.control2,
                controlOut: isLast ? nil : segments[index + 1].control1
            ))
        }

        return points
    }
}
