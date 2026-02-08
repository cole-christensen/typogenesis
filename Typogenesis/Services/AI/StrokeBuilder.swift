import Foundation
import CoreGraphics

/// Converts stroke templates to bezier contours with style-aware expansion.
///
/// This builder takes centerline stroke definitions and expands them into
/// filled contours with proper stroke weight, applying style parameters
/// like contrast, roundness, and serif treatment.
final class StrokeBuilder: Sendable {

    // MARK: - Constants

    /// Stroke dimension parameters
    private enum StrokeDimensions {
        /// Base stroke width as fraction of em (8%)
        static let baseWidthRatio: CGFloat = 0.08
        /// Side bearing as fraction of em (8%)
        static let sideBearingRatio: CGFloat = 0.08
        /// Offset added to stroke weight for minimum thickness
        static let weightOffset: CGFloat = 0.5
    }

    /// Sampling and curve parameters
    private enum CurveParams {
        /// Number of samples along stroke centerline
        static let sampleCount: Int = 10
        /// Threshold for roundness-based point type selection
        static let roundnessThreshold: CGFloat = 0.5
        /// Factor for quadratic-to-cubic bezier conversion (2/3)
        static let quadToCubicFactor: CGFloat = 2.0 / 3.0
        /// Contrast scaling factor for horizontal/vertical stroke variation
        static let contrastScale: CGFloat = 0.5
        /// Handle length as fraction of neighbor distance
        static let handleLengthFactor: CGFloat = 0.3
        /// Distance threshold for deduplicating points
        static let deduplicationThreshold: CGFloat = 0.5
    }

    /// Style parameters for stroke building
    struct StyleParams: Sendable {
        /// Base stroke weight (0-1, where 0.5 is medium)
        var strokeWeight: CGFloat = 0.5

        /// Stroke contrast - variation between thick and thin strokes (0-1)
        var strokeContrast: CGFloat = 0.3

        /// Roundness of corners and terminals (0 = sharp, 1 = fully rounded)
        var roundness: CGFloat = 0.5

        /// Slant angle in degrees (positive = italic right lean)
        var slant: CGFloat = 0

        /// Serif style to apply
        var serifStyle: SerifStyle = .none

        enum SerifStyle: Sendable {
            case none
            case slab
            case bracketed
            case hairline
        }

        static let `default` = StyleParams()

        /// Create from FontStyle
        init(from fontStyle: StyleEncoder.FontStyle) {
            self.strokeWeight = CGFloat(fontStyle.strokeWeight)
            self.strokeContrast = CGFloat(fontStyle.strokeContrast)
            self.roundness = CGFloat(fontStyle.roundness)
            self.slant = CGFloat(fontStyle.slant)
            self.serifStyle = Self.mapSerifStyle(fontStyle.serifStyle)
        }

        init(
            strokeWeight: CGFloat = 0.5,
            strokeContrast: CGFloat = 0.3,
            roundness: CGFloat = 0.5,
            slant: CGFloat = 0,
            serifStyle: SerifStyle = .none
        ) {
            self.strokeWeight = strokeWeight
            self.strokeContrast = strokeContrast
            self.roundness = roundness
            self.slant = slant
            self.serifStyle = serifStyle
        }

        private static func mapSerifStyle(_ style: StyleEncoder.SerifStyle) -> SerifStyle {
            switch style {
            case .sansSerif: return .none
            case .slab: return .slab
            case .oldStyle, .transitional, .modern: return .bracketed
            case .script, .decorative: return .none
            }
        }
    }

    // MARK: - Public API

    /// Build a glyph outline from a template with given metrics and style
    func buildGlyph(
        from template: GlyphTemplate,
        metrics: FontMetrics,
        style: StyleParams = .default
    ) -> GlyphOutline {
        // Handle space character
        if template.strokes.isEmpty {
            return GlyphOutline()
        }

        // Calculate reference heights
        let baseHeight: CGFloat
        if template.isUppercase {
            baseHeight = CGFloat(metrics.capHeight)
        } else {
            baseHeight = CGFloat(metrics.xHeight)
        }

        let actualHeight = baseHeight * template.heightMultiplier
        let glyphWidth = CGFloat(metrics.unitsPerEm) * template.normalizedWidth
        let baselineY = CGFloat(template.baselineOffset) * baseHeight

        // Calculate stroke width based on style
        let baseStrokeWidth = CGFloat(metrics.unitsPerEm) * StrokeDimensions.baseWidthRatio * (StrokeDimensions.weightOffset + style.strokeWeight)

        var allContours: [Contour] = []

        for strokePath in template.strokes {
            let contours = buildStrokeContours(
                strokePath: strokePath,
                width: glyphWidth,
                height: actualHeight,
                baselineY: baselineY,
                strokeWidth: baseStrokeWidth,
                style: style
            )
            allContours.append(contentsOf: contours)
        }

        // Apply slant transformation if needed
        if style.slant != 0 {
            allContours = applySlant(to: allContours, angle: style.slant, baselineY: baselineY)
        }

        return GlyphOutline(contours: allContours)
    }

    /// Calculate the advance width for a generated glyph
    func calculateAdvanceWidth(from template: GlyphTemplate, metrics: FontMetrics, style: StyleParams) -> Int {
        let glyphWidth = CGFloat(metrics.unitsPerEm) * template.normalizedWidth
        let sideBearing = CGFloat(metrics.unitsPerEm) * StrokeDimensions.sideBearingRatio
        let strokeWidth = CGFloat(metrics.unitsPerEm) * StrokeDimensions.baseWidthRatio * (StrokeDimensions.weightOffset + style.strokeWeight)

        return Int(glyphWidth + strokeWidth + sideBearing * 2)
    }

    /// Calculate the left side bearing
    func calculateLeftSideBearing(from template: GlyphTemplate, metrics: FontMetrics, style: StyleParams) -> Int {
        let sideBearing = CGFloat(metrics.unitsPerEm) * StrokeDimensions.sideBearingRatio
        let strokeWidth = CGFloat(metrics.unitsPerEm) * StrokeDimensions.baseWidthRatio * (StrokeDimensions.weightOffset + style.strokeWeight)
        return Int(sideBearing + strokeWidth / 2)
    }

    // MARK: - Private Methods

    private func buildStrokeContours(
        strokePath: StrokePath,
        width: CGFloat,
        height: CGFloat,
        baselineY: CGFloat,
        strokeWidth: CGFloat,
        style: StyleParams
    ) -> [Contour] {
        // Handle closed paths (filled shapes like bowls, ellipses)
        if strokePath.isClosed {
            return buildFilledContour(
                strokePath: strokePath,
                width: width,
                height: height,
                baselineY: baselineY
            )
        }

        // Handle open paths (lines, curves) by expanding to stroke
        return buildExpandedStroke(
            strokePath: strokePath,
            width: width,
            height: height,
            baselineY: baselineY,
            strokeWidth: strokeWidth,
            style: style
        )
    }

    private func buildFilledContour(
        strokePath: StrokePath,
        width: CGFloat,
        height: CGFloat,
        baselineY: CGFloat
    ) -> [Contour] {
        var points: [PathPoint] = []

        for segment in strokePath.segments {
            let scaledPoints = scaleSegment(
                segment,
                width: width,
                height: height,
                baselineY: baselineY
            )
            points.append(contentsOf: scaledPoints)
        }

        // Remove duplicate points at segment boundaries
        points = deduplicatePoints(points)

        guard !points.isEmpty else { return [] }

        return [Contour(points: points, isClosed: true)]
    }

    private func buildExpandedStroke(
        strokePath: StrokePath,
        width: CGFloat,
        height: CGFloat,
        baselineY: CGFloat,
        strokeWidth: CGFloat,
        style: StyleParams
    ) -> [Contour] {
        // Sample points along the stroke centerline
        var centerlinePoints: [(point: CGPoint, direction: CGFloat)] = []

        for segment in strokePath.segments {
            let samples = sampleSegment(
                segment,
                width: width,
                height: height,
                baselineY: baselineY,
                sampleCount: CurveParams.sampleCount
            )
            centerlinePoints.append(contentsOf: samples)
        }

        guard centerlinePoints.count >= 2 else { return [] }

        // Remove duplicates
        centerlinePoints = deduplicateSamples(centerlinePoints)

        guard centerlinePoints.count >= 2 else { return [] }

        // Build offset curves (left and right of centerline)
        var leftPoints: [PathPoint] = []
        var rightPoints: [PathPoint] = []

        for (index, sample) in centerlinePoints.enumerated() {
            // Calculate stroke width with contrast
            let localWidth = calculateLocalStrokeWidth(
                at: sample.direction,
                baseWidth: strokeWidth,
                style: style
            )

            let halfWidth = localWidth / 2

            // Perpendicular direction
            let perpAngle = sample.direction + .pi / 2

            let leftPoint = CGPoint(
                x: sample.point.x + cos(perpAngle) * halfWidth,
                y: sample.point.y + sin(perpAngle) * halfWidth
            )
            let rightPoint = CGPoint(
                x: sample.point.x - cos(perpAngle) * halfWidth,
                y: sample.point.y - sin(perpAngle) * halfWidth
            )

            // Determine point type based on roundness
            let pointType: PathPoint.PointType
            if index == 0 || index == centerlinePoints.count - 1 {
                pointType = style.roundness > CurveParams.roundnessThreshold ? .smooth : .corner
            } else {
                pointType = .smooth
            }

            leftPoints.append(PathPoint(position: leftPoint, type: pointType))
            rightPoints.append(PathPoint(position: rightPoint, type: pointType))
        }

        // Build closed contour: left forward, then right backward
        var contourPoints = leftPoints
        contourPoints.append(contentsOf: rightPoints.reversed())

        // Add smooth handles for curves
        contourPoints = addSmoothHandles(to: contourPoints, roundness: style.roundness)

        return [Contour(points: contourPoints, isClosed: true)]
    }

    private func scaleSegment(
        _ segment: StrokeSegment,
        width: CGFloat,
        height: CGFloat,
        baselineY: CGFloat
    ) -> [PathPoint] {
        func scale(_ point: CGPoint) -> CGPoint {
            CGPoint(
                x: point.x * width,
                y: point.y * height + baselineY
            )
        }

        switch segment {
        case .line(let start, let end):
            return [
                PathPoint(position: scale(start), type: .corner),
                PathPoint(position: scale(end), type: .corner)
            ]

        case .quadCurve(let start, let control, let end):
            // Convert to cubic for consistent handling
            let cp1 = CGPoint(
                x: start.x + CurveParams.quadToCubicFactor * (control.x - start.x),
                y: start.y + CurveParams.quadToCubicFactor * (control.y - start.y)
            )
            let cp2 = CGPoint(
                x: end.x + CurveParams.quadToCubicFactor * (control.x - end.x),
                y: end.y + CurveParams.quadToCubicFactor * (control.y - end.y)
            )

            var startPoint = PathPoint(position: scale(start), type: .smooth)
            startPoint.controlOut = scale(cp1)

            var endPoint = PathPoint(position: scale(end), type: .smooth)
            endPoint.controlIn = scale(cp2)

            return [startPoint, endPoint]

        case .cubicCurve(let start, let control1, let control2, let end):
            var startPoint = PathPoint(position: scale(start), type: .smooth)
            startPoint.controlOut = scale(control1)

            var endPoint = PathPoint(position: scale(end), type: .smooth)
            endPoint.controlIn = scale(control2)

            return [startPoint, endPoint]

        case .arc:
            // Convert arc to bezier approximation (already done in StrokePath.ellipse)
            return []
        }
    }

    private func sampleSegment(
        _ segment: StrokeSegment,
        width: CGFloat,
        height: CGFloat,
        baselineY: CGFloat,
        sampleCount: Int
    ) -> [(point: CGPoint, direction: CGFloat)] {
        func scale(_ point: CGPoint) -> CGPoint {
            CGPoint(
                x: point.x * width,
                y: point.y * height + baselineY
            )
        }

        var samples: [(point: CGPoint, direction: CGFloat)] = []

        switch segment {
        case .line(let start, let end):
            let scaledStart = scale(start)
            let scaledEnd = scale(end)
            let direction = atan2(scaledEnd.y - scaledStart.y, scaledEnd.x - scaledStart.x)

            for i in 0...sampleCount {
                let t = CGFloat(i) / CGFloat(sampleCount)
                let point = CGPoint(
                    x: scaledStart.x + t * (scaledEnd.x - scaledStart.x),
                    y: scaledStart.y + t * (scaledEnd.y - scaledStart.y)
                )
                samples.append((point, direction))
            }

        case .quadCurve(let start, let control, let end):
            let scaledStart = scale(start)
            let scaledControl = scale(control)
            let scaledEnd = scale(end)

            for i in 0...sampleCount {
                let t = CGFloat(i) / CGFloat(sampleCount)
                let point = quadraticBezier(scaledStart, scaledControl, scaledEnd, t: t)
                let tangent = quadraticBezierTangent(scaledStart, scaledControl, scaledEnd, t: t)
                let direction = atan2(tangent.y, tangent.x)
                samples.append((point, direction))
            }

        case .cubicCurve(let start, let control1, let control2, let end):
            let scaledStart = scale(start)
            let scaledControl1 = scale(control1)
            let scaledControl2 = scale(control2)
            let scaledEnd = scale(end)

            for i in 0...sampleCount {
                let t = CGFloat(i) / CGFloat(sampleCount)
                let point = cubicBezier(scaledStart, scaledControl1, scaledControl2, scaledEnd, t: t)
                let tangent = cubicBezierTangent(scaledStart, scaledControl1, scaledControl2, scaledEnd, t: t)
                let direction = atan2(tangent.y, tangent.x)
                samples.append((point, direction))
            }

        case .arc(let center, let radius, let startAngle, let endAngle, let clockwise):
            let scaledCenter = scale(center)
            let scaledRadiusX = radius.width * width
            let scaledRadiusY = radius.height * height

            let angleRange = clockwise ? startAngle - endAngle : endAngle - startAngle

            for i in 0...sampleCount {
                let t = CGFloat(i) / CGFloat(sampleCount)
                let angle = clockwise ? startAngle - t * angleRange : startAngle + t * angleRange
                let point = CGPoint(
                    x: scaledCenter.x + cos(angle) * scaledRadiusX,
                    y: scaledCenter.y + sin(angle) * scaledRadiusY
                )
                let tangentAngle = clockwise ? angle - .pi / 2 : angle + .pi / 2
                samples.append((point, tangentAngle))
            }
        }

        return samples
    }

    private func calculateLocalStrokeWidth(
        at direction: CGFloat,
        baseWidth: CGFloat,
        style: StyleParams
    ) -> CGFloat {
        // Apply contrast: vertical strokes are thicker, horizontal strokes are thinner
        let normalizedAngle = abs(sin(direction))  // 0 = horizontal, 1 = vertical

        // Contrast factor: 1.0 at vertical, reduced at horizontal
        let contrastFactor = 1.0 - style.strokeContrast * (1.0 - normalizedAngle) * CurveParams.contrastScale

        return baseWidth * contrastFactor
    }

    private func applySlant(
        to contours: [Contour],
        angle: CGFloat,
        baselineY: CGFloat
    ) -> [Contour] {
        let shearFactor = tan(angle * .pi / 180)

        return contours.map { contour in
            let slantedPoints = contour.points.map { point -> PathPoint in
                var newPoint = point

                // Shear transformation: x' = x + y * shear
                let yOffset = newPoint.position.y - baselineY
                newPoint.position.x += yOffset * shearFactor

                if let controlIn = point.controlIn {
                    let ciYOffset = controlIn.y - baselineY
                    newPoint.controlIn = CGPoint(
                        x: controlIn.x + ciYOffset * shearFactor,
                        y: controlIn.y
                    )
                }

                if let controlOut = point.controlOut {
                    let coYOffset = controlOut.y - baselineY
                    newPoint.controlOut = CGPoint(
                        x: controlOut.x + coYOffset * shearFactor,
                        y: controlOut.y
                    )
                }

                return newPoint
            }

            return Contour(id: contour.id, points: slantedPoints, isClosed: contour.isClosed)
        }
    }

    private func addSmoothHandles(to points: [PathPoint], roundness: CGFloat) -> [PathPoint] {
        guard points.count >= 3 && roundness > 0 else { return points }

        var result: [PathPoint] = []

        for i in 0..<points.count {
            var point = points[i]

            let prevIndex = (i - 1 + points.count) % points.count
            let nextIndex = (i + 1) % points.count

            let prev = points[prevIndex].position
            let next = points[nextIndex].position
            let curr = point.position

            // Calculate handle length based on distance to neighbors
            let toPrev = CGPoint(x: prev.x - curr.x, y: prev.y - curr.y)
            let toNext = CGPoint(x: next.x - curr.x, y: next.y - curr.y)

            let distToPrev = sqrt(toPrev.x * toPrev.x + toPrev.y * toPrev.y)
            let distToNext = sqrt(toNext.x * toNext.x + toNext.y * toNext.y)

            let handleLength = min(distToPrev, distToNext) * CurveParams.handleLengthFactor * roundness

            if point.type == .smooth || point.type == .symmetric {
                // Calculate smooth handles
                let inDir = atan2(toPrev.y, toPrev.x)
                let outDir = atan2(toNext.y, toNext.x)

                // Average direction for smooth curve
                let avgDir = atan2(
                    sin(inDir) + sin(outDir),
                    cos(inDir) + cos(outDir)
                )

                point.controlIn = CGPoint(
                    x: curr.x - cos(avgDir + .pi) * handleLength,
                    y: curr.y - sin(avgDir + .pi) * handleLength
                )
                point.controlOut = CGPoint(
                    x: curr.x + cos(avgDir + .pi) * handleLength,
                    y: curr.y + sin(avgDir + .pi) * handleLength
                )
            }

            result.append(point)
        }

        return result
    }

    private func deduplicatePoints(_ points: [PathPoint]) -> [PathPoint] {
        guard points.count > 1 else { return points }

        var result: [PathPoint] = [points[0]]

        for i in 1..<points.count {
            let prev = result.last!.position
            let curr = points[i].position

            let dist = sqrt(pow(curr.x - prev.x, 2) + pow(curr.y - prev.y, 2))
            if dist > CurveParams.deduplicationThreshold {
                result.append(points[i])
            }
        }

        return result
    }

    private func deduplicateSamples(_ samples: [(point: CGPoint, direction: CGFloat)]) -> [(point: CGPoint, direction: CGFloat)] {
        guard samples.count > 1 else { return samples }

        var result: [(point: CGPoint, direction: CGFloat)] = [samples[0]]

        for i in 1..<samples.count {
            let prev = result.last!.point
            let curr = samples[i].point

            let dist = sqrt(pow(curr.x - prev.x, 2) + pow(curr.y - prev.y, 2))
            if dist > CurveParams.deduplicationThreshold {
                result.append(samples[i])
            }
        }

        return result
    }

    // MARK: - Bezier Math

    private func quadraticBezier(_ p0: CGPoint, _ p1: CGPoint, _ p2: CGPoint, t: CGFloat) -> CGPoint {
        let mt = 1 - t
        return CGPoint(
            x: mt * mt * p0.x + 2 * mt * t * p1.x + t * t * p2.x,
            y: mt * mt * p0.y + 2 * mt * t * p1.y + t * t * p2.y
        )
    }

    private func quadraticBezierTangent(_ p0: CGPoint, _ p1: CGPoint, _ p2: CGPoint, t: CGFloat) -> CGPoint {
        let mt = 1 - t
        return CGPoint(
            x: 2 * mt * (p1.x - p0.x) + 2 * t * (p2.x - p1.x),
            y: 2 * mt * (p1.y - p0.y) + 2 * t * (p2.y - p1.y)
        )
    }

    private func cubicBezier(_ p0: CGPoint, _ p1: CGPoint, _ p2: CGPoint, _ p3: CGPoint, t: CGFloat) -> CGPoint {
        let mt = 1 - t
        let mt2 = mt * mt
        let mt3 = mt2 * mt
        let t2 = t * t
        let t3 = t2 * t

        return CGPoint(
            x: mt3 * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t3 * p3.x,
            y: mt3 * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t3 * p3.y
        )
    }

    private func cubicBezierTangent(_ p0: CGPoint, _ p1: CGPoint, _ p2: CGPoint, _ p3: CGPoint, t: CGFloat) -> CGPoint {
        let mt = 1 - t
        let mt2 = mt * mt
        let t2 = t * t

        return CGPoint(
            x: 3 * mt2 * (p1.x - p0.x) + 6 * mt * t * (p2.x - p1.x) + 3 * t2 * (p3.x - p2.x),
            y: 3 * mt2 * (p1.y - p0.y) + 6 * mt * t * (p2.y - p1.y) + 3 * t2 * (p3.y - p2.y)
        )
    }
}
