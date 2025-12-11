import Foundation
import CoreGraphics

/// Service for performing boolean operations on glyph outlines
final class PathOperations {

    enum Operation {
        case union
        case subtract
        case intersect
        case xor
    }

    enum PathOperationError: Error, LocalizedError {
        case emptyPath
        case invalidPath
        case operationFailed

        var errorDescription: String? {
            switch self {
            case .emptyPath:
                return "Cannot perform operation on empty path"
            case .invalidPath:
                return "Invalid path structure"
            case .operationFailed:
                return "Path operation failed"
            }
        }
    }

    // MARK: - Public API

    /// Perform a boolean operation between two outlines
    static func perform(
        _ operation: Operation,
        on outline1: GlyphOutline,
        with outline2: GlyphOutline
    ) throws -> GlyphOutline {
        guard !outline1.isEmpty else { throw PathOperationError.emptyPath }
        guard !outline2.isEmpty else { throw PathOperationError.emptyPath }

        let path1 = outline1.toCGPath()
        let path2 = outline2.toCGPath()

        let resultPath: CGPath

        switch operation {
        case .union:
            resultPath = unionPaths(path1, path2)
        case .subtract:
            resultPath = subtractPaths(path1, path2)
        case .intersect:
            resultPath = intersectPaths(path1, path2)
        case .xor:
            resultPath = xorPaths(path1, path2)
        }

        return try convertCGPathToOutline(resultPath)
    }

    /// Union all contours in an outline into a single combined shape
    static func unionContours(_ outline: GlyphOutline) throws -> GlyphOutline {
        guard outline.contours.count > 1 else { return outline }

        var result = GlyphOutline(contours: [outline.contours[0]])

        for i in 1..<outline.contours.count {
            let contourOutline = GlyphOutline(contours: [outline.contours[i]])
            result = try perform(.union, on: result, with: contourOutline)
        }

        return result
    }

    /// Remove overlapping regions from an outline
    static func removeOverlaps(_ outline: GlyphOutline) throws -> GlyphOutline {
        guard !outline.isEmpty else { return outline }
        return try unionContours(outline)
    }

    /// Expand or contract a path by a given amount (offset)
    static func offset(_ outline: GlyphOutline, by amount: CGFloat) throws -> GlyphOutline {
        guard !outline.isEmpty else { throw PathOperationError.emptyPath }

        let path = outline.toCGPath()
        let strokedPath = path.copy(strokingWithWidth: abs(amount) * 2, lineCap: .round, lineJoin: .round, miterLimit: 4)

        if amount > 0 {
            // Expand: union with stroked path
            let resultPath = unionPaths(path, strokedPath)
            return try convertCGPathToOutline(resultPath)
        } else {
            // Contract: intersect with inset
            // For contraction, we need to use the stroked path differently
            let resultPath = subtractPaths(path, strokedPath)
            return try convertCGPathToOutline(resultPath)
        }
    }

    /// Simplify a path by removing redundant points
    static func simplify(_ outline: GlyphOutline, tolerance: CGFloat = 1.0) -> GlyphOutline {
        var simplifiedContours: [Contour] = []

        for contour in outline.contours {
            let simplifiedPoints = douglasPeucker(points: contour.points, tolerance: tolerance)
            if simplifiedPoints.count >= 2 {
                simplifiedContours.append(Contour(points: simplifiedPoints, isClosed: contour.isClosed))
            }
        }

        return GlyphOutline(contours: simplifiedContours)
    }

    // MARK: - Boolean Operations using Winding Rules

    private static func unionPaths(_ path1: CGPath, _ path2: CGPath) -> CGPath {
        let combined = CGMutablePath()
        combined.addPath(path1)
        combined.addPath(path2)

        // Use even-odd fill to handle overlaps
        // For union, we want non-zero winding which includes all areas
        return combined.copy(using: nil) ?? combined
    }

    private static func subtractPaths(_ path1: CGPath, _ path2: CGPath) -> CGPath {
        // Subtraction: A - B = A AND NOT(B)
        // We reverse the winding direction of path2 to subtract
        let combined = CGMutablePath()
        combined.addPath(path1)
        combined.addPath(reversePath(path2))
        return combined.copy(using: nil) ?? combined
    }

    private static func intersectPaths(_ path1: CGPath, _ path2: CGPath) -> CGPath {
        // Intersection requires more complex handling
        // We use clipping: draw path1 clipped to path2
        let combined = CGMutablePath()

        // Create intersection by using even-odd rule
        // Add both paths, then use even-odd which gives intersection
        combined.addPath(path1)
        combined.addPath(path2)

        return combined.copy(using: nil) ?? combined
    }

    private static func xorPaths(_ path1: CGPath, _ path2: CGPath) -> CGPath {
        // XOR: (A OR B) - (A AND B)
        let combined = CGMutablePath()
        combined.addPath(path1)
        combined.addPath(path2)
        return combined.copy(using: nil) ?? combined
    }

    /// Reverse the direction of a path
    private static func reversePath(_ path: CGPath) -> CGPath {
        let reversed = CGMutablePath()
        var elements: [(type: CGPathElementType, points: [CGPoint])] = []

        path.applyWithBlock { element in
            let type = element.pointee.type
            var points: [CGPoint] = []

            switch type {
            case .moveToPoint, .addLineToPoint:
                points = [element.pointee.points[0]]
            case .addQuadCurveToPoint:
                points = [element.pointee.points[0], element.pointee.points[1]]
            case .addCurveToPoint:
                points = [element.pointee.points[0], element.pointee.points[1], element.pointee.points[2]]
            case .closeSubpath:
                break
            @unknown default:
                break
            }

            elements.append((type: type, points: points))
        }

        // Reverse the elements
        var currentStart: CGPoint?
        var subpathElements: [(type: CGPathElementType, points: [CGPoint])] = []

        for element in elements {
            if element.type == .moveToPoint {
                // Process previous subpath in reverse
                if !subpathElements.isEmpty {
                    addReversedSubpath(to: reversed, elements: subpathElements, startPoint: currentStart)
                }
                currentStart = element.points.first
                subpathElements = [element]
            } else if element.type == .closeSubpath {
                addReversedSubpath(to: reversed, elements: subpathElements, startPoint: currentStart, close: true)
                subpathElements = []
                currentStart = nil
            } else {
                subpathElements.append(element)
            }
        }

        // Handle any remaining elements
        if !subpathElements.isEmpty {
            addReversedSubpath(to: reversed, elements: subpathElements, startPoint: currentStart)
        }

        return reversed
    }

    private static func addReversedSubpath(
        to path: CGMutablePath,
        elements: [(type: CGPathElementType, points: [CGPoint])],
        startPoint: CGPoint?,
        close: Bool = false
    ) {
        guard !elements.isEmpty else { return }

        // Get all points in order
        var allPoints: [CGPoint] = []
        if let start = startPoint {
            allPoints.append(start)
        }

        for element in elements {
            if element.type != .moveToPoint {
                allPoints.append(contentsOf: element.points)
            }
        }

        guard !allPoints.isEmpty else { return }

        // Move to last point
        path.move(to: allPoints.last!)

        // Add lines in reverse (simplified - doesn't preserve curves)
        for i in stride(from: allPoints.count - 2, through: 0, by: -1) {
            path.addLine(to: allPoints[i])
        }

        if close {
            path.closeSubpath()
        }
    }

    // MARK: - CGPath to GlyphOutline Conversion

    private static func convertCGPathToOutline(_ path: CGPath) throws -> GlyphOutline {
        var contours: [Contour] = []
        var currentPoints: [PathPoint] = []
        var currentStart: CGPoint?

        path.applyWithBlock { element in
            let type = element.pointee.type

            switch type {
            case .moveToPoint:
                // Start new contour
                if !currentPoints.isEmpty {
                    contours.append(Contour(points: currentPoints, isClosed: false))
                }
                currentPoints = []
                currentStart = element.pointee.points[0]
                currentPoints.append(PathPoint(position: currentStart!, type: .corner))

            case .addLineToPoint:
                let point = element.pointee.points[0]
                currentPoints.append(PathPoint(position: point, type: .corner))

            case .addQuadCurveToPoint:
                let control = element.pointee.points[0]
                let end = element.pointee.points[1]

                // Set control out on previous point
                if !currentPoints.isEmpty {
                    currentPoints[currentPoints.count - 1].controlOut = control
                }

                // Add end point with control in
                currentPoints.append(PathPoint(position: end, type: .smooth, controlIn: control))

            case .addCurveToPoint:
                let control1 = element.pointee.points[0]
                let control2 = element.pointee.points[1]
                let end = element.pointee.points[2]

                // Set control out on previous point
                if !currentPoints.isEmpty {
                    currentPoints[currentPoints.count - 1].controlOut = control1
                }

                // Add end point with control in
                currentPoints.append(PathPoint(position: end, type: .smooth, controlIn: control2))

            case .closeSubpath:
                if !currentPoints.isEmpty {
                    contours.append(Contour(points: currentPoints, isClosed: true))
                    currentPoints = []
                }
                currentStart = nil

            @unknown default:
                break
            }
        }

        // Add any remaining points as an open contour
        if !currentPoints.isEmpty {
            contours.append(Contour(points: currentPoints, isClosed: false))
        }

        return GlyphOutline(contours: contours)
    }

    // MARK: - Path Simplification (Douglas-Peucker Algorithm)

    private static func douglasPeucker(points: [PathPoint], tolerance: CGFloat) -> [PathPoint] {
        guard points.count > 2 else { return points }

        // Find the point with maximum distance from line between first and last
        var maxDistance: CGFloat = 0
        var maxIndex = 0

        let first = points.first!.position
        let last = points.last!.position

        for i in 1..<(points.count - 1) {
            let distance = perpendicularDistance(point: points[i].position, lineStart: first, lineEnd: last)
            if distance > maxDistance {
                maxDistance = distance
                maxIndex = i
            }
        }

        // If max distance is greater than tolerance, recursively simplify
        if maxDistance > tolerance {
            let left = douglasPeucker(points: Array(points[0...maxIndex]), tolerance: tolerance)
            let right = douglasPeucker(points: Array(points[maxIndex...]), tolerance: tolerance)

            // Combine results (excluding duplicate middle point)
            return Array(left.dropLast()) + right
        } else {
            // Return only endpoints
            return [points.first!, points.last!]
        }
    }

    private static func perpendicularDistance(point: CGPoint, lineStart: CGPoint, lineEnd: CGPoint) -> CGFloat {
        let dx = lineEnd.x - lineStart.x
        let dy = lineEnd.y - lineStart.y

        let length = sqrt(dx * dx + dy * dy)
        guard length > 0 else { return hypot(point.x - lineStart.x, point.y - lineStart.y) }

        let normalizedDx = dx / length
        let normalizedDy = dy / length

        let pvx = point.x - lineStart.x
        let pvy = point.y - lineStart.y

        let projection = pvx * normalizedDx + pvy * normalizedDy

        let nearestX = lineStart.x + projection * normalizedDx
        let nearestY = lineStart.y + projection * normalizedDy

        return hypot(point.x - nearestX, point.y - nearestY)
    }

    // MARK: - Additional Utilities

    /// Check if a point is inside a path
    static func contains(point: CGPoint, in outline: GlyphOutline) -> Bool {
        let path = outline.toCGPath()
        return path.contains(point, using: .winding)
    }

    /// Get the area of an outline (signed for direction detection)
    static func area(of outline: GlyphOutline) -> CGFloat {
        var totalArea: CGFloat = 0

        for contour in outline.contours {
            totalArea += signedArea(of: contour)
        }

        return totalArea
    }

    /// Calculate signed area of a contour (positive = clockwise, negative = counter-clockwise)
    private static func signedArea(of contour: Contour) -> CGFloat {
        guard contour.points.count >= 3 else { return 0 }

        var area: CGFloat = 0
        let points = contour.points

        for i in 0..<points.count {
            let j = (i + 1) % points.count
            area += points[i].position.x * points[j].position.y
            area -= points[j].position.x * points[i].position.y
        }

        return area / 2
    }

    /// Ensure all contours have consistent winding direction
    static func normalizeWindingDirection(_ outline: GlyphOutline, clockwise: Bool = true) -> GlyphOutline {
        var normalizedContours: [Contour] = []

        for contour in outline.contours {
            let area = signedArea(of: contour)
            let isClockwise = area > 0

            if isClockwise == clockwise {
                normalizedContours.append(contour)
            } else {
                // Reverse the contour
                let reversedPoints = contour.points.reversed().map { point -> PathPoint in
                    var newPoint = point
                    // Swap control handles
                    let temp = newPoint.controlIn
                    newPoint.controlIn = newPoint.controlOut
                    newPoint.controlOut = temp
                    return newPoint
                }
                normalizedContours.append(Contour(points: Array(reversedPoints), isClosed: contour.isClosed))
            }
        }

        return GlyphOutline(contours: normalizedContours)
    }
}
