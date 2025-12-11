import Foundation
import CoreGraphics

/// Service for detecting edges in binary images
final class EdgeDetector {

    /// Edge point with direction information
    struct EdgePoint: Equatable, Hashable {
        let x: Int
        let y: Int
        let direction: Direction

        enum Direction: Int, CaseIterable {
            case right = 0
            case downRight = 1
            case down = 2
            case downLeft = 3
            case left = 4
            case upLeft = 5
            case up = 6
            case upRight = 7

            var dx: Int {
                switch self {
                case .right, .downRight, .upRight: return 1
                case .left, .downLeft, .upLeft: return -1
                case .up, .down: return 0
                }
            }

            var dy: Int {
                switch self {
                case .down, .downRight, .downLeft: return 1
                case .up, .upRight, .upLeft: return -1
                case .left, .right: return 0
                }
            }

            func next() -> Direction {
                Direction(rawValue: (rawValue + 1) % 8)!
            }

            func previous() -> Direction {
                Direction(rawValue: (rawValue + 7) % 8)!
            }

            func opposite() -> Direction {
                Direction(rawValue: (rawValue + 4) % 8)!
            }
        }
    }

    /// A chain of edge points forming a contour
    struct EdgeChain {
        var points: [EdgePoint]
        var isClosed: Bool

        var cgPoints: [CGPoint] {
            points.map { CGPoint(x: $0.x, y: $0.y) }
        }
    }

    // MARK: - Edge Detection

    /// Detect edges in a binary image using Moore neighborhood tracing
    static func detectEdges(
        binary: [[Bool]],
        width: Int,
        height: Int
    ) -> [EdgeChain] {
        var visited = [[Bool]](repeating: [Bool](repeating: false, count: width), count: height)
        var chains: [EdgeChain] = []

        // Scan for boundary pixels
        for y in 0..<height {
            for x in 0..<width {
                // Look for a foreground pixel that hasn't been visited
                // and has a background neighbor to the left (start of contour)
                if binary[y][x] && !visited[y][x] {
                    let hasBackgroundLeft = x == 0 || !binary[y][x - 1]
                    if hasBackgroundLeft {
                        if let chain = traceContour(binary: binary, startX: x, startY: y, visited: &visited, width: width, height: height) {
                            chains.append(chain)
                        }
                    }
                }
            }
        }

        return chains
    }

    /// Trace a single contour using Moore neighborhood tracing
    private static func traceContour(
        binary: [[Bool]],
        startX: Int,
        startY: Int,
        visited: inout [[Bool]],
        width: Int,
        height: Int
    ) -> EdgeChain? {
        var points: [EdgePoint] = []
        var x = startX
        var y = startY
        var direction = EdgePoint.Direction.right

        // Start by going right from the start point
        let startPoint = EdgePoint(x: x, y: y, direction: direction)
        points.append(startPoint)
        visited[y][x] = true

        // Moore neighbor offsets in clockwise order
        let neighborOffsets: [(Int, Int, EdgePoint.Direction)] = [
            (1, 0, .right),
            (1, 1, .downRight),
            (0, 1, .down),
            (-1, 1, .downLeft),
            (-1, 0, .left),
            (-1, -1, .upLeft),
            (0, -1, .up),
            (1, -1, .upRight)
        ]

        var iterations = 0
        let maxIterations = width * height * 2 // Safety limit

        while iterations < maxIterations {
            iterations += 1

            // Find next boundary pixel by checking neighbors
            // Start from the opposite direction we came from
            let startDir = direction.opposite().next()
            var found = false

            for i in 0..<8 {
                let dirIndex = (startDir.rawValue + i) % 8
                let (dx, dy, dir) = neighborOffsets[dirIndex]
                let nx = x + dx
                let ny = y + dy

                // Check bounds
                guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }

                if binary[ny][nx] {
                    // Found next foreground pixel
                    x = nx
                    y = ny
                    direction = dir

                    // Check if we've returned to start
                    if x == startX && y == startY {
                        return EdgeChain(points: points, isClosed: true)
                    }

                    let point = EdgePoint(x: x, y: y, direction: direction)
                    points.append(point)
                    visited[y][x] = true
                    found = true
                    break
                }
            }

            if !found {
                // Dead end - return open chain
                return EdgeChain(points: points, isClosed: false)
            }
        }

        // Max iterations reached - return what we have
        return points.isEmpty ? nil : EdgeChain(points: points, isClosed: false)
    }

    // MARK: - Contour Simplification

    /// Simplify a contour using Douglas-Peucker algorithm
    static func simplify(chain: EdgeChain, tolerance: CGFloat) -> [CGPoint] {
        let points = chain.cgPoints
        guard points.count > 2 else { return points }

        return douglasPeucker(points: points, tolerance: tolerance)
    }

    private static func douglasPeucker(points: [CGPoint], tolerance: CGFloat) -> [CGPoint] {
        guard points.count > 2 else { return points }

        // Find point with maximum distance from line between first and last
        var maxDistance: CGFloat = 0
        var maxIndex = 0

        let first = points.first!
        let last = points.last!

        for i in 1..<(points.count - 1) {
            let distance = perpendicularDistance(point: points[i], lineStart: first, lineEnd: last)
            if distance > maxDistance {
                maxDistance = distance
                maxIndex = i
            }
        }

        // If max distance exceeds tolerance, recursively simplify
        if maxDistance > tolerance {
            let left = douglasPeucker(points: Array(points[0...maxIndex]), tolerance: tolerance)
            let right = douglasPeucker(points: Array(points[maxIndex...]), tolerance: tolerance)

            return Array(left.dropLast()) + right
        } else {
            return [first, last]
        }
    }

    private static func perpendicularDistance(point: CGPoint, lineStart: CGPoint, lineEnd: CGPoint) -> CGFloat {
        let dx = lineEnd.x - lineStart.x
        let dy = lineEnd.y - lineStart.y
        let length = sqrt(dx * dx + dy * dy)

        guard length > 0 else {
            return sqrt(pow(point.x - lineStart.x, 2) + pow(point.y - lineStart.y, 2))
        }

        let t = max(0, min(1, ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / (length * length)))

        let projX = lineStart.x + t * dx
        let projY = lineStart.y + t * dy

        return sqrt(pow(point.x - projX, 2) + pow(point.y - projY, 2))
    }

    // MARK: - Corner Detection

    /// Detect corner points in a contour
    static func detectCorners(in points: [CGPoint], angleThreshold: CGFloat = 45) -> [Int] {
        guard points.count >= 3 else { return [] }

        var corners: [Int] = []
        let threshold = angleThreshold * .pi / 180 // Convert to radians

        for i in 0..<points.count {
            let prev = points[(i - 1 + points.count) % points.count]
            let curr = points[i]
            let next = points[(i + 1) % points.count]

            // Calculate vectors
            let v1 = CGPoint(x: curr.x - prev.x, y: curr.y - prev.y)
            let v2 = CGPoint(x: next.x - curr.x, y: next.y - curr.y)

            // Calculate angle between vectors
            let angle1 = atan2(v1.y, v1.x)
            let angle2 = atan2(v2.y, v2.x)

            var angleDiff = abs(angle2 - angle1)
            if angleDiff > .pi {
                angleDiff = 2 * .pi - angleDiff
            }

            if angleDiff > threshold {
                corners.append(i)
            }
        }

        return corners
    }

    // MARK: - Utilities

    /// Convert pixel coordinates to glyph coordinates
    static func toGlyphCoordinates(
        points: [CGPoint],
        imageSize: CGSize,
        glyphMetrics: FontMetrics,
        targetHeight: CGFloat? = nil
    ) -> [CGPoint] {
        guard !points.isEmpty else { return [] }

        // Find bounding box
        let minX = points.map { $0.x }.min()!
        let maxX = points.map { $0.x }.max()!
        let minY = points.map { $0.y }.min()!
        let maxY = points.map { $0.y }.max()!

        let width = maxX - minX
        let height = maxY - minY

        guard width > 0, height > 0 else { return points }

        // Calculate scale to fit glyph metrics
        let targetH = targetHeight ?? CGFloat(glyphMetrics.capHeight)
        let scale = targetH / height

        return points.map { point in
            CGPoint(
                x: (point.x - minX) * scale + CGFloat(glyphMetrics.unitsPerEm) * 0.1,
                y: (maxY - point.y) * scale // Flip Y axis
            )
        }
    }
}
