import Foundation
import CoreGraphics
import AppKit

/// Service for tracing contours in binary images and converting to vector paths
final class ContourTracer {

    enum TracerError: Error, LocalizedError {
        case noContoursFound
        case invalidInput

        var errorDescription: String? {
            switch self {
            case .noContoursFound:
                return "No contours found in image"
            case .invalidInput:
                return "Invalid input image"
            }
        }
    }

    struct TracingSettings {
        var simplificationTolerance: CGFloat = 2.0
        var minContourLength: Int = 10
        var detectCorners: Bool = true
        var cornerAngleThreshold: CGFloat = 60
        var smoothCurves: Bool = true

        static let `default` = TracingSettings()
    }

    /// A traced contour with simplified points
    struct TracedContour {
        var points: [CGPoint]
        var cornerIndices: [Int]
        var isClosed: Bool

        /// Convert to GlyphOutline Contour
        func toGlyphContour() -> Contour {
            var pathPoints: [PathPoint] = []

            for (index, point) in points.enumerated() {
                let isCorner = cornerIndices.contains(index)
                let type: PathPoint.PointType = isCorner ? .corner : .smooth

                pathPoints.append(PathPoint(position: point, type: type))
            }

            return Contour(points: pathPoints, isClosed: isClosed)
        }
    }

    // MARK: - Main Tracing API

    /// Trace contours from a binary image
    static func trace(
        binary: [[Bool]],
        width: Int,
        height: Int,
        settings: TracingSettings = .default
    ) throws -> [TracedContour] {
        // Detect edges
        let edgeChains = EdgeDetector.detectEdges(binary: binary, width: width, height: height)

        guard !edgeChains.isEmpty else {
            throw TracerError.noContoursFound
        }

        var contours: [TracedContour] = []

        for chain in edgeChains {
            // Skip short contours
            guard chain.points.count >= settings.minContourLength else { continue }

            // Simplify
            let simplified = EdgeDetector.simplify(chain: chain, tolerance: settings.simplificationTolerance)

            // Detect corners if enabled
            var cornerIndices: [Int] = []
            if settings.detectCorners {
                cornerIndices = EdgeDetector.detectCorners(in: simplified, angleThreshold: settings.cornerAngleThreshold)
            }

            let contour = TracedContour(
                points: simplified,
                cornerIndices: cornerIndices,
                isClosed: chain.isClosed
            )

            contours.append(contour)
        }

        return contours
    }

    /// Trace contours from pixel data
    static func trace(
        pixelData: ImageProcessor.PixelData,
        settings: TracingSettings = .default
    ) throws -> [TracedContour] {
        let binary = pixelData.toBinary()
        return try trace(binary: binary, width: pixelData.width, height: pixelData.height, settings: settings)
    }

    /// Trace contours from a CGImage
    static func trace(
        image: CGImage,
        processingSettings: ImageProcessor.ProcessingSettings = .default,
        tracingSettings: TracingSettings = .default
    ) throws -> [TracedContour] {
        // Process image
        let processed = try ImageProcessor.process(cgImage: image, settings: processingSettings)

        // Get pixel data
        let pixelData = try ImageProcessor.getPixelData(processed)

        // Trace contours
        return try trace(pixelData: pixelData, settings: tracingSettings)
    }

    // MARK: - Convert to GlyphOutline

    /// Convert traced contours to a GlyphOutline
    static func toGlyphOutline(
        contours: [TracedContour],
        metrics: FontMetrics,
        fitToCapHeight: Bool = true
    ) -> GlyphOutline {
        guard !contours.isEmpty else {
            return GlyphOutline()
        }

        // Collect all points to find bounds
        var allPoints: [CGPoint] = []
        for contour in contours {
            allPoints.append(contentsOf: contour.points)
        }

        guard !allPoints.isEmpty else {
            return GlyphOutline()
        }

        // Find bounding box
        let minX = allPoints.map { $0.x }.min()!
        let maxX = allPoints.map { $0.x }.max()!
        let minY = allPoints.map { $0.y }.min()!
        let maxY = allPoints.map { $0.y }.max()!

        let width = maxX - minX
        let height = maxY - minY

        guard width > 0, height > 0 else {
            return GlyphOutline()
        }

        // Calculate scale
        let targetHeight = fitToCapHeight ? CGFloat(metrics.capHeight) : CGFloat(metrics.unitsPerEm)
        let scale = targetHeight / height

        // Transform contours
        var glyphContours: [Contour] = []

        for tracedContour in contours {
            let transformedPoints = tracedContour.points.map { point in
                CGPoint(
                    x: (point.x - minX) * scale + CGFloat(metrics.unitsPerEm) * 0.1,
                    y: (maxY - point.y) * scale // Flip Y and scale
                )
            }

            var pathPoints: [PathPoint] = []
            for (index, point) in transformedPoints.enumerated() {
                let isCorner = tracedContour.cornerIndices.contains(index)
                let type: PathPoint.PointType = isCorner ? .corner : .smooth

                var pathPoint = PathPoint(position: point, type: type)

                // Add control handles for smooth points
                if type == .smooth && transformedPoints.count >= 3 {
                    let prevIndex = (index - 1 + transformedPoints.count) % transformedPoints.count
                    let nextIndex = (index + 1) % transformedPoints.count

                    let prev = transformedPoints[prevIndex]
                    let next = transformedPoints[nextIndex]

                    // Calculate tangent direction
                    let dx = next.x - prev.x
                    let dy = next.y - prev.y
                    let length = sqrt(dx * dx + dy * dy)

                    if length > 0 {
                        let handleLength = length * 0.25
                        let nx = dx / length
                        let ny = dy / length

                        pathPoint.controlIn = CGPoint(
                            x: point.x - nx * handleLength,
                            y: point.y - ny * handleLength
                        )
                        pathPoint.controlOut = CGPoint(
                            x: point.x + nx * handleLength,
                            y: point.y + ny * handleLength
                        )
                    }
                }

                pathPoints.append(pathPoint)
            }

            let contour = Contour(points: pathPoints, isClosed: tracedContour.isClosed)
            glyphContours.append(contour)
        }

        return GlyphOutline(contours: glyphContours)
    }

    // MARK: - Full Pipeline

    /// Complete vectorization pipeline: image -> glyph outline
    static func vectorize(
        image: CGImage,
        metrics: FontMetrics,
        processingSettings: ImageProcessor.ProcessingSettings = .default,
        tracingSettings: TracingSettings = .default
    ) throws -> GlyphOutline {
        let contours = try trace(
            image: image,
            processingSettings: processingSettings,
            tracingSettings: tracingSettings
        )

        return toGlyphOutline(contours: contours, metrics: metrics)
    }

    /// Vectorize from NSImage
    static func vectorize(
        image: NSImage,
        metrics: FontMetrics,
        processingSettings: ImageProcessor.ProcessingSettings = .default,
        tracingSettings: TracingSettings = .default
    ) throws -> GlyphOutline {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw TracerError.invalidInput
        }

        return try vectorize(
            image: cgImage,
            metrics: metrics,
            processingSettings: processingSettings,
            tracingSettings: tracingSettings
        )
    }
}
