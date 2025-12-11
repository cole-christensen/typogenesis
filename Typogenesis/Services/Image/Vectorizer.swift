import Foundation
import CoreGraphics
import AppKit

/// Main service for vectorizing handwritten characters to glyph outlines
final class Vectorizer {

    enum VectorizerError: Error, LocalizedError {
        case invalidImage
        case noCharactersDetected
        case vectorizationFailed(String)

        var errorDescription: String? {
            switch self {
            case .invalidImage:
                return "Invalid or corrupted image"
            case .noCharactersDetected:
                return "No characters detected in image"
            case .vectorizationFailed(let reason):
                return "Vectorization failed: \(reason)"
            }
        }
    }

    /// Settings for the complete vectorization pipeline
    struct VectorizationSettings {
        var imageProcessing: ImageProcessor.ProcessingSettings = .default
        var tracing: ContourTracer.TracingSettings = .default
        var bezierFitting: BezierFitter.FittingSettings = .default
        var useBezierFitting: Bool = true        // Use bezier fitting vs. simple corner detection
        var minCharacterSize: Int = 10           // Minimum character size in pixels
        var characterPadding: Int = 5            // Padding around detected characters

        static let `default` = VectorizationSettings()

        /// Settings optimized for clean handwriting
        static let cleanHandwriting = VectorizationSettings(
            imageProcessing: .init(threshold: 0.4, denoise: true),
            tracing: .init(simplificationTolerance: 1.5, cornerAngleThreshold: 50),
            bezierFitting: .init(errorThreshold: 3.0)
        )

        /// Settings optimized for rough/sketchy handwriting
        static let roughHandwriting = VectorizationSettings(
            imageProcessing: .init(threshold: 0.6, contrast: 1.2, denoise: true),
            tracing: .init(simplificationTolerance: 3.0, cornerAngleThreshold: 45),
            bezierFitting: .init(errorThreshold: 5.0)
        )

        /// Settings for printed/typed characters
        static let printedCharacters = VectorizationSettings(
            imageProcessing: .init(threshold: 0.5, denoise: false),
            tracing: .init(simplificationTolerance: 1.0, cornerAngleThreshold: 70),
            bezierFitting: .init(errorThreshold: 2.0)
        )
    }

    /// Result of vectorizing a single character
    struct VectorizedCharacter {
        let bounds: CGRect
        let outline: GlyphOutline
        var assignedCharacter: Character?
    }

    /// Result of vectorizing an entire image
    struct VectorizationResult {
        let characters: [VectorizedCharacter]
        let processingTime: TimeInterval
        let imageSize: CGSize
    }

    // MARK: - Main Vectorization API

    /// Vectorize all characters in an image
    static func vectorize(
        image: NSImage,
        metrics: FontMetrics,
        settings: VectorizationSettings = .default
    ) async throws -> VectorizationResult {
        let startTime = Date()

        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw VectorizerError.invalidImage
        }

        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)

        // Process the image
        let processed = try ImageProcessor.process(cgImage: cgImage, settings: settings.imageProcessing)

        // Get pixel data
        let pixelData = try ImageProcessor.getPixelData(processed)

        // Detect character bounding boxes
        let bounds = ImageProcessor.detectCharacterBounds(
            in: pixelData,
            minSize: settings.minCharacterSize,
            padding: settings.characterPadding
        )

        guard !bounds.isEmpty else {
            throw VectorizerError.noCharactersDetected
        }

        // Vectorize each detected character
        var characters: [VectorizedCharacter] = []

        for charBounds in bounds {
            do {
                // Extract character image
                let charImage = try ImageProcessor.extractCharacter(from: processed, bounds: charBounds)

                // Vectorize the character
                let outline = try vectorizeCharacter(
                    image: charImage,
                    metrics: metrics,
                    settings: settings
                )

                let character = VectorizedCharacter(
                    bounds: charBounds,
                    outline: outline,
                    assignedCharacter: nil
                )
                characters.append(character)
            } catch {
                // Skip characters that fail to vectorize
                continue
            }
        }

        let processingTime = Date().timeIntervalSince(startTime)

        return VectorizationResult(
            characters: characters,
            processingTime: processingTime,
            imageSize: imageSize
        )
    }

    /// Vectorize a single character image
    static func vectorizeCharacter(
        image: CGImage,
        metrics: FontMetrics,
        settings: VectorizationSettings = .default
    ) throws -> GlyphOutline {
        // Get pixel data
        let pixelData = try ImageProcessor.getPixelData(image)

        // Trace contours
        let tracedContours = try ContourTracer.trace(
            pixelData: pixelData,
            settings: settings.tracing
        )

        guard !tracedContours.isEmpty else {
            throw VectorizerError.vectorizationFailed("No contours found")
        }

        if settings.useBezierFitting {
            // Use bezier curve fitting for smoother results
            return fitBeziersToContours(
                contours: tracedContours,
                metrics: metrics,
                settings: settings.bezierFitting
            )
        } else {
            // Use simple corner detection
            return ContourTracer.toGlyphOutline(
                contours: tracedContours,
                metrics: metrics
            )
        }
    }

    /// Fit bezier curves to traced contours
    private static func fitBeziersToContours(
        contours: [ContourTracer.TracedContour],
        metrics: FontMetrics,
        settings: BezierFitter.FittingSettings
    ) -> GlyphOutline {
        var glyphContours: [Contour] = []

        // Collect all points to find bounds for normalization
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

        // Calculate scale to fit cap height
        let targetHeight = CGFloat(metrics.capHeight)
        let scale = targetHeight / height
        let margin = CGFloat(metrics.unitsPerEm) * 0.1

        for tracedContour in contours {
            // Transform points to glyph coordinates
            let transformedPoints = tracedContour.points.map { point in
                CGPoint(
                    x: (point.x - minX) * scale + margin,
                    y: (maxY - point.y) * scale // Flip Y
                )
            }

            // Fit bezier curves
            let segments = BezierFitter.fitCurves(
                to: transformedPoints,
                isClosed: tracedContour.isClosed,
                settings: settings
            )

            // Convert to PathPoints
            let pathPoints = BezierFitter.toPathPoints(segments: segments)

            if !pathPoints.isEmpty {
                glyphContours.append(Contour(points: pathPoints, isClosed: tracedContour.isClosed))
            }
        }

        return GlyphOutline(contours: glyphContours)
    }

    // MARK: - Sample Sheet Processing

    /// Process a sample sheet with a known grid layout
    static func processSampleSheet(
        image: NSImage,
        characters: String,  // Expected characters in row-major order
        rows: Int,
        cols: Int,
        metrics: FontMetrics,
        settings: VectorizationSettings = .default
    ) async throws -> [(character: Character, outline: GlyphOutline)] {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw VectorizerError.invalidImage
        }

        // Process image
        let processed = try ImageProcessor.process(cgImage: cgImage, settings: settings.imageProcessing)

        // Detect grid cells
        let grid = try ImageProcessor.detectGridCells(in: processed, expectedRows: rows, expectedCols: cols)

        var results: [(character: Character, outline: GlyphOutline)] = []
        let chars = Array(characters)
        var charIndex = 0

        for row in grid {
            for cell in row {
                guard charIndex < chars.count else { break }

                do {
                    // Extract cell image
                    let cellImage = try ImageProcessor.extractCharacter(from: processed, bounds: cell)

                    // Vectorize
                    let outline = try vectorizeCharacter(
                        image: cellImage,
                        metrics: metrics,
                        settings: settings
                    )

                    // Only include non-empty outlines
                    if !outline.isEmpty {
                        results.append((character: chars[charIndex], outline: outline))
                    }
                } catch {
                    // Skip cells that fail
                }

                charIndex += 1
            }
        }

        return results
    }

    // MARK: - Batch Processing

    /// Vectorize multiple images concurrently
    static func vectorizeBatch(
        images: [NSImage],
        metrics: FontMetrics,
        settings: VectorizationSettings = .default
    ) async throws -> [VectorizationResult] {
        try await withThrowingTaskGroup(of: (Int, VectorizationResult).self) { group in
            for (index, image) in images.enumerated() {
                group.addTask {
                    let result = try await vectorize(image: image, metrics: metrics, settings: settings)
                    return (index, result)
                }
            }

            var results = [VectorizationResult?](repeating: nil, count: images.count)

            for try await (index, result) in group {
                results[index] = result
            }

            return results.compactMap { $0 }
        }
    }

    // MARK: - Utilities

    /// Create a glyph from a vectorized character
    static func createGlyph(
        from vectorized: VectorizedCharacter,
        character: Character,
        metrics: FontMetrics
    ) -> Glyph {
        // Calculate advance width based on outline bounds
        let bounds = vectorized.outline.boundingBox
        let advanceWidth = bounds.width + Int(CGFloat(metrics.unitsPerEm) * 0.2) // Add side bearings

        return Glyph(
            character: character,
            outline: vectorized.outline,
            advanceWidth: advanceWidth,
            leftSideBearing: Int(CGFloat(metrics.unitsPerEm) * 0.1)
        )
    }
}
