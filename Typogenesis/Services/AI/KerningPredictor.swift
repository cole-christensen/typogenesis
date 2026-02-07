import Foundation
import CoreGraphics
import AppKit
@preconcurrency import CoreML
import CoreImage
import os.log

/// Service for predicting optimal kerning values using AI
///
/// ## Thread Safety
/// This class conforms to `Sendable` because it has no mutable instance state.
/// All instance properties are immutable (`let` constants), and all methods
/// use only local variables or access shared state through `MainActor`.
final class KerningPredictor: Sendable {

    /// Logger for tracking prediction method and fallback usage
    private static let logger = Logger(subsystem: "com.typogenesis", category: "KerningPredictor")

    // MARK: - Constants

    /// Confidence scores for prediction quality
    private enum Confidence {
        /// High confidence when using trained ML model
        static let withModel: Float = 0.85
        /// Lower confidence for geometric heuristics
        static let geometric: Float = 0.6
    }

    /// Parameters for kerning calculations
    private enum KerningParams {
        /// Base spacing as percentage of em (10%)
        static let baseSpacingRatio: CGFloat = 0.1
        /// Maximum kerning as fraction of em (1/4)
        static let maxKerningDivisor: CGFloat = 4.0
        /// Number of vertical samples for edge analysis
        static let edgeSampleCount: Int = 20
    }

    /// Parameters for glyph pair rendering
    private enum RenderParams {
        /// Output image size in pixels
        static let imageSize: Int = 128
        /// Scale factor for glyph rendering
        static let scaleFactor: CGFloat = 0.8
        /// Baseline position as percentage of image height (70%)
        static let baselineRatio: CGFloat = 0.7
    }

    /// Parameters for CoreML model inference
    private enum ModelInferenceParams {
        /// Maximum kerning value in font units (for denormalization)
        static let maxKerningUnits: Int = 250
        /// Batch size for efficient inference
        static let batchSize: Int = 32
    }

    enum PredictorError: Error, LocalizedError {
        case modelNotLoaded
        case predictionFailed(String)
        case insufficientGlyphs

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "Kerning prediction model is not loaded"
            case .predictionFailed(let reason):
                return "Prediction failed: \(reason)"
            case .insufficientGlyphs:
                return "Not enough glyphs in font to generate kerning"
            }
        }
    }

    /// Settings for kerning prediction
    struct PredictionSettings {
        var minKerningValue: Int = 2           // Minimum kerning to include
        var targetOpticalSpacing: Float = 0.5  // 0 = tight, 1 = loose
        var includePunctuation: Bool = true
        var includeNumbers: Bool = true
        var onlyCriticalPairs: Bool = false    // Only generate most important pairs

        static let `default` = PredictionSettings()

        /// Tight spacing preset
        static let tight = PredictionSettings(
            minKerningValue: 1,
            targetOpticalSpacing: 0.3
        )

        /// Loose spacing preset
        static let loose = PredictionSettings(
            minKerningValue: 3,
            targetOpticalSpacing: 0.7
        )
    }

    /// Result of kerning prediction
    struct PredictionResult {
        let pairs: [KerningPair]
        let predictionTime: TimeInterval
        let confidence: Float
    }

    /// Internal result from batch prediction that includes whether the model was used
    private struct BatchResult {
        let pairs: [KerningPair]
        let usedModel: Bool
    }

    // MARK: - Private Properties

    /// Critical kerning pairs that commonly need adjustment
    private let criticalPairs: [(String, String)] = [
        // Capital-lowercase combinations
        ("A", "v"), ("A", "w"), ("A", "y"),
        ("A", "V"), ("A", "W"), ("A", "Y"), ("A", "T"),
        ("F", "a"), ("F", "e"), ("F", "o"),
        ("L", "T"), ("L", "V"), ("L", "W"), ("L", "Y"),
        ("P", "a"), ("P", "e"), ("P", "o"),
        ("T", "a"), ("T", "e"), ("T", "i"), ("T", "o"), ("T", "r"), ("T", "u"),
        ("V", "a"), ("V", "e"), ("V", "i"), ("V", "o"), ("V", "u"),
        ("W", "a"), ("W", "e"), ("W", "i"), ("W", "o"), ("W", "u"),
        ("Y", "a"), ("Y", "e"), ("Y", "i"), ("Y", "o"), ("Y", "u"),

        // Lowercase combinations
        ("f", "f"), ("f", "i"), ("f", "l"), ("f", "t"),
        ("r", "a"), ("r", "e"), ("r", "o"),
        ("v", "a"), ("v", "e"), ("v", "o"),
        ("w", "a"), ("w", "e"), ("w", "o"),
        ("y", "a"), ("y", "e"), ("y", "o"),

        // Quote/punctuation combinations
        ("\"", "A"), ("\"", "J"), ("\"", "T"),
        ("A", "\""), ("V", "\""), ("W", "\""),
        ("(", "A"), ("(", "J"),

        // Number pairs
        ("1", "1"), ("7", "4")
    ]

    // MARK: - Public API

    /// Predict optimal kerning for a font project
    func predictKerning(
        for project: FontProject,
        settings: PredictionSettings = .default
    ) async throws -> PredictionResult {
        guard project.glyphs.count >= 2 else {
            throw PredictorError.insufficientGlyphs
        }

        let startTime = Date()

        // Get all character pairs to analyze
        let pairs = settings.onlyCriticalPairs ?
            getCriticalPairs(for: project) :
            getAllPairs(for: project, settings: settings)

        // Delegate to internal batch method which returns whether it used the model
        let batchResult = try await predictBatchInternal(
            pairs: pairs,
            project: project,
            settings: settings
        )

        let predictionTime = Date().timeIntervalSince(startTime)

        // Use the usedModel flag from the batch result instead of re-checking
        // model availability (avoids TOCTOU race where model state could change)
        return PredictionResult(
            pairs: batchResult.pairs,
            predictionTime: predictionTime,
            confidence: batchResult.usedModel ? Confidence.withModel : Confidence.geometric
        )
    }

    /// Predict kerning for a single pair
    internal func predictPair(
        left: Character,
        right: Character,
        project: FontProject
    ) async throws -> Int {
        guard project.glyphs[left] != nil,
              project.glyphs[right] != nil else {
            return 0
        }

        let hasModel = await MainActor.run { ModelManager.shared.kerningNet != nil }
        if hasModel {
            return try await predictPairWithModel(
                left: left,
                right: right,
                project: project
            ) ?? 0
        } else {
            return calculateGeometricKerning(
                left: left,
                right: right,
                project: project,
                settings: .default
            )
        }
    }

    /// Check if prediction is available
    var isAvailable: Bool {
        true  // Geometric fallback always available
    }

    // MARK: - Private Methods

    private func getCriticalPairs(for project: FontProject) -> [(Character, Character)] {
        criticalPairs.compactMap { (leftStr, rightStr) -> (Character, Character)? in
            guard let left = leftStr.first,
                  let right = rightStr.first,
                  project.glyphs[left] != nil,
                  project.glyphs[right] != nil else {
                return nil
            }
            return (left, right)
        }
    }

    private func getAllPairs(
        for project: FontProject,
        settings: PredictionSettings
    ) -> [(Character, Character)] {
        let maxPairs = 10000
        var pairs: [(Character, Character)] = []
        let characters = Array(project.glyphs.keys)

        outer: for left in characters {
            for right in characters {
                // Skip if it's punctuation/numbers and those are disabled
                let leftIsLetter = left.isLetter
                let rightIsLetter = right.isLetter
                let leftIsNumber = left.isNumber
                let rightIsNumber = right.isNumber

                if !settings.includePunctuation {
                    if (!leftIsLetter && !leftIsNumber) || (!rightIsLetter && !rightIsNumber) {
                        continue
                    }
                }

                if !settings.includeNumbers {
                    if leftIsNumber || rightIsNumber {
                        continue
                    }
                }

                pairs.append((left, right))

                // Enforce upper bound to prevent O(n^2) blowup on large glyph sets
                if pairs.count >= maxPairs {
                    break outer
                }
            }
        }

        return pairs
    }

    /// Predict kerning for a single pair using the CoreML model
    private func predictPairWithModel(
        left: Character,
        right: Character,
        project: FontProject
    ) async throws -> Int? {
        guard let leftGlyph = project.glyphs[left],
              let rightGlyph = project.glyphs[right] else {
            return nil
        }

        // Render pair to image for model input
        guard let pairImage = renderPair(
            leftGlyph: leftGlyph,
            rightGlyph: rightGlyph,
            metrics: project.metrics,
            spacing: 0
        ) else {
            return nil
        }

        // Get model from ModelManager
        let model = await MainActor.run { ModelManager.shared.kerningNet }
        guard let kerningModel = model else {
            // Fall back to geometric calculation
            Self.logger.info("Using geometric fallback for '\(String(left))'-'\(String(right))' - AI model not loaded")
            return calculateGeometricKerning(
                left: left,
                right: right,
                project: project,
                settings: .default
            )
        }

        do {
            // Create model input using MLDictionaryFeatureProvider (model-generated types unavailable until models compiled)
            let pixelBuffer = try createPixelBuffer(from: pairImage)
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: pixelBuffer)
            ])

            // Run inference - model.prediction is async in Swift 6
            let prediction = try await kerningModel.prediction(from: input)

            // Extract kerning value from output
            guard let outputFeature = prediction.featureValue(for: "kerning"),
                  let kerningArray = outputFeature.multiArrayValue else {
                throw PredictorError.predictionFailed("Model did not produce valid output")
            }

            // Model outputs normalized value [-1, 1], convert to font units
            let normalizedKerning = kerningArray[0].floatValue
            let kerningUnits = Int(normalizedKerning * Float(ModelInferenceParams.maxKerningUnits))

            return kerningUnits
        } catch {
            Self.logger.warning("Model inference failed for '\(String(left))'-'\(String(right))': \(error.localizedDescription)")
            // Fall back to geometric calculation
            return calculateGeometricKerning(
                left: left,
                right: right,
                project: project,
                settings: .default
            )
        }
    }

    /// Predict kerning for multiple pairs in batch (more efficient)
    func predictBatch(
        pairs: [(Character, Character)],
        project: FontProject,
        settings: PredictionSettings = .default
    ) async throws -> [KerningPair] {
        let result = try await predictBatchInternal(pairs: pairs, project: project, settings: settings)
        return result.pairs
    }

    /// Internal batch prediction that also reports whether the model was used
    private func predictBatchInternal(
        pairs: [(Character, Character)],
        project: FontProject,
        settings: PredictionSettings = .default
    ) async throws -> BatchResult {
        guard project.glyphs.count >= 2 else {
            throw PredictorError.insufficientGlyphs
        }

        let model = await MainActor.run { ModelManager.shared.kerningNet }

        if let kerningModel = model {
            // Use batch inference with model
            let pairs = try await predictBatchWithModel(
                pairs: pairs,
                project: project,
                model: kerningModel,
                settings: settings
            )
            return BatchResult(pairs: pairs, usedModel: true)
        } else {
            // Fall back to geometric calculation
            Self.logger.info("Using geometric fallback for batch prediction - AI model not loaded")
            let pairs = predictBatchGeometric(pairs: pairs, project: project, settings: settings)
            return BatchResult(pairs: pairs, usedModel: false)
        }
    }

    /// Batch prediction using CoreML model
    private func predictBatchWithModel(
        pairs: [(Character, Character)],
        project: FontProject,
        model: MLModel,
        settings: PredictionSettings
    ) async throws -> [KerningPair] {
        var results: [KerningPair] = []

        // Process in batches for efficiency
        for batchStart in stride(from: 0, to: pairs.count, by: ModelInferenceParams.batchSize) {
            let batchEnd = min(batchStart + ModelInferenceParams.batchSize, pairs.count)
            let batch = Array(pairs[batchStart..<batchEnd])

            // Render all pairs in batch
            var batchInputs: [(Character, Character, CGImage)] = []
            for (left, right) in batch {
                guard let leftGlyph = project.glyphs[left],
                      let rightGlyph = project.glyphs[right],
                      let pairImage = renderPair(
                          leftGlyph: leftGlyph,
                          rightGlyph: rightGlyph,
                          metrics: project.metrics,
                          spacing: 0
                      ) else {
                    continue
                }
                batchInputs.append((left, right, pairImage))
            }

            // Run inference on batch
            for (left, right, pairImage) in batchInputs {
                do {
                    // Create model input using MLDictionaryFeatureProvider (model-generated types unavailable until models compiled)
                    let pixelBuffer = try createPixelBuffer(from: pairImage)
                    let input = try MLDictionaryFeatureProvider(dictionary: [
                        "image": MLFeatureValue(pixelBuffer: pixelBuffer)
                    ])

                    // Run inference - model.prediction is async in Swift 6
                    let prediction = try await model.prediction(from: input)

                    guard let outputFeature = prediction.featureValue(for: "kerning"),
                          let kerningArray = outputFeature.multiArrayValue else {
                        continue
                    }

                    let normalizedKerning = kerningArray[0].floatValue
                    let kerningUnits = Int(normalizedKerning * Float(ModelInferenceParams.maxKerningUnits))

                    if abs(kerningUnits) >= settings.minKerningValue {
                        results.append(KerningPair(left: left, right: right, value: kerningUnits))
                    }
                } catch {
                    // Skip pair on error, log warning
                    Self.logger.warning("Batch inference failed for '\(String(left))'-'\(String(right))'")
                }
            }
        }

        return results
    }

    /// Batch prediction using geometric heuristics (fallback)
    private func predictBatchGeometric(
        pairs: [(Character, Character)],
        project: FontProject,
        settings: PredictionSettings
    ) -> [KerningPair] {
        var results: [KerningPair] = []

        for (left, right) in pairs {
            let kerning = calculateGeometricKerning(
                left: left,
                right: right,
                project: project,
                settings: settings
            )

            if abs(kerning) >= settings.minKerningValue {
                results.append(KerningPair(left: left, right: right, value: kerning))
            }
        }

        return results
    }

    /// Create CVPixelBuffer from CGImage for model input
    private func createPixelBuffer(from image: CGImage) throws -> CVPixelBuffer {
        let size = RenderParams.imageSize

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw PredictorError.predictionFailed("Failed to create pixel buffer")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw PredictorError.predictionFailed("Failed to create graphics context")
        }

        // Draw image scaled to target size
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        return buffer
    }

    private func calculateGeometricKerning(
        left: Character,
        right: Character,
        project: FontProject,
        settings: PredictionSettings
    ) -> Int {
        guard let leftGlyph = project.glyphs[left],
              let rightGlyph = project.glyphs[right] else {
            return 0
        }

        let leftBounds = leftGlyph.outline.boundingBox
        let rightBounds = rightGlyph.outline.boundingBox

        // Skip empty glyphs
        if leftBounds.width == 0 || rightBounds.width == 0 {
            return 0
        }

        // Analyze right side of left glyph
        let leftRightEdge = analyzeRightEdge(of: leftGlyph.outline)

        // Analyze left side of right glyph
        let rightLeftEdge = analyzeLeftEdge(of: rightGlyph.outline)

        // Calculate optimal spacing adjustment
        let baseSpacing = CGFloat(project.metrics.unitsPerEm) * KerningParams.baseSpacingRatio
        let opticalSpacing = baseSpacing * CGFloat(settings.targetOpticalSpacing)

        // Find minimum gap between edges
        let minGap = calculateMinimumGap(
            leftEdge: leftRightEdge,
            rightEdge: rightLeftEdge,
            leftGlyph: leftGlyph,
            rightGlyph: rightGlyph
        )

        // Calculate kerning to achieve target spacing
        let targetGap = opticalSpacing
        let kerning = targetGap - minGap

        // Clamp to reasonable range
        let maxKerning = CGFloat(project.metrics.unitsPerEm) / KerningParams.maxKerningDivisor
        let clampedKerning = max(-maxKerning, min(maxKerning, kerning))

        return Int(clampedKerning.rounded())
    }

    /// Edge profile for kerning analysis
    private struct EdgeProfile {
        var positions: [(y: CGFloat, x: CGFloat)]  // Sorted by y

        var isEmpty: Bool { positions.isEmpty }

        func xAt(y: CGFloat) -> CGFloat? {
            // Linear interpolation
            guard !positions.isEmpty else { return nil }

            if positions.count == 1 {
                return positions[0].x
            }

            // Find bracketing positions
            var below: (y: CGFloat, x: CGFloat)?
            var above: (y: CGFloat, x: CGFloat)?

            for pos in positions {
                if pos.y <= y {
                    below = pos
                } else if above == nil {
                    above = pos
                }
            }

            if let b = below, let a = above {
                let t = (y - b.y) / (a.y - b.y)
                return b.x + t * (a.x - b.x)
            } else if let b = below {
                return b.x
            } else if let a = above {
                return a.x
            }

            return nil
        }
    }

    /// Which side of the glyph to analyze
    private enum EdgeSide {
        case left   // Find minimum x (leftmost points)
        case right  // Find maximum x (rightmost points)
    }

    /// Analyzes the edge profile of a glyph outline by sampling at various heights.
    /// - Parameters:
    ///   - outline: The glyph outline to analyze
    ///   - side: Which edge to analyze (left finds min x, right finds max x)
    ///   - sampleCount: Number of vertical samples to take
    /// - Returns: Edge profile with (y, x) positions sorted by y
    private func analyzeEdge(
        of outline: GlyphOutline,
        side: EdgeSide,
        sampleCount: Int = KerningParams.edgeSampleCount
    ) -> EdgeProfile {
        var positions: [(y: CGFloat, x: CGFloat)] = []
        let bounds = outline.boundingBox

        guard bounds.height > 0 else {
            return EdgeProfile(positions: [])
        }

        guard sampleCount > 1 else {
            return EdgeProfile(positions: [])
        }

        for i in 0..<sampleCount {
            let y = CGFloat(bounds.minY) + CGFloat(i) / CGFloat(sampleCount - 1) * CGFloat(bounds.height)

            let yTolerance = CGFloat(bounds.height) / CGFloat(sampleCount) * 1.5
            // Initialize to opposite extreme so any real match overrides via min/max.
            // When no points match, this value correctly indicates the glyph recedes
            // at this scanline, producing a larger visual gap for kerning analysis.
            var extremeX: CGFloat = side == .left ? CGFloat(bounds.maxX) : CGFloat(bounds.minX)

            for contour in outline.contours {
                for point in contour.points {
                    if abs(point.position.y - y) < yTolerance {
                        switch side {
                        case .left:
                            extremeX = min(extremeX, point.position.x)
                        case .right:
                            extremeX = max(extremeX, point.position.x)
                        }
                    }
                }
            }

            positions.append((y: y, x: extremeX))
        }

        return EdgeProfile(positions: positions.sorted { $0.y < $1.y })
    }

    // Convenience wrappers for clarity at call sites
    private func analyzeRightEdge(of outline: GlyphOutline) -> EdgeProfile {
        analyzeEdge(of: outline, side: .right)
    }

    private func analyzeLeftEdge(of outline: GlyphOutline) -> EdgeProfile {
        analyzeEdge(of: outline, side: .left)
    }

    private func calculateMinimumGap(
        leftEdge: EdgeProfile,
        rightEdge: EdgeProfile,
        leftGlyph: Glyph,
        rightGlyph: Glyph
    ) -> CGFloat {
        guard !leftEdge.isEmpty && !rightEdge.isEmpty else {
            return CGFloat(leftGlyph.advanceWidth - leftGlyph.leftSideBearing)
        }

        var minGap: CGFloat = .infinity

        // Sample at various y positions
        let sampleCount = KerningParams.edgeSampleCount
        let leftBounds = leftGlyph.outline.boundingBox
        let rightBounds = rightGlyph.outline.boundingBox

        let minY = max(CGFloat(leftBounds.minY), CGFloat(rightBounds.minY))
        let maxY = min(CGFloat(leftBounds.maxY), CGFloat(rightBounds.maxY))

        guard maxY > minY else {
            return CGFloat(leftGlyph.advanceWidth)
        }

        for i in 0..<sampleCount {
            let y = minY + CGFloat(i) / CGFloat(sampleCount - 1) * (maxY - minY)

            if let leftX = leftEdge.xAt(y: y),
               let rightX = rightEdge.xAt(y: y) {
                // Gap is: advance width - right edge of left + left edge of right
                let gap = CGFloat(leftGlyph.advanceWidth) - leftX + (rightX - CGFloat(rightBounds.minX))
                minGap = min(minGap, gap)
            }
        }

        return minGap.isInfinite ? CGFloat(leftGlyph.advanceWidth) : minGap
    }

    private func renderPair(
        leftGlyph: Glyph,
        rightGlyph: Glyph,
        metrics: FontMetrics,
        spacing: Int
    ) -> CGImage? {
        let size = RenderParams.imageSize
        // Guard against division by zero
        let safeUnitsPerEm = max(CGFloat(metrics.unitsPerEm), 1)
        let scale = CGFloat(size) / safeUnitsPerEm * RenderParams.scaleFactor

        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        // White background
        context.setFillColor(CGColor.white)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))

        // Set up transform
        let baseline = CGFloat(size) * RenderParams.baselineRatio
        context.translateBy(x: 10, y: baseline)
        context.scaleBy(x: scale, y: -scale)

        // Draw left glyph
        context.setFillColor(CGColor.black)
        context.addPath(leftGlyph.outline.cgPath)
        context.fillPath()

        // Move to right glyph position
        context.translateBy(x: CGFloat(leftGlyph.advanceWidth + spacing), y: 0)

        // Draw right glyph
        context.addPath(rightGlyph.outline.cgPath)
        context.fillPath()

        return context.makeImage()
    }
}
