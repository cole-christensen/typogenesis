import Foundation
import CoreML
import CoreGraphics
import AppKit

/// Service for predicting optimal kerning values using AI
final class KerningPredictor {

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

        var kerningPairs: [KerningPair] = []

        let hasModel = await MainActor.run { ModelManager.shared.kerningNet != nil }
        if hasModel {
            // Use ML model for prediction
            for (left, right) in pairs {
                if let kerning = try await predictPairWithModel(
                    left: left,
                    right: right,
                    project: project
                ) {
                    if abs(kerning) >= settings.minKerningValue {
                        kerningPairs.append(KerningPair(left: left, right: right, value: kerning))
                    }
                }
            }
        } else {
            // Use geometric heuristics
            for (left, right) in pairs {
                let kerning = calculateGeometricKerning(
                    left: left,
                    right: right,
                    project: project,
                    settings: settings
                )

                if abs(kerning) >= settings.minKerningValue {
                    kerningPairs.append(KerningPair(left: left, right: right, value: kerning))
                }
            }
        }

        let predictionTime = Date().timeIntervalSince(startTime)

        return PredictionResult(
            pairs: kerningPairs,
            predictionTime: predictionTime,
            confidence: hasModel ? 0.85 : 0.6
        )
    }

    /// Predict kerning for a single pair
    func predictPair(
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
        var pairs: [(Character, Character)] = []
        let characters = Array(project.glyphs.keys)

        for left in characters {
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
            }
        }

        return pairs
    }

    private func predictPairWithModel(
        left: Character,
        right: Character,
        project: FontProject
    ) async throws -> Int? {
        guard let leftGlyph = project.glyphs[left],
              let rightGlyph = project.glyphs[right] else {
            return nil
        }

        // Render pair to image (for model input when available)
        guard renderPair(
            leftGlyph: leftGlyph,
            rightGlyph: rightGlyph,
            metrics: project.metrics,
            spacing: 0
        ) != nil else {
            return nil
        }

        // TODO: Run through model when available
        // For now, fall back to geometric calculation
        return calculateGeometricKerning(
            left: left,
            right: right,
            project: project,
            settings: .default
        )
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
        let leftRightEdge = analyzeRightEdge(of: leftGlyph.outline, metrics: project.metrics)

        // Analyze left side of right glyph
        let rightLeftEdge = analyzeLeftEdge(of: rightGlyph.outline, metrics: project.metrics)

        // Calculate optimal spacing adjustment
        let baseSpacing = CGFloat(project.metrics.unitsPerEm) * 0.1  // 10% of em
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
        let maxKerning = CGFloat(project.metrics.unitsPerEm) / 4
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

    private func analyzeRightEdge(of outline: GlyphOutline, metrics: FontMetrics) -> EdgeProfile {
        var positions: [(y: CGFloat, x: CGFloat)] = []

        // Sample at various heights
        let sampleCount = 20
        let bounds = outline.boundingBox

        for i in 0..<sampleCount {
            let y = CGFloat(bounds.minY) + CGFloat(i) / CGFloat(sampleCount - 1) * CGFloat(bounds.height)

            // Find rightmost x at this y
            var maxX: CGFloat = CGFloat(bounds.minX)

            for contour in outline.contours {
                for point in contour.points {
                    if abs(point.position.y - y) < CGFloat(bounds.height) / CGFloat(sampleCount) * 1.5 {
                        maxX = max(maxX, point.position.x)
                    }
                }
            }

            positions.append((y: y, x: maxX))
        }

        return EdgeProfile(positions: positions.sorted { $0.y < $1.y })
    }

    private func analyzeLeftEdge(of outline: GlyphOutline, metrics: FontMetrics) -> EdgeProfile {
        var positions: [(y: CGFloat, x: CGFloat)] = []

        let sampleCount = 20
        let bounds = outline.boundingBox

        for i in 0..<sampleCount {
            let y = CGFloat(bounds.minY) + CGFloat(i) / CGFloat(sampleCount - 1) * CGFloat(bounds.height)

            // Find leftmost x at this y
            var minX: CGFloat = CGFloat(bounds.maxX)

            for contour in outline.contours {
                for point in contour.points {
                    if abs(point.position.y - y) < CGFloat(bounds.height) / CGFloat(sampleCount) * 1.5 {
                        minX = min(minX, point.position.x)
                    }
                }
            }

            positions.append((y: y, x: minX))
        }

        return EdgeProfile(positions: positions.sorted { $0.y < $1.y })
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
        let sampleCount = 20
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
        let size = 128
        let scale = CGFloat(size) / CGFloat(metrics.unitsPerEm) * 0.8

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
        let baseline = CGFloat(size) * 0.7
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
