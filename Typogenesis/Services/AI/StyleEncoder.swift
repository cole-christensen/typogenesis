import Foundation
import CoreML
import CoreGraphics
import AppKit

/// Service for extracting style features from fonts and glyphs
final class StyleEncoder {

    enum StyleEncoderError: Error, LocalizedError {
        case modelNotLoaded
        case invalidInput
        case encodingFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "Style encoder model is not loaded"
            case .invalidInput:
                return "Invalid input for style encoding"
            case .encodingFailed(let reason):
                return "Style encoding failed: \(reason)"
            }
        }
    }

    /// Extracted style features from a font or glyph
    struct FontStyle: Codable, Equatable {
        // Measured properties
        var strokeWeight: Float       // Average stroke thickness (0-1 normalized)
        var strokeContrast: Float     // Variation in stroke thickness (0-1)
        var xHeightRatio: Float       // x-height / cap-height ratio
        var widthRatio: Float         // Average glyph width / height
        var slant: Float              // Italicization angle in degrees

        // Classified properties
        var serifStyle: SerifStyle
        var roundness: Float          // Geometric vs organic curves (0-1)
        var regularity: Float         // Consistency of strokes (0-1)

        // Learned embedding (for ML models)
        var embedding: [Float]

        static let `default` = FontStyle(
            strokeWeight: 0.5,
            strokeContrast: 0.3,
            xHeightRatio: 0.7,
            widthRatio: 0.8,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.5,
            regularity: 0.8,
            embedding: []
        )
    }

    enum SerifStyle: String, Codable, CaseIterable {
        case sansSerif = "Sans Serif"
        case oldStyle = "Old Style"
        case transitional = "Transitional"
        case modern = "Modern"
        case slab = "Slab Serif"
        case script = "Script"
        case decorative = "Decorative"
    }

    // MARK: - Private Properties

    // Representative characters for style analysis
    private let representativeChars: [Character] = [
        "n", "o", "H", "O", "a", "g", "e", "p", "d", "b"
    ]

    // MARK: - Public API

    /// Extract style from a font project
    func extractStyle(from project: FontProject) async throws -> FontStyle {
        // Analyze geometric properties
        let strokeWeight = analyzeStrokeWeight(project)
        let strokeContrast = analyzeStrokeContrast(project)
        let xHeightRatio = Float(project.metrics.xHeight) / Float(project.metrics.capHeight)
        let widthRatio = analyzeWidthRatio(project)
        let slant = measureSlant(project)
        let serifStyle = classifySerifStyle(project)
        let roundness = measureRoundness(project)
        let regularity = measureRegularity(project)

        // Get ML embedding if model is available
        var embedding: [Float] = []
        let hasModel = await MainActor.run { ModelManager.shared.styleEncoder != nil }
        if hasModel {
            embedding = try await encodeWithModel(project)
        }

        return FontStyle(
            strokeWeight: strokeWeight,
            strokeContrast: strokeContrast,
            xHeightRatio: xHeightRatio,
            widthRatio: widthRatio,
            slant: slant,
            serifStyle: serifStyle,
            roundness: roundness,
            regularity: regularity,
            embedding: embedding
        )
    }

    /// Encode a single glyph to a style embedding
    func encodeGlyph(_ glyph: Glyph) async throws -> [Float] {
        let hasModel = await MainActor.run { ModelManager.shared.styleEncoder != nil }
        guard hasModel else {
            throw StyleEncoderError.modelNotLoaded
        }

        // Render glyph to image
        guard let glyphImage = renderGlyphForEncoding(glyph) else {
            throw StyleEncoderError.invalidInput
        }

        // Run through model
        return try await encodeImage(glyphImage)
    }

    /// Compute similarity between two styles (0-1, 1 = identical)
    func similarity(_ styleA: FontStyle, _ styleB: FontStyle) -> Float {
        var totalSimilarity: Float = 0
        var weights: Float = 0

        // Geometric properties
        totalSimilarity += (1 - abs(styleA.strokeWeight - styleB.strokeWeight)) * 2
        weights += 2
        totalSimilarity += (1 - abs(styleA.strokeContrast - styleB.strokeContrast))
        weights += 1
        totalSimilarity += (1 - abs(styleA.xHeightRatio - styleB.xHeightRatio))
        weights += 1
        totalSimilarity += (1 - abs(styleA.widthRatio - styleB.widthRatio))
        weights += 1
        totalSimilarity += (1 - abs(styleA.slant - styleB.slant) / 45)  // Normalize by max reasonable slant
        weights += 1
        totalSimilarity += (1 - abs(styleA.roundness - styleB.roundness))
        weights += 1

        // Serif style (binary match)
        totalSimilarity += (styleA.serifStyle == styleB.serifStyle) ? 2 : 0
        weights += 2

        // Embedding similarity (if both have embeddings)
        if !styleA.embedding.isEmpty && !styleB.embedding.isEmpty &&
           styleA.embedding.count == styleB.embedding.count {
            let cosineSim = cosineSimilarity(styleA.embedding, styleB.embedding)
            totalSimilarity += cosineSim * 3  // Weight embeddings heavily
            weights += 3
        }

        return totalSimilarity / weights
    }

    /// Interpolate between two styles
    func interpolate(_ styleA: FontStyle, _ styleB: FontStyle, t: Float) -> FontStyle {
        let clampedT = max(0, min(1, t))

        return FontStyle(
            strokeWeight: lerp(styleA.strokeWeight, styleB.strokeWeight, clampedT),
            strokeContrast: lerp(styleA.strokeContrast, styleB.strokeContrast, clampedT),
            xHeightRatio: lerp(styleA.xHeightRatio, styleB.xHeightRatio, clampedT),
            widthRatio: lerp(styleA.widthRatio, styleB.widthRatio, clampedT),
            slant: lerp(styleA.slant, styleB.slant, clampedT),
            serifStyle: clampedT < 0.5 ? styleA.serifStyle : styleB.serifStyle,
            roundness: lerp(styleA.roundness, styleB.roundness, clampedT),
            regularity: lerp(styleA.regularity, styleB.regularity, clampedT),
            embedding: interpolateEmbedding(styleA.embedding, styleB.embedding, clampedT)
        )
    }

    // MARK: - Geometric Analysis

    private func analyzeStrokeWeight(_ project: FontProject) -> Float {
        var totalWeight: Float = 0
        var count: Float = 0

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            let weight = estimateStrokeWeight(glyph.outline)
            totalWeight += weight
            count += 1
        }

        return count > 0 ? totalWeight / count : 0.5
    }

    private func analyzeStrokeContrast(_ project: FontProject) -> Float {
        var weights: [Float] = []

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            weights.append(estimateStrokeWeight(glyph.outline))
        }

        guard weights.count >= 2 else { return 0 }

        let minWeight = weights.min() ?? 0
        let maxWeight = weights.max() ?? 1

        return maxWeight > minWeight ? (maxWeight - minWeight) / maxWeight : 0
    }

    private func analyzeWidthRatio(_ project: FontProject) -> Float {
        var totalRatio: Float = 0
        var count: Float = 0

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            let bounds = glyph.outline.boundingBox
            if bounds.height > 0 {
                totalRatio += Float(bounds.width) / Float(bounds.height)
                count += 1
            }
        }

        return count > 0 ? min(2, totalRatio / count) / 2 : 0.5  // Normalize to 0-1
    }

    private func measureSlant(_ project: FontProject) -> Float {
        // Measure average slant angle from vertical strokes
        // For now, return 0 (upright)
        // A full implementation would analyze vertical stroke angles
        return 0
    }

    private func classifySerifStyle(_ project: FontProject) -> SerifStyle {
        // Analyze serif presence and style
        // For now, do simple heuristic based on terminal shapes

        guard let glyph = project.glyphs["n"] ?? project.glyphs["H"] else {
            return .sansSerif
        }

        // Check for serif-like features (endpoints with perpendicular extensions)
        // This is a simplified heuristic
        let hasSerifs = detectSerifs(in: glyph.outline)

        if hasSerifs {
            // Further classify serif type based on shape
            return .transitional  // Default serif type
        }

        return .sansSerif
    }

    private func measureRoundness(_ project: FontProject) -> Float {
        // Measure ratio of curves to corners
        var totalCurves = 0
        var totalCorners = 0

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            for contour in glyph.outline.contours {
                for point in contour.points {
                    switch point.type {
                    case .corner:
                        totalCorners += 1
                    case .smooth, .symmetric:
                        totalCurves += 1
                    }
                }
            }
        }

        let total = totalCurves + totalCorners
        return total > 0 ? Float(totalCurves) / Float(total) : 0.5
    }

    private func measureRegularity(_ project: FontProject) -> Float {
        // Measure consistency of glyph proportions
        // Higher regularity = more consistent width/height ratios
        var ratios: [Float] = []

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            let bounds = glyph.outline.boundingBox
            if bounds.height > 0 {
                ratios.append(Float(bounds.width) / Float(bounds.height))
            }
        }

        guard ratios.count >= 2 else { return 0.5 }

        // Calculate standard deviation
        let mean = ratios.reduce(0, +) / Float(ratios.count)
        let variance = ratios.map { pow($0 - mean, 2) }.reduce(0, +) / Float(ratios.count)
        let stdDev = sqrt(variance)

        // Convert to regularity score (lower stdDev = higher regularity)
        return max(0, min(1, 1 - stdDev))
    }

    // MARK: - Helper Methods

    private func estimateStrokeWeight(_ outline: GlyphOutline) -> Float {
        // Estimate stroke weight from outline area / perimeter ratio
        // This is a rough heuristic
        let bounds = outline.boundingBox
        guard bounds.width > 0 && bounds.height > 0 else { return 0.5 }

        // Larger bounding box relative to point count suggests thicker strokes
        let pointCount = outline.contours.reduce(0) { $0 + $1.points.count }
        guard pointCount > 0 else { return 0.5 }

        let area = Float(bounds.width * bounds.height)
        let density = Float(pointCount) / area * 1000  // Normalize

        return max(0, min(1, 1 - density / 10))  // Invert: more points = finer detail = thinner
    }

    private func detectSerifs(in outline: GlyphOutline) -> Bool {
        // Simplified serif detection
        // Look for terminal points with handles roughly perpendicular to stroke direction

        for contour in outline.contours where contour.isClosed {
            let points = contour.points
            guard points.count >= 3 else { continue }

            for i in 0..<points.count {
                let point = points[i]
                guard point.type == .corner else { continue }

                // Check for perpendicular handles (serif indicator)
                if let controlIn = point.controlIn,
                   let controlOut = point.controlOut {
                    let inAngle = atan2(point.position.y - controlIn.y, point.position.x - controlIn.x)
                    let outAngle = atan2(controlOut.y - point.position.y, controlOut.x - point.position.x)
                    let angleDiff = abs(inAngle - outAngle)

                    // Near perpendicular suggests serif
                    if angleDiff > .pi * 0.4 && angleDiff < .pi * 0.6 {
                        return true
                    }
                }
            }
        }

        return false
    }

    private func renderGlyphForEncoding(_ glyph: Glyph) -> CGImage? {
        // Render glyph to a small image for model input
        let size = 64

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

        // Draw glyph outline
        let bounds = glyph.outline.boundingBox
        guard bounds.width > 0 && bounds.height > 0 else {
            return context.makeImage()
        }

        let scaleX = CGFloat(size - 8) / CGFloat(bounds.width)
        let scaleY = CGFloat(size - 8) / CGFloat(bounds.height)
        let glyphScale = min(scaleX, scaleY)

        context.translateBy(x: 4, y: 4)
        context.scaleBy(x: glyphScale, y: glyphScale)
        context.translateBy(x: CGFloat(-bounds.minX), y: CGFloat(-bounds.minY))

        // Draw filled path
        context.setFillColor(CGColor.black)
        let path = glyph.outline.cgPath
        context.addPath(path)
        context.fillPath()

        return context.makeImage()
    }

    private func encodeWithModel(_ project: FontProject) async throws -> [Float] {
        // Encode representative glyphs and average embeddings
        var embeddings: [[Float]] = []

        for char in representativeChars {
            guard let glyph = project.glyphs[char] else { continue }
            guard let image = renderGlyphForEncoding(glyph) else { continue }

            if let embedding = try? await encodeImage(image) {
                embeddings.append(embedding)
            }
        }

        guard !embeddings.isEmpty else {
            return []
        }

        // Average all embeddings
        return averageEmbeddings(embeddings)
    }

    private func encodeImage(_ image: CGImage) async throws -> [Float] {
        // TODO: Implement actual model inference when models are available
        // For now, return a placeholder embedding

        // Placeholder: generate deterministic "embedding" based on image statistics
        guard let data = image.dataProvider?.data as Data? else {
            return Array(repeating: 0, count: 128)
        }

        var embedding = [Float](repeating: 0, count: 128)
        for (i, byte) in data.prefix(128).enumerated() {
            embedding[i] = Float(byte) / 255.0
        }

        return embedding
    }

    private func averageEmbeddings(_ embeddings: [[Float]]) -> [Float] {
        guard let first = embeddings.first else { return [] }

        var result = [Float](repeating: 0, count: first.count)

        for embedding in embeddings {
            for i in 0..<min(result.count, embedding.count) {
                result[i] += embedding[i]
            }
        }

        let count = Float(embeddings.count)
        return result.map { $0 / count }
    }

    private func interpolateEmbedding(_ a: [Float], _ b: [Float], _ t: Float) -> [Float] {
        guard a.count == b.count else {
            return a.isEmpty ? b : a
        }

        return zip(a, b).map { lerp($0, $1, t) }
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }

        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }

    private func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
        a + (b - a) * t
    }
}
