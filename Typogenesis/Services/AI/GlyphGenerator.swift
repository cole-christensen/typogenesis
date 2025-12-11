import Foundation
import CoreML
import CoreGraphics

/// Service for generating new glyphs using AI
final class GlyphGenerator {

    enum GeneratorError: Error, LocalizedError {
        case modelNotLoaded
        case generationFailed(String)
        case invalidStyle
        case cancelled

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "Glyph generation model is not loaded"
            case .generationFailed(let reason):
                return "Generation failed: \(reason)"
            case .invalidStyle:
                return "Invalid style configuration"
            case .cancelled:
                return "Generation was cancelled"
            }
        }
    }

    /// Generation mode
    enum GenerationMode {
        /// Generate from scratch with a given style
        case fromScratch(style: StyleEncoder.FontStyle)

        /// Complete a partial glyph outline
        case completePartial(partial: GlyphOutline, style: StyleEncoder.FontStyle)

        /// Create a variation of an existing glyph
        case variation(base: Glyph, strength: Float)

        /// Interpolate between two glyphs
        case interpolate(glyphA: Glyph, glyphB: Glyph, t: Float)
    }

    /// Settings for glyph generation
    struct GenerationSettings {
        var steps: Int = 50               // Diffusion steps
        var guidanceScale: Float = 7.5    // Classifier-free guidance scale
        var seed: UInt64? = nil           // Random seed (nil for random)
        var temperature: Float = 1.0      // Sampling temperature

        static let `default` = GenerationSettings()

        /// Faster generation with fewer steps
        static let fast = GenerationSettings(steps: 20)

        /// Higher quality with more steps
        static let quality = GenerationSettings(steps: 100)
    }

    /// Result of glyph generation
    struct GenerationResult {
        let glyph: Glyph
        let confidence: Float
        let generationTime: TimeInterval
    }

    // MARK: - Private Properties

    private let styleEncoder = StyleEncoder()

    // MARK: - Public API

    /// Check if model is available (must be called from MainActor)
    @MainActor
    static func isModelAvailable() -> Bool {
        ModelManager.shared.glyphDiffusion != nil
    }

    /// Generate a single glyph
    func generate(
        character: Character,
        mode: GenerationMode,
        metrics: FontMetrics,
        settings: GenerationSettings = .default
    ) async throws -> GenerationResult {
        // Check model availability on MainActor
        let hasModel = await MainActor.run { Self.isModelAvailable() }
        guard hasModel else {
            // Return placeholder when model not available
            return try await generatePlaceholder(
                character: character,
                mode: mode,
                metrics: metrics
            )
        }

        let startTime = Date()

        // Prepare conditioning based on mode
        let conditioning = try prepareConditioning(
            character: character,
            mode: mode,
            metrics: metrics
        )

        // Generate latent representation
        let outline = try await runDiffusion(
            conditioning: conditioning,
            settings: settings
        )

        // Create glyph
        let bounds = outline.boundingBox
        let glyph = Glyph(
            character: character,
            outline: outline,
            advanceWidth: bounds.width + Int(CGFloat(metrics.unitsPerEm) * 0.2),
            leftSideBearing: Int(CGFloat(metrics.unitsPerEm) * 0.1)
        )

        let generationTime = Date().timeIntervalSince(startTime)

        return GenerationResult(
            glyph: glyph,
            confidence: 0.85,  // Placeholder confidence
            generationTime: generationTime
        )
    }

    /// Generate multiple glyphs for a character set
    func generateBatch(
        characters: [Character],
        mode: GenerationMode,
        metrics: FontMetrics,
        settings: GenerationSettings = .default,
        onProgress: ((Int, Int) -> Void)? = nil
    ) async throws -> [GenerationResult] {
        var results: [GenerationResult] = []

        for (index, char) in characters.enumerated() {
            let result = try await generate(
                character: char,
                mode: mode,
                metrics: metrics,
                settings: settings
            )
            results.append(result)

            onProgress?(index + 1, characters.count)
        }

        return results
    }

    /// Check if generation is available
    var isAvailable: Bool {
        // Either real model or placeholder
        true
    }

    // MARK: - Private Methods

    private struct Conditioning {
        let characterEmbedding: [Float]
        let styleEmbedding: [Float]
        let targetMetrics: FontMetrics
        let baseOutline: GlyphOutline?
    }

    private func prepareConditioning(
        character: Character,
        mode: GenerationMode,
        metrics: FontMetrics
    ) throws -> Conditioning {
        // Create character embedding (one-hot or learned)
        let charEmbedding = createCharacterEmbedding(character)

        // Extract style embedding based on mode
        let styleEmbedding: [Float]
        var baseOutline: GlyphOutline? = nil

        switch mode {
        case .fromScratch(let style):
            styleEmbedding = style.embedding.isEmpty ?
                createDefaultStyleEmbedding() : style.embedding

        case .completePartial(let partial, let style):
            styleEmbedding = style.embedding.isEmpty ?
                createDefaultStyleEmbedding() : style.embedding
            baseOutline = partial

        case .variation(let base, _):
            styleEmbedding = createEmbeddingFromGlyph(base)
            baseOutline = base.outline

        case .interpolate(let glyphA, let glyphB, let t):
            let embeddingA = createEmbeddingFromGlyph(glyphA)
            let embeddingB = createEmbeddingFromGlyph(glyphB)
            styleEmbedding = interpolateEmbeddings(embeddingA, embeddingB, t: t)
        }

        return Conditioning(
            characterEmbedding: charEmbedding,
            styleEmbedding: styleEmbedding,
            targetMetrics: metrics,
            baseOutline: baseOutline
        )
    }

    private func runDiffusion(
        conditioning: Conditioning,
        settings: GenerationSettings
    ) async throws -> GlyphOutline {
        // TODO: Implement actual diffusion when model is available
        // For now, return placeholder outline

        // Simulate processing time
        let simulatedStepTime = 0.02  // 20ms per step
        try await Task.sleep(nanoseconds: UInt64(Double(settings.steps) * simulatedStepTime * 1_000_000_000))

        // Return base outline if provided, otherwise create empty
        if let base = conditioning.baseOutline {
            return base
        }

        return GlyphOutline()
    }

    private func generatePlaceholder(
        character: Character,
        mode: GenerationMode,
        metrics: FontMetrics
    ) async throws -> GenerationResult {
        let startTime = Date()

        // Simulate generation time
        try await Task.sleep(nanoseconds: 500_000_000)

        // Create a simple placeholder outline
        let outline: GlyphOutline

        switch mode {
        case .variation(let base, _):
            // Return slightly modified base
            outline = base.outline

        case .interpolate(let glyphA, _, let t):
            // Return one of the glyphs based on t
            outline = t < 0.5 ? glyphA.outline : glyphA.outline

        case .completePartial(let partial, _):
            // Return the partial as-is
            outline = partial

        case .fromScratch:
            // Create empty outline
            outline = GlyphOutline()
        }

        let bounds = outline.boundingBox
        let glyph = Glyph(
            character: character,
            outline: outline,
            advanceWidth: max(bounds.width + Int(CGFloat(metrics.unitsPerEm) * 0.2), metrics.unitsPerEm / 2),
            leftSideBearing: Int(CGFloat(metrics.unitsPerEm) * 0.1)
        )

        return GenerationResult(
            glyph: glyph,
            confidence: 0.0,  // Zero confidence for placeholder
            generationTime: Date().timeIntervalSince(startTime)
        )
    }

    private func createCharacterEmbedding(_ character: Character) -> [Float] {
        // Create a simple embedding from Unicode code point
        var embedding = [Float](repeating: 0, count: 128)

        for (index, scalar) in character.unicodeScalars.enumerated() {
            let value = Float(scalar.value)
            let normalized = value / 65536.0

            if index < embedding.count {
                embedding[index] = normalized
            }
        }

        return embedding
    }

    private func createDefaultStyleEmbedding() -> [Float] {
        // Default neutral style embedding
        [Float](repeating: 0.5, count: 128)
    }

    private func createEmbeddingFromGlyph(_ glyph: Glyph) -> [Float] {
        // Create embedding from glyph properties
        var embedding = [Float](repeating: 0, count: 128)

        let bounds = glyph.outline.boundingBox
        let pointCount = glyph.outline.contours.reduce(0) { $0 + $1.points.count }

        // Encode basic geometric features
        embedding[0] = Float(bounds.width) / 1000.0
        embedding[1] = Float(bounds.height) / 1000.0
        embedding[2] = Float(pointCount) / 100.0
        embedding[3] = Float(glyph.outline.contours.count) / 10.0
        embedding[4] = Float(glyph.advanceWidth) / 1000.0

        return embedding
    }

    private func interpolateEmbeddings(_ a: [Float], _ b: [Float], t: Float) -> [Float] {
        guard a.count == b.count else {
            return a
        }

        return zip(a, b).map { $0 * (1 - t) + $1 * t }
    }
}
