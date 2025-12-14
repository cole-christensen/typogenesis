import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("GlyphGenerator Tests")
struct GlyphGeneratorTests {

    // MARK: - Helper Methods

    private func createTestMetrics() -> FontMetrics {
        FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -200,
            xHeight: 500,
            capHeight: 700,
            lineGap: 90
        )
    }

    private func createTestGlyph(character: Character) -> Glyph {
        let outline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                ],
                isClosed: true
            )
        ])

        return Glyph(
            character: character,
            outline: outline,
            advanceWidth: 500,
            leftSideBearing: 50
        )
    }

    // MARK: - GenerationSettings Tests

    @Test("GenerationSettings default values are reasonable")
    func testGenerationSettingsDefaults() {
        let settings = GlyphGenerator.GenerationSettings.default

        #expect(settings.steps == 50)
        #expect(settings.guidanceScale == 7.5)
        #expect(settings.seed == nil)
        #expect(settings.temperature == 1.0)
    }

    @Test("GenerationSettings presets have correct steps")
    func testGenerationSettingsPresets() {
        let fast = GlyphGenerator.GenerationSettings.fast
        let quality = GlyphGenerator.GenerationSettings.quality

        #expect(fast.steps == 20)
        #expect(quality.steps == 100)
        #expect(fast.steps < GlyphGenerator.GenerationSettings.default.steps)
        #expect(quality.steps > GlyphGenerator.GenerationSettings.default.steps)
    }

    // MARK: - GenerationMode Tests

    @Test("GenerationMode fromScratch stores style")
    func testGenerationModeFromScratch() {
        let style = StyleEncoder.FontStyle.default
        let mode = GlyphGenerator.GenerationMode.fromScratch(style: style)

        if case .fromScratch(let storedStyle) = mode {
            #expect(storedStyle == style)
        } else {
            Issue.record("Expected fromScratch mode")
        }
    }

    @Test("GenerationMode variation stores base glyph and strength")
    func testGenerationModeVariation() {
        let glyph = createTestGlyph(character: "A")
        let mode = GlyphGenerator.GenerationMode.variation(base: glyph, strength: 0.5)

        if case .variation(let base, let strength) = mode {
            #expect(base.character == "A")
            #expect(strength == 0.5)
        } else {
            Issue.record("Expected variation mode")
        }
    }

    @Test("GenerationMode interpolate stores both glyphs and t value")
    func testGenerationModeInterpolate() {
        let glyphA = createTestGlyph(character: "A")
        let glyphB = createTestGlyph(character: "B")
        let mode = GlyphGenerator.GenerationMode.interpolate(glyphA: glyphA, glyphB: glyphB, t: 0.3)

        if case .interpolate(let a, let b, let t) = mode {
            #expect(a.character == "A")
            #expect(b.character == "B")
            #expect(t == 0.3)
        } else {
            Issue.record("Expected interpolate mode")
        }
    }

    // MARK: - Generator Availability Tests

    @Test("GlyphGenerator is always available (placeholder fallback)")
    func testGeneratorIsAvailable() {
        let generator = GlyphGenerator()
        #expect(generator.isAvailable == true)
    }

    // MARK: - Generation Tests

    @Test("Generate returns result with correct character")
    func testGenerateReturnsCorrectCharacter() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.glyph.character == "A")
    }

    @Test("Generate returns result with generation time")
    func testGenerateReturnsGenerationTime() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "B",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.generationTime > 0)
    }

    @Test("Generate returns result with confidence")
    func testGenerateReturnsConfidence() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "C",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.confidence >= 0 && result.confidence <= 1)
    }

    @Test("Generate variation mode uses base glyph")
    func testGenerateVariationMode() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let baseGlyph = createTestGlyph(character: "X")

        let result = try await generator.generate(
            character: "X",
            mode: .variation(base: baseGlyph, strength: 0.5),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.glyph.character == "X")
        // In placeholder mode, outline should be same as base
        #expect(result.glyph.outline.contours.count == baseGlyph.outline.contours.count)
    }

    @Test("Generate completePartial mode uses partial outline")
    func testGenerateCompletePartialMode() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let partialOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 100, y: 100), type: .corner),
                    PathPoint(position: CGPoint(x: 200, y: 100), type: .corner)
                ],
                isClosed: false
            )
        ])

        let result = try await generator.generate(
            character: "P",
            mode: .completePartial(partial: partialOutline, style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.glyph.character == "P")
        // In placeholder mode, should return the partial
        #expect(result.glyph.outline.contours.count >= 1)
    }

    @Test("Generate with metrics affects advance width")
    func testGenerateWithMetrics() async throws {
        let generator = GlyphGenerator()
        let style = StyleEncoder.FontStyle.default

        let smallMetrics = FontMetrics(unitsPerEm: 500)
        let largeMetrics = FontMetrics(unitsPerEm: 2000)

        let smallResult = try await generator.generate(
            character: "M",
            mode: .fromScratch(style: style),
            metrics: smallMetrics,
            settings: .fast
        )

        let largeResult = try await generator.generate(
            character: "M",
            mode: .fromScratch(style: style),
            metrics: largeMetrics,
            settings: .fast
        )

        // Advance width should scale with unitsPerEm
        #expect(largeResult.glyph.advanceWidth > smallResult.glyph.advanceWidth)
    }

    // MARK: - Batch Generation Tests

    @Test("GenerateBatch returns results for all characters")
    func testGenerateBatchReturnsAllResults() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default
        let characters: [Character] = ["A", "B", "C"]

        let results = try await generator.generateBatch(
            characters: characters,
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(results.count == 3)
        #expect(results[0].glyph.character == "A")
        #expect(results[1].glyph.character == "B")
        #expect(results[2].glyph.character == "C")
    }

    @Test("GenerateBatch calls progress callback")
    func testGenerateBatchProgress() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default
        let characters: [Character] = ["X", "Y", "Z"]

        var progressCalls: [(Int, Int)] = []

        _ = try await generator.generateBatch(
            characters: characters,
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        ) { current, total in
            progressCalls.append((current, total))
        }

        #expect(progressCalls.count == 3)
        #expect(progressCalls[0] == (1, 3))
        #expect(progressCalls[1] == (2, 3))
        #expect(progressCalls[2] == (3, 3))
    }

    @Test("GenerateBatch with empty array returns empty results")
    func testGenerateBatchEmpty() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let results = try await generator.generateBatch(
            characters: [],
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(results.isEmpty)
    }

    // MARK: - GenerationResult Tests

    @Test("GenerationResult contains valid glyph")
    func testGenerationResultGlyph() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "G",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.glyph.advanceWidth > 0)
        #expect(result.glyph.leftSideBearing >= 0)
    }

    // MARK: - Error Handling Tests

    @Test("GeneratorError has correct descriptions")
    func testGeneratorErrorDescriptions() {
        let modelNotLoaded = GlyphGenerator.GeneratorError.modelNotLoaded
        let generationFailed = GlyphGenerator.GeneratorError.generationFailed("Test reason")
        let invalidStyle = GlyphGenerator.GeneratorError.invalidStyle
        let cancelled = GlyphGenerator.GeneratorError.cancelled

        #expect(modelNotLoaded.errorDescription?.contains("not loaded") == true)
        #expect(generationFailed.errorDescription?.contains("Test reason") == true)
        #expect(invalidStyle.errorDescription?.contains("Invalid") == true)
        #expect(cancelled.errorDescription?.contains("cancelled") == true)
    }
}
