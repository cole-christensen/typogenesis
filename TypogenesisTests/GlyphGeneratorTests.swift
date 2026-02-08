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

    @Test("GlyphGenerator isAvailable returns true (placeholder fallback always works)")
    @MainActor
    func testGeneratorIsAvailableWithoutModel() {
        // Without AI models loaded, isAvailable should still return true
        // because the placeholder/template fallback is always ready.
        // This test verifies the fallback design: isAvailable == true even without CoreML models.
        let generator = GlyphGenerator()
        #expect(generator.isAvailable, "isAvailable should be true even without CoreML models (placeholder fallback)")

        // Verify model is NOT actually loaded (the fallback is what makes it available)
        let modelAvailable = GlyphGenerator.isModelAvailable()
        #expect(!modelAvailable, "CoreML model should NOT be available in test environment")
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

    @Test("Generate returns result with valid confidence value")
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

        // Confidence must always be a valid probability between 0 and 1
        #expect(result.confidence >= 0.0, "Confidence must be >= 0.0, got \(result.confidence)")
        #expect(result.confidence <= 1.0, "Confidence must be <= 1.0, got \(result.confidence)")

        // TODO: When real AI models are available, confidence should be > 0 for quality generations.
        // Currently using placeholder generation which correctly reports 0.0 confidence.
        // When models are integrated, add a test that verifies aiGenerated glyphs have confidence > 0.5
    }

    // MARK: - REAL BEHAVIOR TESTS (Commandment VII: Test Real Behavior)

    @Test("Generated glyph has proper renderable content")
    func testGeneratedGlyphHasVisibleContent() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        // The outline must not be empty
        #expect(!result.glyph.outline.isEmpty, "Generated glyph must have visible content, not an empty outline")
        #expect(result.glyph.outline.contours.count >= 1, "Generated glyph must have at least one contour")

        // Each contour must have enough points to form a visible shape
        for (index, contour) in result.glyph.outline.contours.enumerated() {
            #expect(contour.points.count >= 3, "Contour \(index) must have at least 3 points to form a visible shape, got \(contour.points.count)")

            // Contours must be closed for proper glyph rendering (fonts require closed paths)
            #expect(contour.isClosed, "Contour \(index) must be closed for proper font rendering")
        }
    }

    @Test("Generated glyph has typographically reasonable bounds")
    func testGeneratedGlyphHasReasonableBounds() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "H",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        let bounds = result.glyph.outline.boundingBox

        // Bounds must be non-zero and reasonable
        #expect(bounds.width > 100, "Generated glyph must have meaningful width, got \(bounds.width)")
        #expect(bounds.height > 100, "Generated glyph must have meaningful height, got \(bounds.height)")
        #expect(bounds.width < metrics.unitsPerEm, "Width should be less than full em")

        // Height should respect font metrics for uppercase letters
        // "H" is uppercase, so height should be close to cap height (700) with some tolerance
        let expectedMinHeight = Int(Double(metrics.capHeight) * 0.7)  // At least 70% of cap height
        let expectedMaxHeight = metrics.capHeight + 50  // Allow small overshoot
        #expect(bounds.height >= expectedMinHeight,
                "Uppercase 'H' height (\(bounds.height)) should be at least 70% of capHeight (\(metrics.capHeight))")
        #expect(bounds.height <= expectedMaxHeight,
                "Uppercase 'H' height (\(bounds.height)) should not significantly exceed capHeight (\(metrics.capHeight))")

        // Width should be reasonable for character 'H' (typically 60-90% of height for this letter)
        let minReasonableWidth = Int(Double(bounds.height) * 0.4)  // At least 40% of height
        let maxReasonableWidth = Int(Double(bounds.height) * 1.5)  // At most 150% of height
        #expect(bounds.width >= minReasonableWidth,
                "Width (\(bounds.width)) should be at least 40% of height (\(bounds.height)) for 'H'")
        #expect(bounds.width <= maxReasonableWidth,
                "Width (\(bounds.width)) should not exceed 150% of height (\(bounds.height)) for 'H'")
    }

    @Test("Placeholder generation is marked as placeholder source")
    func testPlaceholderMarkedAsPlaceholder() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "X",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        // Placeholder glyphs must be honestly marked so users know they're not AI-generated
        #expect(result.glyph.generatedBy == .placeholder, "Placeholder glyphs must be marked as .placeholder, not hidden")

        // TODO: When real AI models are integrated, add a parallel test:
        // testAIGeneratedMarkedAsAIGenerated() that verifies real AI output is marked as .aiGenerated
        // and has confidence > 0. This ensures we distinguish between placeholder and real AI output.
    }

    @Test("Different characters produce different placeholder widths")
    func testDifferentCharactersProduceDifferentWidths() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let resultA = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        let resultM = try await generator.generate(
            character: "M",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        // Characters should have slightly different widths (variation based on unicode value)
        // This prevents all placeholders from being identical rectangles
        let boundsA = resultA.glyph.outline.boundingBox
        let boundsM = resultM.glyph.outline.boundingBox

        #expect(boundsA.width != boundsM.width || boundsA.height != boundsM.height,
                "Different characters should produce visually distinct placeholders")
    }

    @Test("Uppercase and lowercase have different heights")
    func testUppercaseAndLowercaseDifferentHeights() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let uppercase = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        let lowercase = try await generator.generate(
            character: "a",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        let upperBounds = uppercase.glyph.outline.boundingBox
        let lowerBounds = lowercase.glyph.outline.boundingBox

        // Uppercase should be taller (cap height vs x-height)
        #expect(upperBounds.height > lowerBounds.height,
                "Uppercase 'A' should be taller than lowercase 'a'. Upper: \(upperBounds.height), Lower: \(lowerBounds.height)")
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

        // Use a simple array since the @Sendable callback records synchronously
        // and generateBatch processes sequentially
        let progressCounter = ProgressCounter()

        _ = try await generator.generateBatch(
            characters: characters,
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        ) { current, total in
            Task { await progressCounter.record(current: current, total: total) }
        }

        // Wait for the fire-and-forget tasks to complete.
        // Use a retry loop instead of a fixed sleep to avoid flakiness.
        var progressCalls: [(current: Int, total: Int)] = []
        for _ in 0..<20 {
            progressCalls = await progressCounter.getUpdates()
            if progressCalls.count == 3 { break }
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(progressCalls.count == 3, "Expected 3 progress calls, got \(progressCalls.count)")
        if progressCalls.count == 3 {
            #expect(progressCalls[0] == (1, 3))
            #expect(progressCalls[1] == (2, 3))
            #expect(progressCalls[2] == (3, 3))
        }
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

// MARK: - Thread-safe Progress Counter

/// Actor for safely recording progress updates from @Sendable callbacks
private actor ProgressCounter {
    private var updates: [(current: Int, total: Int)] = []

    func record(current: Int, total: Int) {
        updates.append((current, total))
    }

    func getUpdates() -> [(current: Int, total: Int)] {
        updates
    }
}
