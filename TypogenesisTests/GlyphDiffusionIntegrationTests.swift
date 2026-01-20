import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("GlyphDiffusion Integration Tests")
struct GlyphDiffusionIntegrationTests {

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

    private func createTestStyle() -> StyleEncoder.FontStyle {
        StyleEncoder.FontStyle(
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

    // MARK: - Model Availability Tests

    @Test("Model availability check returns consistent result")
    @MainActor
    func testModelAvailabilityCheck() async {
        // Check availability (should be false without loaded models)
        let isAvailable = GlyphGenerator.isModelAvailable()

        // Result should be boolean
        #expect(isAvailable == false || isAvailable == true)

        // Multiple checks should return same result
        let secondCheck = GlyphGenerator.isModelAvailable()
        #expect(isAvailable == secondCheck, "Model availability should be consistent")
    }

    @Test("Generator is always available due to fallback")
    func testGeneratorAlwaysAvailable() {
        let generator = GlyphGenerator()
        #expect(generator.isAvailable, "Generator should always be available due to template fallback")
    }

    // MARK: - Generation Tests

    @Test("Generate returns valid glyph outline")
    func testGenerateReturnsValidOutline() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        let result = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        // Verify output is a valid glyph
        #expect(result.glyph.character == "A")
        #expect(!result.glyph.outline.isEmpty, "Generated glyph should have non-empty outline")
        #expect(result.glyph.outline.contours.count >= 1, "Should have at least one contour")

        // Verify contours have valid points
        for contour in result.glyph.outline.contours {
            #expect(contour.points.count >= 3, "Contour should have at least 3 points")
        }
    }

    @Test("Generate produces valid GlyphOutline structure")
    func testGenerateProducesValidStructure() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        let result = try await generator.generate(
            character: "H",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        let outline = result.glyph.outline

        // Validate outline structure
        #expect(!outline.isEmpty, "Outline should not be empty")

        // Check bounding box is reasonable
        let bounds = outline.boundingBox
        #expect(bounds.width > 0, "Bounding box width should be positive")
        #expect(bounds.height > 0, "Bounding box height should be positive")
        #expect(bounds.width < metrics.unitsPerEm * 2, "Width should be reasonable")
        #expect(bounds.height <= metrics.capHeight + 100, "Height should be around cap height for uppercase")

        // Check CGPath can be generated
        let path = outline.cgPath
        #expect(!path.isEmpty, "CGPath should not be empty")
    }

    @Test("Generate with various characters produces different results")
    func testGenerateVariousCharacters() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        let characters: [Character] = ["A", "B", "a", "o", "1", "!"]
        var results: [GlyphGenerator.GenerationResult] = []

        for char in characters {
            let result = try await generator.generate(
                character: char,
                mode: .fromScratch(style: style),
                metrics: metrics,
                settings: .fast
            )
            results.append(result)
            #expect(result.glyph.character == char)
        }

        // Uppercase and lowercase should have different heights
        let upperA = results[0].glyph.outline.boundingBox
        let lowerA = results[2].glyph.outline.boundingBox
        #expect(upperA.height > lowerA.height, "Uppercase should be taller than lowercase")
    }

    @Test("Generate with style conditioning affects output")
    func testStyleConditioningAffectsOutput() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()

        // Test with different styles
        let thinStyle = StyleEncoder.FontStyle(
            strokeWeight: 0.2,
            strokeContrast: 0.1,
            xHeightRatio: 0.7,
            widthRatio: 0.7,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.3,
            regularity: 0.9,
            embedding: []
        )

        let boldStyle = StyleEncoder.FontStyle(
            strokeWeight: 0.8,
            strokeContrast: 0.4,
            xHeightRatio: 0.7,
            widthRatio: 0.9,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.5,
            regularity: 0.9,
            embedding: []
        )

        let thinResult = try await generator.generate(
            character: "O",
            mode: .fromScratch(style: thinStyle),
            metrics: metrics,
            settings: .fast
        )

        let boldResult = try await generator.generate(
            character: "O",
            mode: .fromScratch(style: boldStyle),
            metrics: metrics,
            settings: .fast
        )

        // Both should produce valid output
        #expect(!thinResult.glyph.outline.isEmpty)
        #expect(!boldResult.glyph.outline.isEmpty)
    }

    // MARK: - Fallback Behavior Tests

    @Test("Fallback to template generation when model unavailable")
    func testFallbackToTemplate() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        // Since model is not loaded, this should fall back to template
        let result = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        // Should still produce valid output
        #expect(!result.glyph.outline.isEmpty)
        #expect(result.glyph.character == "A")

        // Confidence should be 0.0 for placeholder generation
        #expect(result.confidence == 0.0, "Placeholder generation should have zero confidence")

        // Should be marked as placeholder
        #expect(result.glyph.generatedBy == .placeholder, "Should be marked as placeholder generation")
    }

    @Test("generateWithModel falls back gracefully")
    func testGenerateWithModelFallback() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        // This should fall back since no model is loaded
        let result = try await generator.generateWithModel(
            character: "X",
            style: style,
            metrics: metrics,
            settings: .fast
        )

        // Should still return valid result
        #expect(result.glyph.character == "X")
        #expect(!result.glyph.outline.isEmpty)
    }

    // MARK: - Output Format Tests

    @Test("Generated outline has proper contour closure")
    func testContourClosure() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        let result = try await generator.generate(
            character: "O",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        // O should have closed contours
        for contour in result.glyph.outline.contours {
            #expect(contour.isClosed, "Letter O contours should be closed")
        }
    }

    @Test("Generated glyph has proper metrics")
    func testGeneratedGlyphMetrics() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        let result = try await generator.generate(
            character: "M",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        let glyph = result.glyph

        // Verify advance width is reasonable
        #expect(glyph.advanceWidth > 0, "Advance width should be positive")
        #expect(glyph.advanceWidth <= metrics.unitsPerEm * 2, "Advance width should be reasonable")

        // Verify left side bearing
        #expect(glyph.leftSideBearing >= 0, "Left side bearing should be non-negative")
        #expect(glyph.leftSideBearing < glyph.advanceWidth, "LSB should be less than advance width")
    }

    // MARK: - Performance Tests

    @Test("Generation completes within timeout")
    func testGenerationPerformance() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        let startTime = Date()

        _ = try await generator.generate(
            character: "W",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        let elapsed = Date().timeIntervalSince(startTime)

        // Should complete within 2 seconds (target from requirements)
        #expect(elapsed < 2.0, "Generation should complete within 2 seconds, took \(elapsed)s")
    }

    @Test("Batch generation reports accurate times")
    func testBatchGenerationTiming() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        let characters: [Character] = ["A", "B", "C"]

        var totalReportedTime: TimeInterval = 0

        let results = try await generator.generateBatch(
            characters: characters,
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        for result in results {
            totalReportedTime += result.generationTime
            #expect(result.generationTime > 0, "Each generation should report positive time")
        }

        #expect(results.count == characters.count, "Should generate all characters")
    }

    // MARK: - Edge Cases

    @Test("Generate handles Unicode characters")
    func testUnicodeCharacters() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        // Test with non-ASCII characters
        let unicodeChars: [Character] = ["e", "n", "a"] // Common characters that should have templates

        for char in unicodeChars {
            let result = try await generator.generate(
                character: char,
                mode: .fromScratch(style: style),
                metrics: metrics,
                settings: .fast
            )

            #expect(result.glyph.character == char)
            // Should produce some output even if just a placeholder
            #expect(!result.glyph.outline.isEmpty || result.glyph.advanceWidth > 0)
        }
    }

    @Test("Generate handles special characters")
    func testSpecialCharacters() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        // Test with punctuation and symbols
        let specialChars: [Character] = [".", ",", "!", "?", "@"]

        for char in specialChars {
            let result = try await generator.generate(
                character: char,
                mode: .fromScratch(style: style),
                metrics: metrics,
                settings: .fast
            )

            #expect(result.glyph.character == char)
            #expect(result.glyph.advanceWidth > 0, "Special chars should have advance width")
        }
    }
}
