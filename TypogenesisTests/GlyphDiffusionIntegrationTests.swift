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

        // Without loaded CoreML models, model should not be available
        #expect(!isAvailable, "Model should not be available without loaded CoreML models")

        // Multiple checks should return same result
        let secondCheck = GlyphGenerator.isModelAvailable()
        #expect(isAvailable == secondCheck, "Model availability should be consistent")
    }

    @Test("Model is not available without loaded CoreML models")
    @MainActor
    func testModelNotAvailableWithoutModels() async {
        let isAvailable = GlyphGenerator.isModelAvailable()
        #expect(!isAvailable, "Model should not be available without loaded CoreML models")
    }

    // MARK: - Generation Tests

    // testGenerateReturnsValidOutline removed (C21): duplicates GlyphGeneratorTests.testGeneratedGlyphHasVisibleContent

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

        // Different style parameters should produce different outlines
        let thinBounds = thinResult.glyph.outline.boundingBox
        let boldBounds = boldResult.glyph.outline.boundingBox

        // Bold should be wider or have different point count (stroke weight differs)
        let thinPoints = thinResult.glyph.outline.contours.reduce(0) { $0 + $1.points.count }
        let boldPoints = boldResult.glyph.outline.contours.reduce(0) { $0 + $1.points.count }
        let pointsDiffer = thinPoints != boldPoints
        let boundsDiffer = thinBounds.width != boldBounds.width || thinBounds.height != boldBounds.height

        #expect(pointsDiffer || boundsDiffer,
                "Different styles (thin vs bold) should produce different outlines")
    }

    // MARK: - Fallback Behavior Tests

    // testFallbackToTemplate removed (C21): duplicates GlyphGeneratorTests.testPlaceholderMarkedAsPlaceholder

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

    // testGeneratedGlyphMetrics removed (C21): duplicates GlyphGeneratorTests.testGenerationResultGlyph

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
        #expect(totalReportedTime > 0, "Total reported generation time should be positive")
    }

    // MARK: - Edge Cases

    @Test("Generate handles Unicode characters")
    func testUnicodeCharacters() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = createTestStyle()

        // Test with actual non-ASCII characters and accented Latin
        let unicodeChars: [Character] = ["\u{00E9}", "\u{00F1}", "\u{00E4}"] // é, ñ, ä

        for char in unicodeChars {
            let result = try await generator.generate(
                character: char,
                mode: .fromScratch(style: style),
                metrics: metrics,
                settings: .fast
            )

            #expect(result.glyph.character == char)
            // Unicode characters should produce non-empty outlines
            #expect(!result.glyph.outline.isEmpty, "Unicode character \(char) should produce non-empty outline")
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
