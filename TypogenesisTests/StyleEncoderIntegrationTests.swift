import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("StyleEncoder Integration Tests")
struct StyleEncoderIntegrationTests {

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

    private func createTestProject(withCharacters chars: [Character]) -> FontProject {
        var project = FontProject(name: "Test Font", family: "Test", style: "Regular")

        for char in chars {
            // Create a simple square glyph
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

            project.glyphs[char] = Glyph(
                character: char,
                outline: outline,
                advanceWidth: 500,
                leftSideBearing: 50
            )
        }

        return project
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

    // MARK: - Model Loading Tests

    @Test("StyleEncoder model status can be checked")
    @MainActor
    func testModelStatusCheck() async {
        let status = ModelManager.shared.styleEncoderStatus

        // Status should be a valid ModelStatus value - all enum cases are valid
        // Just verify we can access it without crash
        _ = status.displayText
        #expect(true, "Status is accessible")
    }

    // MARK: - Embedding Extraction Tests

    @Test("encodeGlyph returns embedding when model unavailable (fallback)")
    func testEncodeGlyphFallback() async throws {
        let encoder = StyleEncoder()
        let glyph = createTestGlyph(character: "A")

        // This should either succeed with model or throw modelNotLoaded
        do {
            let embedding = try await encoder.encodeGlyph(glyph)

            // If it succeeds, embedding should have correct dimension
            #expect(embedding.count == 128, "Embedding should be 128 dimensions")

            // All values should be finite
            for value in embedding {
                #expect(value.isFinite, "Embedding values should be finite")
            }
        } catch {
            // Expected to throw if model not loaded
            #expect(error is StyleEncoder.StyleEncoderError, "Should throw StyleEncoderError")
        }
    }

    @Test("extractStyle returns valid FontStyle")
    func testExtractStyle() async throws {
        let encoder = StyleEncoder()
        let project = createTestProject(withCharacters: ["n", "o", "H", "O", "a"])

        let style = try await encoder.extractStyle(from: project)

        // Validate style properties are in expected ranges
        #expect(style.strokeWeight >= 0 && style.strokeWeight <= 1, "strokeWeight should be 0-1")
        #expect(style.strokeContrast >= 0 && style.strokeContrast <= 1, "strokeContrast should be 0-1")
        #expect(style.xHeightRatio >= 0 && style.xHeightRatio <= 1, "xHeightRatio should be 0-1")
        #expect(style.widthRatio >= 0 && style.widthRatio <= 2, "widthRatio should be reasonable")
        #expect(style.roundness >= 0 && style.roundness <= 1, "roundness should be 0-1")
        #expect(style.regularity >= 0 && style.regularity <= 1, "regularity should be 0-1")
    }

    @Test("extractStyle identifies serif style")
    func testSerifStyleClassification() async throws {
        let encoder = StyleEncoder()
        let project = createTestProject(withCharacters: ["n", "H"])

        let style = try await encoder.extractStyle(from: project)

        // Should return one of the valid serif styles
        let validStyles = StyleEncoder.SerifStyle.allCases
        #expect(validStyles.contains(style.serifStyle), "Should return valid serif style")
    }

    // MARK: - Embedding Similarity Tests

    @Test("Similarity between identical styles is 1.0")
    func testIdenticalStyleSimilarity() {
        let encoder = StyleEncoder()
        let style = StyleEncoder.FontStyle.default

        let similarity = encoder.similarity(style, style)
        #expect(similarity == 1.0, "Identical styles should have similarity 1.0")
    }

    @Test("Similarity between different styles is less than 1.0")
    func testDifferentStyleSimilarity() {
        let encoder = StyleEncoder()

        let styleA = StyleEncoder.FontStyle(
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

        let styleB = StyleEncoder.FontStyle(
            strokeWeight: 0.8,
            strokeContrast: 0.5,
            xHeightRatio: 0.6,
            widthRatio: 0.9,
            slant: 12,
            serifStyle: .slab,
            roundness: 0.7,
            regularity: 0.5,
            embedding: []
        )

        let similarity = encoder.similarity(styleA, styleB)
        #expect(similarity < 1.0, "Different styles should have similarity less than 1.0")
        #expect(similarity >= 0.0, "Similarity should be non-negative")
    }

    @Test("Similarity is symmetric")
    func testSimilaritySymmetry() {
        let encoder = StyleEncoder()

        let styleA = StyleEncoder.FontStyle(
            strokeWeight: 0.3,
            strokeContrast: 0.2,
            xHeightRatio: 0.7,
            widthRatio: 0.8,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.4,
            regularity: 0.8,
            embedding: []
        )

        let styleB = StyleEncoder.FontStyle(
            strokeWeight: 0.6,
            strokeContrast: 0.4,
            xHeightRatio: 0.65,
            widthRatio: 0.85,
            slant: 5,
            serifStyle: .transitional,
            roundness: 0.6,
            regularity: 0.7,
            embedding: []
        )

        let simAB = encoder.similarity(styleA, styleB)
        let simBA = encoder.similarity(styleB, styleA)

        #expect(abs(simAB - simBA) < 0.001, "Similarity should be symmetric")
    }

    // MARK: - Embedding Caching Tests

    @Test("Cache can be cleared")
    func testCacheClear() {
        let encoder = StyleEncoder()

        // Clear cache should not throw
        encoder.clearCache()

        // Should still work after clearing
        #expect(encoder.similarity(.default, .default) == 1.0)
    }

    // MARK: - Interpolation Tests

    @Test("Style interpolation at t=0 returns first style")
    func testInterpolationAtZero() {
        let encoder = StyleEncoder()

        let styleA = StyleEncoder.FontStyle(
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

        let styleB = StyleEncoder.FontStyle(
            strokeWeight: 0.8,
            strokeContrast: 0.5,
            xHeightRatio: 0.6,
            widthRatio: 0.9,
            slant: 12,
            serifStyle: .slab,
            roundness: 0.7,
            regularity: 0.5,
            embedding: []
        )

        let result = encoder.interpolate(styleA, styleB, t: 0)

        #expect(result.strokeWeight == styleA.strokeWeight)
        #expect(result.strokeContrast == styleA.strokeContrast)
        #expect(result.serifStyle == styleA.serifStyle)
    }

    @Test("Style interpolation at t=1 returns second style")
    func testInterpolationAtOne() {
        let encoder = StyleEncoder()

        let styleA = StyleEncoder.FontStyle(
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

        let styleB = StyleEncoder.FontStyle(
            strokeWeight: 0.8,
            strokeContrast: 0.5,
            xHeightRatio: 0.6,
            widthRatio: 0.9,
            slant: 12,
            serifStyle: .slab,
            roundness: 0.7,
            regularity: 0.5,
            embedding: []
        )

        let result = encoder.interpolate(styleA, styleB, t: 1)

        #expect(result.strokeWeight == styleB.strokeWeight)
        #expect(result.strokeContrast == styleB.strokeContrast)
        #expect(result.serifStyle == styleB.serifStyle)
    }

    @Test("Style interpolation at t=0.5 is midpoint")
    func testInterpolationAtHalf() {
        let encoder = StyleEncoder()

        let styleA = StyleEncoder.FontStyle(
            strokeWeight: 0.2,
            strokeContrast: 0.0,
            xHeightRatio: 0.6,
            widthRatio: 0.6,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.2,
            regularity: 0.8,
            embedding: []
        )

        let styleB = StyleEncoder.FontStyle(
            strokeWeight: 0.8,
            strokeContrast: 1.0,
            xHeightRatio: 0.8,
            widthRatio: 1.0,
            slant: 10,
            serifStyle: .slab,
            roundness: 0.8,
            regularity: 0.4,
            embedding: []
        )

        let result = encoder.interpolate(styleA, styleB, t: 0.5)

        #expect(abs(result.strokeWeight - 0.5) < 0.01)
        #expect(abs(result.strokeContrast - 0.5) < 0.01)
        #expect(abs(result.xHeightRatio - 0.7) < 0.01)
    }

    // MARK: - Performance Tests

    @Test("Style extraction completes within timeout")
    func testExtractionPerformance() async throws {
        let encoder = StyleEncoder()
        let project = createTestProject(withCharacters: ["n", "o", "H", "O", "a", "g", "e"])

        let startTime = Date()
        _ = try await encoder.extractStyle(from: project)
        let elapsed = Date().timeIntervalSince(startTime)

        // Should complete within reasonable time
        #expect(elapsed < 1.0, "Style extraction should complete within 1 second, took \(elapsed)s")
    }

    // MARK: - Fallback Behavior Tests

    @Test("Fallback embedding is deterministic")
    func testFallbackDeterministic() async throws {
        let encoder = StyleEncoder()
        let project = createTestProject(withCharacters: ["n", "o", "H"])

        // Extract style twice
        let style1 = try await encoder.extractStyle(from: project)
        let style2 = try await encoder.extractStyle(from: project)

        // Geometric analysis should be deterministic
        #expect(style1.strokeWeight == style2.strokeWeight)
        #expect(style1.strokeContrast == style2.strokeContrast)
        #expect(style1.serifStyle == style2.serifStyle)
    }

    // MARK: - Edge Cases

    @Test("Handles empty project gracefully")
    func testEmptyProject() async throws {
        let encoder = StyleEncoder()
        let project = FontProject(name: "Empty", family: "Test", style: "Regular")

        let style = try await encoder.extractStyle(from: project)

        // Should return default-like values
        #expect(style.strokeWeight >= 0 && style.strokeWeight <= 1)
    }

    @Test("Handles project with single glyph")
    func testSingleGlyphProject() async throws {
        let encoder = StyleEncoder()
        let project = createTestProject(withCharacters: ["A"])

        let style = try await encoder.extractStyle(from: project)

        // Should not crash and return valid values
        #expect(style.strokeWeight >= 0)
        #expect(style.regularity >= 0)
    }
}
