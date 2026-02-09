import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("GlyphGenerator Diffusion Tests")
struct GlyphGeneratorDiffusionTests {

    // MARK: - Character-to-Index Mapping

    @Test("charToIndex maps lowercase a-z to 0-25")
    func testLowercaseMapping() {
        let lowercase = "abcdefghijklmnopqrstuvwxyz"
        for (i, char) in lowercase.enumerated() {
            #expect(GlyphGenerator.charToIndex[char] == Int32(i),
                    "Expected '\(char)' → \(i), got \(String(describing: GlyphGenerator.charToIndex[char]))")
        }
    }

    @Test("charToIndex maps uppercase A-Z to 26-51")
    func testUppercaseMapping() {
        let uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for (i, char) in uppercase.enumerated() {
            #expect(GlyphGenerator.charToIndex[char] == Int32(i + 26),
                    "Expected '\(char)' → \(i + 26), got \(String(describing: GlyphGenerator.charToIndex[char]))")
        }
    }

    @Test("charToIndex maps digits 0-9 to 52-61")
    func testDigitMapping() {
        let digits = "0123456789"
        for (i, char) in digits.enumerated() {
            #expect(GlyphGenerator.charToIndex[char] == Int32(i + 52),
                    "Expected '\(char)' → \(i + 52), got \(String(describing: GlyphGenerator.charToIndex[char]))")
        }
    }

    @Test("charToIndex has exactly 62 entries")
    func testMappingSize() {
        #expect(GlyphGenerator.charToIndex.count == 62)
    }

    @Test("charToIndex returns nil for unsupported characters")
    func testUnsupportedCharacters() {
        #expect(GlyphGenerator.charToIndex["!"] == nil)
        #expect(GlyphGenerator.charToIndex[" "] == nil)
        #expect(GlyphGenerator.charToIndex["@"] == nil)
    }

    @Test("charToIndex matches Python config.py ordering")
    func testMatchesPythonOrdering() {
        // Python: CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALL_CHARACTERS)}
        // where ALL_CHARACTERS = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789"
        #expect(GlyphGenerator.charToIndex["a"] == 0)
        #expect(GlyphGenerator.charToIndex["z"] == 25)
        #expect(GlyphGenerator.charToIndex["A"] == 26)
        #expect(GlyphGenerator.charToIndex["Z"] == 51)
        #expect(GlyphGenerator.charToIndex["0"] == 52)
        #expect(GlyphGenerator.charToIndex["9"] == 61)
    }

    // MARK: - Geometric Fallback Still Works

    @Test("Template fallback still generates valid glyphs without model")
    func testFallbackStillWorks() async throws {
        let generator = GlyphGenerator()
        let metrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -200,
            xHeight: 500,
            capHeight: 700,
            lineGap: 90
        )
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        // Fallback should produce visible content with honest confidence
        #expect(result.glyph.character == "A")
        #expect(!result.glyph.outline.isEmpty, "Fallback must produce visible content")
        #expect(result.confidence >= 0.0 && result.confidence <= 1.0)
    }

    @Test("Multiple characters produce distinct fallback glyphs")
    func testDistinctFallbackGlyphs() async throws {
        let generator = GlyphGenerator()
        let metrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -200,
            xHeight: 500,
            capHeight: 700,
            lineGap: 90
        )
        let style = StyleEncoder.FontStyle.default
        let chars: [Character] = ["H", "W", "i", "l"]
        var widths = [Int]()

        for char in chars {
            let result = try await generator.generate(
                character: char,
                mode: .fromScratch(style: style),
                metrics: metrics,
                settings: .fast
            )
            widths.append(result.glyph.advanceWidth)
        }

        // At least some should differ in width
        let uniqueWidths = Set(widths)
        #expect(uniqueWidths.count > 1,
                "Different characters should have varying widths, got \(widths)")
    }

    // MARK: - GenerationSettings

    @Test("Generation settings are preserved correctly")
    func testSettingsPreservation() {
        let settings = GlyphGenerator.GenerationSettings(
            steps: 30,
            guidanceScale: 5.0,
            seed: 42,
            temperature: 0.8
        )
        #expect(settings.steps == 30)
        #expect(settings.guidanceScale == 5.0)
        #expect(settings.seed == 42)
        #expect(settings.temperature == 0.8)
    }

    @Test("Seeded generation is deterministic in fallback mode")
    func testSeededFallback() async throws {
        let generator = GlyphGenerator()
        let metrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -200,
            xHeight: 500,
            capHeight: 700,
            lineGap: 90
        )
        let style = StyleEncoder.FontStyle.default
        let settings = GlyphGenerator.GenerationSettings(steps: 20, seed: 12345)

        let result1 = try await generator.generate(
            character: "X",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: settings
        )
        let result2 = try await generator.generate(
            character: "X",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: settings
        )

        // In fallback mode, same inputs should produce same outputs
        #expect(result1.glyph.outline.contours.count == result2.glyph.outline.contours.count)
        #expect(result1.glyph.advanceWidth == result2.glyph.advanceWidth)
    }

    // MARK: - generateWithModel fallback

    @Test("generateWithModel falls back to template when no model loaded")
    func testGenerateWithModelFallback() async throws {
        let generator = GlyphGenerator()
        let metrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -200,
            xHeight: 500,
            capHeight: 700,
            lineGap: 90
        )
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generateWithModel(
            character: "B",
            style: style,
            metrics: metrics
        )

        // Should have produced a valid glyph via fallback
        #expect(result.glyph.character == "B")
        #expect(!result.glyph.outline.isEmpty)
    }
}
