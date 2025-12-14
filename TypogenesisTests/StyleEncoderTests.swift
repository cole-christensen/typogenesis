import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("StyleEncoder Tests")
struct StyleEncoderTests {

    // MARK: - Helper Methods

    private func createTestProject(withGlyphs glyphs: [Character: GlyphOutline]) -> FontProject {
        var project = FontProject(name: "Test Font", family: "Test", style: "Regular")
        project.metrics = FontMetrics(
            unitsPerEm: 1000,
            ascender: 800,
            descender: -200,
            xHeight: 500,
            capHeight: 700,
            lineGap: 90
        )

        for (char, outline) in glyphs {
            project.glyphs[char] = Glyph(
                character: char,
                outline: outline,
                advanceWidth: 500,
                leftSideBearing: 50
            )
        }

        return project
    }

    private func createSquareOutline(size: CGFloat, origin: CGPoint = .zero) -> GlyphOutline {
        GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: origin.x, y: origin.y), type: .corner),
                    PathPoint(position: CGPoint(x: origin.x + size, y: origin.y), type: .corner),
                    PathPoint(position: CGPoint(x: origin.x + size, y: origin.y + size), type: .corner),
                    PathPoint(position: CGPoint(x: origin.x, y: origin.y + size), type: .corner)
                ],
                isClosed: true
            )
        ])
    }

    private func createRoundOutline(radius: CGFloat, center: CGPoint) -> GlyphOutline {
        // Create a circle-like shape with smooth points
        GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: center.x + radius, y: center.y), type: .smooth),
                    PathPoint(position: CGPoint(x: center.x, y: center.y + radius), type: .smooth),
                    PathPoint(position: CGPoint(x: center.x - radius, y: center.y), type: .smooth),
                    PathPoint(position: CGPoint(x: center.x, y: center.y - radius), type: .smooth)
                ],
                isClosed: true
            )
        ])
    }

    // MARK: - FontStyle Tests

    @Test("FontStyle default values are reasonable")
    func testFontStyleDefaults() {
        let style = StyleEncoder.FontStyle.default

        #expect(style.strokeWeight >= 0 && style.strokeWeight <= 1)
        #expect(style.strokeContrast >= 0 && style.strokeContrast <= 1)
        #expect(style.xHeightRatio > 0 && style.xHeightRatio < 1)
        #expect(style.widthRatio > 0)
        #expect(style.slant == 0)  // Default is upright
        #expect(style.serifStyle == .sansSerif)
        #expect(style.roundness >= 0 && style.roundness <= 1)
        #expect(style.regularity >= 0 && style.regularity <= 1)
    }

    @Test("FontStyle is Codable")
    func testFontStyleCodable() throws {
        let original = StyleEncoder.FontStyle(
            strokeWeight: 0.6,
            strokeContrast: 0.4,
            xHeightRatio: 0.72,
            widthRatio: 0.85,
            slant: 12,
            serifStyle: .transitional,
            roundness: 0.65,
            regularity: 0.9,
            embedding: [0.1, 0.2, 0.3]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(StyleEncoder.FontStyle.self, from: data)

        #expect(decoded == original)
    }

    @Test("FontStyle equality works correctly")
    func testFontStyleEquality() {
        let style1 = StyleEncoder.FontStyle.default
        let style2 = StyleEncoder.FontStyle.default

        #expect(style1 == style2)

        var style3 = style1
        style3.strokeWeight = 0.9

        #expect(style1 != style3)
    }

    // MARK: - Style Extraction Tests

    @Test("Extract style from empty project returns default-like values")
    func testExtractStyleFromEmptyProject() async throws {
        let project = FontProject(name: "Empty", family: "Empty", style: "Regular")
        let encoder = StyleEncoder()

        let style = try await encoder.extractStyle(from: project)

        // Should return reasonable defaults for empty project
        #expect(style.xHeightRatio > 0)
        #expect(style.serifStyle == .sansSerif)
    }

    @Test("Extract style analyzes x-height ratio from metrics")
    func testXHeightRatioFromMetrics() async throws {
        var project = FontProject(name: "Test", family: "Test", style: "Regular")
        project.metrics.xHeight = 500
        project.metrics.capHeight = 700

        let encoder = StyleEncoder()
        let style = try await encoder.extractStyle(from: project)

        let expectedRatio = Float(500) / Float(700)
        #expect(abs(style.xHeightRatio - expectedRatio) < 0.01)
    }

    @Test("Extract style detects roundness from curve points")
    func testRoundnessDetection() async throws {
        // Project with round glyphs (smooth points)
        let roundOutline = createRoundOutline(radius: 200, center: CGPoint(x: 250, y: 350))
        let roundProject = createTestProject(withGlyphs: ["o": roundOutline])

        // Project with square glyphs (corner points)
        let squareOutline = createSquareOutline(size: 400, origin: CGPoint(x: 50, y: 150))
        let squareProject = createTestProject(withGlyphs: ["H": squareOutline])

        let encoder = StyleEncoder()

        let roundStyle = try await encoder.extractStyle(from: roundProject)
        let squareStyle = try await encoder.extractStyle(from: squareProject)

        // Round glyphs should have higher roundness
        #expect(roundStyle.roundness > squareStyle.roundness)
    }

    @Test("Extract style with multiple glyphs")
    func testExtractStyleMultipleGlyphs() async throws {
        var glyphs: [Character: GlyphOutline] = [:]
        glyphs["n"] = createSquareOutline(size: 400, origin: CGPoint(x: 50, y: 100))
        glyphs["o"] = createRoundOutline(radius: 180, center: CGPoint(x: 250, y: 280))
        glyphs["H"] = createSquareOutline(size: 500, origin: CGPoint(x: 50, y: 100))

        let project = createTestProject(withGlyphs: glyphs)
        let encoder = StyleEncoder()

        let style = try await encoder.extractStyle(from: project)

        // Should analyze available representative glyphs
        #expect(style.roundness > 0 && style.roundness < 1)  // Mixed round and square
        #expect(style.regularity > 0)
    }

    // MARK: - Similarity Tests

    @Test("Identical styles have similarity of 1")
    func testIdenticalStyleSimilarity() {
        let encoder = StyleEncoder()
        let style = StyleEncoder.FontStyle.default

        let similarity = encoder.similarity(style, style)

        #expect(similarity > 0.99)
    }

    @Test("Different styles have lower similarity")
    func testDifferentStyleSimilarity() {
        let encoder = StyleEncoder()

        let style1 = StyleEncoder.FontStyle(
            strokeWeight: 0.3,
            strokeContrast: 0.2,
            xHeightRatio: 0.65,
            widthRatio: 0.7,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.8,
            regularity: 0.9,
            embedding: []
        )

        let style2 = StyleEncoder.FontStyle(
            strokeWeight: 0.8,
            strokeContrast: 0.6,
            xHeightRatio: 0.75,
            widthRatio: 0.9,
            slant: 12,
            serifStyle: .transitional,
            roundness: 0.3,
            regularity: 0.5,
            embedding: []
        )

        let similarity = encoder.similarity(style1, style2)

        #expect(similarity < 0.8)
        #expect(similarity > 0)
    }

    @Test("Similarity is symmetric")
    func testSimilaritySymmetric() {
        let encoder = StyleEncoder()

        let style1 = StyleEncoder.FontStyle.default
        var style2 = StyleEncoder.FontStyle.default
        style2.strokeWeight = 0.8

        let sim1to2 = encoder.similarity(style1, style2)
        let sim2to1 = encoder.similarity(style2, style1)

        #expect(abs(sim1to2 - sim2to1) < 0.001)
    }

    @Test("Similarity considers embeddings when present")
    func testSimilarityWithEmbeddings() {
        let encoder = StyleEncoder()

        var style1 = StyleEncoder.FontStyle.default
        style1.embedding = [1.0, 0.0, 0.0, 0.0]

        var style2 = StyleEncoder.FontStyle.default
        style2.embedding = [1.0, 0.0, 0.0, 0.0]  // Same embedding

        var style3 = StyleEncoder.FontStyle.default
        style3.embedding = [0.0, 1.0, 0.0, 0.0]  // Different embedding

        let sim12 = encoder.similarity(style1, style2)
        let sim13 = encoder.similarity(style1, style3)

        #expect(sim12 > sim13)
    }

    // MARK: - Interpolation Tests

    @Test("Interpolate at t=0 returns first style")
    func testInterpolateAtZero() {
        let encoder = StyleEncoder()

        let style1 = StyleEncoder.FontStyle(
            strokeWeight: 0.3,
            strokeContrast: 0.2,
            xHeightRatio: 0.65,
            widthRatio: 0.7,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.8,
            regularity: 0.9,
            embedding: [1.0, 2.0]
        )

        let style2 = StyleEncoder.FontStyle(
            strokeWeight: 0.8,
            strokeContrast: 0.6,
            xHeightRatio: 0.75,
            widthRatio: 0.9,
            slant: 12,
            serifStyle: .transitional,
            roundness: 0.3,
            regularity: 0.5,
            embedding: [3.0, 4.0]
        )

        let result = encoder.interpolate(style1, style2, t: 0)

        #expect(abs(result.strokeWeight - style1.strokeWeight) < 0.001)
        #expect(abs(result.roundness - style1.roundness) < 0.001)
        #expect(result.serifStyle == style1.serifStyle)
    }

    @Test("Interpolate at t=1 returns second style")
    func testInterpolateAtOne() {
        let encoder = StyleEncoder()

        let style1 = StyleEncoder.FontStyle.default
        var style2 = StyleEncoder.FontStyle.default
        style2.strokeWeight = 0.9
        style2.roundness = 0.2
        style2.serifStyle = .slab

        let result = encoder.interpolate(style1, style2, t: 1)

        #expect(abs(result.strokeWeight - style2.strokeWeight) < 0.001)
        #expect(abs(result.roundness - style2.roundness) < 0.001)
        #expect(result.serifStyle == style2.serifStyle)
    }

    @Test("Interpolate at t=0.5 returns midpoint")
    func testInterpolateAtHalf() {
        let encoder = StyleEncoder()

        let style1 = StyleEncoder.FontStyle(
            strokeWeight: 0.2,
            strokeContrast: 0.2,
            xHeightRatio: 0.6,
            widthRatio: 0.6,
            slant: 0,
            serifStyle: .sansSerif,
            roundness: 0.2,
            regularity: 0.6,
            embedding: []
        )

        let style2 = StyleEncoder.FontStyle(
            strokeWeight: 0.8,
            strokeContrast: 0.8,
            xHeightRatio: 0.8,
            widthRatio: 1.0,
            slant: 20,
            serifStyle: .transitional,
            roundness: 0.8,
            regularity: 1.0,
            embedding: []
        )

        let result = encoder.interpolate(style1, style2, t: 0.5)

        #expect(abs(result.strokeWeight - 0.5) < 0.001)
        #expect(abs(result.roundness - 0.5) < 0.001)
        #expect(abs(result.slant - 10) < 0.001)
    }

    @Test("Interpolate clamps t to valid range")
    func testInterpolateClampsT() {
        let encoder = StyleEncoder()

        let style1 = StyleEncoder.FontStyle.default
        var style2 = StyleEncoder.FontStyle.default
        style2.strokeWeight = 1.0

        let resultNegative = encoder.interpolate(style1, style2, t: -0.5)
        let resultOverOne = encoder.interpolate(style1, style2, t: 1.5)

        #expect(abs(resultNegative.strokeWeight - style1.strokeWeight) < 0.001)
        #expect(abs(resultOverOne.strokeWeight - style2.strokeWeight) < 0.001)
    }

    @Test("Interpolate handles embedding arrays")
    func testInterpolateEmbeddings() {
        let encoder = StyleEncoder()

        var style1 = StyleEncoder.FontStyle.default
        style1.embedding = [0.0, 0.0, 0.0]

        var style2 = StyleEncoder.FontStyle.default
        style2.embedding = [1.0, 1.0, 1.0]

        let result = encoder.interpolate(style1, style2, t: 0.5)

        #expect(result.embedding.count == 3)
        for value in result.embedding {
            #expect(abs(value - 0.5) < 0.001)
        }
    }

    // MARK: - SerifStyle Tests

    @Test("SerifStyle has all expected cases")
    func testSerifStyleCases() {
        let allCases = StyleEncoder.SerifStyle.allCases

        #expect(allCases.contains(.sansSerif))
        #expect(allCases.contains(.oldStyle))
        #expect(allCases.contains(.transitional))
        #expect(allCases.contains(.modern))
        #expect(allCases.contains(.slab))
        #expect(allCases.contains(.script))
        #expect(allCases.contains(.decorative))
        #expect(allCases.count == 7)
    }

    @Test("SerifStyle raw values are human readable")
    func testSerifStyleRawValues() {
        #expect(StyleEncoder.SerifStyle.sansSerif.rawValue == "Sans Serif")
        #expect(StyleEncoder.SerifStyle.oldStyle.rawValue == "Old Style")
        #expect(StyleEncoder.SerifStyle.slab.rawValue == "Slab Serif")
    }
}
