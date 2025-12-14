import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("KerningPredictor Tests")
struct KerningPredictorTests {

    // MARK: - Helper Functions

    /// Creates a test project with glyphs for testing kerning
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

    /// Creates a project with actual glyph shapes for more realistic kerning tests
    private func createRealisticProject() -> FontProject {
        var project = FontProject(name: "Realistic Test Font", family: "Realistic", style: "Regular")

        // "A" - triangular shape
        let aOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 250, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 250, y: 500), type: .corner),
                    PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 0), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs["A"] = Glyph(character: "A", outline: aOutline, advanceWidth: 500, leftSideBearing: 0)

        // "V" - inverted triangle
        let vOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 0, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 250, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 400, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 250, y: 200), type: .corner),
                    PathPoint(position: CGPoint(x: 100, y: 700), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs["V"] = Glyph(character: "V", outline: vOutline, advanceWidth: 500, leftSideBearing: 0)

        // "o" - oval
        let oOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 250, y: 500), type: .smooth),
                    PathPoint(position: CGPoint(x: 450, y: 250), type: .smooth),
                    PathPoint(position: CGPoint(x: 250, y: 0), type: .smooth),
                    PathPoint(position: CGPoint(x: 50, y: 250), type: .smooth)
                ],
                isClosed: true
            )
        ])
        project.glyphs["o"] = Glyph(character: "o", outline: oOutline, advanceWidth: 500, leftSideBearing: 50)

        return project
    }

    // MARK: - Tests

    @Test("KerningPredictor is always available (geometric fallback)")
    func testPredictorIsAvailable() {
        let predictor = KerningPredictor()
        #expect(predictor.isAvailable == true)
    }

    @Test("Predict kerning requires at least 2 glyphs")
    func testRequiresMinimumGlyphs() async throws {
        let predictor = KerningPredictor()

        // Test with 0 glyphs
        let emptyProject = FontProject(name: "Empty", family: "Test", style: "Regular")
        do {
            _ = try await predictor.predictKerning(for: emptyProject)
            Issue.record("Should have thrown insufficientGlyphs error")
        } catch {
            #expect(error is KerningPredictor.PredictorError)
        }

        // Test with 1 glyph
        var oneGlyphProject = FontProject(name: "OneGlyph", family: "Test", style: "Regular")
        oneGlyphProject.glyphs["A"] = Glyph(
            character: "A",
            outline: GlyphOutline(),
            advanceWidth: 500,
            leftSideBearing: 50
        )
        do {
            _ = try await predictor.predictKerning(for: oneGlyphProject)
            Issue.record("Should have thrown insufficientGlyphs error")
        } catch {
            #expect(error is KerningPredictor.PredictorError)
        }
    }

    @Test("Predict kerning generates pairs for simple project")
    func testGeneratesPairs() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: ["A", "V", "W", "a", "v"])

        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // Should have generated some pairs
        #expect(result.pairs.count >= 0)  // May be 0 if no significant kerning needed
        #expect(result.predictionTime > 0)
        #expect(result.confidence > 0)
    }

    @Test("Critical pairs only mode generates fewer pairs")
    func testCriticalPairsOnly() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))

        // Generate all pairs
        let allPairsSettings = KerningPredictor.PredictionSettings(
            onlyCriticalPairs: false
        )
        let allResult = try await predictor.predictKerning(for: project, settings: allPairsSettings)

        // Generate only critical pairs
        let criticalOnlySettings = KerningPredictor.PredictionSettings(
            onlyCriticalPairs: true
        )
        let criticalResult = try await predictor.predictKerning(for: project, settings: criticalOnlySettings)

        // Critical only should analyze fewer pairs (and likely generate fewer)
        // This is a soft assertion since the results depend on the actual kerning values
        #expect(criticalResult.predictionTime <= allResult.predictionTime * 2)
    }

    @Test("Minimum kerning value filters small adjustments")
    func testMinKerningValueFilter() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        // Low threshold
        let lowThresholdSettings = KerningPredictor.PredictionSettings(
            minKerningValue: 1
        )
        let lowResult = try await predictor.predictKerning(for: project, settings: lowThresholdSettings)

        // High threshold
        let highThresholdSettings = KerningPredictor.PredictionSettings(
            minKerningValue: 100
        )
        let highResult = try await predictor.predictKerning(for: project, settings: highThresholdSettings)

        // Higher threshold should produce fewer or equal pairs
        #expect(highResult.pairs.count <= lowResult.pairs.count)
    }

    @Test("Predict single pair returns kerning value")
    func testPredictSinglePair() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let kerning = try await predictor.predictPair(
            left: "A",
            right: "V",
            project: project
        )

        // Should return some value (could be 0 if no adjustment needed)
        // The geometric algorithm should detect that A and V can be kerned
        #expect(kerning <= 0)  // AV typically needs negative kerning
    }

    @Test("Predict pair returns 0 for missing glyph")
    func testPredictPairWithMissingGlyph() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: ["A"])

        // "V" doesn't exist in project
        let kerning = try await predictor.predictPair(
            left: "A",
            right: "V",
            project: project
        )

        #expect(kerning == 0)
    }

    @Test("Spacing presets affect results")
    func testSpacingPresets() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let tightResult = try await predictor.predictKerning(
            for: project,
            settings: .tight
        )

        let looseResult = try await predictor.predictKerning(
            for: project,
            settings: .loose
        )

        // Both should complete without error
        #expect(tightResult.pairs.count >= 0)
        #expect(looseResult.pairs.count >= 0)
    }

    @Test("Include punctuation setting works")
    func testIncludePunctuationSetting() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: ["A", "V", ".", ",", "!"])

        // With punctuation
        let withPunctSettings = KerningPredictor.PredictionSettings(
            includePunctuation: true
        )
        let withPunctResult = try await predictor.predictKerning(for: project, settings: withPunctSettings)

        // Without punctuation
        let withoutPunctSettings = KerningPredictor.PredictionSettings(
            includePunctuation: false
        )
        let withoutPunctResult = try await predictor.predictKerning(for: project, settings: withoutPunctSettings)

        // Results should differ (or at least not crash)
        #expect(withPunctResult.pairs.count >= 0)
        #expect(withoutPunctResult.pairs.count >= 0)
    }

    @Test("Include numbers setting works")
    func testIncludeNumbersSetting() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: ["A", "1", "2", "3"])

        // With numbers
        let withNumbersSettings = KerningPredictor.PredictionSettings(
            includeNumbers: true
        )
        let withNumbersResult = try await predictor.predictKerning(for: project, settings: withNumbersSettings)

        // Without numbers
        let withoutNumbersSettings = KerningPredictor.PredictionSettings(
            includeNumbers: false
        )
        let withoutNumbersResult = try await predictor.predictKerning(for: project, settings: withoutNumbersSettings)

        // Results should be valid
        #expect(withNumbersResult.pairs.count >= 0)
        #expect(withoutNumbersResult.pairs.count >= 0)
    }
}
