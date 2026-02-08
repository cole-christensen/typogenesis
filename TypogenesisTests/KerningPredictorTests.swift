import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("KerningPredictor Tests")
struct KerningPredictorTests {

    // MARK: - Helper Functions

    /// Creates a test project with identical rectangular glyphs for testing kerning
    private func createTestProjectWithIdenticalGlyphs(characters chars: [Character]) -> FontProject {
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

    @Test("KerningPredictor isAvailable returns true (geometric fallback always works)")
    @MainActor
    func testPredictorIsAvailableWithoutModel() {
        // Without AI models loaded, isAvailable should still return true
        // because the geometric kerning fallback is always ready.
        // This test verifies the fallback design: isAvailable == true even without CoreML models.
        let predictor = KerningPredictor()
        #expect(predictor.isAvailable, "isAvailable should be true even without CoreML models (geometric fallback)")

        // Verify the AI model is NOT actually loaded (geometric fallback is what makes it available)
        let kerningModel = ModelManager.shared.kerningNet
        #expect(kerningModel == nil, "CoreML kerning model should NOT be loaded in test environment")
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
            if case KerningPredictor.PredictorError.insufficientGlyphs = error {
                // Expected
            } else {
                Issue.record("Expected insufficientGlyphs error, got \(error)")
            }
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
            if case KerningPredictor.PredictorError.insufficientGlyphs = error {
                // Expected
            } else {
                Issue.record("Expected insufficientGlyphs error, got \(error)")
            }
        }
    }

    @Test("Predict kerning generates pairs for realistic glyphs")
    func testGeneratesPairs() async throws {
        let predictor = KerningPredictor()
        // Use realistic project with actual A and V shapes that need kerning
        let project = createRealisticProject()

        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // Verify confidence is in valid range [0, 1]
        #expect(result.confidence >= 0 && result.confidence <= 1, "Confidence should be in valid range [0, 1]")

        // Verify prediction took measurable time
        #expect(result.predictionTime > 0, "Prediction should take measurable time")

        // With realistic A, V, o shapes, the geometric kerning algorithm
        // should detect that AV pair needs adjustment (A's diagonal + V's diagonal creates gap)
        #expect(!result.pairs.isEmpty, "Should produce kerning pairs for A, V, o shapes")
    }

    @Test("Critical pairs only mode generates fewer pairs")
    func testCriticalPairsOnly() async throws {
        let predictor = KerningPredictor()
        // Use the realistic project with actual shaped glyphs (triangular A, inverted-V, oval o)
        // so the geometric kerning algorithm produces non-zero values for critical pairs.
        // Identical square glyphs produce zero kerning for all pairs, making this test meaningless.
        let project = createRealisticProject()

        // Generate all pairs
        let allPairsSettings = KerningPredictor.PredictionSettings(
            minKerningValue: 1,
            onlyCriticalPairs: false
        )
        let allResult = try await predictor.predictKerning(for: project, settings: allPairsSettings)

        // Generate only critical pairs
        let criticalOnlySettings = KerningPredictor.PredictionSettings(
            minKerningValue: 1,
            onlyCriticalPairs: true
        )
        let criticalResult = try await predictor.predictKerning(for: project, settings: criticalOnlySettings)

        // Critical pairs mode should generate fewer or equal pairs than all-pairs mode
        #expect(criticalResult.pairs.count <= allResult.pairs.count,
                "Critical pairs mode should generate fewer or equal pairs than all-pairs mode")
        #expect(criticalResult.pairs.count > 0, "Critical pairs should produce some pairs with realistic glyph shapes")

        // Verify that critical pairs include known problematic pairs like AV
        let hasAV = criticalResult.pairs.contains { $0.left == "A" && $0.right == "V" }
        #expect(hasAV, "AV should be a critical pair that produces non-zero kerning with realistic shapes")
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
        let project = createTestProjectWithIdenticalGlyphs(characters: ["A"])

        // "V" doesn't exist in project
        let kerning = try await predictor.predictPair(
            left: "A",
            right: "V",
            project: project
        )

        #expect(kerning == 0)
    }

    @Test("Spacing presets produce tighter kerning for tight vs loose")
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

        // Both should produce valid results with pairs
        #expect(!tightResult.pairs.isEmpty, "Tight preset should produce kerning pairs")
        #expect(!looseResult.pairs.isEmpty, "Loose preset should produce kerning pairs")

        // For each common pair, tight spacing should produce more negative (or equal) values
        // compared to loose spacing. This is the core semantic guarantee of spacing presets.
        for tightPair in tightResult.pairs {
            if let loosePair = looseResult.pairs.first(where: {
                $0.left == tightPair.left && $0.right == tightPair.right
            }) {
                #expect(tightPair.value <= loosePair.value,
                    "Tight kerning should be <= loose for \(tightPair.left)\(tightPair.right): tight=\(tightPair.value) vs loose=\(loosePair.value)")
            }
        }
    }

    @Test("Include punctuation setting works")
    func testIncludePunctuationSetting() async throws {
        let predictor = KerningPredictor()
        let project = createTestProjectWithIdenticalGlyphs(characters: ["A", "V", ".", ",", "!"])

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

        // Both settings should produce valid results
        #expect(withPunctResult.confidence >= 0 && withPunctResult.confidence <= 1)
        #expect(withoutPunctResult.confidence >= 0 && withoutPunctResult.confidence <= 1)
        // Without punctuation, no pairs involving punctuation marks should exist
        let pairsWithoutPunct = withoutPunctResult.pairs.filter {
            $0.left == "." || $0.left == "," || $0.left == "!" ||
            $0.right == "." || $0.right == "," || $0.right == "!"
        }
        #expect(pairsWithoutPunct.isEmpty, "Without punctuation, no punctuation pairs should exist")
    }

    @Test("Include numbers setting works")
    func testIncludeNumbersSetting() async throws {
        let predictor = KerningPredictor()
        let project = createTestProjectWithIdenticalGlyphs(characters: ["A", "1", "2", "3"])

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

        // Both settings should produce valid results
        #expect(withNumbersResult.confidence >= 0 && withNumbersResult.confidence <= 1)
        #expect(withoutNumbersResult.confidence >= 0 && withoutNumbersResult.confidence <= 1)
        // Without numbers, no pairs involving digits should exist
        let pairsWithoutNumbers = withoutNumbersResult.pairs.filter {
            $0.left.isNumber || $0.right.isNumber
        }
        #expect(pairsWithoutNumbers.isEmpty, "Without numbers, no number pairs should exist")
    }
}
