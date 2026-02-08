import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

@Suite("KerningNet Integration Tests")
struct KerningNetIntegrationTests {

    // MARK: - Helper Methods

    /// Creates a test project with simple rectangular glyphs
    private func createTestProject(withCharacters chars: [Character]) -> FontProject {
        var project = FontProject(name: "Test Font", family: "Test", style: "Regular")

        for char in chars {
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

    /// Creates a project with realistic glyph shapes for kerning tests
    private func createRealisticProject() -> FontProject {
        var project = FontProject(name: "Realistic Test Font", family: "Realistic", style: "Regular")

        // "A" - triangular shape (creates gap with V)
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

        // "V" - inverted triangle (critical pair with A)
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

        // "T" - T-bar creates gap below
        let tOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 0, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 650), type: .corner),
                    PathPoint(position: CGPoint(x: 300, y: 650), type: .corner),
                    PathPoint(position: CGPoint(x: 300, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 200, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 200, y: 650), type: .corner),
                    PathPoint(position: CGPoint(x: 0, y: 650), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs["T"] = Glyph(character: "T", outline: tOutline, advanceWidth: 500, leftSideBearing: 0)

        // "o" - oval (lowercase)
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

        // "a" - lowercase a
        let aLowerOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 400, y: 500), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 500), type: .smooth),
                    PathPoint(position: CGPoint(x: 50, y: 250), type: .smooth)
                ],
                isClosed: true
            )
        ])
        project.glyphs["a"] = Glyph(character: "a", outline: aLowerOutline, advanceWidth: 450, leftSideBearing: 50)

        return project
    }

    // MARK: - Model Loading Tests

    @Test("KerningNet model is not available without loaded CoreML models")
    @MainActor
    func testModelNotAvailableWithoutModels() async {
        let kerningModel = ModelManager.shared.kerningNet
        #expect(kerningModel == nil, "KerningNet model should not be available without loaded CoreML models")
    }

    @Test("Model status can be checked")
    @MainActor
    func testModelStatusCheck() async {
        let status = ModelManager.shared.kerningNetStatus

        // Without trained models, status should not be loaded/available
        #expect(!status.isAvailable, "KerningNet model should not be available without trained models")
        #expect(!status.displayText.isEmpty, "Status display text should not be empty")
    }

    // MARK: - Kerning Prediction Tests

    // testAVKerningNegative removed (C22): duplicates KerningPredictorTests.testPredictSinglePair

    @Test("Predict kerning returns valid result for existing pairs")
    func testPredictExistingPairs() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // Confidence should be in valid range
        #expect(result.confidence >= 0 && result.confidence <= 1,
                "Confidence should be 0-1, got \(result.confidence)")

        // Prediction time should be positive
        #expect(result.predictionTime > 0, "Prediction time should be positive")
    }

    // testMissingGlyphReturnsZero removed (C22): duplicates KerningPredictorTests.testPredictPairWithMissingGlyph

    @Test("Kerning values are within reasonable range")
    func testKerningRange() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // All kerning values should be within 1/4 of unitsPerEm
        let maxKerning = project.metrics.unitsPerEm / 4

        for pair in result.pairs {
            #expect(abs(pair.value) <= maxKerning,
                    "Kerning \(pair.value) for \(pair.left)-\(pair.right) exceeds max \(maxKerning)")
        }
    }

    // MARK: - Batch Prediction Tests

    @Test("Batch prediction processes all pairs")
    func testBatchPredictionProcessesAll() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let pairs: [(Character, Character)] = [
            ("A", "V"),
            ("T", "o"),
            ("V", "a")
        ]

        let results = try await predictor.predictBatch(
            pairs: pairs,
            project: project,
            settings: KerningPredictor.PredictionSettings(minKerningValue: 0) // Include all
        )

        // Should process pairs and return at least some results
        #expect(results.count <= pairs.count, "Should not return more than input pairs")
        #expect(results.count > 0, "Should return at least one kerning pair for known critical pairs")
    }

    @Test("Batch prediction is efficient")
    func testBatchPredictionPerformance() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"))

        let startTime = Date()

        _ = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        let elapsed = Date().timeIntervalSince(startTime)

        // Should complete in reasonable time (< 50ms per pair target, but geometric is fast)
        #expect(elapsed < 10.0, "Batch prediction took too long: \(elapsed)s")
    }

    // MARK: - Settings Tests

    // testCriticalPairsMode removed: duplicates KerningPredictorTests.testCriticalPairsOnly
    // testMinKerningFilter removed: duplicates KerningPredictorTests.testMinKerningValueFilter
    // testSpacingPresets removed: duplicates KerningPredictorTests.testSpacingPresets
    // testPunctuationSetting removed: duplicates KerningPredictorTests.testIncludePunctuationSetting
    // testNumbersSetting removed: duplicates KerningPredictorTests.testIncludeNumbersSetting

    // MARK: - Fallback Behavior Tests

    @Test("Fallback uses geometric calculation")
    func testGeometricFallback() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        // Without model loaded, should use geometric fallback
        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // Geometric fallback has lower confidence
        #expect(result.confidence <= 0.85, "Geometric fallback should have confidence <= 0.85")
    }

    // MARK: - Error Handling Tests

    // testMinimumGlyphsRequired removed (C22): duplicates KerningPredictorTests.testRequiresMinimumGlyphs

    // MARK: - Output Validation Tests

    @Test("Output kerning values are in reasonable range")
    func testKerningValuesInReasonableRange() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // All kerning values should be within reasonable bounds (not extreme)
        let maxReasonable = project.metrics.unitsPerEm  // Should never exceed full em
        for pair in result.pairs {
            #expect(abs(pair.value) <= maxReasonable,
                    "Kerning value \(pair.value) for \(pair.left)-\(pair.right) exceeds reasonable range")
        }
    }

    @Test("Kerning pairs have valid characters")
    func testKerningPairsHaveValidChars() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // All pairs should reference characters that exist in the project
        for pair in result.pairs {
            #expect(project.glyphs[pair.left] != nil,
                    "Left character '\(pair.left)' should exist in project")
            #expect(project.glyphs[pair.right] != nil,
                    "Right character '\(pair.right)' should exist in project")
        }
    }
}
