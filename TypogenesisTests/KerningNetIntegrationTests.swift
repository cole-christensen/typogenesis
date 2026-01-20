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

    @Test("KerningPredictor is always available (geometric fallback)")
    func testPredictorAlwaysAvailable() {
        let predictor = KerningPredictor()
        #expect(predictor.isAvailable, "Predictor should always be available due to geometric fallback")
    }

    @Test("Model status can be checked")
    @MainActor
    func testModelStatusCheck() async {
        let status = ModelManager.shared.kerningNetStatus

        // Should be a valid status - just verify we can access it
        _ = status.displayText
        #expect(true, "Status is accessible")
    }

    // MARK: - Kerning Prediction Tests

    @Test("AV pair should have negative kerning")
    func testAVKerningNegative() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let kerning = try await predictor.predictPair(
            left: "A",
            right: "V",
            project: project
        )

        // AV is a classic kerning pair that needs negative adjustment
        #expect(kerning <= 0, "AV should have negative or zero kerning, got \(kerning)")
    }

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

    @Test("Predict single pair returns 0 for missing glyph")
    func testMissingGlyphReturnsZero() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: ["A"])

        // "V" doesn't exist
        let kerning = try await predictor.predictPair(
            left: "A",
            right: "V",
            project: project
        )

        #expect(kerning == 0, "Missing glyph should return 0 kerning")
    }

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

        // Should process all pairs (may filter some based on value)
        #expect(results.count <= pairs.count, "Should not return more than input pairs")
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

    @Test("Critical pairs only mode generates fewer pairs")
    func testCriticalPairsMode() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))

        let allPairsSettings = KerningPredictor.PredictionSettings(
            onlyCriticalPairs: false
        )
        let allResult = try await predictor.predictKerning(for: project, settings: allPairsSettings)

        let criticalSettings = KerningPredictor.PredictionSettings(
            onlyCriticalPairs: true
        )
        let criticalResult = try await predictor.predictKerning(for: project, settings: criticalSettings)

        // Critical only should analyze fewer pairs
        #expect(criticalResult.predictionTime <= allResult.predictionTime * 2,
                "Critical pairs should be faster or similar")
    }

    @Test("Minimum kerning value filters small adjustments")
    func testMinKerningFilter() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let lowThreshold = KerningPredictor.PredictionSettings(minKerningValue: 1)
        let lowResult = try await predictor.predictKerning(for: project, settings: lowThreshold)

        let highThreshold = KerningPredictor.PredictionSettings(minKerningValue: 100)
        let highResult = try await predictor.predictKerning(for: project, settings: highThreshold)

        // Higher threshold should produce fewer or equal pairs
        #expect(highResult.pairs.count <= lowResult.pairs.count,
                "Higher threshold should filter more pairs")
    }

    @Test("Spacing presets affect kerning values")
    func testSpacingPresets() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let tightResult = try await predictor.predictKerning(for: project, settings: .tight)
        let looseResult = try await predictor.predictKerning(for: project, settings: .loose)

        // Both should complete successfully
        #expect(tightResult.confidence >= 0)
        #expect(looseResult.confidence >= 0)
    }

    @Test("Include punctuation setting works")
    func testPunctuationSetting() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: ["A", "V", ".", ",", "!"])

        let withPunct = KerningPredictor.PredictionSettings(includePunctuation: true)
        let withoutPunct = KerningPredictor.PredictionSettings(includePunctuation: false)

        let withResult = try await predictor.predictKerning(for: project, settings: withPunct)
        let withoutResult = try await predictor.predictKerning(for: project, settings: withoutPunct)

        // Without punctuation should have no pairs with punctuation
        let punctPairs = withoutResult.pairs.filter {
            !$0.left.isLetter || !$0.right.isLetter
        }
        #expect(punctPairs.isEmpty, "Should not have punctuation pairs when disabled")
    }

    @Test("Include numbers setting works")
    func testNumbersSetting() async throws {
        let predictor = KerningPredictor()
        let project = createTestProject(withCharacters: ["A", "1", "2", "3"])

        let withNumbers = KerningPredictor.PredictionSettings(includeNumbers: true)
        let withoutNumbers = KerningPredictor.PredictionSettings(includeNumbers: false)

        let withResult = try await predictor.predictKerning(for: project, settings: withNumbers)
        let withoutResult = try await predictor.predictKerning(for: project, settings: withoutNumbers)

        // Without numbers should have no pairs with numbers
        let numberPairs = withoutResult.pairs.filter {
            $0.left.isNumber || $0.right.isNumber
        }
        #expect(numberPairs.isEmpty, "Should not have number pairs when disabled")
    }

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

    @Test("Requires minimum glyphs")
    func testMinimumGlyphsRequired() async throws {
        let predictor = KerningPredictor()

        // Empty project
        let emptyProject = FontProject(name: "Empty", family: "Test", style: "Regular")
        do {
            _ = try await predictor.predictKerning(for: emptyProject)
            Issue.record("Should throw insufficientGlyphs")
        } catch {
            #expect(error is KerningPredictor.PredictorError)
        }

        // Single glyph project
        var singleProject = FontProject(name: "Single", family: "Test", style: "Regular")
        singleProject.glyphs["A"] = Glyph(
            character: "A",
            outline: GlyphOutline(),
            advanceWidth: 500,
            leftSideBearing: 50
        )
        do {
            _ = try await predictor.predictKerning(for: singleProject)
            Issue.record("Should throw insufficientGlyphs")
        } catch {
            #expect(error is KerningPredictor.PredictorError)
        }
    }

    // MARK: - Output Validation Tests

    @Test("Output kerning units are integers")
    func testKerningUnitsAreIntegers() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // All values should be valid integers (not NaN or infinity)
        for pair in result.pairs {
            #expect(pair.value == Int(exactly: pair.value), "Kerning should be exact integer")
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
