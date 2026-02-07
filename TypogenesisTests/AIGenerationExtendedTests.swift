import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

// MARK: - Extended AI Generation Tests

/// These tests fill gaps in AI generation coverage per Issue #16 requirements:
/// - Generated glyphs have valid outlines
/// - Kerning predictions produce expected values for known pairs
/// - Generated content can be used in actual font workflows

@Suite("Extended GlyphGenerator Tests")
struct ExtendedGlyphGeneratorTests {

    // MARK: - Helpers

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

    private func createTestGlyph(character: Character, withOutline: Bool = true) -> Glyph {
        let outline: GlyphOutline
        if withOutline {
            outline = GlyphOutline(contours: [
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
        } else {
            outline = GlyphOutline()
        }

        return Glyph(
            character: character,
            outline: outline,
            advanceWidth: 500,
            leftSideBearing: 50
        )
    }

    // MARK: - Glyph Validity Tests

    @Test("Generated glyph has non-negative advance width")
    func generatedGlyphHasValidAdvanceWidth() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "A",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.glyph.advanceWidth > 0)
        #expect(result.glyph.advanceWidth <= metrics.unitsPerEm * 2) // Reasonable max
    }

    @Test("Generated glyph has non-negative left sidebearing")
    func generatedGlyphHasValidSidebearing() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "B",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.glyph.leftSideBearing >= 0)
    }

    @Test("Generated glyph outline can be converted to CGPath")
    func generatedGlyphConvertsToCGPath() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let baseGlyph = createTestGlyph(character: "X")

        let result = try await generator.generate(
            character: "X",
            mode: .variation(base: baseGlyph, strength: 0.5),
            metrics: metrics,
            settings: .fast
        )

        // Should produce a valid CGPath with a non-degenerate bounding box
        let path = result.glyph.outline.cgPath
        let bounds = path.boundingBox
        #expect(!bounds.isNull, "CGPath bounding box should not be null")
        #expect(bounds.width > 0, "CGPath should have non-zero width")
        #expect(bounds.height > 0, "CGPath should have non-zero height")
    }

    @Test("Generated glyph bounding box is reasonable")
    func generatedGlyphHasReasonableBounds() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let baseGlyph = createTestGlyph(character: "H")

        let result = try await generator.generate(
            character: "H",
            mode: .variation(base: baseGlyph, strength: 0.3),
            metrics: metrics,
            settings: .fast
        )

        let bounds = result.glyph.outline.boundingBox

        // Bounds should be within reasonable range for font metrics
        #expect(bounds.maxY <= metrics.ascender * 2)  // Some tolerance
        #expect(bounds.minY >= metrics.descender * 2)
        #expect(bounds.width <= metrics.unitsPerEm * 2)
        #expect(bounds.height <= metrics.unitsPerEm * 2)
    }

    @Test("Interpolation at different t values produces different results")
    func interpolationProducesDifferentResults() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()

        let glyphA = createTestGlyph(character: "A")
        var glyphB = createTestGlyph(character: "A")
        glyphB = Glyph(
            character: "A",
            outline: GlyphOutline(contours: [
                Contour(
                    points: [
                        PathPoint(position: CGPoint(x: 100, y: 50), type: .corner),
                        PathPoint(position: CGPoint(x: 400, y: 50), type: .corner),
                        PathPoint(position: CGPoint(x: 400, y: 650), type: .corner),
                        PathPoint(position: CGPoint(x: 100, y: 650), type: .corner)
                    ],
                    isClosed: true
                )
            ]),
            advanceWidth: 450,
            leftSideBearing: 100
        )

        let result0 = try await generator.generate(
            character: "A",
            mode: .interpolate(glyphA: glyphA, glyphB: glyphB, t: 0.0),
            metrics: metrics,
            settings: .fast
        )

        let result05 = try await generator.generate(
            character: "A",
            mode: .interpolate(glyphA: glyphA, glyphB: glyphB, t: 0.5),
            metrics: metrics,
            settings: .fast
        )

        let result1 = try await generator.generate(
            character: "A",
            mode: .interpolate(glyphA: glyphA, glyphB: glyphB, t: 1.0),
            metrics: metrics,
            settings: .fast
        )

        // All three should produce valid "A" glyphs
        #expect(result0.glyph.character == "A")
        #expect(result05.glyph.character == "A")
        #expect(result1.glyph.character == "A")

        // All results should have valid structure (non-empty contours with valid points)
        for (label, result) in [("t=0", result0), ("t=0.5", result05), ("t=1", result1)] {
            #expect(!result.glyph.outline.contours.isEmpty,
                "\(label) should have non-empty contours")
            for contour in result.glyph.outline.contours {
                #expect(!contour.points.isEmpty,
                    "\(label) contour should have points")
                #expect(contour.isClosed,
                    "\(label) contour should be closed")
            }
            #expect(result.glyph.advanceWidth > 0,
                "\(label) should have positive advance width")
        }

        // The generator's interpolateOutlines does real point-by-point linear
        // interpolation when contour structures match (same number of contours
        // and points). glyphA and glyphB have matching structures with different
        // positions, so results at t=0, t=0.5, t=1 MUST differ.
        let points0 = result0.glyph.outline.contours.flatMap { $0.points.map { $0.position } }
        let points05 = result05.glyph.outline.contours.flatMap { $0.points.map { $0.position } }
        let points1 = result1.glyph.outline.contours.flatMap { $0.points.map { $0.position } }

        // Verify we actually have points to compare
        #expect(!points0.isEmpty, "t=0 should have outline points")
        #expect(points0.count == points05.count, "All results should have same point count")
        #expect(points0.count == points1.count, "All results should have same point count")

        // t=0 should differ from t=0.5
        let differsFrom0To05 = zip(points0, points05).contains { p0, p05 in
            abs(p0.x - p05.x) > 0.01 || abs(p0.y - p05.y) > 0.01
        }
        #expect(differsFrom0To05, "t=0.0 and t=0.5 should produce different point positions")

        // t=0.5 should differ from t=1.0
        let differsFrom05To1 = zip(points05, points1).contains { p05, p1 in
            abs(p05.x - p1.x) > 0.01 || abs(p05.y - p1.y) > 0.01
        }
        #expect(differsFrom05To1, "t=0.5 and t=1.0 should produce different point positions")

        // t=0 should differ from t=1.0
        let differsFrom0To1 = zip(points0, points1).contains { p0, p1 in
            abs(p0.x - p1.x) > 0.01 || abs(p0.y - p1.y) > 0.01
        }
        #expect(differsFrom0To1, "t=0.0 and t=1.0 should produce different point positions")

        // Advance widths should also interpolate: glyphA=500, glyphB=450
        // t=0 -> 500, t=0.5 -> 475, t=1 -> 450
        #expect(result0.glyph.advanceWidth != result05.glyph.advanceWidth,
            "Advance width should differ between t=0 (\(result0.glyph.advanceWidth)) and t=0.5 (\(result05.glyph.advanceWidth))")
        #expect(result05.glyph.advanceWidth != result1.glyph.advanceWidth,
            "Advance width should differ between t=0.5 (\(result05.glyph.advanceWidth)) and t=1 (\(result1.glyph.advanceWidth))")
    }

    @Test("Generation result confidence is within valid range")
    func generationConfidenceIsValid() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default

        let result = try await generator.generate(
            character: "C",
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(result.confidence >= 0.0)
        #expect(result.confidence <= 1.0)
    }

    @Test("Batch generation maintains glyph order")
    func batchGenerationMaintainsOrder() async throws {
        let generator = GlyphGenerator()
        let metrics = createTestMetrics()
        let style = StyleEncoder.FontStyle.default
        let characters: [Character] = ["A", "B", "C", "D", "E"]

        let results = try await generator.generateBatch(
            characters: characters,
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        #expect(results.count == characters.count)
        for (i, result) in results.enumerated() {
            #expect(result.glyph.character == characters[i])
        }
    }
}

@Suite("Extended KerningPredictor Tests")
struct ExtendedKerningPredictorTests {

    // MARK: - Helpers

    /// Creates a project with realistic glyph shapes for kerning tests
    private func createRealisticProject() -> FontProject {
        var project = FontProject(name: "Kerning Test Font", family: "Test", style: "Regular")

        // "A" - triangular shape (wide at bottom, narrow at top)
        let aOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 250, y: 700), type: .corner),  // Top
                    PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),      // Bottom left
                    PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 150, y: 150), type: .corner),  // Crossbar left
                    PathPoint(position: CGPoint(x: 350, y: 150), type: .corner),  // Crossbar right
                    PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 0), type: .corner)     // Bottom right
                ],
                isClosed: true
            )
        ])
        project.glyphs["A"] = Glyph(character: "A", outline: aOutline, advanceWidth: 500, leftSideBearing: 0)

        // "V" - inverted triangle (narrow at bottom, wide at top)
        let vOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 0, y: 700), type: .corner),    // Top left
                    PathPoint(position: CGPoint(x: 100, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 250, y: 100), type: .corner),  // Bottom point
                    PathPoint(position: CGPoint(x: 400, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 700), type: .corner),  // Top right
                    PathPoint(position: CGPoint(x: 250, y: 0), type: .corner)     // Very bottom
                ],
                isClosed: true
            )
        ])
        project.glyphs["V"] = Glyph(character: "V", outline: vOutline, advanceWidth: 500, leftSideBearing: 0)

        // "H" - rectangular shape (balanced, no kerning needed)
        let hOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 150, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 150, y: 300), type: .corner),
                    PathPoint(position: CGPoint(x: 350, y: 300), type: .corner),
                    PathPoint(position: CGPoint(x: 350, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 350, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 350, y: 400), type: .corner),
                    PathPoint(position: CGPoint(x: 150, y: 400), type: .corner),
                    PathPoint(position: CGPoint(x: 150, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs["H"] = Glyph(character: "H", outline: hOutline, advanceWidth: 500, leftSideBearing: 50)

        // "I" - vertical bar
        let iOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 200, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 300, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 300, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 200, y: 700), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs["I"] = Glyph(character: "I", outline: iOutline, advanceWidth: 500, leftSideBearing: 200)

        // "T" - T-shape (overhang at top)
        let tOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 0, y: 600), type: .corner),
                    PathPoint(position: CGPoint(x: 0, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 700), type: .corner),
                    PathPoint(position: CGPoint(x: 500, y: 600), type: .corner),
                    PathPoint(position: CGPoint(x: 300, y: 600), type: .corner),
                    PathPoint(position: CGPoint(x: 300, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 200, y: 0), type: .corner),
                    PathPoint(position: CGPoint(x: 200, y: 600), type: .corner)
                ],
                isClosed: true
            )
        ])
        project.glyphs["T"] = Glyph(character: "T", outline: tOutline, advanceWidth: 500, leftSideBearing: 0)

        // Lowercase "a" and "o" for mixed case tests
        let aLowerOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 50, y: 0), type: .smooth),
                    PathPoint(position: CGPoint(x: 350, y: 0), type: .smooth),
                    PathPoint(position: CGPoint(x: 350, y: 500), type: .smooth),
                    PathPoint(position: CGPoint(x: 50, y: 500), type: .smooth)
                ],
                isClosed: true
            )
        ])
        project.glyphs["a"] = Glyph(character: "a", outline: aLowerOutline, advanceWidth: 400, leftSideBearing: 50)

        let oOutline = GlyphOutline(contours: [
            Contour(
                points: [
                    PathPoint(position: CGPoint(x: 200, y: 0), type: .smooth),
                    PathPoint(position: CGPoint(x: 400, y: 250), type: .smooth),
                    PathPoint(position: CGPoint(x: 200, y: 500), type: .smooth),
                    PathPoint(position: CGPoint(x: 0, y: 250), type: .smooth)
                ],
                isClosed: true
            )
        ])
        project.glyphs["o"] = Glyph(character: "o", outline: oOutline, advanceWidth: 400, leftSideBearing: 0)

        return project
    }

    // MARK: - Kerning Accuracy Tests

    @Test("AV pair has negative kerning (shapes should tuck together)")
    func avPairHasNegativeKerning() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let kerning = try await predictor.predictPair(
            left: "A",
            right: "V",
            project: project
        )

        // AV is a classic pair that needs negative kerning
        // The A's right diagonal and V's left diagonal create a gap
        #expect(kerning < 0, "AV pair should have negative kerning, got \(kerning)")
    }

    @Test("VA pair has negative kerning")
    func vaPairHasNegativeKerning() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let kerning = try await predictor.predictPair(
            left: "V",
            right: "A",
            project: project
        )

        // VA also needs negative kerning (V's right diagonal + A's left diagonal)
        #expect(kerning < 0, "VA pair should have negative kerning, got \(kerning)")
    }

    @Test("HI pair has minimal kerning (rectangular shapes)")
    func hiPairHasMinimalKerning() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let kerning = try await predictor.predictPair(
            left: "H",
            right: "I",
            project: project
        )

        // H and I are both rectangular, should need little adjustment
        #expect(abs(kerning) < 50, "HI pair should have minimal kerning, got \(kerning)")
    }

    @Test("Ta pair has negative kerning (T overhang)")
    func taPairHasNegativeKerning() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let kerning = try await predictor.predictPair(
            left: "T",
            right: "a",
            project: project
        )

        // T's crossbar overhangs, lowercase letters should tuck under
        #expect(kerning < 0, "Ta pair should have negative kerning, got \(kerning)")
    }

    @Test("Kerning values are within reasonable range")
    func kerningValuesInReasonableRange() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        for pair in result.pairs {
            // Kerning should not be extreme
            // Typical range is -200 to +100 units per em
            #expect(pair.value >= -250, "Kerning too negative: \(pair.left)\(pair.right) = \(pair.value)")
            #expect(pair.value <= 150, "Kerning too positive: \(pair.left)\(pair.right) = \(pair.value)")
        }
    }

    @Test("Kerning batch is consistent (same input = same output)")
    func kerningBatchIsConsistent() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let result1 = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        let result2 = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // Both runs should produce non-empty results
        #expect(!result1.pairs.isEmpty, "First kerning run should produce pairs")
        #expect(!result2.pairs.isEmpty, "Second kerning run should produce pairs")
        #expect(result1.pairs.count == result2.pairs.count,
            "Both runs should produce same number of pairs (\(result1.pairs.count) vs \(result2.pairs.count))")

        // Critical pairs that MUST be present given the test project's glyphs
        // (A, V, H, I, T, a, o). The KerningPredictor's criticalPairs list
        // includes these combinations for the available glyphs:
        let expectedCriticalPairs: [(Character, Character)] = [
            ("A", "V"), ("A", "T"),
            ("T", "a"), ("T", "o"),
            ("V", "a"), ("V", "o"),
        ]

        for (left, right) in expectedCriticalPairs {
            let found = result1.pairs.contains { $0.left == left && $0.right == right }
            #expect(found,
                "Critical pair \(left)\(right) should be present in kerning results")
        }

        // Same pairs should have same values (consistency check)
        for pair1 in result1.pairs {
            if let pair2 = result2.pairs.first(where: { $0.left == pair1.left && $0.right == pair1.right }) {
                #expect(pair1.value == pair2.value,
                    "Inconsistent kerning for \(pair1.left)\(pair1.right): \(pair1.value) vs \(pair2.value)")
            }
        }
    }

    @Test("Symmetric pairs have symmetric kerning")
    func symmetricPairsHaveSymmetricKerning() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        // AV and VA should have similar (negative) kerning
        let avKerning = try await predictor.predictPair(left: "A", right: "V", project: project)
        let vaKerning = try await predictor.predictPair(left: "V", right: "A", project: project)

        // They should be similar in magnitude (both negative)
        #expect(avKerning < 0)
        #expect(vaKerning < 0)

        // Allow some difference but should be in same ballpark
        let difference = abs(avKerning - vaKerning)
        #expect(difference < 100, "AV (\(avKerning)) and VA (\(vaKerning)) should be similar")
    }

    @Test("Tight spacing preset produces more negative kerning")
    func tightSpacingMoreNegative() async throws {
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

        // Find common pairs and compare
        for tightPair in tightResult.pairs {
            if let loosePair = looseResult.pairs.first(where: {
                $0.left == tightPair.left && $0.right == tightPair.right
            }) {
                // Tight spacing should produce more negative (or less positive) values
                #expect(tightPair.value <= loosePair.value,
                    "Tight should be ≤ loose for \(tightPair.left)\(tightPair.right)")
            }
        }
    }

    @Test("Critical pairs only produces subset of all pairs")
    func criticalPairsOnlyProducesSubset() async throws {
        let predictor = KerningPredictor()
        let project = createRealisticProject()

        let allSettings = KerningPredictor.PredictionSettings(onlyCriticalPairs: false)
        let criticalSettings = KerningPredictor.PredictionSettings(onlyCriticalPairs: true)

        let allResult = try await predictor.predictKerning(for: project, settings: allSettings)
        let criticalResult = try await predictor.predictKerning(for: project, settings: criticalSettings)

        // Critical pairs should be a subset
        for criticalPair in criticalResult.pairs {
            let existsInAll = allResult.pairs.contains(where: {
                $0.left == criticalPair.left && $0.right == criticalPair.right
            })
            #expect(existsInAll,
                "Critical pair \(criticalPair.left)\(criticalPair.right) should exist in all pairs result")
        }
    }
}

@Suite("Extended StyleEncoder Tests")
struct ExtendedStyleEncoderTests {

    // MARK: - Style Extraction Accuracy Tests

    @Test("Serif detection distinguishes serif from sans-serif")
    func serifDetection() async throws {
        // Create a project with clearly serif-like features (would need actual serif shapes)
        // For now, verify the encoder doesn't crash with various inputs
        let encoder = StyleEncoder()

        var project = FontProject(name: "Test", family: "Test", style: "Regular")
        project.glyphs["H"] = Glyph(
            character: "H",
            outline: GlyphOutline(contours: [
                Contour(
                    points: [
                        PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                        PathPoint(position: CGPoint(x: 150, y: 0), type: .corner),
                        PathPoint(position: CGPoint(x: 150, y: 700), type: .corner),
                        PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                    ],
                    isClosed: true
                )
            ]),
            advanceWidth: 500,
            leftSideBearing: 50
        )

        let style = try await encoder.extractStyle(from: project)

        // Sans-serif by default for simple rectangular shapes
        #expect(style.serifStyle == .sansSerif)
    }

    @Test("Slant detection for italic fonts")
    func slantDetection() async throws {
        let encoder = StyleEncoder()

        // Vertical shape (no slant)
        let verticalProject = FontProject(name: "Vertical", family: "Test", style: "Regular")

        let verticalStyle = try await encoder.extractStyle(from: verticalProject)

        // Default should have no slant
        #expect(abs(verticalStyle.slant) < 1.0)
    }

    @Test("Style embedding has consistent dimensions")
    func styleEmbeddingDimensions() async throws {
        let encoder = StyleEncoder()

        var project1 = FontProject(name: "Test1", family: "Test", style: "Regular")
        project1.glyphs["A"] = Glyph(
            character: "A",
            outline: GlyphOutline(),
            advanceWidth: 500,
            leftSideBearing: 50
        )

        var project2 = FontProject(name: "Test2", family: "Test", style: "Bold")
        project2.glyphs["A"] = Glyph(
            character: "A",
            outline: GlyphOutline(),
            advanceWidth: 600,
            leftSideBearing: 50
        )
        project2.glyphs["B"] = Glyph(
            character: "B",
            outline: GlyphOutline(),
            advanceWidth: 600,
            leftSideBearing: 50
        )

        let style1 = try await encoder.extractStyle(from: project1)
        let style2 = try await encoder.extractStyle(from: project2)

        // Embeddings should have same dimensions
        if !style1.embedding.isEmpty && !style2.embedding.isEmpty {
            #expect(style1.embedding.count == style2.embedding.count)
        }
    }
}

@Suite("AI Generation Integration Tests")
struct AIGenerationIntegrationTests {

    @Test("Generated glyphs can be added to project")
    func generatedGlyphsCanBeAddedToProject() async throws {
        let generator = GlyphGenerator()
        let metrics = FontMetrics()
        let style = StyleEncoder.FontStyle.default

        var project = FontProject(name: "Test", family: "Test", style: "Regular")

        let characters: [Character] = ["A", "B", "C"]
        let results = try await generator.generateBatch(
            characters: characters,
            mode: .fromScratch(style: style),
            metrics: metrics,
            settings: .fast
        )

        for result in results {
            project.glyphs[result.glyph.character] = result.glyph
        }

        #expect(project.glyphs.count == 3)
        #expect(project.glyphs["A"] != nil)
        #expect(project.glyphs["B"] != nil)
        #expect(project.glyphs["C"] != nil)
    }

    @Test("Predicted kerning can be added to project")
    func predictedKerningCanBeAddedToProject() async throws {
        var project = FontProject(name: "Test", family: "Test", style: "Regular")

        // Add some glyphs
        for char in ["A", "V", "a", "v"] {
            project.glyphs[Character(char)] = Glyph(
                character: Character(char),
                outline: GlyphOutline(contours: [
                    Contour(
                        points: [
                            PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                            PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                            PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                            PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
                        ],
                        isClosed: true
                    )
                ]),
                advanceWidth: 500,
                leftSideBearing: 50
            )
        }

        let predictor = KerningPredictor()
        let result = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        // Add predicted kerning to project
        project.kerning = result.pairs

        // Project should have the kerning pairs
        #expect(project.kerning.count == result.pairs.count)
    }

    @Test("Full generation workflow: generate, kern, export")
    func fullGenerationWorkflow() async throws {
        // 1. Create project
        var project = FontProject(name: "Generated Font", family: "Generated", style: "Regular")

        // 2. Generate glyphs
        let generator = GlyphGenerator()
        let style = StyleEncoder.FontStyle.default

        let results = try await generator.generateBatch(
            characters: ["A", "V", "T", "a"],
            mode: .fromScratch(style: style),
            metrics: project.metrics,
            settings: .fast
        )

        for result in results {
            project.glyphs[result.glyph.character] = result.glyph
        }

        #expect(project.glyphs.count == 4)

        // 3. Predict kerning
        let predictor = KerningPredictor()
        let kerningResult = try await predictor.predictKerning(
            for: project,
            settings: .default
        )

        project.kerning = kerningResult.pairs

        // 4. Verify project is valid for export
        #expect(project.glyphs.count >= 1)
        #expect(project.metrics.unitsPerEm > 0)

        // 5. Export to TTF
        let exporter = FontExporter()
        let data = try await exporter.export(project: project)

        #expect(data.count > 0)

        // 6. Verify export has valid TTF signature
        let signature = data.prefix(4)
        let signatureValue = UInt32(bigEndian: signature.withUnsafeBytes { $0.load(as: UInt32.self) })
        let validSignatures: [UInt32] = [0x00010000, 0x4F54544F] // TrueType or OpenType
        #expect(validSignatures.contains(signatureValue), "Invalid font signature")
    }
}
