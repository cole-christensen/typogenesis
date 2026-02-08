import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

// MARK: - Glyph Editor E2E Tests

/// Single comprehensive E2E test for the glyph editing workflow.
/// Tests the complete cycle from creation through editing to export.

@Suite("Glyph Editor E2E Tests")
struct GlyphEditorE2ETests {

    @Test("Complete glyph editing workflow from creation to export")
    func completeGlyphEditingWorkflow() async throws {
        // =====================================================
        // PHASE 1: Create a new font project
        // =====================================================
        var project = FontProject(name: "E2E Test Font", family: "E2ETest", style: "Regular")

        // Configure metrics
        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200
        project.metrics.capHeight = 700
        project.metrics.xHeight = 500
        project.metrics.lineGap = 90

        #expect(project.name == "E2E Test Font")
        #expect(project.metrics.unitsPerEm == 1000)

        // =====================================================
        // PHASE 2: Create glyphs with various outline types
        // =====================================================

        // Create a simple rectangular glyph (like 'H')
        let hOutline = GlyphOutline(contours: [
            Contour(points: [
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
            ], isClosed: true)
        ])
        project.glyphs["H"] = Glyph(character: "H", outline: hOutline, advanceWidth: 500, leftSideBearing: 50)

        // Create a glyph with multiple contours (like 'O' with counter)
        let oOuterContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: 50, y: 350), type: .smooth),
                PathPoint(position: CGPoint(x: 250, y: 700), type: .smooth),
                PathPoint(position: CGPoint(x: 450, y: 350), type: .smooth),
                PathPoint(position: CGPoint(x: 250, y: 0), type: .smooth)
            ],
            isClosed: true
        )
        let oInnerContour = Contour(
            points: [
                PathPoint(position: CGPoint(x: 150, y: 350), type: .smooth),
                PathPoint(position: CGPoint(x: 250, y: 550), type: .smooth),
                PathPoint(position: CGPoint(x: 350, y: 350), type: .smooth),
                PathPoint(position: CGPoint(x: 250, y: 150), type: .smooth)
            ],
            isClosed: true
        )
        let oOutline = GlyphOutline(contours: [oOuterContour, oInnerContour])
        project.glyphs["O"] = Glyph(character: "O", outline: oOutline, advanceWidth: 500, leftSideBearing: 50)

        // Create a glyph with curve control points
        let sContour = Contour(
            points: [
                PathPoint(
                    position: CGPoint(x: 400, y: 600),
                    type: .smooth,
                    controlIn: CGPoint(x: 450, y: 650),
                    controlOut: CGPoint(x: 350, y: 550)
                ),
                PathPoint(
                    position: CGPoint(x: 250, y: 500),
                    type: .smooth,
                    controlIn: CGPoint(x: 350, y: 500),
                    controlOut: CGPoint(x: 150, y: 500)
                ),
                PathPoint(
                    position: CGPoint(x: 100, y: 400),
                    type: .smooth,
                    controlIn: CGPoint(x: 50, y: 450),
                    controlOut: CGPoint(x: 150, y: 350)
                ),
                PathPoint(
                    position: CGPoint(x: 250, y: 350),
                    type: .smooth,
                    controlIn: CGPoint(x: 150, y: 350),
                    controlOut: CGPoint(x: 350, y: 350)
                ),
                PathPoint(
                    position: CGPoint(x: 400, y: 250),
                    type: .smooth,
                    controlIn: CGPoint(x: 450, y: 300),
                    controlOut: CGPoint(x: 350, y: 200)
                ),
                PathPoint(
                    position: CGPoint(x: 250, y: 150),
                    type: .smooth,
                    controlIn: CGPoint(x: 350, y: 150),
                    controlOut: CGPoint(x: 150, y: 150)
                ),
                PathPoint(
                    position: CGPoint(x: 100, y: 100),
                    type: .smooth,
                    controlIn: CGPoint(x: 50, y: 50),
                    controlOut: CGPoint(x: 150, y: 150)
                )
            ],
            isClosed: false
        )
        project.glyphs["S"] = Glyph(character: "S", outline: GlyphOutline(contours: [sContour]), advanceWidth: 450, leftSideBearing: 50)

        #expect(project.glyphs.count == 3)
        #expect(project.glyphs["H"]?.outline.contours.count == 1)
        #expect(project.glyphs["O"]?.outline.contours.count == 2)

        // =====================================================
        // PHASE 3: Edit glyph outlines
        // =====================================================

        // Add a point to H glyph
        var hGlyph = project.glyphs["H"]!
        var hContour = hGlyph.outline.contours[0]
        let originalPointCount = hContour.points.count
        hContour.points.insert(
            PathPoint(position: CGPoint(x: 250, y: 350), type: .corner),
            at: 5
        )
        hGlyph.outline.contours[0] = hContour
        project.glyphs["H"] = hGlyph

        #expect(project.glyphs["H"]!.outline.contours[0].points.count == originalPointCount + 1)

        // Modify a point position
        project.glyphs["H"]!.outline.contours[0].points[0].position = CGPoint(x: 60, y: 0)
        #expect(project.glyphs["H"]!.outline.contours[0].points[0].position.x == 60)

        // Change point type
        project.glyphs["H"]!.outline.contours[0].points[0].type = .smooth
        #expect(project.glyphs["H"]!.outline.contours[0].points[0].type == .smooth)

        // Modify advance width
        project.glyphs["H"]!.advanceWidth = 520
        #expect(project.glyphs["H"]!.advanceWidth == 520)

        // Modify left side bearing
        project.glyphs["H"]!.leftSideBearing = 60
        #expect(project.glyphs["H"]!.leftSideBearing == 60)

        // =====================================================
        // PHASE 4: Test bounding box calculations
        // =====================================================

        let oBbox = project.glyphs["O"]!.outline.boundingBox
        #expect(oBbox.minX == 50)
        #expect(oBbox.maxX == 450)
        #expect(oBbox.width == 400)
        #expect(oBbox.height > 0)

        // =====================================================
        // PHASE 5: Test path operations
        // =====================================================

        // Create two overlapping outlines for union test
        let rect1 = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 100), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 100), type: .corner)
            ], isClosed: true)
        ])
        let rect2 = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 50, y: 50), type: .corner),
                PathPoint(position: CGPoint(x: 150, y: 50), type: .corner),
                PathPoint(position: CGPoint(x: 150, y: 150), type: .corner),
                PathPoint(position: CGPoint(x: 50, y: 150), type: .corner)
            ], isClosed: true)
        ])

        let unionResult = try PathOperations.perform(.union, on: rect1, with: rect2)
        #expect(!unionResult.isEmpty)

        // Test simplification
        let complexOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 25, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 75, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 100), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 100), type: .corner)
            ], isClosed: true)
        ])
        let simplified = PathOperations.simplify(complexOutline, tolerance: 1.0)
        #expect(simplified.contours[0].points.count <= complexOutline.contours[0].points.count)

        // =====================================================
        // PHASE 6: Convert to CGPath for rendering
        // =====================================================

        let cgPath = project.glyphs["O"]!.outline.cgPath
        #expect(!cgPath.isEmpty)
        #expect(cgPath.boundingBox.width > 0)
        #expect(cgPath.boundingBox.height > 0)

        // =====================================================
        // PHASE 7: Test GlyphEditorViewModel
        // =====================================================

        let glyph = project.glyphs["H"]!
        let viewModel = await MainActor.run {
            GlyphEditorViewModel(glyph: glyph)
        }

        let vmGlyph = await MainActor.run { viewModel.glyph }
        #expect(vmGlyph.character == "H")
        #expect(vmGlyph.advanceWidth == 520)

        // =====================================================
        // PHASE 8: Add kerning pairs
        // =====================================================

        project.kerning = [
            KerningPair(left: "H", right: "O", value: -20),
            KerningPair(left: "O", right: "S", value: -15)
        ]

        #expect(project.kerning.count == 2)
        let hoPair = project.kerning.first { $0.left == "H" && $0.right == "O" }
        #expect(hoPair?.value == -20)

        // Modify kerning
        if let index = project.kerning.firstIndex(where: { $0.left == "H" && $0.right == "O" }) {
            project.kerning[index] = KerningPair(left: "H", right: "O", value: -25)
        }
        let updatedPair = project.kerning.first { $0.left == "H" && $0.right == "O" }
        #expect(updatedPair?.value == -25)

        // =====================================================
        // PHASE 9: AI kerning prediction
        // =====================================================

        let predictor = KerningPredictor()
        let predictionResult = try await predictor.predictKerning(for: project)
        // Geometric fallback returns a confidence in (0, 1) (no AI model loaded).
        #expect(predictionResult.confidence > 0 && predictionResult.confidence < 1, "Geometric fallback confidence should be in (0, 1), got \(predictionResult.confidence)")

        // =====================================================
        // PHASE 10: Export the font
        // =====================================================

        let exporter = FontExporter()
        let ttfData = try await exporter.export(project: project)
        #expect(!ttfData.isEmpty)

        // Export to WOFF
        let webExporter = WebFontExporter()
        let woffData = try await webExporter.exportWOFF(ttfData: ttfData)
        #expect(!woffData.isEmpty)

        // Verify WOFF signature
        let signature = woffData.prefix(4)
        let signatureValue = signature.withUnsafeBytes { $0.load(as: UInt32.self) }
        let bigEndianSignature = UInt32(bigEndian: signatureValue)
        #expect(bigEndianSignature == 0x774F4646, "Should have WOFF signature")

        // Export to UFO
        let ufoExporter = UFOExporter()
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_\(UUID().uuidString).ufo")
        try await ufoExporter.export(project: project, to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))
        try? FileManager.default.removeItem(at: tempURL)

        // =====================================================
        // VERIFICATION: All phases completed successfully
        // =====================================================

        #expect(project.glyphs.count == 3)
        #expect(project.kerning.count == 2)
        #expect(project.metrics.unitsPerEm == 1000)
    }
}

// MARK: - Kerning Editor E2E Tests

@Suite("Kerning Editor E2E Tests")
struct KerningEditorE2ETests {

    @Test("Complete kerning workflow from creation to prediction")
    func completeKerningWorkflow() async throws {
        // =====================================================
        // PHASE 1: Create font with glyphs
        // =====================================================
        var project = FontProject(name: "Kerning Test", family: "KernTest", style: "Regular")

        // Add glyphs with differentiated shapes that produce meaningful kerning values.
        // Identical rectangles produce zero kerning for all pairs, making the test vacuous.

        // "A" - triangular shape
        let aOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 250, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 250, y: 500), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 500, y: 0), type: .corner)
            ], isClosed: true)
        ])
        project.glyphs["A"] = Glyph(character: "A", outline: aOutline, advanceWidth: 500, leftSideBearing: 0)

        // "V" - inverted triangle
        let vOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 250, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 500, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 250, y: 200), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 700), type: .corner)
            ], isClosed: true)
        ])
        project.glyphs["V"] = Glyph(character: "V", outline: vOutline, advanceWidth: 500, leftSideBearing: 0)

        // "T" - T-bar shape
        let tOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 500, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 500, y: 650), type: .corner),
                PathPoint(position: CGPoint(x: 300, y: 650), type: .corner),
                PathPoint(position: CGPoint(x: 300, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 200, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 200, y: 650), type: .corner),
                PathPoint(position: CGPoint(x: 0, y: 650), type: .corner)
            ], isClosed: true)
        ])
        project.glyphs["T"] = Glyph(character: "T", outline: tOutline, advanceWidth: 500, leftSideBearing: 0)

        // "o" - oval
        let oOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 250, y: 500), type: .smooth),
                PathPoint(position: CGPoint(x: 450, y: 250), type: .smooth),
                PathPoint(position: CGPoint(x: 250, y: 0), type: .smooth),
                PathPoint(position: CGPoint(x: 50, y: 250), type: .smooth)
            ], isClosed: true)
        ])
        project.glyphs["o"] = Glyph(character: "o", outline: oOutline, advanceWidth: 500, leftSideBearing: 50)

        // "a" - lowercase a
        let aLowerOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 400, y: 500), type: .corner),
                PathPoint(position: CGPoint(x: 50, y: 500), type: .smooth),
                PathPoint(position: CGPoint(x: 50, y: 250), type: .smooth)
            ], isClosed: true)
        ])
        project.glyphs["a"] = Glyph(character: "a", outline: aLowerOutline, advanceWidth: 450, leftSideBearing: 50)

        // "W" - double-V shape
        let wOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 125, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 250, y: 500), type: .corner),
                PathPoint(position: CGPoint(x: 375, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 500, y: 700), type: .corner)
            ], isClosed: true)
        ])
        project.glyphs["W"] = Glyph(character: "W", outline: wOutline, advanceWidth: 600, leftSideBearing: 0)

        // "Y" - Y shape
        let yOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 0, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 200, y: 350), type: .corner),
                PathPoint(position: CGPoint(x: 200, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 300, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 300, y: 350), type: .corner),
                PathPoint(position: CGPoint(x: 500, y: 700), type: .corner)
            ], isClosed: true)
        ])
        project.glyphs["Y"] = Glyph(character: "Y", outline: yOutline, advanceWidth: 500, leftSideBearing: 0)

        #expect(project.glyphs.count == 7)

        // =====================================================
        // PHASE 2: Add manual kerning pairs
        // =====================================================

        let manualPairs = [
            ("A", "V", -50),
            ("V", "A", -50),
            ("T", "o", -30),
            ("A", "T", -20),
            ("A", "W", -40),
            ("W", "a", -25),
            ("Y", "o", -35)
        ]

        for (left, right, value) in manualPairs {
            project.kerning.append(KerningPair(left: Character(left), right: Character(right), value: value))
        }

        #expect(project.kerning.count == 7)

        // =====================================================
        // PHASE 3: Query and modify kerning
        // =====================================================

        // Find specific pair
        let avPair = project.kerning.first { $0.left == "A" && $0.right == "V" }
        #expect(avPair != nil)
        #expect(avPair?.value == -50)

        // Modify a pair
        if let index = project.kerning.firstIndex(where: { $0.left == "A" && $0.right == "V" }) {
            project.kerning[index] = KerningPair(left: "A", right: "V", value: -60)
        }

        let updatedAV = project.kerning.first { $0.left == "A" && $0.right == "V" }
        #expect(updatedAV?.value == -60)

        // Remove a pair
        project.kerning.removeAll { $0.left == "Y" && $0.right == "o" }
        #expect(project.kerning.count == 6)

        // =====================================================
        // PHASE 4: Calculate text width with kerning
        // =====================================================

        // "AV" without kerning
        let aWidth = project.glyphs["A"]!.advanceWidth
        let vWidth = project.glyphs["V"]!.advanceWidth
        let widthWithoutKerning = aWidth + vWidth

        // With kerning
        let avKern = project.kerning.first { $0.left == "A" && $0.right == "V" }?.value ?? 0
        let widthWithKerning = widthWithoutKerning + avKern

        #expect(widthWithKerning < widthWithoutKerning)
        #expect(widthWithKerning == 940) // 500 + 500 - 60

        // =====================================================
        // PHASE 5: AI kerning prediction
        // =====================================================

        let predictor = KerningPredictor()
        let result = try await predictor.predictKerning(for: project)

        // Geometric fallback returns a confidence in (0, 1) (no AI model loaded).
        #expect(result.confidence > 0 && result.confidence < 1, "Geometric fallback confidence should be in (0, 1), got \(result.confidence)")
        // Prediction should take measurable time
        #expect(result.predictionTime > 0, "Prediction should take measurable time, got \(result.predictionTime)")

        // =====================================================
        // PHASE 6: Export with kerning
        // =====================================================

        let exporter = FontExporter()
        let ttfData = try await exporter.export(project: project)
        #expect(!ttfData.isEmpty)
    }
}

// MARK: - Metrics Editor E2E Tests

@Suite("Metrics Editor E2E Tests")
struct MetricsEditorE2ETests {

    @Test("Complete metrics configuration workflow")
    func completeMetricsWorkflow() async throws {
        // =====================================================
        // PHASE 1: Create project with default metrics
        // =====================================================
        var project = FontProject(name: "Metrics Test", family: "MetricsTest", style: "Regular")

        // Default metrics should be set
        #expect(project.metrics.unitsPerEm == 1000)

        // =====================================================
        // PHASE 2: Configure all metrics
        // =====================================================

        project.metrics.unitsPerEm = 2048
        project.metrics.ascender = 1638
        project.metrics.descender = -410
        project.metrics.capHeight = 1434
        project.metrics.xHeight = 1024
        project.metrics.lineGap = 184

        #expect(project.metrics.unitsPerEm == 2048)
        #expect(project.metrics.ascender == 1638)
        #expect(project.metrics.descender == -410)
        #expect(project.metrics.capHeight == 1434)
        #expect(project.metrics.xHeight == 1024)
        #expect(project.metrics.lineGap == 184)

        // Verify total height
        let totalHeight = project.metrics.ascender - project.metrics.descender
        #expect(totalHeight == 2048) // Should equal unitsPerEm

        // =====================================================
        // PHASE 3: Add glyphs that respect metrics
        // =====================================================

        // Capital H at cap height
        let hOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 300, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 300, y: 1434), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 1434), type: .corner)
            ], isClosed: true)
        ])
        project.glyphs["H"] = Glyph(character: "H", outline: hOutline, advanceWidth: 1400, leftSideBearing: 100)

        // Lowercase x at x-height
        let xOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 100, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 900, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 900, y: 1024), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 1024), type: .corner)
            ], isClosed: true)
        ])
        project.glyphs["x"] = Glyph(character: "x", outline: xOutline, advanceWidth: 1000, leftSideBearing: 100)

        // Descender glyph (p)
        let pOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 100, y: -410), type: .corner),
                PathPoint(position: CGPoint(x: 300, y: -410), type: .corner),
                PathPoint(position: CGPoint(x: 300, y: 1024), type: .corner),
                PathPoint(position: CGPoint(x: 100, y: 1024), type: .corner)
            ], isClosed: true)
        ])
        project.glyphs["p"] = Glyph(character: "p", outline: pOutline, advanceWidth: 1100, leftSideBearing: 100)

        // =====================================================
        // PHASE 4: Verify glyphs fit within metrics
        // =====================================================

        let hBbox = project.glyphs["H"]!.outline.boundingBox
        #expect(hBbox.maxY <= project.metrics.capHeight + 50) // Small tolerance
        #expect(hBbox.minY >= project.metrics.descender)

        let xBbox = project.glyphs["x"]!.outline.boundingBox
        #expect(xBbox.maxY <= project.metrics.xHeight + 50)

        let pBbox = project.glyphs["p"]!.outline.boundingBox
        #expect(pBbox.minY >= project.metrics.descender - 50)

        // =====================================================
        // PHASE 5: Change metrics and verify impact
        // =====================================================

        // Switch to smaller unitsPerEm
        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200
        project.metrics.capHeight = 700
        project.metrics.xHeight = 500
        project.metrics.lineGap = 90

        // Note: In real usage, glyphs would need to be scaled
        // Here we're just testing that metrics can be modified
        #expect(project.metrics.unitsPerEm == 1000)

        // =====================================================
        // PHASE 6: Export with configured metrics
        // =====================================================

        let exporter = FontExporter()
        let ttfData = try await exporter.export(project: project)
        #expect(!ttfData.isEmpty)

        // Export to UFO (preserves metrics)
        let ufoExporter = UFOExporter()
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_metrics_\(UUID().uuidString).ufo")
        try await ufoExporter.export(project: project, to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))
        try? FileManager.default.removeItem(at: tempURL)
    }
}
