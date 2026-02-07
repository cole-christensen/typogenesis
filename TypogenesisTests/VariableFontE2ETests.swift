import Testing
import Foundation
import CoreGraphics
@testable import Typogenesis

// MARK: - Variable Font Editor E2E Tests

/// Single comprehensive E2E test for the variable font workflow.
/// Tests the complete cycle from axis configuration through master creation to export.

@Suite("Variable Font Editor E2E Tests")
struct VariableFontEditorE2ETests {

    @Test("Complete variable font workflow from configuration to export")
    func completeVariableFontWorkflow() async throws {
        // =====================================================
        // PHASE 1: Create a font project for variable font
        // =====================================================
        var project = FontProject(name: "Variable E2E Font", family: "VarE2E", style: "Regular")

        project.metrics.unitsPerEm = 1000
        project.metrics.ascender = 800
        project.metrics.descender = -200
        project.metrics.capHeight = 700
        project.metrics.xHeight = 500

        #expect(project.name == "Variable E2E Font")

        // =====================================================
        // PHASE 2: Configure variable font axes
        // =====================================================

        // Add weight axis
        let weightAxis = VariationAxis(
            tag: "wght",
            name: "Weight",
            minValue: 100,
            defaultValue: 400,
            maxValue: 900
        )
        project.variableConfig.axes.append(weightAxis)

        // Add width axis
        let widthAxis = VariationAxis(
            tag: "wdth",
            name: "Width",
            minValue: 75,
            defaultValue: 100,
            maxValue: 125
        )
        project.variableConfig.axes.append(widthAxis)

        // Enable variable font
        project.variableConfig.isVariableFont = true

        #expect(project.variableConfig.axes.count == 2)
        #expect(project.variableConfig.axes[0].tag == "wght")
        #expect(project.variableConfig.axes[1].tag == "wdth")
        #expect(project.variableConfig.isVariableFont)

        // Verify axis ranges
        let wghtAxis = project.variableConfig.axes.first { $0.tag == "wght" }
        #expect(wghtAxis?.minValue == 100)
        #expect(wghtAxis?.maxValue == 900)
        #expect(wghtAxis?.defaultValue == 400)

        // =====================================================
        // PHASE 3: Create base glyphs (for default master)
        // =====================================================

        // Create H glyph for default master (Regular weight)
        let hRegularOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 50, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 130, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 130, y: 300), type: .corner),
                PathPoint(position: CGPoint(x: 370, y: 300), type: .corner),
                PathPoint(position: CGPoint(x: 370, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 450, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 450, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 370, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 370, y: 380), type: .corner),
                PathPoint(position: CGPoint(x: 130, y: 380), type: .corner),
                PathPoint(position: CGPoint(x: 130, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 50, y: 700), type: .corner)
            ], isClosed: true)
        ])
        let hRegularGlyph = Glyph(character: "H", outline: hRegularOutline, advanceWidth: 500, leftSideBearing: 50)
        project.glyphs["H"] = hRegularGlyph

        // Create O glyph for default master
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
                PathPoint(position: CGPoint(x: 130, y: 350), type: .smooth),
                PathPoint(position: CGPoint(x: 250, y: 580), type: .smooth),
                PathPoint(position: CGPoint(x: 370, y: 350), type: .smooth),
                PathPoint(position: CGPoint(x: 250, y: 120), type: .smooth)
            ],
            isClosed: true
        )
        let oRegularOutline = GlyphOutline(contours: [oOuterContour, oInnerContour])
        project.glyphs["O"] = Glyph(character: "O", outline: oRegularOutline, advanceWidth: 500, leftSideBearing: 50)

        #expect(project.glyphs.count == 2)

        // =====================================================
        // PHASE 4: Define masters with their own glyph sets
        // =====================================================

        // Light H - thinner stems
        let hLightOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 70, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 110, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 110, y: 320), type: .corner),
                PathPoint(position: CGPoint(x: 390, y: 320), type: .corner),
                PathPoint(position: CGPoint(x: 390, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 430, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 430, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 390, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 390, y: 360), type: .corner),
                PathPoint(position: CGPoint(x: 110, y: 360), type: .corner),
                PathPoint(position: CGPoint(x: 110, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 70, y: 700), type: .corner)
            ], isClosed: true)
        ])
        let hLightGlyph = Glyph(character: "H", outline: hLightOutline, advanceWidth: 500, leftSideBearing: 70)

        // Bold H - thicker stems
        let hBoldOutline = GlyphOutline(contours: [
            Contour(points: [
                PathPoint(position: CGPoint(x: 30, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 170, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 170, y: 280), type: .corner),
                PathPoint(position: CGPoint(x: 330, y: 280), type: .corner),
                PathPoint(position: CGPoint(x: 330, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 470, y: 0), type: .corner),
                PathPoint(position: CGPoint(x: 470, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 330, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 330, y: 420), type: .corner),
                PathPoint(position: CGPoint(x: 170, y: 420), type: .corner),
                PathPoint(position: CGPoint(x: 170, y: 700), type: .corner),
                PathPoint(position: CGPoint(x: 30, y: 700), type: .corner)
            ], isClosed: true)
        ])
        let hBoldGlyph = Glyph(character: "H", outline: hBoldOutline, advanceWidth: 500, leftSideBearing: 30)

        // Create masters with their glyphs
        let lightMaster = FontMaster(
            name: "Light",
            location: ["wght": 100, "wdth": 100],
            glyphs: ["H": hLightGlyph],
            metrics: project.metrics
        )
        project.variableConfig.masters.append(lightMaster)

        let regularMaster = FontMaster(
            name: "Regular",
            location: ["wght": 400, "wdth": 100],
            glyphs: ["H": hRegularGlyph],
            metrics: project.metrics
        )
        project.variableConfig.masters.append(regularMaster)

        let boldMaster = FontMaster(
            name: "Bold",
            location: ["wght": 900, "wdth": 100],
            glyphs: ["H": hBoldGlyph],
            metrics: project.metrics
        )
        project.variableConfig.masters.append(boldMaster)

        #expect(project.variableConfig.masters.count == 3)

        // =====================================================
        // PHASE 5: Calculate glyph variations (deltas)
        // =====================================================

        // Calculate variation from regular to light
        let lightVariation = GlyphVariation.calculate(
            character: "H",
            source: hRegularGlyph,
            target: hLightGlyph,
            sourceMasterID: regularMaster.id,
            targetMasterID: lightMaster.id
        )

        #expect(lightVariation != nil)
        #expect(lightVariation?.pointDeltas.count == 1)  // One contour

        // Calculate variation from regular to bold
        let boldVariation = GlyphVariation.calculate(
            character: "H",
            source: hRegularGlyph,
            target: hBoldGlyph,
            sourceMasterID: regularMaster.id,
            targetMasterID: boldMaster.id
        )

        #expect(boldVariation != nil)

        // =====================================================
        // PHASE 6: Verify interpolation
        // =====================================================

        // Test applying variation at 50%
        if let variation = boldVariation {
            let interpolated = variation.apply(to: hRegularGlyph, factor: 0.5)
            #expect(interpolated.outline.contours.count == hRegularGlyph.outline.contours.count)
        }

        // =====================================================
        // PHASE 7: Add named instances
        // =====================================================

        project.variableConfig.instances.append(NamedInstance.thin())
        project.variableConfig.instances.append(NamedInstance.light())
        project.variableConfig.instances.append(NamedInstance.regular())
        project.variableConfig.instances.append(NamedInstance.bold())
        project.variableConfig.instances.append(NamedInstance.black())

        #expect(project.variableConfig.instances.count == 5)

        // =====================================================
        // PHASE 8: Verify variable font configuration
        // =====================================================

        #expect(project.variableConfig.isVariableFont)
        #expect(project.variableConfig.axes.count == 2)
        #expect(project.variableConfig.masters.count == 3)
        #expect(project.variableConfig.instances.count == 5)

        // Verify master has glyphs
        let regular = project.variableConfig.masters.first { $0.name == "Regular" }
        #expect(regular?.glyphs.count == 1)
        #expect(regular?.glyphs["H"] != nil)

        // =====================================================
        // PHASE 9: Test preset configurations
        // =====================================================

        // Test weight-only preset
        let weightOnlyConfig = VariableFontConfig.weightOnly()
        #expect(weightOnlyConfig.isVariableFont)
        #expect(weightOnlyConfig.axes.count == 1)
        #expect(weightOnlyConfig.axes[0].tag == "wght")
        #expect(weightOnlyConfig.instances.count >= 3)

        // Test weight + width preset
        let weightWidthConfig = VariableFontConfig.weightAndWidth()
        #expect(weightWidthConfig.axes.count == 2)
        #expect(weightWidthConfig.masters.count == 4)  // 2x2 corners

        // =====================================================
        // PHASE 10: Export as variable font
        // =====================================================

        let exporter = FontExporter()
        let options = FontExporter.ExportOptions(format: .ttf, exportAsVariable: true)
        let ttfData = try await exporter.export(project: project, options: options)

        #expect(!ttfData.isEmpty)

        // Validate TTF magic bytes
        let magic = ttfData.prefix(4).withUnsafeBytes { $0.load(as: UInt32.self) }
        #expect(UInt32(bigEndian: magic) == 0x00010000, "Should have TrueType magic bytes")

        // =====================================================
        // PHASE 11: Export to UFO
        // =====================================================

        let ufoExporter = UFOExporter()
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("var_test_\(UUID().uuidString).ufo")
        try await ufoExporter.export(project: project, to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))
        try? FileManager.default.removeItem(at: tempURL)

        // =====================================================
        // VERIFICATION: Complete variable font workflow succeeded
        // =====================================================

        #expect(project.glyphs.count == 2)
        #expect(project.variableConfig.axes.count == 2)
        #expect(project.variableConfig.masters.count == 3)
        #expect(project.variableConfig.instances.count == 5)
    }
}
